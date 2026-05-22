"""AnalyticLM solver — NumPy + analytic Jacobian + L-BFGS-B → TRF/LM.

The reference solver: exploits the analytical Jacobian computed via the
implicit function theorem. Two-stage strategy:

1. **Warm phase** — L-BFGS-B on the SSE loss using analytic gradient
   ``∇L = (2/N) Jᵀ r``. Robust to far-from-optimum starts, handles
   the bound constraints natively.
2. **Polish phase** — :func:`scipy.optimize.least_squares` with
   ``method="trf"`` (Trust-Region-Reflective, LM-like with bounds)
   using the analytic Jacobian. Quadratic convergence near the optimum.

On the 100-mol CHAOS benchmark this is roughly 8× faster than the JAX
autodiff variants while reaching lower RMSE.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as so

from powelleem.jacobian import loss_and_grad, residuals_and_jacobian
from powelleem.solvers.base import Solver, SolverConfig, random_initial_x

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


class AnalyticLM(Solver):
    """NumPy + analytic Jacobian + L-BFGS-B → TRF/LM solver.

    Parameters
    ----------
    config
        :class:`SolverConfig` (bounds, max_iter, seed, verbose).
    maxiter_lbfgs
        L-BFGS-B warm iterations.
    maxiter_lm
        Maximum ``nfev`` for the TRF/LM polish.
    skip_lbfgs
        If ``True``, jumps straight to TRF/LM from ``x0`` (useful when
        you have a warm start already).
    """

    name = "AnalyticLM"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        maxiter_lbfgs: int = 200,
        maxiter_lm: int = 200,
        skip_lbfgs: bool = False,
    ) -> None:
        super().__init__(config)
        self.maxiter_lbfgs = maxiter_lbfgs
        self.maxiter_lm = maxiter_lm
        self.skip_lbfgs = skip_lbfgs

    def fit(
        self,
        model: EEMModel,
        dataset: Dataset,
        *,
        x0: NDArray[np.float64] | None = None,
    ) -> FitResult:
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)
        if x0 is None:
            x0 = random_initial_x(n_types, self.config)

        trajectory: list[float] = []

        # Wrap loss/grad to also record convergence trajectory.
        loss_evals = {"count": 0}
        jac_evals = {"count": 0}

        def loss_grad(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            l, g = loss_and_grad(x, dataset, n_types)
            loss_evals["count"] += 1
            jac_evals["count"] += 1
            trajectory.append(l)
            return l, g

        loss0 = loss_grad(x0)[0]

        # -------- Phase 1: L-BFGS-B warm start --------
        t0 = time.perf_counter()
        if not self.skip_lbfgs:
            res_lbfgs = so.minimize(
                loss_grad,
                x0,
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(lo, hi, strict=True)),
                options={"maxiter": self.maxiter_lbfgs},
            )
            x_warm = res_lbfgs.x
            loss_warm = float(res_lbfgs.fun)
            n_iter_lbfgs = int(res_lbfgs.nit)
        else:
            x_warm = x0
            loss_warm = loss0
            n_iter_lbfgs = 0
        t_lbfgs = time.perf_counter() - t0

        # -------- Phase 2: TRF/LM polish using analytic Jacobian --------
        def res_fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            r, _ = residuals_and_jacobian(x, dataset, n_types)
            loss_evals["count"] += 1
            trajectory.append(float((r * r).mean()))
            return r

        def jac_fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            _, J = residuals_and_jacobian(x, dataset, n_types)
            jac_evals["count"] += 1
            return J

        t0 = time.perf_counter()
        res_lm = so.least_squares(
            fun=res_fn,
            x0=x_warm,
            jac=jac_fn,
            bounds=(lo, hi),
            method="trf",
            max_nfev=self.maxiter_lm,
        )
        t_lm = time.perf_counter() - t0

        x_opt = res_lm.x
        loss_final = float((res_lm.fun * res_lm.fun).mean())
        rmse = float(np.sqrt(loss_final))

        params = ParamSet.from_vector(x_opt, model.atom_types)
        return FitResult(
            params=params,
            rmse=rmse,
            loss_initial=float(loss0),
            loss_final=loss_final,
            solver_name=self.name,
            solver_metadata={
                "maxiter_lbfgs": self.maxiter_lbfgs,
                "maxiter_lm": self.maxiter_lm,
                "loss_after_lbfgs": loss_warm,
                "n_iter_lbfgs": n_iter_lbfgs,
                "lm_status": int(res_lm.status),
                "lm_message": str(res_lm.message),
                "wall_lbfgs_s": t_lbfgs,
                "wall_lm_s": t_lm,
            },
            wall_time_s=t_lbfgs + t_lm,
            n_function_evals=loss_evals["count"],
            n_jacobian_evals=jac_evals["count"],
            converged=bool(res_lm.success),
            message=str(res_lm.message),
            loss_trajectory=trajectory,
        )
