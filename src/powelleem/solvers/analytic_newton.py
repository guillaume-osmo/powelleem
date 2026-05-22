"""AnalyticNewton solver — full Hessian via implicit-fn theorem + trust-region Newton.

Uses the analytical Hessian (from :mod:`powelleem.hessian`) and SciPy's
``trust-ncg`` / ``trust-exact`` to take second-order Newton steps. This
should converge quadratically near the optimum and out-perform the
Gauss-Newton approximation (LM/TRF) on datasets where the residuals
are not vanishingly small at the minimum.

Strategy:

1. **Warm phase** — L-BFGS-B with analytic gradient (quick descent
   to a basin), capped at ``maxiter_lbfgs``.
2. **Newton phase** — ``scipy.optimize.minimize(method='trust-exact')``
   with our analytical Hessian. ``trust-exact`` is robust to
   indefinite Hessians (it solves the constrained subproblem exactly).

Bounds are emulated via a quadratic boundary penalty (small constant)
since trust-exact doesn't take bounds natively. Final clip ensures we
stay inside the box.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as so

from powelleem.solvers.base import Solver, SolverConfig, random_initial_x

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


class AnalyticNewton(Solver):
    """Trust-region Newton with analytical Jacobian + Hessian via implicit-fn theorem."""

    name = "AnalyticNewton"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        maxiter_lbfgs: int = 100,
        maxiter_newton: int = 200,
        bound_penalty: float = 1e2,
        method: str = "trust-exact",
        skip_lbfgs: bool = False,
    ) -> None:
        super().__init__(config)
        self.maxiter_lbfgs = maxiter_lbfgs
        self.maxiter_newton = maxiter_newton
        self.bound_penalty = bound_penalty
        self.method = method
        self.skip_lbfgs = skip_lbfgs

    def fit(
        self,
        model: EEMModel,
        dataset: Dataset,
        *,
        x0: NDArray[np.float64] | None = None,
    ) -> FitResult:
        from powelleem.hessian import loss_grad_hessian
        from powelleem.jacobian import loss_and_grad
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)
        if x0 is None:
            x0 = random_initial_x(n_types, self.config)

        # ----- closures with smooth boundary penalty -----
        def penalty(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
            """(p_value, p_grad, p_hess_diag) for a quadratic boundary penalty."""
            below = np.maximum(0.0, lo - x)
            above = np.maximum(0.0, x - hi)
            p_val = float((below * below).sum() + (above * above).sum())
            p_grad = -2.0 * below + 2.0 * above
            p_hess = 2.0 * ((below > 0) | (above > 0)).astype(np.float64)
            return p_val, p_grad, p_hess

        trajectory: list[float] = []
        loss_evals = {"count": 0}
        jac_evals = {"count": 0}
        hess_evals = {"count": 0}

        def loss_grad_with_pen(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            l, g = loss_and_grad(x, dataset, n_types)
            pv, pg, _ = penalty(x)
            loss_evals["count"] += 1
            jac_evals["count"] += 1
            trajectory.append(l)
            return l + self.bound_penalty * pv, g + self.bound_penalty * pg

        def hess_with_pen(x: NDArray[np.float64]) -> NDArray[np.float64]:
            _, _, H = loss_grad_hessian(x, dataset, n_types)
            _, _, ph = penalty(x)
            hess_evals["count"] += 1
            return H + self.bound_penalty * np.diag(ph)

        loss0 = loss_grad_with_pen(x0)[0]

        # ----- Phase 1: L-BFGS-B warm start -----
        t0 = time.perf_counter()
        if not self.skip_lbfgs:
            res_l = so.minimize(
                loss_grad_with_pen,
                x0,
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(lo, hi, strict=True)),
                options={"maxiter": self.maxiter_lbfgs},
            )
            x_warm = res_l.x
            loss_warm = float(res_l.fun)
            n_iter_lbfgs = int(res_l.nit)
        else:
            x_warm = x0
            loss_warm = loss0
            n_iter_lbfgs = 0
        t_lbfgs = time.perf_counter() - t0

        # ----- Phase 2: trust-region Newton with full Hessian -----
        def loss_only(x: NDArray[np.float64]) -> float:
            return loss_grad_with_pen(x)[0]

        def grad_only(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return loss_grad_with_pen(x)[1]

        t0 = time.perf_counter()
        res_n = so.minimize(
            loss_only,
            x_warm,
            jac=grad_only,
            hess=hess_with_pen,
            method=self.method,
            options={"maxiter": self.maxiter_newton},
        )
        t_newton = time.perf_counter() - t0

        x_opt = np.clip(res_n.x, lo, hi)
        # Evaluate the pure (no-penalty) loss at the final point.
        l_final, _ = loss_and_grad(x_opt, dataset, n_types)
        rmse = float(np.sqrt(l_final))

        params = ParamSet.from_vector(x_opt, model.atom_types)
        return FitResult(
            params=params,
            rmse=rmse,
            loss_initial=loss0,
            loss_final=l_final,
            solver_name=self.name,
            solver_metadata={
                "maxiter_lbfgs": self.maxiter_lbfgs,
                "maxiter_newton": self.maxiter_newton,
                "newton_method": self.method,
                "n_iter_lbfgs": n_iter_lbfgs,
                "n_iter_newton": int(res_n.nit),
                "newton_status": int(res_n.status),
                "newton_message": str(res_n.message),
                "loss_after_lbfgs": loss_warm,
                "wall_lbfgs_s": t_lbfgs,
                "wall_newton_s": t_newton,
            },
            wall_time_s=t_lbfgs + t_newton,
            n_function_evals=loss_evals["count"],
            n_jacobian_evals=jac_evals["count"],
            converged=bool(res_n.success),
            message=str(res_n.message),
            loss_trajectory=trajectory,
        )
