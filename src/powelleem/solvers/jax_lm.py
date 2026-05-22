"""JaxLM solver — JAX autodiff + L-BFGS-B → TRF/LM polish.

Same two-stage strategy as :class:`AnalyticLM` but with the Jacobian
obtained via :func:`jax.jacrev` (reverse-mode autodiff through
``jax.numpy.linalg.solve``). Useful when the analytical Jacobian is hard
to write (richer parametrisations, additional regularisers, etc.) but
generally slower than the analytic variant on small problems because
of JIT compile + trace overhead.
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


class JaxLM(Solver):
    """JAX autodiff + L-BFGS-B → TRF/LM (autodiff variant of :class:`AnalyticLM`)."""

    name = "JaxLM"

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
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "JAX is required for JaxLM. Install with `pip install powelleem[jax]`."
            ) from exc

        from powelleem.model import predict_charges_jax_factory
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)
        if x0 is None:
            x0 = random_initial_x(n_types, self.config)

        predict_one = predict_charges_jax_factory(n_types)
        blocks = tuple(
            (
                jnp.asarray(m.inv_r),
                jnp.asarray(m.atom_types),
                jnp.asarray(m.target_charges),
                float(m.formal_charge),
                int(m.n_atoms),
            )
            for m in dataset.molecules
        )

        def residuals(x):  # type: ignore[no-untyped-def]
            rs = []
            for inv_r, type_idx, target, q_tot, n in blocks:
                q = predict_one(x, inv_r, type_idx, q_tot, n)
                rs.append(q - target)
            return jnp.concatenate(rs)

        def loss(x):  # type: ignore[no-untyped-def]
            r = residuals(x)
            return jnp.mean(r * r)

        loss_jit = jax.jit(loss)
        grad_jit = jax.jit(jax.grad(loss))
        res_jit = jax.jit(residuals)
        jac_jit = jax.jit(jax.jacrev(residuals))

        np_loss = lambda x: float(loss_jit(x))  # noqa: E731
        np_grad = lambda x: np.asarray(grad_jit(x))  # noqa: E731
        np_res = lambda x: np.asarray(res_jit(x))  # noqa: E731
        np_jac = lambda x: np.asarray(jac_jit(x))  # noqa: E731

        trajectory: list[float] = []

        def loss_recording(x: NDArray[np.float64]) -> float:
            v = np_loss(x)
            trajectory.append(v)
            return v

        loss0 = loss_recording(x0)

        # ---- Phase 1: L-BFGS-B warm start ----
        t0 = time.perf_counter()
        if not self.skip_lbfgs:
            res_lbfgs = so.minimize(
                loss_recording,
                x0,
                jac=np_grad,
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

        # ---- Phase 2: TRF/LM polish ----
        t0 = time.perf_counter()
        res_lm = so.least_squares(
            fun=np_res,
            x0=x_warm,
            jac=np_jac,
            bounds=(lo, hi),
            method="trf",
            max_nfev=self.maxiter_lm,
        )
        t_lm = time.perf_counter() - t0

        x_opt = res_lm.x
        loss_final = float((res_lm.fun ** 2).mean())
        rmse = float(np.sqrt(loss_final))

        params = ParamSet.from_vector(x_opt, model.atom_types)
        return FitResult(
            params=params,
            rmse=rmse,
            loss_initial=loss0,
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
            n_function_evals=len(trajectory) + int(res_lm.nfev),
            n_jacobian_evals=n_iter_lbfgs + int(res_lm.njev or 0),
            converged=bool(res_lm.success),
            message=str(res_lm.message),
            loss_trajectory=trajectory,
        )
