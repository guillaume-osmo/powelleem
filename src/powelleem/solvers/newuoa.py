"""Newuoa solver — Powell's NEWUOA via PDFO.

NEWUOA (NEW Unconstrained Optimization Algorithm, Powell 2006) is the
derivative-free trust-region solver used by the original MATLAB
``DE_UOA_FINAL.m`` pipeline. PDFO (Ragonneau & Zhang 2024) is the
modern Fortran reference re-implementation with Python bindings.

NEWUOA is *unconstrained*; bound constraints are emulated by adding a
quadratic penalty when the parameters drift outside ``[lo, hi]``.
For native bound handling prefer :class:`Bobyqa`.

References
----------
* Powell, M. J. D. *The NEWUOA software for unconstrained optimization
  without derivatives.* In *Large-Scale Nonlinear Optimization*,
  Springer 2006, 255–297.
* Ragonneau & Zhang, *PDFO*, Math. Prog. Comput. 2024.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from powelleem.solvers.base import Solver, SolverConfig, random_initial_x

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


class Newuoa(Solver):
    """Powell's NEWUOA via PDFO, with quadratic bound penalty."""

    name = "Newuoa"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        max_fev: int = 10000,
        bound_penalty: float = 1e3,
        radius_init: float | None = None,
        radius_final: float = 1e-6,
    ) -> None:
        super().__init__(config)
        self.max_fev = max_fev
        self.bound_penalty = bound_penalty
        self.radius_init = radius_init
        self.radius_final = radius_final

    def fit(
        self,
        model: EEMModel,
        dataset: Dataset,
        *,
        x0: NDArray[np.float64] | None = None,
    ) -> FitResult:
        try:
            from pdfo import pdfo  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PDFO is required for the NEWUOA solver. "
                "Install with `pip install powelleem[powell]`."
            ) from exc

        from powelleem.jacobian import residuals_and_jacobian
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)
        if x0 is None:
            x0 = random_initial_x(n_types, self.config)

        trajectory: list[float] = []
        loss_evals = {"count": 0}

        def objective(x: NDArray[np.float64]) -> float:
            r, _ = residuals_and_jacobian(x, dataset, n_types)
            loss = float((r * r).mean())
            # Quadratic penalty for bound violation (NEWUOA is unconstrained).
            penalty = float(
                (np.maximum(0.0, lo - x) ** 2).sum()
                + (np.maximum(0.0, x - hi) ** 2).sum()
            )
            total = loss + self.bound_penalty * penalty
            loss_evals["count"] += 1
            trajectory.append(loss)
            return total

        loss0 = objective(x0.copy())

        radius_init = (
            self.radius_init if self.radius_init is not None else float(np.min(hi - lo) / 10)
        )
        options = {
            "maxfev": self.max_fev,
            "radius_init": radius_init,
            "radius_final": self.radius_final,
        }

        t0 = time.perf_counter()
        res = pdfo(objective, x0, method="newuoa", options=options)
        wall = time.perf_counter() - t0

        x_opt = np.clip(res.x, lo, hi)
        r_final, _ = residuals_and_jacobian(x_opt, dataset, n_types)
        loss_final = float((r_final * r_final).mean())
        rmse = float(np.sqrt(loss_final))

        params = ParamSet.from_vector(x_opt, model.atom_types)
        return FitResult(
            params=params,
            rmse=rmse,
            loss_initial=loss0,
            loss_final=loss_final,
            solver_name=self.name,
            solver_metadata={
                "max_fev": self.max_fev,
                "radius_init": radius_init,
                "radius_final": self.radius_final,
                "bound_penalty": self.bound_penalty,
                "pdfo_status": int(getattr(res, "status", -1)),
                "pdfo_message": str(getattr(res, "message", "")),
                "n_fev_pdfo": int(getattr(res, "nfev", 0)),
            },
            wall_time_s=wall,
            n_function_evals=loss_evals["count"],
            n_jacobian_evals=0,
            converged=bool(getattr(res, "success", False)),
            message=str(getattr(res, "message", "")),
            loss_trajectory=trajectory,
        )
