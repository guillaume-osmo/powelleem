"""Bobyqa solver — Powell's BOBYQA via PDFO.

BOBYQA (Bound Optimization BY Quadratic Approximation, Powell 2009) is
the bound-constrained sibling of NEWUOA. Same derivative-free
trust-region machinery, but handles ``[lo, hi]`` boxes natively so no
penalty term is needed.

Reference
---------
* Powell, M. J. D. *The BOBYQA algorithm for bound constrained
  optimization without derivatives.* Cambridge NA Report NA2009/06.
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


class Bobyqa(Solver):
    """Powell's BOBYQA via PDFO (native bound constraints)."""

    name = "Bobyqa"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        max_fev: int = 10000,
        radius_init: float | None = None,
        radius_final: float = 1e-6,
    ) -> None:
        super().__init__(config)
        self.max_fev = max_fev
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
                "PDFO is required for the BOBYQA solver. "
                "Install with `pip install powelleem[powell]`."
            ) from exc

        from powelleem.jacobian import residuals_and_jacobian
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)
        if x0 is None:
            x0 = random_initial_x(n_types, self.config)
        x0 = np.clip(x0, lo, hi)

        trajectory: list[float] = []
        loss_evals = {"count": 0}

        def objective(x: NDArray[np.float64]) -> float:
            r, _ = residuals_and_jacobian(x, dataset, n_types)
            loss = float((r * r).mean())
            loss_evals["count"] += 1
            trajectory.append(loss)
            return loss

        loss0 = objective(x0.copy())
        radius_init = (
            self.radius_init if self.radius_init is not None else float(np.min(hi - lo) / 10)
        )
        options = {
            "maxfev": self.max_fev,
            "radius_init": radius_init,
            "radius_final": self.radius_final,
        }

        # PDFO accepts bounds as a list of (lo, hi) scalar tuples.
        bounds_pairs = list(zip(lo.tolist(), hi.tolist(), strict=True))
        t0 = time.perf_counter()
        res = pdfo(
            objective, x0, method="bobyqa",
            bounds=bounds_pairs,
            options=options,
        )
        wall = time.perf_counter() - t0

        x_opt = res.x
        loss_final = float(objective(x_opt))
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
