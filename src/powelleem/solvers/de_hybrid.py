"""DEHybrid solver — DE outer loop + NEWUOA polish.

Reproduces the exact strategy of the MATLAB ``DE_UOA_FINAL.m`` driver:

1. **Latin Hypercube Sampling** of ``NP=400`` initial points within bounds.
2. **Differential Evolution** outer loop (``DE_MUT.m`` style) for ``n_iter``
   generations, occasionally polishing promising points with NEWUOA when
   the molecule-averaged Pearson r² exceeds ``polish_r2_threshold``.
3. **Final NEWUOA polish** of the best individual.

This is the "honest" baseline that we benchmark the analytical-Jacobian
:class:`AnalyticLM` against — same algorithm as the MATLAB original,
just modernised.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from powelleem.solvers.base import Solver, SolverConfig, latin_hypercube_initial_xs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


class DEHybrid(Solver):
    """DE outer + NEWUOA polish — matches MATLAB ``DE_UOA_FINAL.m``."""

    name = "DEHybrid"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        population_size: int = 400,
        n_generations: int = 1000,
        mutation_F: float = 0.5,
        crossover_CR: float = 0.7,
        polish_r2_threshold: float = 0.6,
        polish_max_fev: int = 200,
        final_polish_max_fev: int = 20000,
        bound_penalty: float = 1e3,
    ) -> None:
        super().__init__(config)
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_F = mutation_F
        self.crossover_CR = crossover_CR
        self.polish_r2_threshold = polish_r2_threshold
        self.polish_max_fev = polish_max_fev
        self.final_polish_max_fev = final_polish_max_fev
        self.bound_penalty = bound_penalty

    def fit(
        self,
        model: EEMModel,
        dataset: Dataset,
        *,
        x0: NDArray[np.float64] | None = None,
    ) -> FitResult:
        from powelleem.jacobian import residuals_and_jacobian
        from powelleem.types import FitResult, ParamSet

        # PRIMA is optional — DE itself works without it, but the polish does need it.
        try:
            from prima import minimize as prima_minimize  # type: ignore

            have_prima = True
        except ImportError:
            have_prima = False

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)

        # ----- Stage 0: LHS initial population --------------------------
        population = latin_hypercube_initial_xs(n_types, self.config, self.population_size)
        if x0 is not None:
            population[0] = np.clip(x0, lo, hi)

        # ----- Helpers --------------------------------------------------
        def evaluate(x: NDArray[np.float64]) -> tuple[float, float]:
            """Return (rmse, mean_pearson_r) over the dataset."""
            r, _ = residuals_and_jacobian(x, dataset, n_types)
            rmse = float(np.sqrt((r * r).mean()))
            # Per-molecule Pearson r aggregated like in structureB.cpp:calculate_r
            offset = 0
            r_acc = 0.0
            r_count = 0
            for mol in dataset.molecules:
                n = mol.n_atoms
                qp = mol.target_charges + r[offset : offset + n]
                offset += n
                qr = mol.target_charges
                dx = qp - qp.mean()
                dy = qr - qr.mean()
                cov_xx = float((dx * dx).sum())
                cov_yy = float((dy * dy).sum())
                if cov_xx * cov_yy > 1e-12:
                    r_acc += float((dx * dy).sum()) / np.sqrt(cov_xx * cov_yy)
                    r_count += 1
            r_mean = r_acc / r_count if r_count else 0.0
            return rmse, r_mean

        fitness = np.array([evaluate(x)[0] for x in population])
        best_idx = int(np.argmin(fitness))
        best_x = population[best_idx].copy()
        best_f = float(fitness[best_idx])
        loss0 = float(fitness.mean())

        trajectory: list[float] = [best_f]
        rng = np.random.default_rng(self.config.seed)
        loss_evals = {"count": self.population_size}

        t_start = time.perf_counter()

        # ----- Stage 1: DE outer loop ----------------------------------
        for gen in range(self.n_generations):
            for i in range(self.population_size):
                # DE/rand/1/bin
                idxs = rng.choice(
                    np.delete(np.arange(self.population_size), i), size=3, replace=False
                )
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                v = a + self.mutation_F * (b - c)
                v = np.clip(v, lo, hi)
                # Crossover
                mask = rng.random(v.shape) < self.crossover_CR
                trial = np.where(mask, v, population[i])
                rmse_trial, r_trial = evaluate(trial)
                loss_evals["count"] += 1
                if rmse_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = rmse_trial
                    if rmse_trial < best_f:
                        best_f = rmse_trial
                        best_x = trial.copy()
                        # Polish promising candidates with NEWUOA
                        if have_prima and r_trial > self.polish_r2_threshold:
                            best_x, best_f = self._polish(
                                best_x, best_f, dataset, n_types, lo, hi,
                                prima_minimize, self.polish_max_fev,
                            )
            trajectory.append(best_f)

        # ----- Stage 2: final NEWUOA polish on best individual ----------
        if have_prima:
            best_x, best_f = self._polish(
                best_x, best_f, dataset, n_types, lo, hi,
                prima_minimize, self.final_polish_max_fev,
            )

        wall = time.perf_counter() - t_start

        rmse = best_f
        loss_final = rmse * rmse
        params = ParamSet.from_vector(best_x, model.atom_types)
        return FitResult(
            params=params,
            rmse=rmse,
            loss_initial=loss0,
            loss_final=loss_final,
            solver_name=self.name,
            solver_metadata={
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                "mutation_F": self.mutation_F,
                "crossover_CR": self.crossover_CR,
                "polish_r2_threshold": self.polish_r2_threshold,
                "have_prima": have_prima,
            },
            wall_time_s=wall,
            n_function_evals=loss_evals["count"],
            n_jacobian_evals=0,
            converged=True,
            message="DE+NEWUOA hybrid finished",
            loss_trajectory=trajectory,
        )

    @staticmethod
    def _polish(
        x: NDArray[np.float64],
        f_x: float,
        dataset: Dataset,
        n_types: int,
        lo: NDArray[np.float64],
        hi: NDArray[np.float64],
        prima_minimize,  # type: ignore[no-untyped-def]
        max_fev: int,
    ) -> tuple[NDArray[np.float64], float]:
        from powelleem.jacobian import residuals_and_jacobian

        bound_penalty = 1e3

        def obj(z: NDArray[np.float64]) -> float:
            r, _ = residuals_and_jacobian(z, dataset, n_types)
            loss = float((r * r).mean())
            pen = float(
                (np.maximum(0.0, lo - z) ** 2).sum()
                + (np.maximum(0.0, z - hi) ** 2).sum()
            )
            return loss + bound_penalty * pen

        rhobeg = float(np.min(hi - lo) / 20)
        res = prima_minimize(
            obj, x, method="newuoa",
            options={"maxfev": max_fev, "rhobeg": rhobeg, "rhoend": 1e-6},
        )
        x_new = np.clip(res.x, lo, hi)
        r_new, _ = residuals_and_jacobian(x_new, dataset, n_types)
        f_new = float(np.sqrt((r_new * r_new).mean()))
        if f_new < f_x:
            return x_new, f_new
        return x, f_x
