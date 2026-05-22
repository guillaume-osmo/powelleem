"""DENewton solver — DE global search bridged with analytical-Newton polish.

The motivation, from the NEEMP set01 benchmark:

* `DEHybrid`        : RMSE 0.094, wall 68 s, κ = 0.44   ← global minimum
* `AnalyticNewton`  : RMSE 0.366, wall 3.6 s, κ = 0.53  ← right κ basin, wrong α/β still

DE finds the right basin; AnalyticNewton refines a basin to its true
minimum using the analytical Hessian (quadratic convergence). The
hybrid:

1. Run differential evolution with a small population (default 50) for a
   limited number of generations (default 20) — cheap exploration that
   discovers the κ-basin within ~5 s of wall.
2. Pick the best individual.
3. Polish it with :class:`AnalyticNewton` (L-BFGS-B warm + trust-exact
   Newton with full analytical Hessian) — ~3-5 s of wall.

Total wall ≈ 10 s for the best-of-both-worlds outcome. The DE phase
escapes the local-minimum trap that catches all first-order methods;
the Newton polish leverages curvature to drop the RMSE the rest of the
way without paying for thousands of DE/NEWUOA evals.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from powelleem.solvers.analytic_newton import AnalyticNewton
from powelleem.solvers.base import Solver, SolverConfig, latin_hypercube_initial_xs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


class DENewton(Solver):
    """DE global search + AnalyticNewton polish — bridges DEHybrid and AnalyticNewton."""

    name = "DENewton"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_F: float = 0.5,
        crossover_CR: float = 0.7,
        # Newton polish stage
        maxiter_lbfgs: int = 100,
        maxiter_newton: int = 100,
    ) -> None:
        super().__init__(config)
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_F = mutation_F
        self.crossover_CR = crossover_CR
        self.maxiter_lbfgs = maxiter_lbfgs
        self.maxiter_newton = maxiter_newton

    def fit(
        self,
        model: EEMModel,
        dataset: Dataset,
        *,
        x0: NDArray[np.float64] | None = None,
    ) -> FitResult:
        from powelleem.jacobian import residuals_and_jacobian
        from powelleem.types import FitResult

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)

        # ---------- Stage 1: DE global search ----------
        rng = np.random.default_rng(self.config.seed)
        population = latin_hypercube_initial_xs(n_types, self.config, self.population_size)
        if x0 is not None:
            population[0] = np.clip(x0, lo, hi)

        def evaluate(x: NDArray[np.float64]) -> float:
            r, _ = residuals_and_jacobian(x, dataset, n_types)
            return float(np.sqrt((r * r).mean()))

        fitness = np.array([evaluate(x) for x in population])
        best_idx = int(np.argmin(fitness))
        best_x = population[best_idx].copy()
        best_f = float(fitness[best_idx])
        loss0 = float(fitness.mean() ** 2)

        trajectory: list[float] = [best_f * best_f]
        n_de_evals = self.population_size

        t_de_start = time.perf_counter()
        for _gen in range(self.n_generations):
            for i in range(self.population_size):
                idxs = rng.choice(
                    np.delete(np.arange(self.population_size), i), size=3, replace=False
                )
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                v = a + self.mutation_F * (b - c)
                v = np.clip(v, lo, hi)
                mask = rng.random(v.shape) < self.crossover_CR
                trial = np.where(mask, v, population[i])
                trial_fitness = evaluate(trial)
                n_de_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_f:
                        best_f = trial_fitness
                        best_x = trial.copy()
            trajectory.append(best_f * best_f)
        t_de = time.perf_counter() - t_de_start

        # ---------- Stage 2: AnalyticNewton polish ----------
        polish_solver = AnalyticNewton(
            config=SolverConfig(seed=self.config.seed),
            maxiter_lbfgs=self.maxiter_lbfgs,
            maxiter_newton=self.maxiter_newton,
        )
        t_newton_start = time.perf_counter()
        polish_result = polish_solver.fit(model, dataset, x0=best_x)
        t_newton = time.perf_counter() - t_newton_start

        # Merge trajectories
        trajectory.extend(polish_result.loss_trajectory)

        return FitResult(
            params=polish_result.params,
            rmse=polish_result.rmse,
            loss_initial=loss0,
            loss_final=polish_result.loss_final,
            solver_name=self.name,
            solver_metadata={
                "stage1_de_rmse": best_f,
                "stage1_n_evals": n_de_evals,
                "stage1_wall_s": t_de,
                "stage2_rmse": polish_result.rmse,
                "stage2_wall_s": t_newton,
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                **{"newton_" + k: v for k, v in polish_result.solver_metadata.items()},
            },
            wall_time_s=t_de + t_newton,
            n_function_evals=n_de_evals + polish_result.n_function_evals,
            n_jacobian_evals=polish_result.n_jacobian_evals,
            converged=polish_result.converged,
            message=f"DE({n_de_evals} evals) → {polish_result.message}",
            loss_trajectory=trajectory,
        )
