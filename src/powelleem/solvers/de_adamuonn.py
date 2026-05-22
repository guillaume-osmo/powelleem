"""DEAdaMuonn solver — DE global search bridged with JaxMuonN polish.

Sibling of :class:`DENewton`. Same two-stage pattern, but the polish
phase uses Guillaume's ``JaxMuonN`` (AdaMuonn variant) instead of
``AnalyticNewton``. Useful when:

* the analytical Hessian is not available (richer parametrisations,
  added regularisers, etc.);
* you want to compare a curvature-aware (``DENewton``) polish vs a
  spectrally-orthogonalised gradient (``DEAdaMuonn``) polish from the
  same DE warm start.

Pipeline:

1. DE with Latin-Hypercube initial population (default 50 individuals,
   20 generations) locates the right κ-basin in ~3 s.
2. Best individual is fed to a ``JaxMuonN`` run (default 500 iterations,
   ``lr=0.005`` for a finer step than the cold-start defaults). The
   spectrally-orthogonalised steps escape any narrow local minima inside
   the basin without needing curvature information.

Reference for the polish step: ``JaxMuonN`` (this package) ported from
Guillaume's PyTorch ``MuonN`` in ``mlxmolkit/tools/torch_optimizers.py``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from powelleem.solvers.base import Solver, SolverConfig, latin_hypercube_initial_xs
from powelleem.solvers.jax_muonn import JaxMuonN

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


class DEAdaMuonn(Solver):
    """DE global search + JaxMuonN (AdaMuonn) polish."""

    name = "DEAdaMuonn"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_F: float = 0.5,
        crossover_CR: float = 0.7,
        # JaxMuonN polish stage
        polish_iterations: int = 500,
        polish_learning_rate: float = 0.005,
        polish_betas: tuple[float, float, float] = (0.9, 0.1, 0.999),
        polish_ns_steps: int = 5,
    ) -> None:
        super().__init__(config)
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_F = mutation_F
        self.crossover_CR = crossover_CR
        self.polish_iterations = polish_iterations
        self.polish_learning_rate = polish_learning_rate
        self.polish_betas = polish_betas
        self.polish_ns_steps = polish_ns_steps

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

        # ---------- Stage 2: JaxMuonN (AdaMuonn) polish ----------
        polish_solver = JaxMuonN(
            config=SolverConfig(seed=self.config.seed),
            n_iterations=self.polish_iterations,
            learning_rate=self.polish_learning_rate,
            betas=self.polish_betas,
            ns_steps=self.polish_ns_steps,
        )
        t_polish_start = time.perf_counter()
        polish_result = polish_solver.fit(model, dataset, x0=best_x)
        t_polish = time.perf_counter() - t_polish_start

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
                "stage2_wall_s": t_polish,
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                **{"polish_" + k: v for k, v in polish_result.solver_metadata.items()},
            },
            wall_time_s=t_de + t_polish,
            n_function_evals=n_de_evals + polish_result.n_function_evals,
            n_jacobian_evals=polish_result.n_jacobian_evals,
            converged=polish_result.converged,
            message=f"DE({n_de_evals} evals) → {polish_result.message}",
            loss_trajectory=trajectory,
        )
