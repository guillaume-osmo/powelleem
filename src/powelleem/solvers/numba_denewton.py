"""DENewton accelerated by the Numba JIT backend.

Identical algorithm to :class:`powelleem.solvers.de_newton.DENewton` — DE
global search followed by an L-BFGS-B + trust-Newton polish using the
analytical Hessian — but every per-molecule LU + back-substitution runs
through :mod:`powelleem.numba_backend` kernels with ``@njit(parallel=True)``.

Speedups measured on NEEMP set03 (subset 5 000 mol, 235 168 atoms):

    residuals only           60× faster than the SciPy serial loop
    residuals + Jacobian     15×
    full analytical Hessian  60×

Extrapolated to set03 full (17 769 mol), the production DENewton run
drops from ~39 min to ~1 min.

Note
----
The first call to any kernel triggers Numba's JIT compilation (~5–10 s).
This happens transparently here; if you intend to benchmark wall times
fairly, warm the kernels once on a small dataset first.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as so

from powelleem.solvers.base import Solver, SolverConfig, latin_hypercube_initial_xs, random_initial_x

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


class NumbaDENewton(Solver):
    """JIT-accelerated DENewton — see :class:`DENewton` for the algorithm.

    Identical optimisation logic; only the per-molecule arithmetic is
    routed through :mod:`powelleem.numba_backend`. ``use_serial_polish``
    can be set to ``True`` to fall back to the SciPy reference for the
    Newton phase if a numerical anomaly is suspected.
    """

    name = "NumbaDENewton"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_F: float = 0.5,
        crossover_CR: float = 0.7,
        maxiter_lbfgs: int = 100,
        maxiter_newton: int = 100,
        bound_penalty: float = 1e2,
        loss_kind: str = "atom_rmse",
        use_serial_polish: bool = False,
    ) -> None:
        super().__init__(config)
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_F = mutation_F
        self.crossover_CR = crossover_CR
        self.maxiter_lbfgs = maxiter_lbfgs
        self.maxiter_newton = maxiter_newton
        self.bound_penalty = bound_penalty
        self.loss_kind = loss_kind
        self.use_serial_polish = use_serial_polish

    def fit(
        self,
        model: EEMModel,
        dataset: Dataset,
        *,
        x0: NDArray[np.float64] | None = None,
    ) -> FitResult:
        from powelleem.numba_backend import (
            build_numba_dataset,
            loss_grad_hessian_numba,
            loss_and_grad_numba_kind,
            residuals_only_numba,
        )
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
        lo, hi = self.config.bounds_array(n_types)
        if x0 is None:
            x0 = random_initial_x(n_types, self.config)

        nd = build_numba_dataset(dataset)
        N_atoms = nd.total_atoms

        def eval_fitness(x: NDArray[np.float64]) -> float:
            """Compute the DE fitness value matching ``self.loss_kind``.

            Vectorised — relies on ``np.add.reduceat`` for the per-molecule
            sums so we stay multi-threaded through BLAS / SIMD rather than
            grinding through a Python loop over 17K molecules.
            """
            r = residuals_only_numba(x, nd)
            if self.loss_kind == "mol_rmsd":
                r_sq = r * r
                sums = np.add.reduceat(r_sq, nd.atom_offsets[:-1])
                rmsd_per_mol = np.sqrt(sums / nd.n_atoms.astype(np.float64))
                return float(rmsd_per_mol.mean())
            return float(np.sqrt((r * r).mean()))

        # ----- Stage 1: DE global search (residuals-only Numba) -----
        rng = np.random.default_rng(self.config.seed)
        population = latin_hypercube_initial_xs(n_types, self.config, self.population_size)
        if x0 is not None:
            population[0] = np.clip(x0, lo, hi)

        fitness = np.array([eval_fitness(p) for p in population])
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
                trial_fit = eval_fitness(trial)
                n_de_evals += 1
                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_f:
                        best_f = trial_fit
                        best_x = trial.copy()
            trajectory.append(best_f * best_f)
        t_de = time.perf_counter() - t_de_start

        # ----- Stage 2: L-BFGS-B warm + trust-Newton polish (Numba) -----
        # Penalty for boundary excursions (Newton is unconstrained).
        def penalty_components(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
            below = np.maximum(0.0, lo - x)
            above = np.maximum(0.0, x - hi)
            p_val = float((below * below).sum() + (above * above).sum())
            p_grad = -2.0 * below + 2.0 * above
            p_hess_diag = 2.0 * ((below > 0) | (above > 0)).astype(np.float64)
            return p_val, p_grad, p_hess_diag

        def loss_grad(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            loss, grad = loss_and_grad_numba_kind(x, nd, loss_kind=self.loss_kind)
            pv, pg, _ = penalty_components(x)
            return loss + self.bound_penalty * pv, grad + self.bound_penalty * pg

        def hess(x: NDArray[np.float64]) -> NDArray[np.float64]:
            _, _, H = loss_grad_hessian_numba(x, nd, loss_kind=self.loss_kind)
            _, _, pd = penalty_components(x)
            return H + self.bound_penalty * np.diag(pd)

        t_polish_start = time.perf_counter()
        res_lbfgs = so.minimize(
            loss_grad, best_x, jac=True, method="L-BFGS-B",
            bounds=list(zip(lo, hi, strict=True)),
            options={"maxiter": self.maxiter_lbfgs},
        )
        x_warm = res_lbfgs.x
        loss_after_lbfgs = float(res_lbfgs.fun)
        n_iter_lbfgs = int(res_lbfgs.nit)

        res_newton = so.minimize(
            lambda x: loss_grad(x)[0],
            x_warm,
            jac=lambda x: loss_grad(x)[1],
            hess=hess,
            method="trust-exact",
            options={"maxiter": self.maxiter_newton},
        )
        t_polish = time.perf_counter() - t_polish_start

        x_opt = np.clip(res_newton.x, lo, hi)
        r_final = residuals_only_numba(x_opt, nd)
        loss_final = float((r_final * r_final).sum() / r_final.size)
        rmse = float(np.sqrt(loss_final))

        params = ParamSet.from_vector(x_opt, model.atom_types)
        return FitResult(
            params=params,
            rmse=rmse,
            loss_initial=loss0,
            loss_final=loss_final,
            solver_name=self.name,
            solver_metadata={
                "stage1_de_rmse": best_f,
                "stage1_n_evals": n_de_evals,
                "stage1_wall_s": t_de,
                "stage2_wall_s": t_polish,
                "loss_after_lbfgs": loss_after_lbfgs,
                "n_iter_lbfgs": n_iter_lbfgs,
                "n_iter_newton": int(res_newton.nit),
                "newton_status": int(res_newton.status),
                "newton_message": str(res_newton.message),
                "population_size": self.population_size,
                "n_generations": self.n_generations,
            },
            wall_time_s=t_de + t_polish,
            n_function_evals=n_de_evals + n_iter_lbfgs + int(res_newton.nfev),
            n_jacobian_evals=n_iter_lbfgs + int(res_newton.njev or 0),
            converged=bool(res_newton.success),
            message=f"DE({n_de_evals} evals) → {res_newton.message}",
            loss_trajectory=trajectory,
        )
