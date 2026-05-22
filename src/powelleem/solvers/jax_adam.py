"""JaxAdam solver — JAX autodiff + Adam.

Reproduces the spirit of ``pure_jax_gradient_optimizer.py`` from
the original ``gasteiger/`` directory: pure-JAX forward, ``jax.grad``
for the loss gradient, Adam optimizer with optional clipping to bounds.

Slower than :class:`AnalyticLM` on small problems (CG/dense solve in JAX
+ trace overhead) but useful when the analytical Jacobian is impractical
or when the per-element model grows to thousands of parameters.
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


class JaxAdam(Solver):
    """JAX autodiff + Adam optimiser with optional bound clipping."""

    name = "JaxAdam"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        clip_to_bounds: bool = True,
    ) -> None:
        super().__init__(config)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.clip_to_bounds = clip_to_bounds

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
                "JAX is required for JaxAdam. Install with `pip install powelleem[jax]`."
            ) from exc

        from powelleem.model import predict_charges_jax_factory
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
        if x0 is None:
            x0 = random_initial_x(n_types, self.config)

        predict_one = predict_charges_jax_factory(n_types)

        # Pre-marshal dataset into a static tuple of JAX arrays.
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

        def loss(x):  # type: ignore[no-untyped-def]
            total = 0.0
            n_total = 0
            for inv_r, type_idx, target, q_tot, n in blocks:
                q = predict_one(x, inv_r, type_idx, q_tot, n)
                diff = q - target
                total = total + jnp.sum(diff * diff)
                n_total += n
            return total / n_total

        loss_jit = jax.jit(loss)
        grad_jit = jax.jit(jax.grad(loss))

        lo, hi = self.config.bounds_array(n_types)
        lo_j = jnp.asarray(lo)
        hi_j = jnp.asarray(hi)

        x = jnp.asarray(x0)
        loss0 = float(loss_jit(x))
        trajectory: list[float] = [loss0]

        # Adam state
        b1, b2, eps = 0.9, 0.999, 1e-8
        m_state = jnp.zeros_like(x)
        v_state = jnp.zeros_like(x)

        t0 = time.perf_counter()
        for t in range(1, self.n_iterations + 1):
            g = grad_jit(x)
            m_state = b1 * m_state + (1 - b1) * g
            v_state = b2 * v_state + (1 - b2) * g * g
            m_hat = m_state / (1 - b1 ** t)
            v_hat = v_state / (1 - b2 ** t)
            x = x - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            if self.clip_to_bounds:
                x = jnp.clip(x, lo_j, hi_j)
            if t % 50 == 0:
                trajectory.append(float(loss_jit(x)))
        wall = time.perf_counter() - t0

        x_np = np.asarray(x)
        loss_final = float(loss_jit(x))
        trajectory.append(loss_final)
        rmse = float(np.sqrt(loss_final))

        params = ParamSet.from_vector(x_np, model.atom_types)
        return FitResult(
            params=params,
            rmse=rmse,
            loss_initial=loss0,
            loss_final=loss_final,
            solver_name=self.name,
            solver_metadata={
                "n_iterations": self.n_iterations,
                "learning_rate": self.learning_rate,
                "clip_to_bounds": self.clip_to_bounds,
            },
            wall_time_s=wall,
            n_function_evals=self.n_iterations,
            n_jacobian_evals=self.n_iterations,
            converged=True,
            message="adam-fixed-iterations",
            loss_trajectory=trajectory,
        )
