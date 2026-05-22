"""JaxAdaMuon solver — AdaMuon (Liu et al. 2025) in JAX.

AdaMuon combines two ideas:

- **Muon** (Jordan 2024) replaces SGD's per-coordinate update with a
  spectrally-normalised step. Concretely, the momentum buffer is fed
  through Newton-Schulz iterations to approximate the polar/orthogonal
  factor before being applied. The "step direction" is therefore a
  unit-norm matrix (or unit-norm vector for non-matrix params).
- **Ada** adds Adam-style per-coordinate adaptive scaling on top, using
  the running second moment of the *raw* gradient ``g²``.

For a flat parameter vector ``x ∈ ℝ^P`` (our case: ``P = 1 + 2T`` for an
EEM fit with ``T`` atom types), the Newton-Schulz iteration on a vector
degenerates to a single L2-normalisation. The full update reads::

    g_t  = ∇L(x_{t-1})
    m_t  = β1·m_{t-1} + (1-β1)·g_t         # 1st moment
    v_t  = β2·v_{t-1} + (1-β2)·g_t²        # 2nd moment (element-wise)
    m̂_t  = m_t / max(‖m_t‖₂, ε)            # Muon orthogonalisation (vector form)
    v̂_t  = v_t / (1 − β2^t)                # Adam bias correction
    x_t  = clip(x_{t-1} − η·m̂_t / (√v̂_t + ε), lo, hi)

Compared to plain Adam, AdaMuon decouples the *direction* (purely
momentum, L2-normalised) from the *magnitude* (Adam-style 1/√v scaling).
Reported in the AdaMuon paper to converge faster than Adam on transformer
training; for non-convex small-dim least-squares like ours it is
essentially Lion-flavoured Adam.
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


class JaxAdaMuon(Solver):
    """AdaMuon (Adam + Muon orthogonalisation) implemented in JAX."""

    name = "JaxAdaMuon"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        n_iterations: int = 1000,
        learning_rate: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip_to_bounds: bool = True,
    ) -> None:
        super().__init__(config)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
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
                "JAX is required for JaxAdaMuon. Install with `pip install powelleem[jax]`."
            ) from exc

        from powelleem.model import predict_charges_jax_factory
        from powelleem.types import FitResult, ParamSet

        n_types = model.n_types
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

        b1, b2, eps = self.beta1, self.beta2, self.eps
        m_state = jnp.zeros_like(x)
        v_state = jnp.zeros_like(x)

        t0 = time.perf_counter()
        for t in range(1, self.n_iterations + 1):
            g = grad_jit(x)
            m_state = b1 * m_state + (1 - b1) * g
            v_state = b2 * v_state + (1 - b2) * g * g

            # Bias-correct
            m_hat = m_state / (1 - b1 ** t)
            v_hat = v_state / (1 - b2 ** t)

            # AdaMuon: Adam-scale the momentum *first*, then unit-normalise the
            # resulting direction (Muon's polar step). This decouples the step
            # *magnitude* (controlled by lr only) from the *direction* (which
            # adaptively weights coordinates by their inverse-variance).
            scaled = m_hat / (jnp.sqrt(v_hat) + eps)
            direction = scaled / (jnp.linalg.norm(scaled) + eps)

            # Scale by lr and sqrt(P) so the per-coordinate step magnitude is
            # comparable to Adam's effective per-coordinate step (Adam: lr per
            # coordinate; AdaMuon: lr * sqrt(P) / P = lr/sqrt(P) per coordinate
            # without the rescale, restored here).
            step = self.learning_rate * jnp.sqrt(direction.size) * direction

            x = x - step
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
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.eps,
                "clip_to_bounds": self.clip_to_bounds,
            },
            wall_time_s=wall,
            n_function_evals=self.n_iterations,
            n_jacobian_evals=self.n_iterations,
            converged=True,
            message="adamuon-fixed-iterations",
            loss_trajectory=trajectory,
        )
