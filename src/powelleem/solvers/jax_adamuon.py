"""JaxAdaMuon solver — faithful port of AdaMuonOfficial (Liu et al. 2025).

Mirrors the reference PyTorch implementation found in
``mlxmolkit/tools/torch_optimizers.py:AdaMuonOfficial`` (Apache-2.0):

1. Momentum buffer (Nesterov-style):  m ← μ·m + g;  direction = g + μ·m
2. Newton-Schulz quintic on ``sign(direction)`` (5 steps by default)
   with coefficients (a, b, c) = (3.4445, -4.7750, 2.0315). The matrix
   is first L2-normalised so the iteration is in the contraction regime.
3. Variance buffer on the *post-NS* direction:
       v ← μ·v + (1−μ)·(d⊗d)
4. ``flat = direction / (√v + ε)``
5. Spectral rescale: direction · (s · √(r·c) / (‖direction‖ + ε))
   with ``s = scale_coeff = 0.2`` and ``r, c`` being the matrix dims.
6. ``x ← x − lr · direction``

For our flat 1+2T-dim parameter vector we reshape to ``(1, P)`` so the
Newton-Schulz iteration runs on a degenerate 1×P matrix — that still
orthogonalises via row-normalisation, which on a single row collapses
to ``v / ‖v‖``. The quintic coefficients give a sharper polar
approximation than a single L2-normalise.

Reference
---------
- AdaMuonOfficial in ``mlxmolkit/tools/torch_optimizers.py`` (Guillaume's
  fork, May 2026), itself faithful to:
  Liu, Y. et al. *AdaMuon: Adaptive Muon Optimizer.* 2025.
  Jordan, K. *Muon: An optimizer for hidden layers in neural networks.*
  GitHub, 2024.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

from powelleem.solvers.base import Solver, SolverConfig, random_initial_x

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


# Newton-Schulz quintic coefficients used by common Muon implementations.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def _zeropower_newton_schulz_jax(matrix, steps: int, eps: float = 1e-7):  # type: ignore[no-untyped-def]
    """Quintic Newton-Schulz iteration in JAX. ``matrix`` is shape (r, c)."""
    import jax.numpy as jnp

    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix = matrix / (jnp.linalg.norm(matrix) + eps)
    for _ in range(int(steps)):
        gram = matrix @ matrix.T
        matrix = _NS_A * matrix + (_NS_B * gram + _NS_C * (gram @ gram)) @ matrix
    if transposed:
        matrix = matrix.T
    return matrix


class JaxAdaMuon(Solver):
    """AdaMuon (Liu et al. 2025) in JAX, faithful to AdaMuonOfficial."""

    name = "JaxAdaMuon"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        momentum: float = 0.95,
        eps: float = 1e-8,
        ns_steps: int = 5,
        scale_coeff: float = 0.2,
        nesterov: bool = True,
        clip_to_bounds: bool = True,
    ) -> None:
        super().__init__(config)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.eps = eps
        self.ns_steps = ns_steps
        self.scale_coeff = scale_coeff
        self.nesterov = nesterov
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
        P = x.shape[0]
        loss0 = float(loss_jit(x))
        trajectory: list[float] = [loss0]

        mu = self.momentum
        eps = self.eps
        m_buf = jnp.zeros_like(x)
        v_buf = jnp.zeros_like(x)

        # For our (1, P) reshape, r = 1, c = P, so spectral rescale factor is
        #   s · √(min(r,c)·max(r,c)) / (||direction|| + ε) = s · √P / (...)
        rc_factor = math.sqrt(max(1, min(1, P)) * max(1, P))

        t0 = time.perf_counter()
        for _t in range(1, self.n_iterations + 1):
            g = grad_jit(x)

            # 1. Momentum buffer + Nesterov-style direction
            m_buf = mu * m_buf + g
            direction_flat = g + mu * m_buf if self.nesterov else m_buf

            # 2. Newton-Schulz quintic on sign(direction), reshaped to (1, P)
            sign_dir = jnp.sign(direction_flat).reshape(1, -1)
            polished = _zeropower_newton_schulz_jax(sign_dir, self.ns_steps, eps=1e-7)
            direction_flat = polished.reshape(-1)

            # 3. Variance buffer on post-NS direction (shared β with momentum)
            v_buf = mu * v_buf + (1.0 - mu) * direction_flat * direction_flat

            # 4. Adam-style 1/√v scaling
            direction_flat = direction_flat / (jnp.sqrt(v_buf) + eps)

            # 5. Spectral rescale
            scale = self.scale_coeff * rc_factor / (jnp.linalg.norm(direction_flat) + eps)
            direction_flat = direction_flat * scale

            # 6. Param update
            x = x - self.learning_rate * direction_flat
            if self.clip_to_bounds:
                x = jnp.clip(x, lo_j, hi_j)

            if _t % 50 == 0:
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
                "momentum": self.momentum,
                "ns_steps": self.ns_steps,
                "scale_coeff": self.scale_coeff,
                "nesterov": self.nesterov,
            },
            wall_time_s=wall,
            n_function_evals=self.n_iterations,
            n_jacobian_evals=self.n_iterations,
            converged=True,
            message="adamuon-fixed-iterations",
            loss_trajectory=trajectory,
        )
