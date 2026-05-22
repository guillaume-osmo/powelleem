"""JaxMuonN solver — Guillaume's "AdaMuonn" variant.

Faithful JAX port of the PyTorch ``MuonN`` class
(``mlxmolkit/tools/torch_optimizers.py:237``). Differs from the
official AdaMuon (Liu et al. 2025) in several deliberate ways:

* **Triple-β nested AdamN EMA** (β_grad, β_nested, β_sq), default
  (0.9, 0.1, 0.999):

      grad_avg   ← β_grad·grad_avg   + (1−β_grad)·grad
      nested_avg ← β_nested·nested   + (1−β_nested)·grad_avg
      sq_avg     ← β_sq·sq_avg       + (1−β_sq)·grad²

  The nested 1st moment ``nested_avg`` is the directional signal.

* **Exact (closed-form) bias correction** for the nested moment::

      cross  = (1−β_n) · β_g · (β_n^t − β_g^t) / (β_n − β_g)
      bias1  = 1 − β_n^t − cross
      direction = nested_avg / bias1

  (or simple Adam-style ``1 − β_n^t`` if ``bias_correction='simple'``).

* **Variance normalisation** on the *raw gradient's* 2nd moment (not
  the post-NS direction): ``direction /= √(sq_avg/bias2) + ε``.

* **Newton-Schulz on the direction itself** (not on ``sign(direction)``
  as AdaMuonOfficial does). Quintic coefficients (3.4445, -4.7750,
  2.0315) with 5 iterations by default.

* **Final rescale** by ``√max(1, rows/cols)`` (aspect-ratio compensation,
  not the spectral ``√(r·c)`` of AdaMuonOfficial).

For our flat 1+2T parameter vector the rows/cols heuristic and the NS
step are both degenerate, but we ship the solver so the user can
benchmark it on matrix-shaped fits (per-element params reshaped as
a (T, 2) matrix of (α, β) for example).

Reference: ``MuonN`` in ``mlxmolkit/tools/torch_optimizers.py`` —
Apache-2.0 (Guillaume Godin's variant extending Liu et al. 2025 +
Jordan 2024 + AdamN nested bias-correction trick).
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


_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def _nested_bias_correction(step: int, beta_grad: float, beta_nested: float) -> float:
    """Exact bias correction for ``nested_avg = β_nested·avg + (1-β_nested)·grad_avg``."""
    if step <= 0:
        return 0.0
    if abs(beta_nested - beta_grad) < 1.0e-12:
        beta = beta_grad
        return (1.0 - beta ** step) - (1.0 - beta) * step * (beta ** step)
    grad_pow = beta_grad ** step
    nested_pow = beta_nested ** step
    cross = (1.0 - beta_nested) * beta_grad * (nested_pow - grad_pow) / (beta_nested - beta_grad)
    return 1.0 - nested_pow - cross


def _zeropower_newton_schulz_jax(matrix, steps: int, eps: float = 1e-7):  # type: ignore[no-untyped-def]
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


class JaxMuonN(Solver):
    """Guillaume's ``MuonN`` ("AdaMuonn") variant — nested AdamN EMA + NS-on-direction."""

    name = "JaxMuonN"

    def __init__(
        self,
        config: SolverConfig | None = None,
        *,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        betas: tuple[float, float, float] = (0.9, 0.1, 0.999),
        eps: float = 1e-8,
        ns_steps: int = 5,
        bias_correction: str = "exact",
        variance_normalize: bool = True,
        clip_to_bounds: bool = True,
    ) -> None:
        super().__init__(config)
        if len(betas) != 3:
            raise ValueError("betas must be (beta_grad, beta_nested, beta_sq)")
        if bias_correction not in {"exact", "simple"}:
            raise ValueError("bias_correction must be 'exact' or 'simple'")
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.betas = tuple(float(b) for b in betas)
        self.eps = eps
        self.ns_steps = ns_steps
        self.bias_correction = bias_correction
        self.variance_normalize = variance_normalize
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
                "JAX is required for JaxMuonN. Install with `pip install powelleem[jax]`."
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

        b_grad, b_nest, b_sq = self.betas
        eps = self.eps
        grad_avg = jnp.zeros_like(x)
        nested_avg = jnp.zeros_like(x)
        sq_avg = jnp.zeros_like(x)

        # rows/cols compensation for (1, P) reshape
        aspect_scale = math.sqrt(max(1.0, 1.0 / max(1, P)))  # = sqrt(1/P) for r=1, c=P

        t0 = time.perf_counter()
        for t in range(1, self.n_iterations + 1):
            g = grad_jit(x)

            # Triple-β nested AdamN EMA
            grad_avg = b_grad * grad_avg + (1.0 - b_grad) * g
            nested_avg = b_nest * nested_avg + (1.0 - b_nest) * grad_avg
            sq_avg = b_sq * sq_avg + (1.0 - b_sq) * g * g

            # Bias correction on nested moment
            if self.bias_correction == "exact":
                bias1 = max(_nested_bias_correction(t, b_grad, b_nest), 1e-16)
            else:
                bias1 = max(1.0 - b_nest ** t, 1e-16)
            direction = nested_avg / bias1

            if self.variance_normalize:
                bias2 = max(1.0 - b_sq ** t, 1e-16)
                direction = direction / (jnp.sqrt(sq_avg / bias2) + eps)

            # NS on direction directly (not on sign), reshape to (1, P)
            polished = _zeropower_newton_schulz_jax(direction.reshape(1, -1), self.ns_steps)
            direction = polished.reshape(-1)

            # Aspect-ratio rescale (MuonN-specific; for r=1,c=P this is sqrt(1/P))
            direction = direction * aspect_scale

            x = x - self.learning_rate * direction
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
                "betas": self.betas,
                "ns_steps": self.ns_steps,
                "bias_correction": self.bias_correction,
                "variance_normalize": self.variance_normalize,
            },
            wall_time_s=wall,
            n_function_evals=self.n_iterations,
            n_jacobian_evals=self.n_iterations,
            converged=True,
            message="muonn-fixed-iterations",
            loss_trajectory=trajectory,
        )
