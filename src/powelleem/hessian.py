"""Analytical Hessian of EEM loss w.r.t. parameters.

Math
----
Loss     :  L(x) = (1/N) Σ_i r_i(x)²    where r_i = q_i^pred - q_i^ref
Gradient :  ∇L = (2/N) J^T r
Hessian  :  ∇²L = (2/N) ( J^T J + Σ_i r_i · ∇² r_i )

The EEM linear system A(x)·y(x) = b(x) makes second derivatives of q
remarkably simple. Because A is **linear** in (κ, β) and b is **linear**
in α, every second derivative of A and b w.r.t. parameters is zero.
Hence the implicit-function-theorem second-derivative reduces to

    y_pq = − A⁻¹ ( ∂A/∂x_p · y_q + ∂A/∂x_q · y_p )

with the simplifying observation that ∂A/∂α_k = 0. The non-zero pair
classes (p, q) and their right-hand sides are:

    (κ, κ)        :  −2 A⁻¹ (M · y_κ)              with M = inv_r padded
    (κ, α_k)      :  − A⁻¹ (M · y_{α_k})
    (κ, β_k)      :  − A⁻¹ (M · y_{β_k} + diag(1_{t=k}) · y_κ)
    (α_j, α_k)    :  0
    (α_j, β_k)    :  − A⁻¹ (diag(1_{t=k}) · y_{α_j})
    (β_j, β_k)    :  − A⁻¹ (diag(1_{t=j}) · y_{β_k} + diag(1_{t=k}) · y_{β_j})

Per molecule cost = 1 LU factorisation (reused from forward pass)
+ (1 + 2T) back-substitutions for first derivatives (Jacobian)
+ at most P(P+1)/2 − T(T+1)/2 back-substitutions for the Hessian.

For T = 12 (NEEMP set01), P = 25, that's at most 247 extra back-subs
per molecule — still trivial vs the LU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray

from powelleem.jacobian import jacobian_one
from powelleem.model import predict_charges_numpy

if TYPE_CHECKING:
    from powelleem.types import Dataset, MoleculeData


def _per_atom_residual_hessian_one(
    x: NDArray[np.float64],
    mol: MoleculeData,
    n_types: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (q_pred, J_q, H_q) for one molecule.

    H_q is the per-atom residual Hessian, shape (n, P, P), where
    ``H_q[i, p, q] = ∂²q_i / ∂x_p∂x_q``.
    """
    solved = predict_charges_numpy(x, mol, n_types)
    q, J = jacobian_one(x, mol, n_types, solved=solved)
    lu_piv = solved.lu_piv
    y = solved.y
    n = mol.n_atoms
    P = 1 + 2 * n_types

    # ----------- y_p for every p (full (n+1)-vectors, not just the q part) --
    # Re-solve via lu_piv to obtain the full Lagrange-multiplier-extended y_p.
    y_p_full = np.empty((n + 1, P), dtype=np.float64)

    # κ column
    rhs = np.zeros(n + 1)
    rhs[:n] = -mol.inv_r @ y[:n]
    y_p_full[:, 0] = sla.lu_solve(lu_piv, rhs)

    # α columns
    rhs_a = np.zeros((n + 1, n_types))
    for k in range(n_types):
        mask = (mol.atom_types == (k + 1)).astype(np.float64)
        rhs_a[:n, k] = -mask
    y_p_full[:, 1 : 1 + n_types] = sla.lu_solve(lu_piv, rhs_a)

    # β columns
    rhs_b = np.zeros((n + 1, n_types))
    for k in range(n_types):
        mask = (mol.atom_types == (k + 1)).astype(np.float64)
        rhs_b[:n, k] = -mask * q
    y_p_full[:, 1 + n_types :] = sla.lu_solve(lu_piv, rhs_b)

    # ----------- second derivatives -------------------------------------------
    H_q = np.zeros((n, P, P), dtype=np.float64)

    def apply_kappa_term(vec_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply (∂A/∂κ) · v = M_ext · v on a padded (n+1,) vector."""
        out = np.zeros(n + 1)
        out[:n] = mol.inv_r @ vec_full[:n]
        return out

    def apply_beta_term(k_idx: int, vec_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply diag(1_{type=k+1}) ⊕ 0 on the padded vector."""
        mask = (mol.atom_types == (k_idx + 1)).astype(np.float64)
        out = np.zeros(n + 1)
        out[:n] = mask * vec_full[:n]
        return out

    # Precompute lots of small RHS vectors and back-solve as a single block.
    rhs_pairs: list[tuple[tuple[int, int], NDArray[np.float64]]] = []

    # (κ, κ)
    rhs_pairs.append(((0, 0), -2.0 * apply_kappa_term(y_p_full[:, 0])))
    # (κ, α_k)
    for k in range(n_types):
        rhs_pairs.append(((0, 1 + k), -apply_kappa_term(y_p_full[:, 1 + k])))
    # (κ, β_k)
    for k in range(n_types):
        rhs_pairs.append(
            ((0, 1 + n_types + k),
             -(apply_kappa_term(y_p_full[:, 1 + n_types + k])
               + apply_beta_term(k, y_p_full[:, 0]))),
        )
    # (α_j, β_k)
    for j in range(n_types):
        for k in range(n_types):
            rhs_pairs.append(
                ((1 + j, 1 + n_types + k),
                 -apply_beta_term(k, y_p_full[:, 1 + j])),
            )
    # (β_j, β_k) for j <= k
    for j in range(n_types):
        for k in range(j, n_types):
            if j == k:
                rhs_pairs.append(
                    ((1 + n_types + j, 1 + n_types + k),
                     -2.0 * apply_beta_term(j, y_p_full[:, 1 + n_types + j])),
                )
            else:
                rhs_pairs.append(
                    ((1 + n_types + j, 1 + n_types + k),
                     -(apply_beta_term(j, y_p_full[:, 1 + n_types + k])
                       + apply_beta_term(k, y_p_full[:, 1 + n_types + j]))),
                )

    if rhs_pairs:
        rhs_matrix = np.stack([rhs for _, rhs in rhs_pairs], axis=1)
        sol_matrix = sla.lu_solve(lu_piv, rhs_matrix)
        for idx, ((p, qj), _) in enumerate(rhs_pairs):
            H_q[:, p, qj] = sol_matrix[:n, idx]
            if p != qj:
                H_q[:, qj, p] = sol_matrix[:n, idx]
    # (α_j, α_k) terms are zero by construction → already zeros in H_q.

    return q, J, H_q


def loss_grad_hessian(
    x: NDArray[np.float64],
    dataset: Dataset,
    n_types: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return (loss, gradient, Hessian) of the sum-of-squares loss.

    Loss  = (1/N) Σ_i r_i²
    Grad  = (2/N) J^T r
    Hess  = (2/N) (J^T J + Σ_i r_i · ∇² r_i)
    """
    P = 1 + 2 * n_types
    N = dataset.total_atoms
    r_total = np.empty(N, dtype=np.float64)
    JtJ_total = np.zeros((P, P), dtype=np.float64)
    JtR_total = np.zeros(P, dtype=np.float64)
    second_corr_total = np.zeros((P, P), dtype=np.float64)

    offset = 0
    for mol in dataset.molecules:
        q_pred, J_block, H_block = _per_atom_residual_hessian_one(x, mol, n_types)
        r_block = q_pred - mol.target_charges
        n = mol.n_atoms
        r_total[offset : offset + n] = r_block
        offset += n
        JtJ_total += J_block.T @ J_block
        JtR_total += J_block.T @ r_block
        # second_corr[p, q] = Σ_i r_i · H_block[i, p, q]
        second_corr_total += np.einsum("i,ipq->pq", r_block, H_block)

    loss = float((r_total * r_total).sum() / N)
    grad = (2.0 / N) * JtR_total
    H = (2.0 / N) * (JtJ_total + second_corr_total)
    return loss, grad, H


def check_hessian(
    x: NDArray[np.float64],
    dataset: Dataset,
    n_types: int,
    *,
    eps: float = 1e-5,
    atol: float = 1e-4,
) -> dict[str, float]:
    """Compare analytical Hessian to central finite differences on the gradient."""
    from powelleem.jacobian import loss_and_grad

    _, _, H_analytic = loss_grad_hessian(x, dataset, n_types)
    P = x.size
    H_fd = np.empty((P, P), dtype=np.float64)
    for p in range(P):
        e = np.zeros_like(x)
        e[p] = eps
        _, g_plus = loss_and_grad(x + e, dataset, n_types)
        _, g_minus = loss_and_grad(x - e, dataset, n_types)
        H_fd[:, p] = (g_plus - g_minus) / (2 * eps)
    # Symmetrise
    H_fd = 0.5 * (H_fd + H_fd.T)

    diff = H_analytic - H_fd
    max_abs = float(np.max(np.abs(diff)))
    mask = np.abs(H_fd) > 1e-6
    max_rel = float(np.max(np.abs(diff[mask]) / np.abs(H_fd[mask]))) if mask.any() else 0.0
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "atol_satisfied": max_abs < atol,
    }
