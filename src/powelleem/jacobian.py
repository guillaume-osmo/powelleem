"""Analytical Jacobian of EEM charges with respect to parameters.

Derivation (implicit function theorem)
--------------------------------------
Given the linear system per molecule

    A(x) · y(x) = b(x),

where ``y = (q ; λ) ∈ ℝ^{n+1}`` and ``x = (κ, α_1..α_T, β_1..β_T)``, taking
the derivative of both sides w.r.t. an arbitrary parameter ``x_p`` yields

    (∂A/∂x_p) · y + A · (∂y/∂x_p) = ∂b/∂x_p,

hence

    ∂y/∂x_p = A⁻¹ · (∂b/∂x_p − (∂A/∂x_p) · y).

The three partial derivatives are sparse and known in closed form:

- ``∂A/∂κ``       = ``M_ext`` where ``M_ext`` is ``inv_r`` padded with
  zeros on the last row/col. ``∂b/∂κ = 0``.
- ``∂A/∂α_k``     = ``0``. ``∂b/∂α_k`` = ``-1[atom_type = k]``
  padded with a 0 in the Lagrange-multiplier row.
- ``∂A/∂β_k``     = ``diag(1[atom_type = k])`` padded with 0 on last row/col.
  ``∂b/∂β_k = 0``.

Per molecule we therefore need

    1 LU factorization of A (already done by the forward pass)
    + (1 + 2T) back-substitutions to obtain all Jacobian columns.

We only return the first ``n`` rows (q part), giving
``J_q ∈ ℝ^{n × (1 + 2T)}``.

Stacking over molecules yields the full residual Jacobian
``J_total ∈ ℝ^{N_atoms × (1 + 2T)}`` which is exactly what
:func:`scipy.optimize.least_squares` consumes for LM / TRF steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray

from powelleem.model import _Solved, predict_charges_numpy

if TYPE_CHECKING:
    from powelleem.types import Dataset, MoleculeData


def jacobian_one(
    x: NDArray[np.float64],
    mol: MoleculeData,
    n_types: int,
    solved: _Solved | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(q_pred, J_q)`` for one molecule.

    Parameters
    ----------
    x
        Packed parameter vector ``(κ, α_1..α_T, β_1..β_T)``.
    mol
        Molecule data (precomputed ``inv_r``, ``atom_types``, etc.).
    n_types
        Number of atom types.
    solved
        Optional cached output of :func:`predict_charges_numpy` to avoid
        redoing the LU factorization.

    Returns
    -------
    q_pred : (n,) ndarray
    J_q    : (n, 1 + 2T) ndarray   columns = (κ, α_1..α_T, β_1..β_T)
    """
    if solved is None:
        solved = predict_charges_numpy(x, mol, n_types)
    lu_piv = solved.lu_piv
    y = solved.y
    q = solved.q
    n = mol.n_atoms
    P = 1 + 2 * n_types

    J = np.empty((n, P), dtype=np.float64)

    # ----- κ column ----------------------------------------------------
    # ∂y/∂κ = - A⁻¹ · (M_ext · y);   M_ext is inv_r padded with zeros on last row/col.
    rhs_kappa = np.zeros(n + 1, dtype=np.float64)
    rhs_kappa[:n] = -mol.inv_r @ y[:n]
    J[:, 0] = sla.lu_solve(lu_piv, rhs_kappa)[:n]

    # ----- α columns ---------------------------------------------------
    # ∂y/∂α_k = A⁻¹ · (-1[atom_type = k] padded with 0 in λ row).
    rhs_alpha = np.zeros((n + 1, n_types), dtype=np.float64)
    for k in range(n_types):
        mask = (mol.atom_types == (k + 1)).astype(np.float64)
        rhs_alpha[:n, k] = -mask
    Jq_alpha = sla.lu_solve(lu_piv, rhs_alpha)[:n, :]
    J[:, 1 : 1 + n_types] = Jq_alpha

    # ----- β columns ---------------------------------------------------
    # ∂y/∂β_k = -A⁻¹ · (diag(1[type = k]) · y_padded).
    rhs_beta = np.zeros((n + 1, n_types), dtype=np.float64)
    for k in range(n_types):
        mask = (mol.atom_types == (k + 1)).astype(np.float64)
        rhs_beta[:n, k] = -mask * q
    Jq_beta = sla.lu_solve(lu_piv, rhs_beta)[:n, :]
    J[:, 1 + n_types :] = Jq_beta

    return q, J


def residuals_and_jacobian(
    x: NDArray[np.float64],
    dataset: Dataset,
    n_types: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Concatenated residuals and Jacobian across all molecules.

    Returns
    -------
    r : (N_atoms,) ndarray of ``q_pred - q_ref``
    J : (N_atoms, 1 + 2T) ndarray of Jacobian rows
    """
    N = dataset.total_atoms
    P = 1 + 2 * n_types
    r = np.empty(N, dtype=np.float64)
    J = np.empty((N, P), dtype=np.float64)
    offset = 0
    for mol in dataset.molecules:
        q_pred, J_block = jacobian_one(x, mol, n_types)
        n = mol.n_atoms
        r[offset : offset + n] = q_pred - mol.target_charges
        J[offset : offset + n, :] = J_block
        offset += n
    return r, J


def loss_and_grad(
    x: NDArray[np.float64],
    dataset: Dataset,
    n_types: int,
) -> tuple[float, NDArray[np.float64]]:
    """Sum-of-squares loss + analytic gradient w.r.t. parameters.

    Loss is the atom-averaged squared error,
    ``L(x) = (1/N) Σ_i r_i(x)²`` where ``N = N_atoms``.

    Gradient is ``∇L = (2/N) Jᵀ r``.
    """
    r, J = residuals_and_jacobian(x, dataset, n_types)
    N = r.size
    loss = float((r * r).sum() / N)
    grad = (2.0 / N) * (J.T @ r)
    return loss, grad


def check_jacobian(
    x: NDArray[np.float64],
    dataset: Dataset,
    n_types: int,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, float]:
    """Sanity-check analytical Jacobian against central finite differences.

    Returns a dict with ``max_abs_err``, ``max_rel_err`` between the analytic
    Jacobian and the finite-difference approximation. Useful for tests.
    """
    _, J_analytic = residuals_and_jacobian(x, dataset, n_types)

    P = 1 + 2 * n_types
    N = dataset.total_atoms
    J_fd = np.empty((N, P), dtype=np.float64)
    for p in range(P):
        e = np.zeros_like(x)
        e[p] = eps
        r_plus, _ = residuals_and_jacobian(x + e, dataset, n_types)
        r_minus, _ = residuals_and_jacobian(x - e, dataset, n_types)
        J_fd[:, p] = (r_plus - r_minus) / (2 * eps)

    diff = J_analytic - J_fd
    max_abs = float(np.max(np.abs(diff)))
    # Relative error only meaningful where J_fd is not vanishingly small —
    # gate on |J_fd| > 1e-6 to avoid 0/0 blowups for parameters that
    # legitimately have zero sensitivity (e.g. ∂q_H_water / ∂α_C).
    mask = np.abs(J_fd) > 1e-6
    if mask.any():
        max_rel = float(np.max(np.abs(diff[mask]) / np.abs(J_fd[mask])))
    else:
        max_rel = 0.0
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "atol_satisfied": max_abs < atol,
        "rtol_satisfied": max_rel < rtol,
    }
