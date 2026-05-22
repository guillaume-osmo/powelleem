"""PyTorch backend — batched EEM forward + analytical Jacobian + Hessian.

Why PyTorch (CPU, not MPS)
--------------------------
On Apple Silicon M-series, ``torch.linalg.solve`` calls into Accelerate's
optimised batched LAPACK routine (gesv with stride-batching). On the
NEEMP set03 workload — 17 769 molecules with ~30 atoms each — a single
batched solve completes in ~30 ms whereas the SciPy per-molecule loop
takes ~200 ms (5–6× slower) and ``numpy.linalg.solve`` is much slower
still due to a different code path. MPS (GPU) dispatch overhead
dominates for matrices below ~100×100 and is actually *slower* here.

Padding strategy
----------------
Molecules have variable atom counts. We pad each to ``N_max + 1`` (the
extra row holds the Lagrange multiplier for charge conservation):

* Padded atoms ``i`` get ``A[batch, i, i] = 1`` and ``A[batch, i, j] = 0``
  for ``j ≠ i``; ``b[batch, i] = 0``. The padded ``q_i`` solves to 0 and
  does not perturb real atoms.
* The Lagrange row ``A[batch, N_max, j] = 1`` only for real atoms
  ``j < n``; ``0`` elsewhere. The Lagrange column matches by symmetry.
  ``A[batch, N_max, N_max] = 0``.

We carry an ``atom_mask`` (B, N_max) boolean to extract real residuals
after the solve. Padding waste is acceptable: even at N_max = 80 on
set03 (vs median 30), the batched solve still wins by an order of
magnitude over the per-mol Python loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.types import Dataset


def _ensure_torch():  # type: ignore[no-untyped-def]
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the torch backend. "
            "Install with `pip install powelleem[torch]`."
        ) from exc


# ---------------------------------------------------------------------------
# Pre-padded dataset tensors (one-time cost; the dataset is fixed during fit)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TorchDataset:
    """Padded batched tensors for the whole dataset, plus bookkeeping."""

    inv_r: object  # (B, N_max, N_max) torch.float64
    atom_type_idx: object  # (B, N_max) torch.int64; padding atoms have value 0
    atom_mask: object  # (B, N_max) torch.bool — True for real atoms
    n_atoms: object  # (B,) torch.int64
    q_total: object  # (B,) torch.float64
    target_q: object  # (B, N_max) torch.float64; padded entries are 0
    n_types: int
    N_max: int
    total_real_atoms: int

    def build_A_b(self, x: object) -> tuple[object, object]:  # type: ignore[override]
        """Construct ``(A, b)`` tensors for the EEM linear system.

        x is shape (P,) torch.float64 with layout (κ, α_1..α_T, β_1..β_T).
        Returns A of shape (B, N_max + 1, N_max + 1) and b of shape
        (B, N_max + 1).
        """
        torch = _ensure_torch()
        B = self.inv_r.shape[0]
        N = self.N_max
        kappa = x[0]
        alpha = x[1 : 1 + self.n_types]
        beta = x[1 + self.n_types :]

        # Atom-type index for padding atoms doesn't matter (they're masked).
        # We index alpha/beta with the atom_type_idx; padded entries pick
        # alpha[-1]/beta[-1] arbitrarily but get overwritten below.
        beta_per_atom = beta[(self.atom_type_idx - 1).clamp(min=0)]
        alpha_per_atom = alpha[(self.atom_type_idx - 1).clamp(min=0)]

        # ---- Build A_core (B, N, N) ----
        A_core = kappa * self.inv_r
        # Set diagonal = β for real atoms, 1 for padding (so q_pad solves to 0)
        diag = torch.where(self.atom_mask, beta_per_atom, torch.ones_like(beta_per_atom))
        A_core = A_core.diagonal_scatter(diag, dim1=1, dim2=2)
        # Zero out rows/cols of padding atoms (except diagonal) to fully
        # decouple them. Multiplication by atom_mask broadcasts.
        mask_f = self.atom_mask.to(A_core.dtype)
        A_core = A_core * mask_f.unsqueeze(2) * mask_f.unsqueeze(1)
        # Restore the unit diagonal for padding atoms
        A_core = A_core.diagonal_scatter(diag, dim1=1, dim2=2)

        # ---- Embed into (N+1, N+1) with constraint row/col ----
        A = torch.zeros(B, N + 1, N + 1, dtype=A_core.dtype)
        A[:, :N, :N] = A_core
        # last column / last row = 1 for real atoms, 0 for padding
        A[:, :N, N] = mask_f
        A[:, N, :N] = mask_f
        # A[:, N, N] = 0 already

        # ---- b vector ----
        b = torch.zeros(B, N + 1, dtype=A_core.dtype)
        b[:, :N] = torch.where(self.atom_mask, -alpha_per_atom, torch.zeros_like(alpha_per_atom))
        b[:, N] = self.q_total
        return A, b


def build_torch_dataset(dataset: Dataset) -> TorchDataset:
    """Convert a powelleem :class:`Dataset` into padded torch tensors."""
    torch = _ensure_torch()
    n_atoms_list = [m.n_atoms for m in dataset.molecules]
    N_max = max(n_atoms_list)
    B = len(dataset.molecules)
    n_types = dataset.n_types

    inv_r = np.zeros((B, N_max, N_max), dtype=np.float64)
    atom_type_idx = np.zeros((B, N_max), dtype=np.int64)
    atom_mask = np.zeros((B, N_max), dtype=bool)
    target_q = np.zeros((B, N_max), dtype=np.float64)
    q_total = np.zeros(B, dtype=np.float64)
    n_atoms_arr = np.zeros(B, dtype=np.int64)

    for i, mol in enumerate(dataset.molecules):
        n = mol.n_atoms
        inv_r[i, :n, :n] = mol.inv_r
        atom_type_idx[i, :n] = mol.atom_types
        atom_mask[i, :n] = True
        target_q[i, :n] = mol.target_charges
        q_total[i] = mol.formal_charge
        n_atoms_arr[i] = n

    return TorchDataset(
        inv_r=torch.from_numpy(inv_r),
        atom_type_idx=torch.from_numpy(atom_type_idx),
        atom_mask=torch.from_numpy(atom_mask),
        n_atoms=torch.from_numpy(n_atoms_arr),
        q_total=torch.from_numpy(q_total),
        target_q=torch.from_numpy(target_q),
        n_types=n_types,
        N_max=N_max,
        total_real_atoms=int(n_atoms_arr.sum()),
    )


# ---------------------------------------------------------------------------
# Forward + Jacobian + Hessian — all batched
# ---------------------------------------------------------------------------

def residuals_and_jacobian_torch(
    x_np: NDArray[np.float64],
    td: TorchDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Concatenated residuals + Jacobian over the dataset, batched."""
    torch = _ensure_torch()
    x = torch.from_numpy(np.asarray(x_np, dtype=np.float64))

    A, b = td.build_A_b(x)  # (B, N+1, N+1), (B, N+1)

    # ---- LU factorize once, reuse for all RHS ----
    LU, piv = torch.linalg.lu_factor(A)

    # ---- Forward: solve A y = b ----
    y = torch.linalg.lu_solve(LU, piv, b.unsqueeze(2)).squeeze(2)  # (B, N+1)
    q = y[:, : td.N_max]  # (B, N)

    # Residuals only for real atoms
    r_full = (q - td.target_q) * td.atom_mask.to(q.dtype)
    # Flatten to a 1D vector over real atoms
    r_flat = r_full[td.atom_mask].numpy()  # (total_real_atoms,)

    # ---- Jacobian via implicit fn theorem ----
    # ∂y/∂κ = - A⁻¹ (M · y) where M = inv_r padded with 0 on Lagrange row/col
    # ∂y/∂α_k = - A⁻¹ · 1[type=k] (with 0 on Lagrange row)
    # ∂y/∂β_k = - A⁻¹ · diag(1[type=k]) · y
    B = td.inv_r.shape[0]
    N = td.N_max
    P = 1 + 2 * td.n_types
    T = td.n_types

    # RHS bundle (B, N+1, P)
    rhs = torch.zeros(B, N + 1, P, dtype=A.dtype)

    # κ column: rhs[:, :N, 0] = -(inv_r · q)
    rhs[:, :N, 0] = -(td.inv_r @ q.unsqueeze(2)).squeeze(2)
    # apply atom_mask to ignore padding contributions
    rhs[:, :N, 0] = rhs[:, :N, 0] * td.atom_mask.to(A.dtype)

    # α columns: rhs[:, :N, 1+k] = -(atom_type==k+1)
    # one-hot of atom_type_idx
    types = td.atom_type_idx  # (B, N), 1-based; padding has value 0
    for k in range(T):
        mask_k = (types == (k + 1)).to(A.dtype)  # (B, N)
        rhs[:, :N, 1 + k] = -mask_k

    # β columns: rhs[:, :N, 1+T+k] = -(atom_type==k+1) · q
    for k in range(T):
        mask_k = (types == (k + 1)).to(A.dtype)
        rhs[:, :N, 1 + T + k] = -mask_k * q

    # ---- One batched LU solve for the (P) columns at once ----
    J_full = torch.linalg.lu_solve(LU, piv, rhs)  # (B, N+1, P)
    J_q_full = J_full[:, :N, :]  # (B, N, P) — drop Lagrange row

    # Zero out padding rows so they don't get picked up by mask
    J_masked = J_q_full * td.atom_mask.to(A.dtype).unsqueeze(2)
    # Pick only real-atom rows
    J_flat = J_masked[td.atom_mask].numpy()  # (total_real_atoms, P)

    return r_flat, J_flat


def loss_grad_torch(
    x_np: NDArray[np.float64],
    td: TorchDataset,
) -> tuple[float, NDArray[np.float64]]:
    """Sum-of-squares loss + analytic gradient (atom-mean normalised)."""
    r, J = residuals_and_jacobian_torch(x_np, td)
    N = r.size
    loss = float((r * r).sum() / N)
    grad = (2.0 / N) * (J.T @ r)
    return loss, grad


def loss_grad_hessian_torch(
    x_np: NDArray[np.float64],
    td: TorchDataset,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Sum-of-squares loss + analytic gradient + analytic Hessian.

    Uses the implicit-fn-theorem decomposition (see powelleem.hessian).
    The Hessian = (2/N) (Jᵀ J + Σᵢ rᵢ · ∇²rᵢ).
    """
    torch = _ensure_torch()
    x = torch.from_numpy(np.asarray(x_np, dtype=np.float64))

    A, b = td.build_A_b(x)
    LU, piv = torch.linalg.lu_factor(A)
    y = torch.linalg.lu_solve(LU, piv, b.unsqueeze(2)).squeeze(2)
    q = y[:, : td.N_max]
    r_full = (q - td.target_q) * td.atom_mask.to(q.dtype)
    r_flat = r_full[td.atom_mask].numpy()

    B = td.inv_r.shape[0]
    N = td.N_max
    T = td.n_types
    P = 1 + 2 * T
    dtype = A.dtype

    # ---- First derivatives y_p (B, N+1, P) ----
    rhs1 = torch.zeros(B, N + 1, P, dtype=dtype)
    rhs1[:, :N, 0] = -(td.inv_r @ q.unsqueeze(2)).squeeze(2) * td.atom_mask.to(dtype)
    types = td.atom_type_idx
    for k in range(T):
        mask_k = (types == (k + 1)).to(dtype)
        rhs1[:, :N, 1 + k] = -mask_k
        rhs1[:, :N, 1 + T + k] = -mask_k * q
    Y_p = torch.linalg.lu_solve(LU, piv, rhs1)  # (B, N+1, P)
    J_q = Y_p[:, :N, :] * td.atom_mask.to(dtype).unsqueeze(2)
    J_flat = J_q[td.atom_mask].numpy()

    # ---- Hessian: build second-derivative RHS for non-zero pair classes ----
    # Pair indexing — we'll fill (P, P) upper-triangular into H_q[B, N, P, P].
    pair_list: list[tuple[int, int]] = []
    rhs_list: list[object] = []

    # apply_kappa(v) = inv_r @ v[:n]   (Lagrange row stays 0)
    def apply_kappa(v_full: object) -> object:  # (B, N+1)
        out = torch.zeros_like(v_full)
        out[:, :N] = (td.inv_r @ v_full[:, :N].unsqueeze(2)).squeeze(2)
        out[:, :N] = out[:, :N] * td.atom_mask.to(dtype)
        return out

    # apply_beta_k(v) = diag(mask_k) on the n-block, 0 on Lagrange
    def apply_beta_k(k: int, v_full: object) -> object:
        out = torch.zeros_like(v_full)
        mask_k = (types == (k + 1)).to(dtype)
        out[:, :N] = mask_k * v_full[:, :N]
        return out

    # (κ, κ)
    pair_list.append((0, 0))
    rhs_list.append(-2.0 * apply_kappa(Y_p[:, :, 0]))
    # (κ, α_k)
    for k in range(T):
        pair_list.append((0, 1 + k))
        rhs_list.append(-apply_kappa(Y_p[:, :, 1 + k]))
    # (κ, β_k)
    for k in range(T):
        pair_list.append((0, 1 + T + k))
        rhs_list.append(
            -(apply_kappa(Y_p[:, :, 1 + T + k]) + apply_beta_k(k, Y_p[:, :, 0]))
        )
    # (α_j, β_k) — all T·T pairs (none zero in general)
    for j in range(T):
        for k in range(T):
            pair_list.append((1 + j, 1 + T + k))
            rhs_list.append(-apply_beta_k(k, Y_p[:, :, 1 + j]))
    # (β_j, β_k) for j <= k
    for j in range(T):
        for k in range(j, T):
            pair_list.append((1 + T + j, 1 + T + k))
            if j == k:
                rhs_list.append(-2.0 * apply_beta_k(j, Y_p[:, :, 1 + T + j]))
            else:
                rhs_list.append(
                    -(apply_beta_k(j, Y_p[:, :, 1 + T + k])
                      + apply_beta_k(k, Y_p[:, :, 1 + T + j]))
                )

    # Stack RHS along last axis: (B, N+1, n_pairs)
    rhs2 = torch.stack(rhs_list, dim=-1)
    Y_pq = torch.linalg.lu_solve(LU, piv, rhs2)  # (B, N+1, n_pairs)
    H_q_pairs = Y_pq[:, :N, :] * td.atom_mask.to(dtype).unsqueeze(2)
    # (real_atoms, n_pairs)
    H_pair_flat = H_q_pairs[td.atom_mask].numpy()

    # Assemble J^T J + sum_i r_i · ∇²r_i_pq
    N_atoms = r_flat.size
    JtJ = J_flat.T @ J_flat
    second_corr = np.zeros((P, P), dtype=np.float64)
    for idx, (p, qj) in enumerate(pair_list):
        contrib = float((r_flat * H_pair_flat[:, idx]).sum())
        second_corr[p, qj] += contrib
        if p != qj:
            second_corr[qj, p] += contrib
    # (α_j, α_k) pairs are zero by EEM linearity → already zero in second_corr.

    loss = float((r_flat * r_flat).sum() / N_atoms)
    grad = (2.0 / N_atoms) * (J_flat.T @ r_flat)
    H = (2.0 / N_atoms) * (JtJ + second_corr)
    return loss, grad, H
