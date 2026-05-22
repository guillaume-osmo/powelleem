"""Numba JIT backend — parallel per-molecule loop without fork or GIL.

Numba's ``@njit(parallel=True)`` compiles a Python loop into LLVM bitcode
with OpenMP threading. Each ``prange`` iteration runs on its own thread
without holding the GIL, and without the macOS Accelerate fork-unsafety
issue that plagues :mod:`powelleem.parallel`.

For our problem (17 769 molecules × ~30 atoms each, 19 atom types), the
inner work is one LU + back-substitutions per molecule. ``np.linalg.solve``
is supported by Numba and dispatches to LAPACK; combined with ``prange``,
we get near-linear scaling across the 11 P-cores on Apple-Silicon M-series.

Workflow:

1. Convert a :class:`powelleem.Dataset` to flat ragged-storage NumPy
   arrays via :func:`build_numba_dataset`. One-time cost ≈ O(total_atoms).
2. Call :func:`residuals_and_jacobian_numba(x, nd)` from the solvers.
   First call triggers JIT compilation (~5-10 s); subsequent calls are
   pure native code.

The math is identical to :mod:`powelleem.jacobian`; we cross-check at
the test level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.types import Dataset


def _ensure_numba():  # type: ignore[no-untyped-def]
    try:
        import numba

        return numba
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Numba is required for the numba backend. "
            "Install with `pip install numba>=0.59`."
        ) from exc


@dataclass(slots=True)
class NumbaDataset:
    """Flat ragged-storage representation of a :class:`Dataset` for Numba kernels.

    - ``inv_r_flat``      : concatenated ``mol.inv_r.reshape(-1)`` over all mols
    - ``inv_r_offsets``   : ``(B+1,)`` offsets into ``inv_r_flat``
    - ``atom_type_idx``   : flat 1-based atom-type indices, length ``total_atoms``
    - ``atom_offsets``    : ``(B+1,)`` offsets into ``atom_type_idx``
    - ``target_q``        : flat reference charges, length ``total_atoms``
    - ``q_total``         : per-molecule total charge, shape ``(B,)``
    - ``n_atoms``         : per-molecule atom count, shape ``(B,)``
    - ``n_types``         : number of atom types T
    - ``total_atoms``     : sum of n_atoms
    """

    inv_r_flat: NDArray[np.float64]
    inv_r_offsets: NDArray[np.int64]
    atom_type_idx: NDArray[np.int64]
    atom_offsets: NDArray[np.int64]
    target_q: NDArray[np.float64]
    q_total: NDArray[np.float64]
    n_atoms: NDArray[np.int64]
    n_types: int
    total_atoms: int


def build_numba_dataset(dataset: Dataset) -> NumbaDataset:
    B = dataset.n_mols
    n_atoms_list = [m.n_atoms for m in dataset.molecules]
    total_atoms = sum(n_atoms_list)
    total_inv_r = sum(n * n for n in n_atoms_list)

    inv_r_flat = np.empty(total_inv_r, dtype=np.float64)
    inv_r_offsets = np.zeros(B + 1, dtype=np.int64)
    atom_type_idx = np.empty(total_atoms, dtype=np.int64)
    atom_offsets = np.zeros(B + 1, dtype=np.int64)
    target_q = np.empty(total_atoms, dtype=np.float64)
    q_total = np.zeros(B, dtype=np.float64)
    n_atoms = np.zeros(B, dtype=np.int64)

    inv_r_pos = 0
    atom_pos = 0
    for i, mol in enumerate(dataset.molecules):
        n = mol.n_atoms
        n_atoms[i] = n
        inv_r_offsets[i + 1] = inv_r_offsets[i] + n * n
        atom_offsets[i + 1] = atom_offsets[i] + n
        inv_r_flat[inv_r_pos : inv_r_pos + n * n] = mol.inv_r.reshape(-1)
        atom_type_idx[atom_pos : atom_pos + n] = mol.atom_types
        target_q[atom_pos : atom_pos + n] = mol.target_charges
        q_total[i] = mol.formal_charge
        inv_r_pos += n * n
        atom_pos += n

    return NumbaDataset(
        inv_r_flat=inv_r_flat,
        inv_r_offsets=inv_r_offsets,
        atom_type_idx=atom_type_idx,
        atom_offsets=atom_offsets,
        target_q=target_q,
        q_total=q_total,
        n_atoms=n_atoms,
        n_types=dataset.n_types,
        total_atoms=total_atoms,
    )


# ---------------------------------------------------------------------------
# JIT kernels — built lazily so import doesn't require Numba unless used.
# ---------------------------------------------------------------------------

_CACHED_KERNELS: dict[str, object] = {}


def _kernels():  # type: ignore[no-untyped-def]
    """Lazily build and cache the JIT-compiled kernels."""
    if "rj" in _CACHED_KERNELS:
        return _CACHED_KERNELS

    numba = _ensure_numba()
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)
    def _kernel_residuals_and_jacobian(
        kappa,
        alpha,  # (T,)
        beta,   # (T,)
        inv_r_flat,
        inv_r_offsets,
        atom_type_idx,
        atom_offsets,
        target_q,
        q_total,
        n_atoms,
        r_out,      # (total_atoms,)  — output residuals
        J_out,      # (total_atoms, P) — output Jacobian (P = 1 + 2*T)
    ):
        B = n_atoms.shape[0]
        T = alpha.shape[0]
        P = 1 + 2 * T
        for mol_idx in prange(B):
            n = n_atoms[mol_idx]
            atom_off = atom_offsets[mol_idx]
            inv_off = inv_r_offsets[mol_idx]
            np1 = n + 1

            # Build (n+1, n+1) A
            A = np.zeros((np1, np1))
            for i in range(n):
                t_i = atom_type_idx[atom_off + i] - 1
                A[i, i] = beta[t_i]
                for j in range(n):
                    if i != j:
                        A[i, j] = kappa * inv_r_flat[inv_off + i * n + j]
                A[i, n] = 1.0
                A[n, i] = 1.0
            # A[n, n] = 0 by zeros() init

            # Build RHS bundle: (n+1, 1 + P)
            #   column 0    : b (forward), -alpha for atoms, q_total for Lagrange
            #   column 1    : -inv_r · q   (for κ Jacobian — depends on q, computed below after forward solve)
            #   columns 2..1+T  : -1[type=k] (for α_k Jacobian columns)
            #   columns 1+T..1+2T : -1[type=k] · q (for β_k columns — depends on q)
            #
            # We do two passes: forward solve first (to get q), then assemble
            # the Jacobian-dependent RHS columns.

            # Forward solve
            b = np.zeros(np1)
            for i in range(n):
                t_i = atom_type_idx[atom_off + i] - 1
                b[i] = -alpha[t_i]
            b[n] = q_total[mol_idx]
            y = np.linalg.solve(A, b)
            # q = y[:n]

            # Residuals
            for i in range(n):
                r_out[atom_off + i] = y[i] - target_q[atom_off + i]

            # ---- Jacobian via implicit fn theorem ----
            # Build (n+1, P) RHS for Jq columns
            J_rhs = np.zeros((np1, P))
            # κ column: -inv_r · q  (Lagrange row stays 0)
            for i in range(n):
                acc = 0.0
                for j in range(n):
                    if i != j:
                        acc += inv_r_flat[inv_off + i * n + j] * y[j]
                J_rhs[i, 0] = -acc
            # α columns: -1[type=k] for atom of type k+1
            for i in range(n):
                t_i = atom_type_idx[atom_off + i] - 1
                J_rhs[i, 1 + t_i] = -1.0
            # β columns: -1[type=k] * q
            for i in range(n):
                t_i = atom_type_idx[atom_off + i] - 1
                J_rhs[i, 1 + T + t_i] = -y[i]

            Y_jac = np.linalg.solve(A, J_rhs)
            for i in range(n):
                for p in range(P):
                    J_out[atom_off + i, p] = Y_jac[i, p]

    _CACHED_KERNELS["rj"] = _kernel_residuals_and_jacobian
    return _CACHED_KERNELS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def residuals_and_jacobian_numba(
    x: NDArray[np.float64], nd: NumbaDataset
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """JIT-parallel residuals + analytical Jacobian.

    First call triggers Numba compilation (~5-10 s). Subsequent calls are
    pure native code. Caches the compiled kernel in module memory + on disk
    (Numba ``cache=True``) so a fresh interpreter only recompiles once.
    """
    kappa = float(x[0])
    T = nd.n_types
    alpha = np.ascontiguousarray(x[1 : 1 + T], dtype=np.float64)
    beta = np.ascontiguousarray(x[1 + T :], dtype=np.float64)
    P = 1 + 2 * T

    r = np.empty(nd.total_atoms, dtype=np.float64)
    J = np.empty((nd.total_atoms, P), dtype=np.float64)

    kernels = _kernels()
    kernels["rj"](
        kappa,
        alpha,
        beta,
        nd.inv_r_flat,
        nd.inv_r_offsets,
        nd.atom_type_idx,
        nd.atom_offsets,
        nd.target_q,
        nd.q_total,
        nd.n_atoms,
        r,
        J,
    )
    return r, J


def loss_and_grad_numba(
    x: NDArray[np.float64], nd: NumbaDataset
) -> tuple[float, NDArray[np.float64]]:
    """Atom-mean-square loss + analytic gradient via Numba kernel."""
    r, J = residuals_and_jacobian_numba(x, nd)
    N = r.size
    loss = float((r * r).sum() / N)
    grad = (2.0 / N) * (J.T @ r)
    return loss, grad
