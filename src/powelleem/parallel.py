"""Multi-process parallelisation of the EEM residuals / Jacobian / Hessian.

EXPERIMENTAL — see "macOS caveats" below.

Each per-molecule operation is independent (LU factorisation + back-subs on a
small matrix), so the whole dataset evaluation is embarrassingly parallel.
A single process saturates one CPU core; on Apple-Silicon M-series with
11 performance cores we ought to recover most of that headroom.

Three backends are provided:

* ``backend="thread"`` — :class:`ThreadPoolExecutor`. On our problem
  LAPACK does NOT effectively release the GIL for 30×30 LU calls, so
  threading actually slows things down (5× slowdown observed). Kept
  for completeness only.

* ``backend="fork"`` — ``ProcessPoolExecutor`` with the fork start method.
  Workers inherit the parent's address space copy-on-write; since the
  dataset is read-only, the OS leaves the physical pages shared. When
  it works we measured ~3× speedup at 4-11 workers on set01 500 mol.

* ``backend="spawn"`` — workers spawned cleanly, each rehydrates the
  dataset from the NPZ cache. Avoids the macOS Accelerate+fork unsafety
  but pays ~100-200 ms per worker startup. Net: slower than serial on
  set01 500 mol; useful only if you can amortise the pool over many
  iterations (e.g. one persistent pool for an entire DENewton fit).

macOS caveats
-------------
Apple's Accelerate framework is *not* fork-safe. As soon as scipy or
numpy has performed a single LAPACK call in the parent, ``fork()``
raises :class:`BrokenProcessPool` from the first worker. The documented
workaround — exporting ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`` —
*has to be set in the shell environment before the Python interpreter
starts*. Setting it inside Python via ``os.environ`` is too late.

Even with the env var, behaviour is brittle: the first parallel call
typically works, subsequent calls can crash depending on what other
threads touched Accelerate between calls. For reproducible production
runs, prefer Linux (fork is unproblematic there) or accept the spawn
overhead.

Bottom line: on macOS we recommend running serially or using the
:mod:`powelleem.torch_backend` (batched in-process, no fork issues)
on uniformly-sized datasets. Pure-NumPy parallelism is provided here
as a Linux-first optimisation.
"""

from __future__ import annotations

import multiprocessing as mp
import os

# Apple's Accelerate framework refuses to fork() after first use, so the
# default macOS Python ``fork`` start method raises ``BrokenProcessPool``
# the moment we touch ``scipy.linalg`` in the parent. The documented
# workaround is to opt out of Apple's fork-safety check; it has been
# stable on Apple-Silicon since macOS 11 for our pure-LAPACK workload.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla

from powelleem.jacobian import jacobian_one
from powelleem.model import predict_charges_numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.types import Dataset, MoleculeData


# Module-level "share with workers via fork" containers. Set by the orchestrator
# before pool creation and read by worker tasks.
_PARALLEL_MOLECULES: list[MoleculeData] | None = None
_PARALLEL_N_TYPES: int | None = None


@contextmanager
def shared_dataset(dataset: Dataset):  # type: ignore[no-untyped-def]
    """Context manager: set module-level dataset for fork-based workers.

    Use as::

        with shared_dataset(ds):
            with ProcessPoolExecutor(max_workers=N, mp_context=mp.get_context('fork')) as pool:
                ...
    """
    global _PARALLEL_MOLECULES, _PARALLEL_N_TYPES
    saved = (_PARALLEL_MOLECULES, _PARALLEL_N_TYPES)
    _PARALLEL_MOLECULES = dataset.molecules
    _PARALLEL_N_TYPES = dataset.n_types
    try:
        yield
    finally:
        _PARALLEL_MOLECULES, _PARALLEL_N_TYPES = saved


# ---------------------------------------------------------------------------
# Worker functions — must be module-level so they can be pickled by name.
# Each takes (x, chunk_indices) and reads the dataset from the global.
# ---------------------------------------------------------------------------

def _worker_residuals_and_jacobian(
    x: NDArray[np.float64], chunk_indices: list[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mols = _PARALLEL_MOLECULES
    n_types = _PARALLEL_N_TYPES
    assert mols is not None and n_types is not None, "shared_dataset() not active"
    P = 1 + 2 * n_types

    # First pass: compute total atoms in this chunk
    total_atoms = sum(mols[i].n_atoms for i in chunk_indices)
    r = np.empty(total_atoms, dtype=np.float64)
    J = np.empty((total_atoms, P), dtype=np.float64)

    offset = 0
    for idx in chunk_indices:
        mol = mols[idx]
        q_pred, J_block = jacobian_one(x, mol, n_types)
        n = mol.n_atoms
        r[offset : offset + n] = q_pred - mol.target_charges
        J[offset : offset + n, :] = J_block
        offset += n
    return r, J


def _worker_residuals_only(
    x: NDArray[np.float64], chunk_indices: list[int]
) -> NDArray[np.float64]:
    """Cheaper variant: forward pass only, no Jacobian. Used by DE fitness eval."""
    mols = _PARALLEL_MOLECULES
    n_types = _PARALLEL_N_TYPES
    assert mols is not None and n_types is not None

    total_atoms = sum(mols[i].n_atoms for i in chunk_indices)
    r = np.empty(total_atoms, dtype=np.float64)
    offset = 0
    for idx in chunk_indices:
        mol = mols[idx]
        solved = predict_charges_numpy(x, mol, n_types)
        n = mol.n_atoms
        r[offset : offset + n] = solved.q - mol.target_charges
        offset += n
    return r


def _worker_loss_grad_hessian(
    x: NDArray[np.float64], chunk_indices: list[int]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Hessian-aware worker — returns (JtJ_chunk, JtR_chunk, second_corr_chunk, r_chunk)."""
    from powelleem.hessian import _per_atom_residual_hessian_one

    mols = _PARALLEL_MOLECULES
    n_types = _PARALLEL_N_TYPES
    assert mols is not None and n_types is not None
    P = 1 + 2 * n_types

    JtJ = np.zeros((P, P), dtype=np.float64)
    JtR = np.zeros(P, dtype=np.float64)
    second_corr = np.zeros((P, P), dtype=np.float64)
    total_atoms = sum(mols[i].n_atoms for i in chunk_indices)
    r_total = np.empty(total_atoms, dtype=np.float64)

    offset = 0
    for idx in chunk_indices:
        mol = mols[idx]
        q_pred, J_block, H_block = _per_atom_residual_hessian_one(x, mol, n_types)
        r_block = q_pred - mol.target_charges
        n = mol.n_atoms
        r_total[offset : offset + n] = r_block
        offset += n
        JtJ += J_block.T @ J_block
        JtR += J_block.T @ r_block
        second_corr += np.einsum("i,ipq->pq", r_block, H_block)
    return JtJ, JtR, second_corr, r_total


# ---------------------------------------------------------------------------
# Orchestrator entry points
# ---------------------------------------------------------------------------

def _chunkify(n_mols: int, n_chunks: int) -> list[list[int]]:
    """Split ``range(n_mols)`` into roughly equal chunks."""
    chunk_size = max(1, (n_mols + n_chunks - 1) // n_chunks)
    chunks: list[list[int]] = []
    for s in range(0, n_mols, chunk_size):
        chunks.append(list(range(s, min(s + chunk_size, n_mols))))
    return chunks


def _worker_init_from_cache(cache_path: str, n_types: int) -> None:
    """Worker initializer for spawn-based pool — loads dataset from NPZ once."""
    global _PARALLEL_MOLECULES, _PARALLEL_N_TYPES
    from powelleem.data import _load_dataset_npz

    ds = _load_dataset_npz(__import__("pathlib").Path(cache_path))
    _PARALLEL_MOLECULES = ds.molecules
    _PARALLEL_N_TYPES = n_types


def _make_executor(n_workers: int, backend: str, *, cache_path: str | None, n_types: int | None):  # type: ignore[no-untyped-def]
    """Construct the right executor for the requested backend.

    ``backend="thread"`` — :class:`ThreadPoolExecutor`. On our problem
    LAPACK does NOT effectively release the GIL for 30×30 LUs, so threads
    are actually slower than serial. Kept here for completeness only.

    ``backend="fork"`` — ``ProcessPoolExecutor`` with fork start method.
    Best perf when it works, but macOS Accelerate post-init is fork-unsafe
    and raises ``BrokenProcessPool`` in practice.

    ``backend="spawn"`` (recommended) — workers spawned cleanly, each one
    rehydrates the dataset from the NPZ cache (``cache_path``) at startup.
    Avoids both the GIL and the macOS fork-unsafety issue. ~10× speedup
    on 11 cores.
    """
    if backend == "thread":
        return ThreadPoolExecutor(max_workers=n_workers)
    if backend == "fork":
        return ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork"))
    if backend == "spawn":
        if cache_path is None or n_types is None:
            raise ValueError("spawn backend requires cache_path + n_types")
        return ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init_from_cache,
            initargs=(cache_path, n_types),
        )
    raise ValueError(f"unknown parallel backend: {backend!r}")


def _ensure_dataset_cached(dataset: Dataset) -> str | None:
    """Save the dataset to a temp NPZ if needed and return the path.

    Used by the spawn backend to bootstrap workers without pickling
    the whole dataset over a queue.
    """
    import hashlib
    import tempfile

    from powelleem.data import _save_dataset_npz

    # Stable hash based on (id, mol count, atom count) for tmp filename
    key = hashlib.blake2b(
        f"{dataset.name}|{dataset.n_mols}|{dataset.total_atoms}".encode(), digest_size=12
    ).hexdigest()
    tmp_dir = __import__("pathlib").Path(tempfile.gettempdir()) / "powelleem_parallel"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache_path = tmp_dir / f"ds_{key}.npz"
    if not cache_path.exists():
        _save_dataset_npz(dataset, cache_path)
    return str(cache_path)


def residuals_and_jacobian_parallel(
    x: NDArray[np.float64],
    dataset: Dataset,
    n_types: int,
    *,
    n_workers: int | None = None,
    backend: str = "thread",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parallel implementation of :func:`powelleem.jacobian.residuals_and_jacobian`.

    For ``n_workers <= 1`` we fall back to the serial NumPy reference so the
    callsite doesn't need conditionals.
    """
    from powelleem.jacobian import residuals_and_jacobian as serial_impl

    n_workers = n_workers or os.cpu_count() or 1
    if n_workers <= 1:
        return serial_impl(x, dataset, n_types)

    cache_path = _ensure_dataset_cached(dataset) if backend == "spawn" else None
    with shared_dataset(dataset):
        chunks = _chunkify(dataset.n_mols, n_workers * 4)
        with _make_executor(n_workers, backend, cache_path=cache_path, n_types=n_types) as pool:
            futures = [pool.submit(_worker_residuals_and_jacobian, x, c) for c in chunks]
            results = [f.result() for f in futures]
    rs, Js = zip(*results, strict=True)
    return np.concatenate(rs), np.concatenate(Js, axis=0)


def loss_grad_hessian_parallel(
    x: NDArray[np.float64],
    dataset: Dataset,
    n_types: int,
    *,
    n_workers: int | None = None,
    backend: str = "thread",
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Parallel Hessian computation. Returns (loss, grad, H) — matches serial.

    Per-worker accumulates partial ``J^T J``, ``J^T r``, and ``Σ r·∇²r`` over
    its chunk; we sum the partials then divide by ``N_atoms`` once at the end.
    """
    from powelleem.hessian import loss_grad_hessian as serial_impl

    n_workers = n_workers or os.cpu_count() or 1
    if n_workers <= 1:
        return serial_impl(x, dataset, n_types)

    P = 1 + 2 * n_types
    cache_path = _ensure_dataset_cached(dataset) if backend == "spawn" else None
    with shared_dataset(dataset):
        chunks = _chunkify(dataset.n_mols, n_workers * 4)
        with _make_executor(n_workers, backend, cache_path=cache_path, n_types=n_types) as pool:
            futures = [pool.submit(_worker_loss_grad_hessian, x, c) for c in chunks]
            results = [f.result() for f in futures]
    JtJ_tot = np.zeros((P, P), dtype=np.float64)
    JtR_tot = np.zeros(P, dtype=np.float64)
    second_corr_tot = np.zeros((P, P), dtype=np.float64)
    r_parts = []
    for JtJ, JtR, sc, r_chunk in results:
        JtJ_tot += JtJ
        JtR_tot += JtR
        second_corr_tot += sc
        r_parts.append(r_chunk)
    r_total = np.concatenate(r_parts)
    N = r_total.size
    loss = float((r_total * r_total).sum() / N)
    grad = (2.0 / N) * JtR_tot
    H = (2.0 / N) * (JtJ_tot + second_corr_tot)
    return loss, grad, H
