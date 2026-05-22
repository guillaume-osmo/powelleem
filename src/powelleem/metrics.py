"""NEEMP-compatible statistics for direct comparison with Raček 2016.

Mirrors the metric definitions in NEEMP's ``src/statistics.c`` exactly:

* ``rmsd``      — mean over molecules of per-molecule RMSD
                  ``(1/M) Σ_m sqrt((1/n_m) Σ_i r_i²)``
* ``rmsd_avg``  — mean over atom-type classes of per-class RMSD
                  ``(1/T) Σ_t sqrt(MSE_t)``
* ``log_rmsd``  — like ``rmsd`` but on ``log|r|`` (clipped at ε to avoid −∞)
* ``r``         — mean over molecules of per-molecule Pearson correlation
* ``r2``        — square of ``r``
* ``rw``        — NEEMP's custom weighted R, faithfully ported (see source)
* ``sp``        — mean over molecules of per-molecule Spearman correlation
* ``d_avg``     — mean over molecules of per-molecule MAE
                  ``(1/M) Σ_m (1/n_m) Σ_i |r_i|``
* ``d_max``     — mean over molecules of per-molecule max |r|

Atom-flat RMSE (the loss we currently optimise) is exposed as
``atom_rmse`` for sanity-checking against the ``rmsd`` value.

Reference: Raček et al., *NEEMP*, J. Cheminform. 8:57 (2016),
``neemp/src/statistics.c``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.types import Dataset


@dataclass(slots=True)
class MetricsReport:
    """Container for full (overall + per-element) NEEMP-style metrics."""

    kappa: float
    overall: dict[str, float] = field(default_factory=dict)
    per_element: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_neemp_string(self) -> str:
        """Render in NEEMP's output format for direct diff."""
        o = self.overall
        lines = [
            f"K: {self.kappa:.4f} |  R: {o['r']:.4f}  R2: {o['r2']:.4f}  RW: {o['rw']:.4f}  "
            f"Sp: {o['sp']:.4f}  RMSD: {o['rmsd']:.4f}  D_avg: {o['d_avg']:.4f}  D_max: {o['d_max']:.4f}",
            f"Atom type            R      R2      Sp      RMSD    D_avg   D_max",
        ]
        for t, m in self.per_element.items():
            lines.append(
                f"  {t:>10s}       {m['r']:.4f}  {m['r2']:.4f}  {m['sp']:.4f}    "
                f"{m['rmsd']:.4f}   {m['d_avg']:.4f}  {m['d_max']:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-molecule basic stats (small helpers, vectorised over atoms within a mol)
# ---------------------------------------------------------------------------

def _per_mol_rmsd(r: NDArray[np.float64]) -> float:
    return float(np.sqrt((r * r).mean()))


def _per_mol_d_avg(r: NDArray[np.float64]) -> float:
    return float(np.mean(np.abs(r)))


def _per_mol_d_max(r: NDArray[np.float64]) -> float:
    return float(np.max(np.abs(r))) if r.size else 0.0


def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 2:
        return 0.0
    dx = x - x.mean()
    dy = y - y.mean()
    cov_xx = float((dx * dx).sum())
    cov_yy = float((dy * dy).sum())
    if cov_xx * cov_yy <= 0.0:
        return 0.0
    return float((dx * dy).sum() / np.sqrt(cov_xx * cov_yy))


def _spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """NEEMP-style: tie-averaged ranks then Pearson."""
    if x.size < 2:
        return 0.0
    rx = _rank_ties(x)
    ry = _rank_ties(y)
    return _pearson(rx, ry)


def _rank_ties(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """NEEMP-faithful tie-handling: identical values share the average rank."""
    order = np.argsort(a)
    ranks = np.empty_like(a, dtype=np.float64)
    i = 0
    n = a.size
    while i < n:
        j = 1
        while i + j < n and abs(a[order[i]] - a[order[i + j]]) < 1e-5:
            j += 1
        avg_rank = (2.0 * (i + 1) + j - 1) / 2.0  # ranks 1-based
        for k in range(j):
            ranks[order[i + k]] = avg_rank
        i += j
    return ranks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    q_pred_per_mol: list[NDArray[np.float64]],
    dataset: Dataset,
    kappa: float,
    *,
    log_eps: float = 1e-6,
) -> MetricsReport:
    """Compute the full NEEMP-style report.

    ``q_pred_per_mol`` is a list aligned with ``dataset.molecules`` giving the
    predicted atomic charges for each molecule (shape ``(n_atoms_m,)``).
    """
    M = dataset.n_mols
    types = dataset.atom_types

    rmsd_per_mol = np.empty(M)
    d_avg_per_mol = np.empty(M)
    d_max_per_mol = np.empty(M)
    r_per_mol = np.empty(M)
    sp_per_mol = np.empty(M)

    # For log-RMSD: use atom-flat aggregation since per-mol log-of-residual
    # is dominated by the molecule with the smallest non-zero residual and
    # tends to NaN.  We still report it under "log_rmsd".

    # Per-element accumulators
    per_elem_r: dict[str, list[float]] = {t: [] for t in types}
    per_elem_diffs: dict[str, list[float]] = {t: [] for t in types}
    per_elem_preds: dict[str, list[float]] = {t: [] for t in types}
    per_elem_refs: dict[str, list[float]] = {t: [] for t in types}

    all_diffs: list[float] = []

    for m_idx, mol in enumerate(dataset.molecules):
        q_pred = q_pred_per_mol[m_idx]
        q_ref = mol.target_charges
        diff = q_pred - q_ref
        rmsd_per_mol[m_idx] = _per_mol_rmsd(diff)
        d_avg_per_mol[m_idx] = _per_mol_d_avg(diff)
        d_max_per_mol[m_idx] = _per_mol_d_max(diff)
        r_per_mol[m_idx] = _pearson(q_pred, q_ref)
        sp_per_mol[m_idx] = _spearman(q_pred, q_ref)

        for i, t_idx in enumerate(mol.atom_types):
            t = types[t_idx - 1]
            per_elem_diffs[t].append(float(diff[i]))
            per_elem_preds[t].append(float(q_pred[i]))
            per_elem_refs[t].append(float(q_ref[i]))
            all_diffs.append(float(diff[i]))

    # Overall NEEMP-style values
    r_overall = float(r_per_mol.mean())
    overall = {
        "kappa": kappa,
        "rmsd": float(rmsd_per_mol.mean()),
        "d_avg": float(d_avg_per_mol.mean()),
        "d_max": float(d_max_per_mol.mean()),
        "r": r_overall,
        "r2": r_overall * r_overall,
        "sp": float(sp_per_mol.mean()),
        # Atom-flat (our default loss) for cross-check:
        "atom_rmse": float(np.sqrt(np.mean(np.array(all_diffs) ** 2))),
        "log_rmsd": float(
            np.sqrt(np.mean(np.log(np.abs(np.array(all_diffs)) + log_eps) ** 2))
        ),
    }

    # Per-element metrics
    per_elem: dict[str, dict[str, float]] = {}
    rmsd_per_class_list: list[float] = []
    for t in types:
        diffs = np.array(per_elem_diffs[t], dtype=np.float64)
        preds = np.array(per_elem_preds[t], dtype=np.float64)
        refs = np.array(per_elem_refs[t], dtype=np.float64)
        if diffs.size == 0:
            continue
        rmsd_t = float(np.sqrt((diffs * diffs).mean()))
        rmsd_per_class_list.append(rmsd_t)
        r_t = _pearson(preds, refs)
        per_elem[t] = {
            "rmsd": rmsd_t,
            "d_avg": float(np.mean(np.abs(diffs))),
            "d_max": float(np.max(np.abs(diffs))),
            "r": r_t,
            "r2": r_t * r_t,
            "sp": _spearman(preds, refs),
            "n_atoms": int(diffs.size),
        }
    overall["rmsd_avg"] = float(np.mean(rmsd_per_class_list)) if rmsd_per_class_list else 0.0
    overall["rw"] = _neemp_rw(overall, per_elem)
    return MetricsReport(kappa=kappa, overall=overall, per_element=per_elem)


def _neemp_rw(overall: dict[str, float], per_elem: dict[str, dict[str, float]]) -> float:
    """Faithful port of NEEMP's ``set_total_R_w`` heuristic.

    From ``neemp/src/statistics.c`` lines 93–110::

        weighted_corr_sum = 0
        for each atom type i:
            weighted_corr_sum += weight * per_at_R[i]
            weighted_corr_sum -= per_at_RMSD[i]
        weighted_corr_sum += 3 * (0.5 ** R²) * R² - RMSD / 3
        R_w = weighted_corr_sum / (n_atom_types * 0.5 + 1.5)

    The per-type ``weight`` in the original code is 0.5 (cross-checked
    against NEEMP's source for the typical run).
    """
    weight = 0.5
    s = 0.0
    for t, m in per_elem.items():
        s += weight * m["r"]
        s -= m["rmsd"]
    n_types = max(1, len(per_elem))
    s += 3.0 * (0.5 ** overall["r2"]) * overall["r2"] - overall["rmsd"] / 3.0
    return s / (n_types * 0.5 + 1.5)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def report_metrics(
    model,  # type: ignore[no-untyped-def]
    params,  # type: ignore[no-untyped-def]
    dataset: Dataset,
    *,
    use_numba: bool = True,
) -> MetricsReport:
    """End-to-end: predict charges then compute the full report.

    Switches to the Numba forward kernel when available for speed.
    """
    if use_numba:
        try:
            from powelleem.numba_backend import build_numba_dataset, residuals_only_numba

            nd = build_numba_dataset(dataset)
            r_flat = residuals_only_numba(params.to_vector(), nd)
            # Build per-mol q_pred from residuals + targets
            q_pred_per_mol: list[NDArray[np.float64]] = []
            offset = 0
            for mol in dataset.molecules:
                n = mol.n_atoms
                q_pred_per_mol.append(r_flat[offset : offset + n] + mol.target_charges)
                offset += n
            return compute_metrics(q_pred_per_mol, dataset, kappa=params.kappa)
        except ImportError:
            pass
    # Fallback: NumPy model.predict
    q_pred_per_mol = model.predict(params, dataset)
    return compute_metrics(q_pred_per_mol, dataset, kappa=params.kappa)
