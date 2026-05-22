"""Dataset loaders.

Three input paths are supported:

1. **CHAOS** (``chaos.zip``) — 53k molecules with DFT atomic charges
   (Mulliken, APT, COSMO-segment-integrated).
2. **NEEMP legacy** — the original Račkov 2016 format: ``set.sdf`` for
   geometries + ``set.chg`` for reference charges + ``set.typ`` for
   per-atom NEEMP atom types.
3. **Generic** — an SDF (one or many molecules) + a CSV of reference
   charges aligned by (mol_id, atom_index).

All loaders return a :class:`Dataset` with a *fixed atom-type
vocabulary* — molecules contain integer type indices into
``dataset.atom_types``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray

from powelleem.types import Dataset, MoleculeData

ChaosTarget = Literal["apt", "mulliken", "apt_heavy", "mulliken_heavy", "cosmo"]


# ---------------------------------------------------------------------------
# CHAOS loader
# ---------------------------------------------------------------------------

def _build_inv_r(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse pairwise distance matrix, zero on diagonal."""
    diff = coords[:, None, :] - coords[None, :, :]
    r = np.sqrt((diff * diff).sum(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r = np.where(r > 1e-8, 1.0 / r, 0.0)
    np.fill_diagonal(inv_r, 0.0)
    return inv_r


def _iter_chaos_json(zip_path: Path, limit: int | None = None) -> Iterator[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            (n for n in zf.namelist() if n.endswith(".json")),
            key=lambda n: int(Path(n).stem),
        )
        if limit is not None:
            names = names[:limit]
        for name in names:
            with zf.open(name) as fh:
                yield json.loads(fh.read())


def _extract_chaos_charges(entry: dict, target: ChaosTarget) -> NDArray[np.float64] | None:
    elec = entry.get("electronic", {})
    sol = entry.get("solvation", {})
    if target == "apt":
        v = elec.get("PartChargeAPT")
    elif target == "mulliken":
        v = elec.get("PartChargeMulliken")
    elif target == "apt_heavy":
        v = elec.get("PartChargeAPTHeavy")
    elif target == "mulliken_heavy":
        v = elec.get("PartChargeMullikenHeavy")
    elif target == "cosmo":
        atom_cosmo = sol.get("AtomCOSMOCharge")
        if not atom_cosmo:
            return None
        v = [a["charge"] for a in atom_cosmo]
    else:
        raise ValueError(f"Unknown CHAOS target: {target!r}")
    if v is None:
        return None
    return np.asarray(v, dtype=np.float64)


def load_chaos(
    zip_path: str | Path,
    *,
    n_mols: int | None = None,
    max_n_atoms: int = 50,
    target: ChaosTarget = "apt",
    skip_non_converged: bool = True,
    name: str = "CHAOS",
) -> Dataset:
    """Load a CHAOS subset into a :class:`Dataset`.

    Parameters
    ----------
    zip_path
        Path to the ``CHAOS.zip`` archive.
    n_mols
        If set, stop after that many *successfully-loaded* molecules.
    max_n_atoms
        Skip molecules larger than this (memory/speed cap).
    target
        Which reference-charge column to use as the target.
    skip_non_converged
        Drop molecules with ``general.not_converged == True``.
    name
        Stored on the returned :class:`Dataset` for bookkeeping.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    raw: list[tuple[dict, NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]] = []
    seen_Z: set[int] = set()
    for entry in _iter_chaos_json(zip_path):
        if skip_non_converged and entry["general"].get("not_converged"):
            continue
        atoms = entry["general"]["AtomList"]
        if len(atoms) > max_n_atoms:
            continue
        coords = np.asarray(entry["structural"]["Coordinates"], dtype=np.float64)
        if coords.shape != (len(atoms), 3):
            continue
        q_ref = _extract_chaos_charges(entry, target)
        if q_ref is None or q_ref.shape[0] != len(atoms):
            continue
        atomic_nums = np.asarray([a["atomic_number"] for a in atoms], dtype=np.int64)
        seen_Z.update(int(z) for z in atomic_nums)
        raw.append((entry, coords, atomic_nums, q_ref))
        if n_mols is not None and len(raw) >= n_mols:
            break

    # Build atom-type vocabulary in periodic-table order (atomic-number ascending).
    sorted_Z = sorted(seen_Z)
    type_strs = tuple(_z_to_symbol(z) for z in sorted_Z)
    z_to_type_idx = {z: i + 1 for i, z in enumerate(sorted_Z)}

    molecules: list[MoleculeData] = []
    for entry, coords, atomic_nums, q_ref in raw:
        type_idx = np.asarray([z_to_type_idx[int(z)] for z in atomic_nums], dtype=np.int64)
        molecules.append(
            MoleculeData(
                smiles=entry["general"]["CanonicalSMILES"],
                atom_types=type_idx,
                inv_r=_build_inv_r(coords),
                target_charges=q_ref,
                formal_charge=float(entry["electronic"].get("Charge", 0)),
                metadata={"source": "CHAOS", "target": target},
            )
        )

    return Dataset(
        molecules=molecules,
        atom_types=type_strs,
        name=name,
        metadata={"source": "CHAOS", "zip_path": str(zip_path), "target": target},
    )


def _z_to_symbol(z: int) -> str:
    # Compact lookup for the common subset; falls back to "Z<n>" otherwise.
    symbols = {
        1: "H", 2: "He",
        3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
        19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
        27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
        35: "Br", 36: "Kr", 51: "Sb", 53: "I",
    }
    return symbols.get(z, f"Z{z}")


# ---------------------------------------------------------------------------
# NEEMP legacy loader (Račkov 2016) — .chg / .typ / .sdf triplet
# ---------------------------------------------------------------------------

def load_neemp_legacy(
    sdf_path: str | Path,
    chg_path: str | Path,
    typ_path: str | Path,
    *,
    name: str = "NEEMP-legacy",
) -> Dataset:
    """Load a NEEMP-format dataset: SDF + ``.chg`` + ``.typ`` triplet.

    The NEEMP files use a custom whitespace format where ``.chg`` contains
    one line per atom with ``(mol_id, atom_idx, charge)`` and ``.typ``
    contains ``(mol_id, atom_idx, element, neemp_type)``.

    Requires RDKit for SDF parsing.
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RDKit is required for the NEEMP legacy loader. "
            "Install with `pip install powelleem[rdkit]`."
        ) from exc

    # Stub — full implementation reads the three files and aligns by mol_id.
    raise NotImplementedError(
        "NEEMP legacy loader is scaffolded but not yet implemented in v0.1.0a0. "
        "Contributions welcome — see `data.load_chaos` for the reference pattern."
    )


# ---------------------------------------------------------------------------
# Generic loader — SDF + CSV
# ---------------------------------------------------------------------------

def load_generic_sdf_csv(
    sdf_path: str | Path,
    charges_csv: str | Path,
    *,
    mol_id_col: str = "mol_id",
    atom_idx_col: str = "atom_idx",
    charge_col: str = "charge",
    name: str = "generic",
) -> Dataset:
    """Load arbitrary SDF + CSV of reference charges.

    The CSV must have at least three columns: ``mol_id``, ``atom_idx``
    (0-based), and ``charge``. SDF mol order determines mol_id.
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RDKit is required for the generic SDF+CSV loader. "
            "Install with `pip install powelleem[rdkit]`."
        ) from exc

    raise NotImplementedError(
        "Generic SDF+CSV loader is scaffolded for v0.1.0a0. See `load_chaos` for "
        "the reference pattern."
    )
