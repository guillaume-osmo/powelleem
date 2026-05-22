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

def _strip_name_prefix(name: str) -> str:
    """Normalise mol-name conventions across NEEMP set variants.

    * set01 names look like ``NSC_100000`` → kept as-is (SDF agrees).
    * set02 names look like ``N000`` or ``N00C`` (chg/typ) vs bare ``000``
      / ``00C`` (sdf) — strip the leading ``N`` whenever the suffix is
      pure alphanumeric (no underscore, no further punctuation).
    * set03 names look like ``NAME:000`` (chg/typ) vs bare ``000`` (sdf)
      → strip the ``NAME:`` prefix.
    """
    if name.upper().startswith("NAME:"):
        return name.split(":", 1)[1].strip()
    # set02-style: starts with N and the suffix is alphanumeric only (no underscore).
    if (
        name.startswith("N")
        and len(name) > 1
        and name[1:].isalnum()
        and "_" not in name
    ):
        return name[1:]
    return name


def _parse_neemp_chg(path: Path) -> dict[str, NDArray[np.float64]]:
    """Parse a NEEMP ``.chg`` file.

    Two block layouts are supported:

    Layout A (set01/set02)::

        NSC_100000           ← mol name (bare token)
        29                   ← number of atoms
             1  N   -0.812377
             ...

    Layout B (set03, with ``$$$$`` separators)::

        NAME:000             ← mol name (may contain ':')
        9
             1  C    0.97509
             ...
        $$$$                 ← record terminator
        NAME:001
        ...
    """
    out: dict[str, NDArray[np.float64]] = {}
    with path.open() as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "$$$$":
            i += 1
            continue
        # Normalise: strip ``NAME:`` (set03) and bare ``N`` prefix (set02) so
        # chg/typ keys match the SDF's bare numeric names.
        name = _strip_name_prefix(line)
        i += 1
        # Next non-blank, non-separator line is the atom count, optionally
        # prefixed with "NATO:" (set03 uses both ``9`` and ``NATO:53`` forms).
        while i < len(lines) and (not lines[i].strip() or lines[i].strip() == "$$$$"):
            i += 1
        if i >= len(lines):
            break
        raw_n = lines[i].strip()
        if raw_n.upper().startswith("NATO:"):
            raw_n = raw_n.split(":", 1)[1].strip()
        n = int(raw_n)
        i += 1
        charges = np.empty(n, dtype=np.float64)
        for j in range(n):
            parts = lines[i].split()
            charges[j] = float(parts[2])
            i += 1
        out[name] = charges
    return out


def _parse_neemp_typ(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Parse a NEEMP ``.typ`` file.

    Two layouts supported:

    Layout A (set01/set02)::

        NSC_100000              ← bare token
           1   N   1
           2   O   2
           ...

    Layout B (set03, count line + ``$$$$`` separator)::

        NAME:000
        9                       ← atom count (skipped during parse)
           1   C   2
           ...
        $$$$

    The third column is the bond-order class (1 for single, 2 for double,
    1.5 for aromatic, 3 for triple).
    """
    out: dict[str, list[tuple[str, str]]] = {}
    with path.open() as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "$$$$":
            i += 1
            continue
        # Normalise across set01/set02/set03 conventions.
        name = _strip_name_prefix(line)
        i += 1
        atoms: list[tuple[str, str]] = []
        while i < len(lines):
            row = lines[i].strip()
            if not row:
                i += 1
                continue
            if row == "$$$$":
                break
            parts = row.split()
            # Skip optional atom-count line ("9" or "NATO:53")
            if len(parts) == 1 and (
                parts[0].isdigit() or parts[0].upper().startswith("NATO:")
            ):
                i += 1
                continue
            # An atom-row has ≥ 3 fields and starts with an integer index.
            if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
                # Next molecule's name; break (don't consume).
                break
            atoms.append((parts[1], parts[2]))
            i += 1
        out[name] = atoms
    return out


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "powelleem" / "neemp"


def _neemp_cache_key(sdf: Path, chg: Path, typ: Path, limit: int | None) -> str:
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for p in (sdf, chg, typ):
        st = p.stat()
        h.update(str(p).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime_ns)).encode())
    h.update(str(limit).encode())
    return h.hexdigest()


def _save_dataset_npz(ds: Dataset, path: Path) -> None:
    """Serialise a :class:`Dataset` as a single .npz (much faster than pickle).

    Layout: flat concatenated atom-axis arrays + per-mol offsets + atom-type
    vocabulary + per-mol scalars + per-mol metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_mols = ds.n_mols
    n_atoms = np.array([m.n_atoms for m in ds.molecules], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_atoms)]).astype(np.int64)

    atom_types_flat = np.concatenate(
        [m.atom_types for m in ds.molecules], dtype=np.int64
    )
    target_q_flat = np.concatenate(
        [m.target_charges for m in ds.molecules], dtype=np.float64
    )
    formal_q = np.array([m.formal_charge for m in ds.molecules], dtype=np.float64)

    # Per-mol inv_r matrices: flatten with offsets in a second array.
    inv_r_flat = np.concatenate(
        [m.inv_r.reshape(-1) for m in ds.molecules], dtype=np.float64
    )
    inv_r_offsets = np.concatenate(
        [[0], np.cumsum([m.inv_r.size for m in ds.molecules])]
    ).astype(np.int64)

    smiles = np.array([m.smiles for m in ds.molecules], dtype=object)
    mol_names = np.array(
        [str(m.metadata.get("name", "")) for m in ds.molecules], dtype=object
    )
    atom_type_strs = np.array(ds.atom_types, dtype=object)

    np.savez(
        path,
        n_atoms=n_atoms,
        offsets=offsets,
        atom_types_flat=atom_types_flat,
        target_q_flat=target_q_flat,
        formal_q=formal_q,
        inv_r_flat=inv_r_flat,
        inv_r_offsets=inv_r_offsets,
        smiles=smiles,
        mol_names=mol_names,
        atom_type_strs=atom_type_strs,
        name=np.array(ds.name, dtype=object),
        metadata_keys=np.array(list(ds.metadata.keys()), dtype=object),
        metadata_vals=np.array([str(v) for v in ds.metadata.values()], dtype=object),
    )


def _load_dataset_npz(path: Path) -> Dataset:
    """Inverse of :func:`_save_dataset_npz`. Reconstructs the full Dataset."""
    d = np.load(path, allow_pickle=True)
    n_atoms = d["n_atoms"]
    offsets = d["offsets"]
    atom_types_flat = d["atom_types_flat"]
    target_q_flat = d["target_q_flat"]
    formal_q = d["formal_q"]
    inv_r_flat = d["inv_r_flat"]
    inv_r_offsets = d["inv_r_offsets"]
    smiles = d["smiles"]
    mol_names = d["mol_names"]
    atom_type_strs = tuple(str(s) for s in d["atom_type_strs"])

    molecules: list[MoleculeData] = []
    for i in range(len(n_atoms)):
        n = int(n_atoms[i])
        inv_r = inv_r_flat[inv_r_offsets[i] : inv_r_offsets[i + 1]].reshape(n, n)
        molecules.append(
            MoleculeData(
                smiles=str(smiles[i]),
                atom_types=atom_types_flat[offsets[i] : offsets[i + 1]],
                inv_r=inv_r,
                target_charges=target_q_flat[offsets[i] : offsets[i + 1]],
                formal_charge=float(formal_q[i]),
                metadata={"source": "NEEMP", "name": str(mol_names[i])},
            )
        )

    meta_keys = [str(k) for k in d["metadata_keys"]]
    meta_vals = [str(v) for v in d["metadata_vals"]]
    return Dataset(
        molecules=molecules,
        atom_types=atom_type_strs,
        name=str(d["name"]),
        metadata=dict(zip(meta_keys, meta_vals, strict=True)),
    )


_CHAOS_CACHE_DIR = Path.home() / ".cache" / "powelleem" / "chaos"


def _chaos_cache_key(
    zip_path: Path,
    n_mols: int | None,
    target: str,
    contains_element: int | None,
    max_n_atoms: int,
) -> str:
    import hashlib

    st = zip_path.stat()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(zip_path).encode())
    h.update(str(st.st_size).encode())
    h.update(f"{n_mols}|{target}|{contains_element}|{max_n_atoms}".encode())
    return h.hexdigest()


def load_chaos(
    zip_path: str | Path,
    *,
    n_mols: int | None = None,
    max_n_atoms: int = 50,
    target: str = "apt",
    contains_element: int | None = None,
    skip_non_converged: bool = True,
    name: str = "CHAOS",
    cache_dir: str | Path | None = _CHAOS_CACHE_DIR,
    use_cache: bool = True,
) -> Dataset:
    """Load a CHAOS subset (Computed High-Accuracy Observables and Sigma-profiles).

    CHAOS (Raček-independent, 53,091 mols, ωB97X-D / def2-TZVP + C-PCM) ships
    atomic charges (Mulliken, APT) plus per-atom COSMO surface charges in
    one JSON per molecule. We extract:

    - ``structural.Coordinates``     → atom xyz (Å), for inv_r
    - ``general.AtomList``           → element + atomic_number per atom
    - ``electronic.PartChargeAPT``   → target ``target="apt"``  (default)
    - ``electronic.PartChargeMulliken`` → target ``target="mulliken"``
    - ``solvation.AtomCOSMOCharge``  → target ``target="cosmo"`` (≈ COSMO-screened)

    ``contains_element=53`` filters to iodine-containing molecules only — useful
    for fitting an iodine-specific EEM parameter set (NEEMP CCD_gen did not
    include iodine).

    NPZ cache: ``~/.cache/powelleem/chaos/<hash>.npz`` (saves the ~minutes of
    ZIP streaming + JSON parsing).
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    if use_cache and cache_dir is not None:
        key = _chaos_cache_key(zip_path, n_mols, target, contains_element, max_n_atoms)
        cp = Path(cache_dir) / f"{key}.npz"
        if cp.exists():
            return _load_dataset_npz(cp)
    else:
        cp = None  # type: ignore[assignment]

    import json
    import zipfile

    raw: list[tuple[str, NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], int]] = []
    seen_Z: set[int] = set()

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted((n for n in zf.namelist() if n.endswith(".json")),
                       key=lambda n: int(Path(n).stem))
        for member in names:
            with zf.open(member) as fh:
                entry = json.loads(fh.read())
            if skip_non_converged and entry["general"].get("not_converged"):
                continue
            atoms = entry["general"]["AtomList"]
            n = len(atoms)
            if n > max_n_atoms:
                continue
            atomic_nums = np.asarray([a["atomic_number"] for a in atoms], dtype=np.int64)
            if contains_element is not None and contains_element not in atomic_nums:
                continue
            coords = np.asarray(entry["structural"]["Coordinates"], dtype=np.float64)
            if coords.shape != (n, 3):
                continue
            try:
                if target == "apt":
                    q = np.asarray(entry["electronic"]["PartChargeAPT"], dtype=np.float64)
                elif target == "mulliken":
                    q = np.asarray(entry["electronic"]["PartChargeMulliken"], dtype=np.float64)
                elif target == "cosmo":
                    cs = entry["solvation"]["AtomCOSMOCharge"]
                    q = np.asarray([a["charge"] for a in cs], dtype=np.float64)
                else:
                    raise ValueError(f"unknown CHAOS target: {target!r}")
            except (KeyError, TypeError):
                continue
            if q.shape[0] != n:
                continue
            formal_q = int(entry["electronic"].get("Charge", 0))
            mol_name = Path(member).stem
            seen_Z.update(int(z) for z in atomic_nums)
            raw.append((mol_name, coords, atomic_nums, q, formal_q))
            if n_mols is not None and len(raw) >= n_mols:
                break

    sorted_Z = sorted(seen_Z)
    z_to_type_idx = {z: i + 1 for i, z in enumerate(sorted_Z)}
    type_strs = tuple(_z_to_symbol(z) for z in sorted_Z)

    molecules: list[MoleculeData] = []
    for mol_name, coords, atomic_nums, q, formal_q in raw:
        type_idx = np.asarray([z_to_type_idx[int(z)] for z in atomic_nums], dtype=np.int64)
        molecules.append(
            MoleculeData(
                smiles="",
                atom_types=type_idx,
                inv_r=_build_inv_r(coords),
                target_charges=q,
                formal_charge=float(formal_q),
                metadata={"source": "CHAOS", "name": mol_name, "target": target},
            )
        )

    ds = Dataset(
        molecules=molecules,
        atom_types=type_strs,
        name=name,
        metadata={
            "source": "CHAOS",
            "zip": str(zip_path),
            "target": target,
            "contains_element": str(contains_element),
            "n_loaded": len(molecules),
            "level_of_theory": "ωB97X-D/def2-TZVP + C-PCM",
        },
    )
    if cp is not None:
        try:
            _save_dataset_npz(ds, cp)
        except Exception:
            pass
    return ds


def _z_to_symbol(z: int) -> str:
    symbols = {
        1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
        14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
        33: "As", 34: "Se", 11: "Na", 19: "K", 20: "Ca", 12: "Mg",
        26: "Fe", 29: "Cu", 30: "Zn",
    }
    return symbols.get(z, f"Z{z}")


def load_neemp(
    sdf_path: str | Path,
    chg_path: str | Path,
    typ_path: str | Path,
    *,
    name: str = "NEEMP",
    limit: int | None = None,
    cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    aromatic_to_double: bool = True,
) -> Dataset:
    """Load a NEEMP-format dataset: SDF + ``.chg`` + ``.typ`` triplet.

    Atom typing follows NEEMP's *ElemBond* convention — atoms are typed by
    ``(element, bond_order_class)`` (e.g. ``"C-1"``, ``"C-1.5"``, ``"O-2"``).
    The bond-order class comes straight from the ``.typ`` file.

    Caching
    -------
    SDF parsing + ``inv_r`` matrix construction is dominated by RDKit
    (set03 17,769 mol takes ~6 min). On first load we serialise the
    fully-parsed dataset as an NPZ in ``cache_dir`` keyed by the source
    files' (size, mtime); subsequent loads of the same dataset complete
    in < 5 s. Pass ``use_cache=False`` to disable, or ``cache_dir=None``
    to skip caching entirely.

    Requires RDKit for SDF parsing (only on cache miss).
    """
    sdf_path = Path(sdf_path)
    chg_path = Path(chg_path)
    typ_path = Path(typ_path)

    # ---- Try cache first ----
    cache_path: Path | None = None
    if use_cache and cache_dir is not None:
        cache_path = Path(cache_dir) / f"{_neemp_cache_key(sdf_path, chg_path, typ_path, limit)}.npz"
        if cache_path.exists():
            return _load_dataset_npz(cache_path)

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RDKit is required for the NEEMP loader. "
            "Install with `pip install powelleem[rdkit]`."
        ) from exc

    charges_by_name = _parse_neemp_chg(chg_path)
    types_by_name = _parse_neemp_typ(typ_path)

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    raw: list[tuple[str, NDArray[np.float64], list[tuple[str, str]], NDArray[np.float64]]] = []
    seen_types: set[tuple[str, str]] = set()
    for mol in suppl:
        if mol is None:
            continue
        name_id = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if not name_id:
            continue
        if name_id not in charges_by_name or name_id not in types_by_name:
            continue
        conf = mol.GetConformer(0)
        coords = np.asarray(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float64
        )
        if coords.shape[0] != mol.GetNumAtoms():
            continue
        atom_types_meta = types_by_name[name_id]
        if len(atom_types_meta) != coords.shape[0]:
            continue
        q_ref = charges_by_name[name_id]
        if q_ref.shape[0] != coords.shape[0]:
            continue
        seen_types.update(atom_types_meta)
        raw.append((name_id, coords, atom_types_meta, q_ref))
        if limit is not None and len(raw) >= limit:
            break

    def _bo_remap(b: str) -> str:
        """NEEMP-style bond-order coarsening: aromatic (1.5) → double (2)."""
        if aromatic_to_double and b in ("1.5", "1.5 "):
            return "2"
        return b

    type_strs = tuple(sorted({f"{e}-{_bo_remap(b)}" for e, b in seen_types}))
    type_idx = {t: i + 1 for i, t in enumerate(type_strs)}

    molecules: list[MoleculeData] = []
    for name_id, coords, atype_meta, q_ref in raw:
        molecules.append(
            MoleculeData(
                smiles="",  # SDF doesn't carry SMILES — could re-compute with Chem.MolToSmiles
                atom_types=np.asarray(
                    [type_idx[f"{e}-{_bo_remap(b)}"] for e, b in atype_meta], dtype=np.int64
                ),
                inv_r=_build_inv_r(coords),
                target_charges=q_ref,
                metadata={"source": "NEEMP", "name": name_id},
            )
        )

    ds = Dataset(
        molecules=molecules,
        atom_types=type_strs,
        name=name,
        metadata={
            "source": "NEEMP",
            "sdf": str(sdf_path),
            "chg": str(chg_path),
            "typ": str(typ_path),
            "n_loaded": len(molecules),
        },
    )

    # ---- Save cache on miss ----
    if cache_path is not None:
        try:
            _save_dataset_npz(ds, cache_path)
        except Exception:  # pragma: no cover
            pass  # cache write is best-effort

    return ds


# Back-compat alias
load_neemp_legacy = load_neemp


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
