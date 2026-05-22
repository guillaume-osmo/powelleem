"""MMFF94 atom-typing as an alternative to NEEMP's element+bond-order scheme.

NEEMP uses 15 atom types on set03 — element + (1, 2, 3, aromatic-mapped-to-2).
MMFF94 (Halgren 1996) defines 95 atom types based on richer chemical context:
hybridisation, formal charge, ring membership, neighbour identities, etc. A
single ``C`` element can map to ``CR`` (alkyl), ``C=C`` (vinyl), ``CB``
(aromatic benzene), ``CGD+`` (guanidinium central), ``C5`` (5-ring aromatic),
``CSP`` (sp carbon), etc.

The hypothesis to test here: more atom types ⇒ more degrees of freedom in the
EEM model ⇒ lower fitting error. With set03's 821 418 atoms and ~191
parameters (κ + 2 × 95), each parameter still gets ~4 300 atoms of training
data, so overfitting is unlikely.

This module wraps RDKit's ``rdMolDescriptors.MMFFGetMoleculeProperties`` to
extract the atom types per molecule given an SDF. The output replaces our
NEEMP-derived ``atom_types`` indexing while keeping every other field of
``MoleculeData`` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from powelleem.types import Dataset


# 95 MMFF94 atom-type symbol → numeric type
# Source: rdkit MMFFGetMoleculeProperties().GetMMFFAtomType()
# Numbers run 1..99 (with gaps for unused historic entries; 95 active types).
_MMFF94_SYMBOLS: tuple[str, ...] = (
    "CR", "C=C", "CSP2", "C=O", "C=N", "NC=O", "CSP", "=C=", "CR3R", "CR4R",
    "CE4R", "CB", "C5A", "C5B", "C5", "C=C(C)C", "CR4E", "C=S", "CSO2", "CGD",
    "CGD+", "HC", "HO", "HN", "HOH", "HOCO", "HN=", "HNN+", "HSP2", "HP",
    "O", "OR", "OC=O", "OC=C", "OC=N", "OC=S", "OSO3", "OSO2", "OSO", "-O-",
    "O=C", "O=N", "O=S", "OM", "OM2", "ON+", "O+", "NR", "N=C", "N=N",
    "NC=C", "NC=N", "NC=S", "NSP", "NAZT", "=N=", "NCN+", "N5A", "N5B", "N5+",
    "N5AX", "N5BX", "N5OX", "N5M", "NIM+", "NSO2", "NPYL", "NPYD", "NPD+",
    "NR+", "N+=C", "NCN+", "N5+", "N+", "F", "CL", "BR", "I", "F-", "CL-",
    "BR-", "FE+2", "FE+3", "S", "S=C", "S=O", "SO2", "S=N", "SO3", "SO4", "SI",
    "P", "PO4", "PO3", "PO2", "-P=C"
)


def load_neemp_with_mmff94(
    sdf_path: str | Path,
    chg_path: str | Path,
    typ_path: str | Path,
    *,
    name: str = "NEEMP-MMFF94",
    limit: int | None = None,
) -> Dataset:
    """Like :func:`powelleem.data.load_neemp`, but re-types atoms via MMFF94.

    Charges are still taken from the ``.chg`` file; only the ``atom_types``
    field of every :class:`MoleculeData` is replaced. The atom-type
    vocabulary becomes the subset of MMFF94 types actually present in the
    dataset (≤ 95).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RDKit is required for MMFF94 typing. "
            "Install with `pip install powelleem[rdkit]`."
        ) from exc

    from powelleem.data import _parse_neemp_chg, _build_inv_r
    from powelleem.types import Dataset, MoleculeData

    sdf_path = Path(sdf_path)
    chg_path = Path(chg_path)

    charges_by_name = _parse_neemp_chg(chg_path)

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    raw_mols: list[tuple[str, NDArray[np.float64], list[int], NDArray[np.float64]]] = []
    seen_types: set[int] = set()
    skipped = 0
    for mol in suppl:
        if mol is None:
            continue
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if mol_name not in charges_by_name:
            continue
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                skipped += 1
                continue
            atom_types_int = [int(props.GetMMFFAtomType(i)) for i in range(mol.GetNumAtoms())]
            if any(t == 0 for t in atom_types_int):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        conf = mol.GetConformer(0)
        coords = np.asarray(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float64
        )
        q_ref = charges_by_name[mol_name]
        if q_ref.shape[0] != coords.shape[0]:
            continue
        seen_types.update(atom_types_int)
        raw_mols.append((mol_name, coords, atom_types_int, q_ref))
        if limit is not None and len(raw_mols) >= limit:
            break

    # Vocabulary = sorted numeric types in this dataset
    sorted_types = sorted(seen_types)
    type_to_idx = {t: i + 1 for i, t in enumerate(sorted_types)}
    type_strs = tuple(f"MMFF{t}" for t in sorted_types)

    molecules: list[MoleculeData] = []
    for name_id, coords, atype_ints, q_ref in raw_mols:
        molecules.append(
            MoleculeData(
                smiles="",
                atom_types=np.asarray(
                    [type_to_idx[t] for t in atype_ints], dtype=np.int64
                ),
                inv_r=_build_inv_r(coords),
                target_charges=q_ref,
                metadata={"source": "NEEMP+MMFF94", "name": name_id},
            )
        )

    return Dataset(
        molecules=molecules,
        atom_types=type_strs,
        name=name,
        metadata={
            "source": "NEEMP-charges+MMFF94-typing",
            "sdf": str(sdf_path),
            "chg": str(chg_path),
            "n_loaded": len(molecules),
            "n_skipped_mmff_failed": skipped,
        },
    )
