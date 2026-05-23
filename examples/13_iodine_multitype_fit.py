"""Multi-type iodine fit — split I into chemical-environment classes.

Audit (examples/12) showed three large populations within the single
``I`` element class:

    I_sp3_C   441 atoms   RMSD 0.061 (well fit)
    I_aryl_C  227 atoms   RMSD 0.112 (q surestimé de 0.11 e)
    I_sp2_C    67 atoms   RMSD 0.106 (vinyl, similar to aryl)

Plus several small classes (Si/Ge/anion etc.) which we collapse with
``I_sp3_C`` if their reference charges are similar enough, otherwise
treat as additional types.

This script re-loads the 763 CHAOS iodine molecules, recomputes the
iodine atom type per-atom from RDKit hybridisation, and refits
NumbaDENewton with 3 separate iodine types. We then report whether
splitting actually lowers per-atom RMSD on test.
"""

from __future__ import annotations

import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from powelleem import EEMModel
from powelleem.data import _build_inv_r, load_chaos
from powelleem.metrics import report_metrics
from powelleem.solvers import NumbaDENewton, SolverConfig
from powelleem.types import Dataset, MoleculeData

if TYPE_CHECKING:
    pass

CHAOS = Path("/Volumes/RescueCopy/Github/data_archives/CHAOS.zip")


def _classify_i_subtype(mol, atom_idx: int) -> str:  # type: ignore[no-untyped-def]
    """Return a sub-type label for an iodine atom (sp3 / aryl / sp2 / sp / anion / other)."""
    from rdkit.Chem import HybridizationType

    a = mol.GetAtomWithIdx(atom_idx)
    if a.GetFormalCharge() == -1 and a.GetDegree() == 0:
        return "I_anion"
    if a.GetDegree() == 0:
        return "I_atomic"
    nbr = a.GetNeighbors()[0]
    if nbr.GetIsAromatic():
        return "I_aryl"
    if nbr.GetHybridization() == HybridizationType.SP3:
        return "I_sp3"
    if nbr.GetHybridization() == HybridizationType.SP2:
        return "I_sp2"
    if nbr.GetHybridization() == HybridizationType.SP:
        return "I_sp"
    return "I_other"


def _rebuild_with_split_I(ds: Dataset, smiles_by_id: dict[str, str]) -> Dataset:
    """Return a new Dataset with the iodine class split by neighbour hybridisation."""
    from rdkit import Chem

    # New vocabulary: keep all non-I types unchanged, replace single "I" with sub-types we encounter.
    base_types = [t for t in ds.atom_types if t != "I"]
    extended: set[str] = set(base_types)
    # First pass: classify each I atom in every mol to find which sub-types actually appear.
    classifications: list[list[str | None]] = []
    i_idx_in_old = ds.atom_types.index("I") + 1
    for mol_idx, mol in enumerate(ds.molecules):
        cid = mol.metadata["name"]
        smi = smiles_by_id.get(cid)
        if smi is None:
            classifications.append([None] * mol.n_atoms)
            continue
        rdmol = Chem.MolFromSmiles(smi)
        if rdmol is None:
            classifications.append([None] * mol.n_atoms)
            continue
        rdmol = Chem.AddHs(rdmol)
        rd_i_indices = [a.GetIdx() for a in rdmol.GetAtoms() if a.GetSymbol() == "I"]
        chaos_i_positions = [i for i, t in enumerate(mol.atom_types) if t == i_idx_in_old]
        if len(rd_i_indices) != len(chaos_i_positions):
            classifications.append([None] * mol.n_atoms)
            continue
        per_atom: list[str | None] = [None] * mol.n_atoms
        for rd_i, chaos_i in zip(rd_i_indices, chaos_i_positions, strict=True):
            sub = _classify_i_subtype(rdmol, rd_i)
            per_atom[chaos_i] = sub
            extended.add(sub)
        classifications.append(per_atom)

    # Determine final ordered vocabulary
    extra_i = sorted(t for t in extended if t.startswith("I_"))
    final_types = tuple(base_types + extra_i)
    type_to_idx = {t: i + 1 for i, t in enumerate(final_types)}

    # Old non-I type names → new index
    old_name_to_new_idx = {}
    for old_i, name in enumerate(ds.atom_types):
        if name == "I":
            continue
        old_name_to_new_idx[old_i + 1] = type_to_idx[name]

    new_mols: list[MoleculeData] = []
    n_skipped = 0
    for mol, classes in zip(ds.molecules, classifications, strict=True):
        new_idx = np.empty(mol.n_atoms, dtype=np.int64)
        skip_mol = False
        for j, t in enumerate(mol.atom_types):
            if t == i_idx_in_old:
                cls = classes[j]
                if cls is None:
                    skip_mol = True
                    break
                new_idx[j] = type_to_idx[cls]
            else:
                new_idx[j] = old_name_to_new_idx[int(t)]
        if skip_mol:
            n_skipped += 1
            continue
        new_mols.append(
            MoleculeData(
                smiles=mol.smiles,
                atom_types=new_idx,
                inv_r=mol.inv_r,
                target_charges=mol.target_charges,
                formal_charge=mol.formal_charge,
                metadata=dict(mol.metadata),
            )
        )
    return Dataset(
        molecules=new_mols,
        atom_types=final_types,
        name=f"{ds.name}[I-split,skipped={n_skipped}]",
        metadata={**ds.metadata, "I_split": True, "skipped_smiles_mismatch": n_skipped},
    )


def main() -> None:
    print("Loading CHAOS iodine subset (cached)…", flush=True)
    ds = load_chaos(CHAOS, n_mols=5000, contains_element=53, target="apt")
    print(f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types", flush=True)

    print("Streaming CHAOS for SMILES…", flush=True)
    chaos_ids = {m.metadata["name"] for m in ds.molecules}
    smiles_by_id: dict[str, str] = {}
    with zipfile.ZipFile(CHAOS) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            cid = Path(name).stem
            if cid not in chaos_ids:
                continue
            with zf.open(name) as fh:
                raw = fh.read(4096).decode("utf-8", errors="replace")
            idx = raw.find('"CanonicalSMILES"')
            if idx < 0:
                continue
            colon = raw.find(":", idx)
            q1 = raw.find('"', colon + 1)
            q2 = raw.find('"', q1 + 1)
            smiles_by_id[cid] = raw[q1 + 1 : q2]
    print(f"  Got SMILES for {len(smiles_by_id)}/{ds.n_mols}", flush=True)

    print("Rebuilding dataset with split-I typing…", flush=True)
    ds_split = _rebuild_with_split_I(ds, smiles_by_id)
    print(f"  {ds_split.n_mols} mols (skipped {ds_split.metadata['skipped_smiles_mismatch']}), {ds_split.n_types} types", flush=True)
    print(f"  vocabulary: {ds_split.atom_types}", flush=True)

    # Fit both versions for direct comparison
    for label, dataset in [("single-I", ds), ("split-I", ds_split)]:
        print(f"\n--- Fitting {label} ---", flush=True)
        train, test = dataset.split(test_fraction=0.2, seed=42)
        model = EEMModel(atom_types=dataset.atom_types)
        s = NumbaDENewton(
            config=SolverConfig(seed=42),
            population_size=50, n_generations=20, loss_kind="mol_rmsd",
        )
        t0 = time.perf_counter()
        res = s.fit(model, train)
        wall = time.perf_counter() - t0
        tr = report_metrics(model, res.params, train).overall
        te = report_metrics(model, res.params, test).overall
        print(f"  P={1 + 2 * dataset.n_types}  κ={res.params.kappa:.4f}  wall={wall:.1f}s")
        print(f"  TRAIN: RMSD={tr['rmsd']:.4f}  R={tr['r']:.4f}  D_avg={tr['d_avg']:.4f}")
        print(f"  TEST : RMSD={te['rmsd']:.4f}  R={te['r']:.4f}  D_avg={te['d_avg']:.4f}  Δ={(te['rmsd']-tr['rmsd'])/tr['rmsd']*100:+.1f}%")
        # Per-element I-class metrics
        rep_test = report_metrics(model, res.params, test)
        d = res.params.to_dict()
        print(f"  Iodine subtypes (test set):")
        print(f"  {'type':>12s}  {'α':>8s}  {'β':>8s}  {'n':>5s}  {'RMSD':>8s}  {'R':>8s}")
        for t in dataset.atom_types:
            if not t.startswith("I"):
                continue
            if t in rep_test.per_element:
                pe = rep_test.per_element[t]
                print(f"  {t:>12s}  {d['alpha'][t]:>8.4f}  {d['beta'][t]:>8.4f}  {pe['n_atoms']:>5d}  {pe['rmsd']:>8.4f}  {pe['r']:>8.4f}")


if __name__ == "__main__":
    main()
