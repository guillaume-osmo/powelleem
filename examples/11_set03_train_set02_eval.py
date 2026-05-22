"""Out-of-distribution validation — train on set03, evaluate on set02.

set03 (17,769 mol) is NEEMP's CCD_gen training set. set02 (4,443 mol)
is an *independent* dataset used by NEEMP for separate parameter
training. A model fitted on set03 has *never seen* set02 — so
evaluating it on set02 measures genuine out-of-distribution
generalisation, not just train/test variance on the same distribution.

We use the union of the two sets' atom-type vocabularies (so the model
can predict charges on set02 atoms whose exact NEEMP type might appear
in set02 but not set03, or vice-versa).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from powelleem import EEMModel
from powelleem.data import load_neemp
from powelleem.metrics import report_metrics
from powelleem.solvers import NumbaDENewton, SolverConfig
from powelleem.types import Dataset, MoleculeData

NEEMP_EXAMPLES = Path(
    os.environ.get(
        "POWELLEEM_NEEMP_DIR",
        "/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples",
    )
)


def _remap_to_union(ds: Dataset, union: tuple[str, ...]) -> Dataset:
    """Rebuild a Dataset with atom-type indices pointing into the union vocab."""
    type_to_new_idx = {t: i + 1 for i, t in enumerate(union)}
    new_mols: list[MoleculeData] = []
    skipped = 0
    for mol in ds.molecules:
        try:
            new_idx = [type_to_new_idx[ds.atom_types[t - 1]] for t in mol.atom_types]
        except KeyError:
            skipped += 1
            continue
        import numpy as np

        new_mols.append(
            MoleculeData(
                smiles=mol.smiles,
                atom_types=np.asarray(new_idx, dtype=mol.atom_types.dtype),
                inv_r=mol.inv_r,
                target_charges=mol.target_charges,
                formal_charge=mol.formal_charge,
                metadata=dict(mol.metadata),
            )
        )
    return Dataset(
        molecules=new_mols,
        atom_types=union,
        name=f"{ds.name}[remap_to_union; skipped={skipped}]",
        metadata={**ds.metadata, "vocab_remap_skipped": skipped},
    )


def main() -> None:
    print("Loading set03 (NEEMP training, 17,769 mol)…", flush=True)
    ds_03 = load_neemp(
        NEEMP_EXAMPLES / "set03.sdf",
        NEEMP_EXAMPLES / "set03.chg",
        NEEMP_EXAMPLES / "set03.typ",
        limit=17769,
    )
    print(f"  set03: {ds_03.n_mols} mols, {ds_03.total_atoms} atoms, {ds_03.n_types} types", flush=True)

    print("Loading set02 (NEEMP OOD test, 4,443 mol)…", flush=True)
    ds_02 = load_neemp(
        NEEMP_EXAMPLES / "set02.sdf",
        NEEMP_EXAMPLES / "set02.chg",
        NEEMP_EXAMPLES / "set02.typ",
        limit=4443,
    )
    print(f"  set02: {ds_02.n_mols} mols, {ds_02.total_atoms} atoms, {ds_02.n_types} types", flush=True)

    union = tuple(sorted(set(ds_03.atom_types) | set(ds_02.atom_types)))
    print(f"\nUnion atom-type vocabulary ({len(union)} types): {union}", flush=True)

    ds_03_u = _remap_to_union(ds_03, union)
    ds_02_u = _remap_to_union(ds_02, union)
    print(f"  set03 remapped: {ds_03_u.n_mols} mols (skipped {ds_03_u.metadata['vocab_remap_skipped']})", flush=True)
    print(f"  set02 remapped: {ds_02_u.n_mols} mols (skipped {ds_02_u.metadata['vocab_remap_skipped']})", flush=True)

    model = EEMModel(atom_types=union)

    for loss_kind in ("atom_rmse", "mol_rmsd"):
        print(f"\n--- Fitting on set03 with loss={loss_kind} ---", flush=True)
        s = NumbaDENewton(
            config=SolverConfig(seed=42),
            population_size=50, n_generations=20, loss_kind=loss_kind,
        )
        t0 = time.perf_counter()
        res = s.fit(model, ds_03_u)
        wall = time.perf_counter() - t0
        print(f"  fit wall: {wall:.1f}s, κ={res.params.kappa:.4f}", flush=True)

        train_metrics = report_metrics(model, res.params, ds_03_u).overall
        ood_metrics = report_metrics(model, res.params, ds_02_u).overall

        print(f"\n=== loss={loss_kind}: train(set03) vs OOD(set02) ===")
        print(f"  {'metric':<12s}  {'train(set03)':>14s}  {'OOD(set02)':>14s}    Δ%")
        for key in ("rmsd", "r", "r2", "sp", "d_avg", "d_max", "atom_rmse"):
            tr = train_metrics[key]; oo = ood_metrics[key]
            delta = (oo - tr) / tr * 100 if tr != 0 else 0.0
            print(f"  {key:<12s}  {tr:>14.4f}  {oo:>14.4f}   {delta:>+5.1f} %", flush=True)


if __name__ == "__main__":
    main()
