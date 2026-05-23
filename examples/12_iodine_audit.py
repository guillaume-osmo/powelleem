"""Iodine environment + per-atom error audit on the CHAOS iodine fit.

Re-streams the CHAOS database for iodine-containing molecules; for each
iodine atom, looks up its local environment via RDKit on the canonical
SMILES (hybridisation of bonded neighbour, ring membership, formal
charge, etc.). Then loads the previously-fitted EEM parameters and
computes per-atom errors so we can see:

  - distribution of iodine environments in the dataset
  - mean / worst / best per-environment prediction error
  - which kinds of iodine are over- or under-fit by the single ``I`` type
  - worst-100 individual atoms (for outlier diagnosis)

The output drives the question "do we need more than one iodine type?".
"""

from __future__ import annotations

import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from powelleem import EEMModel
from powelleem.data import load_chaos
from powelleem.metrics import report_metrics
from powelleem.solvers import NumbaDENewton, SolverConfig

CHAOS = Path("/Volumes/RescueCopy/Github/data_archives/CHAOS.zip")


def classify_iodine_env(mol, atom_idx: int) -> str:  # type: ignore[no-untyped-def]
    """Return a short label for the chemical environment of an I atom."""
    from rdkit.Chem import HybridizationType

    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() == -1 and atom.GetDegree() == 0:
        return "I_anion"
    neighbours = atom.GetNeighbors()
    if not neighbours:
        return "I_isolated"
    nbr = neighbours[0]
    nbr_sym = nbr.GetSymbol()
    nbr_hyb = nbr.GetHybridization()
    nbr_aromatic = nbr.GetIsAromatic()

    if nbr_aromatic:
        return f"I_aryl_{nbr_sym}"
    if nbr_hyb == HybridizationType.SP3:
        return f"I_sp3_{nbr_sym}"
    if nbr_hyb == HybridizationType.SP2:
        return f"I_sp2_{nbr_sym}"
    if nbr_hyb == HybridizationType.SP:
        return f"I_sp_{nbr_sym}"
    return f"I_other_{nbr_sym}_{nbr_hyb}"


def main() -> None:
    print("Loading CHAOS iodine subset (cached)…", flush=True)
    ds = load_chaos(CHAOS, n_mols=5000, contains_element=53, target="apt")
    print(f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types", flush=True)

    print("\nRe-streaming CHAOS for SMILES (to feed RDKit)…", flush=True)
    smiles_by_chaos_id: dict[str, str] = {}
    chaos_ids_we_have = {m.metadata["name"] for m in ds.molecules}
    t0 = time.perf_counter()
    with zipfile.ZipFile(CHAOS) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            cid = Path(name).stem
            if cid not in chaos_ids_we_have:
                continue
            try:
                with zf.open(name) as fh:
                    raw = fh.read(4096).decode("utf-8", errors="replace")
                idx = raw.find('"CanonicalSMILES"')
                if idx < 0:
                    continue
                colon = raw.find(":", idx)
                q1 = raw.find('"', colon + 1)
                q2 = raw.find('"', q1 + 1)
                smi = raw[q1 + 1 : q2]
                smiles_by_chaos_id[cid] = smi
            except Exception:
                continue
    print(f"  SMILES extracted for {len(smiles_by_chaos_id)}/{ds.n_mols} mols ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("\nFitting CHAOS iodine subset (mol_rmsd) for per-atom error…", flush=True)
    model = EEMModel(atom_types=ds.atom_types)
    res = NumbaDENewton(
        config=SolverConfig(seed=42),
        population_size=50, n_generations=20, loss_kind="mol_rmsd",
    ).fit(model, ds)
    print(f"  κ={res.params.kappa:.4f}, total RMSD={res.rmse:.4f}", flush=True)
    rep = report_metrics(model, res.params, ds)
    print(f"  per-element I: RMSD={rep.per_element['I']['rmsd']:.4f}", flush=True)

    # Compute predicted charges (via the model)
    preds = model.predict(res.params, ds)

    from rdkit import Chem

    env_counts: Counter[str] = Counter()
    env_errors: dict[str, list[float]] = defaultdict(list)
    env_qref: dict[str, list[float]] = defaultdict(list)
    env_qpred: dict[str, list[float]] = defaultdict(list)
    worst_atoms: list[tuple[float, str, str, int, float, float]] = []  # (|err|, env, mol_name, i, qref, qpred)

    i_type_idx = ds.atom_types.index("I") + 1
    unclassified = 0
    for mol_idx, mol in enumerate(ds.molecules):
        cid = mol.metadata["name"]
        smi = smiles_by_chaos_id.get(cid)
        if smi is None:
            unclassified += sum(1 for t in mol.atom_types if t == i_type_idx)
            continue
        rdmol = Chem.MolFromSmiles(smi)
        if rdmol is None:
            unclassified += sum(1 for t in mol.atom_types if t == i_type_idx)
            continue
        rdmol = Chem.AddHs(rdmol)
        # Build I-atom-index list in the RDKit mol order
        rd_i_indices = [a.GetIdx() for a in rdmol.GetAtoms() if a.GetSymbol() == "I"]
        # CHAOS I atom indices in our atom_types
        chaos_i_indices = [i for i, t in enumerate(mol.atom_types) if t == i_type_idx]
        if len(rd_i_indices) != len(chaos_i_indices):
            unclassified += len(chaos_i_indices)
            continue
        # Assume the SDF order matches SMILES order — true for CHAOS's generation pipeline
        for rd_i, chaos_i in zip(rd_i_indices, chaos_i_indices, strict=True):
            env = classify_iodine_env(rdmol, rd_i)
            env_counts[env] += 1
            qref = float(mol.target_charges[chaos_i])
            qpred = float(preds[mol_idx][chaos_i])
            err = qpred - qref
            env_errors[env].append(err)
            env_qref[env].append(qref)
            env_qpred[env].append(qpred)
            worst_atoms.append((abs(err), env, cid, chaos_i, qref, qpred))

    print(f"\n=== Iodine environment distribution (total {sum(env_counts.values())} I atoms classified, {unclassified} unclassified) ===")
    print(f"  {'env':>20s}  {'count':>6s}  {'mean_err':>10s}  {'rmsd':>8s}  {'mean_qref':>10s}  {'mean_qpred':>10s}")
    for env, n in env_counts.most_common():
        errs = np.array(env_errors[env])
        qrefs = np.array(env_qref[env])
        qpreds = np.array(env_qpred[env])
        print(
            f"  {env:>20s}  {n:>6d}  {errs.mean():>+10.4f}  "
            f"{float(np.sqrt((errs * errs).mean())):>8.4f}  "
            f"{qrefs.mean():>+10.4f}  {qpreds.mean():>+10.4f}"
        )

    print("\n=== Top 20 worst-fit iodine atoms ===")
    worst_atoms.sort(reverse=True)
    print(f"  {'|err|':>8s}  {'env':>20s}  {'chaos_id':>12s}  {'atom_i':>6s}  {'qref':>10s}  {'qpred':>10s}")
    for abserr, env, cid, ai, qref, qpred in worst_atoms[:20]:
        print(f"  {abserr:>8.4f}  {env:>20s}  {cid:>12s}  {ai:>6d}  {qref:>+10.4f}  {qpred:>+10.4f}")


if __name__ == "__main__":
    main()
