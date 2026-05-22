"""Generalisation study — train/test split on every headline benchmark.

A single fitted parameter set is only useful if it generalises beyond
the molecules it was trained on. This script reproduces the four
headline fits with an 80/20 random train/test split and reports
metrics on **both** subsets:

  1. NEEMP set03 17 769 mol (atom_rmse loss)
  2. NEEMP set03 17 769 mol (mol_rmsd loss — the headline)
  3. CHAOS iodine 500 mol  (mol_rmsd loss)
  4. CHAOS iodine 500 mol  (atom_rmse loss)

The train/test RMSD gap quantifies overfitting. For a well-specified
EEM model on a representative training set, train and test RMSD should
be within ~5% of each other; a larger gap indicates either (i) too many
free parameters relative to the training-set chemistry or (ii) a test
subset with chemistry not represented in train.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from powelleem import EEMModel
from powelleem.data import load_chaos, load_neemp
from powelleem.metrics import report_metrics
from powelleem.solvers import NumbaDENewton, SolverConfig

NEEMP_EXAMPLES = Path(
    os.environ.get(
        "POWELLEEM_NEEMP_DIR",
        "/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples",
    )
)
CHAOS = Path(os.environ.get("CHAOS_ZIP", "/Volumes/RescueCopy/Github/data_archives/CHAOS.zip"))


def _fit_and_report(
    train, test, model, loss_kind: str, pop: int, gen: int
) -> tuple[float, float, dict, dict]:
    solver = NumbaDENewton(
        config=SolverConfig(seed=42),
        population_size=pop, n_generations=gen,
        loss_kind=loss_kind,
    )
    t0 = time.perf_counter()
    res = solver.fit(model, train)
    wall = time.perf_counter() - t0
    train_metrics = report_metrics(model, res.params, train).overall
    test_metrics = report_metrics(model, res.params, test).overall
    return wall, res.params.kappa, train_metrics, test_metrics


def _summary(label: str, wall: float, kappa: float, train: dict, test: dict) -> None:
    gap = (test["rmsd"] - train["rmsd"]) / train["rmsd"] * 100
    print(f"\n=== {label} ===")
    print(f"  κ = {kappa:.4f}   wall = {wall:.1f} s")
    print(f"  {'metric':<12s}  {'train':>10s}  {'test':>10s}   gap")
    for key in ("rmsd", "r", "r2", "sp", "d_avg", "d_max", "atom_rmse"):
        tr = train[key]; te = test[key]
        delta = (te - tr) / tr * 100 if tr != 0 else 0.0
        print(f"  {key:<12s}  {tr:>10.4f}  {te:>10.4f}   {delta:>+5.1f} %")
    print(f"  → train/test RMSD gap: {gap:+.1f} %")


def main() -> None:
    print("Loading NEEMP set03 17 769 mol…", flush=True)
    ds_neemp = load_neemp(
        NEEMP_EXAMPLES / "set03.sdf",
        NEEMP_EXAMPLES / "set03.chg",
        NEEMP_EXAMPLES / "set03.typ",
        limit=17769,
    )
    print(f"  {ds_neemp.n_mols} mols, {ds_neemp.total_atoms} atoms, {ds_neemp.n_types} types", flush=True)
    neemp_train, neemp_test = ds_neemp.split(test_fraction=0.2, seed=42)
    print(f"  train: {neemp_train.n_mols} mols ({neemp_train.total_atoms} atoms)", flush=True)
    print(f"  test : {neemp_test.n_mols} mols ({neemp_test.total_atoms} atoms)", flush=True)
    neemp_model = EEMModel(atom_types=ds_neemp.atom_types)

    print("Loading CHAOS iodine subset (500 mol)…", flush=True)
    ds_iodine = load_chaos(CHAOS, n_mols=500, contains_element=53, target="apt")
    print(f"  {ds_iodine.n_mols} mols, {ds_iodine.total_atoms} atoms, {ds_iodine.n_types} types", flush=True)
    iodine_train, iodine_test = ds_iodine.split(test_fraction=0.2, seed=42)
    iodine_model = EEMModel(atom_types=ds_iodine.atom_types)

    # 1+2: NEEMP set03 — both loss kinds
    for loss_kind in ("atom_rmse", "mol_rmsd"):
        wall, k, tr, te = _fit_and_report(
            neemp_train, neemp_test, neemp_model, loss_kind, pop=50, gen=20
        )
        _summary(f"NEEMP set03  loss={loss_kind}", wall, k, tr, te)

    # 3+4: CHAOS iodine — both loss kinds
    for loss_kind in ("mol_rmsd", "atom_rmse"):
        wall, k, tr, te = _fit_and_report(
            iodine_train, iodine_test, iodine_model, loss_kind, pop=50, gen=20
        )
        _summary(f"CHAOS iodine  loss={loss_kind}", wall, k, tr, te)


if __name__ == "__main__":
    main()
