"""Validation against the original MATLAB DE_UOA_FINAL.m + NEEMP set01 dataset.

This script loads NEEMP's reference dataset (500 molecules, B3LYP/6-311G NPA
reference charges, atom typing by element+bond-order) and runs all six
solvers on a small subset to verify:

1. The PDFO Newuoa/Bobyqa bindings actually run (the package name is
   only honest if they work).
2. The analytical-Jacobian AnalyticLM converges and ties or beats the
   derivative-free solvers, matching the gist of NEEMP's published
   `CCD_gen_DE_RMSD_B3LYP_6311G_NPA.par` (Raček et al. 2016).

Run with::

    python examples/02_validate_against_matlab.py --n-mols 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from powelleem import EEMModel
from powelleem.data import load_neemp
from powelleem.solvers import (
    AnalyticLM,
    AnalyticNewton,
    Bobyqa,
    DEHybrid,
    JaxAdam,
    JaxAdaMuon,
    JaxLM,
    Newuoa,
    SolverConfig,
)


NEEMP_EXAMPLES = Path("/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples")
SDF = NEEMP_EXAMPLES / "set01.sdf"
CHG = NEEMP_EXAMPLES / "set01.chg"
TYP = NEEMP_EXAMPLES / "set01.typ"
REF_PAR = Path(
    "/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/data/Parameters/"
    "CCD_gen_DE_RMSD_B3LYP_6311G_NPA.par"
)


def main(n_mols: int = 50, seed: int = 42, skip_de: bool = True, de_timeout_s: float = 300.0) -> None:
    print(f"Loading NEEMP set01 (limit={n_mols})…")
    t0 = time.perf_counter()
    ds = load_neemp(SDF, CHG, TYP, limit=n_mols, name="NEEMP-set01")
    print(
        f"  → {ds.n_mols} mols, {ds.total_atoms} atoms, "
        f"{ds.n_types} types: {ds.atom_types}"
    )
    print(f"  load time: {time.perf_counter() - t0:.1f}s")
    print(f"Reference parameter file: {REF_PAR.name}")
    print()

    model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)
    config = SolverConfig(seed=seed)

    solvers: list = [
        ("AnalyticLM",     AnalyticLM(config=config, maxiter_lbfgs=100, maxiter_lm=100)),
        ("AnalyticNewton", AnalyticNewton(config=config, maxiter_lbfgs=100, maxiter_newton=100)),
        ("Newuoa",         Newuoa(config=config, max_fev=3000)),
        ("Bobyqa",         Bobyqa(config=config, max_fev=3000)),
        ("JaxLM",          JaxLM(config=config, maxiter_lbfgs=100, maxiter_lm=100)),
        ("JaxAdam",        JaxAdam(config=config, n_iterations=500, learning_rate=0.02)),
        ("JaxAdaMuon",     JaxAdaMuon(config=config, n_iterations=500, learning_rate=0.05)),
    ]
    if not skip_de:
        solvers.append(
            ("DEHybrid", DEHybrid(config=config, population_size=30, n_generations=10))
        )

    results = []
    for name, solver in solvers:
        print(f"=== {name} ===")
        t0 = time.perf_counter()
        # Soft per-solver wall-clock budget — used to display a warning,
        # not to interrupt mid-run (interrupting NumPy/JAX cleanly is brittle).
        budget = de_timeout_s if name == "DEHybrid" else 60.0
        try:
            res = solver.fit(model, ds)
            wall = time.perf_counter() - t0
            tag = "" if wall < budget else f" [over budget {budget:.0f}s]"
            print(
                f"  RMSE = {res.rmse:.5f}    wall = {wall:.2f}s{tag}    "
                f"loss0 = {res.loss_initial:.5f}    nfev = {res.n_function_evals}"
            )
            results.append((name, res))
        except Exception as exc:  # pragma: no cover
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results.append((name, None))
        print()

    # Summary
    print("=" * 70)
    print(f"{'Solver':<14s} {'RMSE':>10s} {'wall(s)':>10s} {'κ':>8s} {'#nfev':>8s}")
    print("-" * 70)
    for name, res in sorted(
        ((n, r) for n, r in results if r is not None), key=lambda x: x[1].rmse
    ):
        print(
            f"{name:<14s} {res.rmse:>10.5f} {res.wall_time_s:>10.2f} "
            f"{res.params.kappa:>8.4f} {res.n_function_evals:>8d}"
        )

    # Display the AnalyticLM params side-by-side with NEEMP reference for sanity
    print()
    print("AnalyticLM fitted parameters (the reference solver):")
    al_res = next(r for n, r in results if n == "AnalyticLM" and r is not None)
    d = al_res.params.to_dict()
    print(f"  κ = {d['kappa']:.4f}    (NEEMP CCD_gen reference: κ = 0.5125)")
    print(f"  {'type':>8s}  {'α (fitted)':>12s}  {'β (fitted)':>12s}")
    for t in al_res.params.atom_types:
        print(f"  {t:>8s}  {d['alpha'][t]:>12.4f}  {d['beta'][t]:>12.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-mols", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-de", action="store_true", help="Include DEHybrid (slow)")
    args = p.parse_args()
    main(n_mols=args.n_mols, seed=args.seed, skip_de=not args.with_de)
