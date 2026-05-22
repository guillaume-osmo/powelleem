"""DENewton-only scale bench — skip single-start methods that get stuck.

On set01 30 mol AnalyticLM finds κ=1.39 and AnalyticNewton finds κ=0.53
(close to NEEMP ref); on set01 500 mol BOTH get stuck at κ ≈ 2.5 because
the loss landscape geometry changes with dataset size. At set03 scale
(17,769 mol) they would also be stuck and just waste compute.

DENewton (DE warm + AnalyticNewton polish) escapes the local minima
trap via the population search. This script runs only DENewton on the
requested NEEMP set, with progress prints from the underlying DE loop.

Usage::

    # Quick check on 500 mol
    python examples/06_scale_denewton_only.py --set 01 --n-mols 500

    # Production: full NEEMP CCD_gen training set
    python examples/06_scale_denewton_only.py --set 03 --n-mols 17769
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import os

from powelleem import EEMModel
from powelleem.data import load_neemp
from powelleem.solvers import DENewton, SolverConfig

NEEMP_EXAMPLES = Path(
    os.environ.get(
        "POWELLEEM_NEEMP_DIR",
        "/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples",
    )
)


def main(set_id: str, n_mols: int, seed: int = 42) -> None:
    sdf = NEEMP_EXAMPLES / f"set{set_id}.sdf"
    chg = NEEMP_EXAMPLES / f"set{set_id}.chg"
    typ = NEEMP_EXAMPLES / f"set{set_id}.typ"

    print(f"Loading NEEMP set{set_id} (limit={n_mols})…", flush=True)
    t0 = time.perf_counter()
    ds = load_neemp(sdf, chg, typ, limit=n_mols, name=f"set{set_id}-{n_mols}")
    print(
        f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types "
        f"({time.perf_counter() - t0:.1f}s load)",
        flush=True,
    )
    print(f"  atom types: {ds.atom_types}", flush=True)

    # Auto-scale DE budget — same heuristic as 05_scale_numpy.py.
    if ds.n_mols <= 100:
        de_pop, de_gen = 50, 20
    elif ds.n_mols <= 1000:
        de_pop, de_gen = 30, 15
    elif ds.n_mols <= 5000:
        de_pop, de_gen = 20, 10
    else:
        de_pop, de_gen = 15, 7

    print(f"\nDE configuration: pop={de_pop}, gen={de_gen}", flush=True)
    model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)
    solver = DENewton(
        config=SolverConfig(seed=seed),
        population_size=de_pop,
        n_generations=de_gen,
        maxiter_lbfgs=100,
        maxiter_newton=100,
    )

    print("\nRunning DENewton…", flush=True)
    t0 = time.perf_counter()
    res = solver.fit(model, ds)
    wall = time.perf_counter() - t0

    meta = res.solver_metadata
    print(f"\n=== DENewton on set{set_id} ({ds.n_mols} mols) ===", flush=True)
    print(f"  RMSE              : {res.rmse:.5f}")
    print(f"  κ (NEEMP ref 0.5125): {res.params.kappa:.4f}")
    print(f"  Wall total        : {wall:.1f}s")
    print(f"  Stage 1 (DE) wall : {meta.get('stage1_wall_s'):.1f}s  ({meta.get('stage1_n_evals')} evals)")
    print(f"  Stage 2 (Newton)  : {meta.get('stage2_wall_s'):.1f}s")
    print(f"  Stage1 RMSE       : {meta.get('stage1_de_rmse'):.5f}")
    print()
    print("Per-element fitted parameters:")
    d = res.params.to_dict()
    print(f"  {'type':>10s}  {'α':>10s}  {'β':>10s}")
    for t in res.params.atom_types:
        print(f"  {t:>10s}  {d['alpha'][t]:>10.4f}  {d['beta'][t]:>10.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", type=str, default="01")
    p.add_argument("--n-mols", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(set_id=args.set, n_mols=args.n_mols, seed=args.seed)
