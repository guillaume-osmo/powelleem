"""End-to-end Numba-accelerated DENewton on NEEMP set03 17,769 mol.

Baseline (serial NumPy, committed earlier in benchmarks/results/scale_set03_denewton_only_17769mols.txt):
    RMSE 0.0542  wall 2362 s (~39 min)  κ = 0.4509

Target with NumbaDENewton: < 5 min wall, same RMSE.

Usage::

    python examples/08_numba_denewton_set03.py --n-mols 17769
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from powelleem import EEMModel
from powelleem.data import load_neemp
from powelleem.solvers import NumbaDENewton, SolverConfig

NEEMP_EXAMPLES = Path(
    os.environ.get(
        "POWELLEEM_NEEMP_DIR",
        "/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples",
    )
)


def main(set_id: str, n_mols: int, seed: int = 42) -> None:
    print(f"Loading NEEMP set{set_id} (limit={n_mols})…", flush=True)
    t0 = time.perf_counter()
    ds = load_neemp(
        NEEMP_EXAMPLES / f"set{set_id}.sdf",
        NEEMP_EXAMPLES / f"set{set_id}.chg",
        NEEMP_EXAMPLES / f"set{set_id}.typ",
        limit=n_mols,
        name=f"set{set_id}-{n_mols}",
    )
    print(
        f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types "
        f"(loaded in {time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)

    # Auto-scaled DE budget — same heuristic as 05/06 for direct comparability.
    if ds.n_mols <= 100:
        de_pop, de_gen = 50, 20
    elif ds.n_mols <= 1000:
        de_pop, de_gen = 30, 15
    elif ds.n_mols <= 5000:
        de_pop, de_gen = 20, 10
    else:
        de_pop, de_gen = 15, 7

    print(f"DE configuration: pop={de_pop}, gen={de_gen}\n", flush=True)

    solver = NumbaDENewton(
        config=SolverConfig(seed=seed),
        population_size=de_pop,
        n_generations=de_gen,
        maxiter_lbfgs=100,
        maxiter_newton=100,
    )

    print("Running NumbaDENewton… (first call triggers JIT compile ~7s)", flush=True)
    t0 = time.perf_counter()
    res = solver.fit(model, ds)
    wall = time.perf_counter() - t0

    meta = res.solver_metadata
    print(f"\n=== NumbaDENewton on set{set_id} ({ds.n_mols} mols) ===")
    print(f"  RMSE                  : {res.rmse:.5f}")
    print(f"  κ (NEEMP ref 0.5125)  : {res.params.kappa:.4f}")
    print(f"  Wall total            : {wall:.1f}s")
    print(f"  Stage 1 (DE) wall     : {meta.get('stage1_wall_s'):.1f}s  ({meta.get('stage1_n_evals')} evals)")
    print(f"  Stage 2 (Newton) wall : {meta.get('stage2_wall_s'):.1f}s")
    print(f"  Stage1 DE RMSE        : {meta.get('stage1_de_rmse'):.5f}")
    print(f"  L-BFGS loss           : {meta.get('loss_after_lbfgs'):.5f}")
    print(f"  Newton final message  : {meta.get('newton_message')}")
    print()
    print("Per-element fitted parameters:")
    d = res.params.to_dict()
    print(f"  {'type':>10s}  {'α':>10s}  {'β':>10s}")
    for t in res.params.atom_types:
        print(f"  {t:>10s}  {d['alpha'][t]:>10.4f}  {d['beta'][t]:>10.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", type=str, default="03")
    p.add_argument("--n-mols", type=int, default=17769)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(set_id=args.set, n_mols=args.n_mols, seed=args.seed)
