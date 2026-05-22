"""Replicate the Raček 2016 NEEMP CCD_gen κ = 0.5125 on set03 17,769 mol.

The auto-scaled DE budget (pop=15, gen=7) only does ~120 trial evals on
set03 and lands at κ ≈ 0.45-0.46. To match Raček's published κ = 0.5125
we need substantially more exploration of the multi-modal landscape.

The original MATLAB `DE_UOA_FINAL.m` used:
    NP = 400  population
    1001 generations
    NEWUOA polish whenever r² > 0.6

That is ~400 000 residual evaluations, ~5 hours on Numba — feasible but
long for a single run.

A smarter approach: multi-seed runs at a moderate budget. With
NumbaDENewton finishing in ~64 s for pop=15, we can afford 10–20 seeds
at a larger pop and pick the best — covers more basins than a single
long run.

This script runs several `(population_size, n_generations, seed)` combos
and reports the κ progression.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from powelleem import EEMModel
from powelleem.data import load_neemp
from powelleem.solvers import NumbaDENewton, SolverConfig

NEEMP_EXAMPLES = Path(
    os.environ.get(
        "POWELLEEM_NEEMP_DIR",
        "/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples",
    )
)


SWEEP = [
    # (population_size, n_generations, n_seeds)  — total evals per run + seeds
    (50, 30,  3),  # 1500 evals × 3 = 4.5K evals  — ~3 min/seed
    (100, 50, 3),  # 5000 evals × 3 = 15K evals   — ~9 min/seed
    (200, 100, 1),  # 20K evals × 1                — ~15 min
]


def main(n_mols: int) -> None:
    print(f"Loading NEEMP set03 (limit={n_mols})…", flush=True)
    t0 = time.perf_counter()
    ds = load_neemp(
        NEEMP_EXAMPLES / "set03.sdf",
        NEEMP_EXAMPLES / "set03.chg",
        NEEMP_EXAMPLES / "set03.typ",
        limit=n_mols,
        name=f"set03-{n_mols}",
    )
    print(
        f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types  "
        f"(loaded {time.perf_counter() - t0:.1f}s)\n",
        flush=True,
    )

    model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)
    results: list[dict] = []

    for pop, gen, n_seeds in SWEEP:
        for seed in range(42, 42 + n_seeds):
            print(f"--- pop={pop:>3d} gen={gen:>3d} seed={seed} ---", flush=True)
            solver = NumbaDENewton(
                config=SolverConfig(seed=seed),
                population_size=pop,
                n_generations=gen,
                maxiter_lbfgs=100,
                maxiter_newton=100,
            )
            t0 = time.perf_counter()
            res = solver.fit(model, ds)
            wall = time.perf_counter() - t0
            print(
                f"  RMSE = {res.rmse:.5f}    κ = {res.params.kappa:.4f}    "
                f"wall = {wall:.1f}s   (DE {res.solver_metadata.get('stage1_wall_s'):.0f}s + "
                f"Newton {res.solver_metadata.get('stage2_wall_s'):.0f}s)",
                flush=True,
            )
            results.append(
                {
                    "pop": pop, "gen": gen, "seed": seed,
                    "rmse": res.rmse, "kappa": res.params.kappa,
                    "wall_s": wall,
                    "stage1_de_rmse": res.solver_metadata.get("stage1_de_rmse"),
                }
            )

    # Rank by RMSE
    print("\n" + "=" * 78)
    print("FULL RANKING (lowest RMSE first):")
    print(f"  {'pop':>4s} {'gen':>4s} {'seed':>5s}  {'RMSE':>8s}  {'κ':>8s}  {'wall(s)':>8s}")
    for r in sorted(results, key=lambda r: r["rmse"]):
        kappa_marker = "  ✓ NEEMP" if abs(r["kappa"] - 0.5125) < 0.02 else ""
        print(
            f"  {r['pop']:>4d} {r['gen']:>4d} {r['seed']:>5d}  "
            f"{r['rmse']:>8.5f}  {r['kappa']:>8.4f}  {r['wall_s']:>8.1f}{kappa_marker}"
        )

    best = min(results, key=lambda r: r["rmse"])
    print(f"\nBest RMSE: {best['rmse']:.5f}  κ = {best['kappa']:.4f}  "
          f"(NEEMP CCD_gen reference κ = 0.5125)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-mols", type=int, default=17769)
    args = p.parse_args()
    main(n_mols=args.n_mols)
