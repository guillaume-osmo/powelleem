"""Focused 6-solver benchmark — keep only the methods that actually win.

After the 9-solver sweep on NEEMP set01, the empirical ranking is:

  Tier 1 (best RMSE, escape local minima)
    - DENewton        : DE-LHS + AnalyticNewton polish        ★ best overall
    - DEHybrid        : DE + NEWUOA (= MATLAB DE_UOA_FINAL)   ← global reference

  Tier 2 (Hessian-aware single-start)
    - AnalyticNewton  : NumPy + analytic Hessian + trust-Newton
    - AnalyticLM      : NumPy + analytic Jacobian + L-BFGS-B → TRF/LM (fastest)

  Tier 3 (orthogonalised gradient, for comparison)
    - JaxAdaMuon      : official AdaMuon (Liu et al. 2025)
    - JaxMuonN        : Guillaume's AdaMuonn variant (nested AdamN EMA + NS-on-direction)

We drop the "trucs chelou" (Newuoa/Bobyqa without polish, JaxLM without
analytic Jacobian, JaxAdam plain) since they're dominated by the above
on this dataset.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from powelleem import EEMModel
from powelleem.data import load_neemp
from powelleem.solvers import (
    AnalyticLM,
    AnalyticNewton,
    DEAdaMuonn,
    DEHybrid,
    DENewton,
    JaxAdaMuon,
    JaxMuonN,
    SolverConfig,
)

NEEMP_EXAMPLES = Path("/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples")


def main(n_mols: int, seed: int = 42) -> None:
    print(f"Loading NEEMP set01 (limit={n_mols})…", flush=True)
    t0 = time.perf_counter()
    ds = load_neemp(
        NEEMP_EXAMPLES / "set01.sdf",
        NEEMP_EXAMPLES / "set01.chg",
        NEEMP_EXAMPLES / "set01.typ",
        limit=n_mols,
        name=f"NEEMP-set01-{n_mols}",
    )
    print(
        f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types  "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)
    config = SolverConfig(seed=seed)

    solvers = [
        # Tier 1: DE-warm + curvature-aware polish
        ("DENewton",       DENewton(config=config, population_size=50, n_generations=20,
                                    maxiter_lbfgs=100, maxiter_newton=100)),
        ("DEAdaMuonn",     DEAdaMuonn(config=config, population_size=50, n_generations=20)),
        # ↑ DEAdaMuonn now uses tuned defaults (n_iter=2000, betas=(0.9, 0.0, 0.999))
        ("DEHybrid",       DEHybrid(config=config, population_size=30, n_generations=10)),
        # Tier 2: single-start, Hessian-aware
        ("AnalyticNewton", AnalyticNewton(config=config, maxiter_lbfgs=100, maxiter_newton=100)),
        ("AnalyticLM",     AnalyticLM(config=config, maxiter_lbfgs=100, maxiter_lm=100)),
        # Tier 3: orthogonalised gradient (no DE warm)
        ("JaxMuonN",       JaxMuonN(config=config, n_iterations=500, learning_rate=0.01)),
        ("JaxAdaMuon",     JaxAdaMuon(config=config, n_iterations=500, learning_rate=0.01)),
    ]

    results = []
    for name, solver in solvers:
        print(f"\n=== {name} ===", flush=True)
        t = time.perf_counter()
        try:
            res = solver.fit(model, ds)
            wall = time.perf_counter() - t
            print(
                f"  RMSE = {res.rmse:.5f}    wall = {wall:.2f}s    "
                f"κ = {res.params.kappa:.4f}    nfev = {res.n_function_evals}",
                flush=True,
            )
            results.append((name, res))
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    # Sorted summary
    print("\n" + "=" * 70)
    print(f"{'Solver':<18s} {'RMSE':>10s} {'wall(s)':>10s} {'κ':>10s} {'nfev':>8s}")
    print("-" * 70)
    for name, res in sorted(results, key=lambda kv: kv[1].rmse):
        print(
            f"{name:<18s} {res.rmse:>10.5f} {res.wall_time_s:>10.2f} "
            f"{res.params.kappa:>10.4f} {res.n_function_evals:>8d}"
        )
    print("\nNEEMP CCD_gen reference: κ = 0.5125 (Raček et al. 2016).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-mols", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(n_mols=args.n_mols, seed=args.seed)
