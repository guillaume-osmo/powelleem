"""Scale-up benchmark — NumPy-only solvers on NEEMP set01 (500 mol) and set03 (17769 mol).

The JAX backends (JaxAdaMuon, JaxMuonN, JaxLM, JaxAdam) hit a wall on
large datasets because XLA must re-JIT the loss function over the
whole training tensor — compile time scales with the number of molecule
blocks (a ``jit_loss`` trace of 17k blocks takes minutes per compile).

For production runs we therefore restrict ourselves to four NumPy
solvers whose runtime scales linearly with the number of molecules
without any JIT overhead:

* `DENewton`       — DE + analytical-Hessian polish (top of focused bench)
* `DEHybrid`       — DE + NEWUOA (the original MATLAB pipeline)
* `AnalyticNewton` — single-start trust-region Newton (Hessian-aware)
* `AnalyticLM`     — single-start L-BFGS-B + TRF/LM (Gauss-Newton)

Output is dropped to ``benchmarks/results/scale_set0{N}_{Nmols}.txt`` so
multiple runs accumulate. The script prints per-stage timings (DE
warm vs polish, L-BFGS warm vs Newton/LM polish) so we can see where
the wall is spent on each set.

Examples
--------
    # Baseline
    python examples/05_scale_numpy.py --set 01 --n-mols 500

    # Mid-scale
    python examples/05_scale_numpy.py --set 02 --n-mols 1000

    # NEEMP CCD_gen training set (full set03 = 17,769 mol)
    python examples/05_scale_numpy.py --set 03 --n-mols 17769
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
    DEHybrid,
    DENewton,
    SolverConfig,
)

NEEMP_EXAMPLES = Path("/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples")


def main(set_id: str, n_mols: int, seed: int = 42, skip_dehybrid: bool = False) -> None:
    sdf = NEEMP_EXAMPLES / f"set{set_id}.sdf"
    chg = NEEMP_EXAMPLES / f"set{set_id}.chg"
    typ = NEEMP_EXAMPLES / f"set{set_id}.typ"
    if not sdf.exists():
        raise FileNotFoundError(sdf)

    print(f"Loading NEEMP set{set_id} (limit={n_mols})…", flush=True)
    t0 = time.perf_counter()
    ds = load_neemp(sdf, chg, typ, limit=n_mols, name=f"NEEMP-set{set_id}-{n_mols}")
    print(
        f"  → {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    print(f"  atom types: {ds.atom_types}", flush=True)

    # Scale DE population/generations down for large datasets to keep wall
    # reasonable. Per-eval cost is O(N_mol · n³); we hold total eval budget
    # roughly constant.
    if ds.n_mols <= 100:
        de_pop, de_gen = 50, 20
    elif ds.n_mols <= 1000:
        de_pop, de_gen = 30, 15
    elif ds.n_mols <= 5000:
        de_pop, de_gen = 20, 10
    else:
        de_pop, de_gen = 15, 7

    print(f"\nDE configuration: pop={de_pop}, gen={de_gen} (auto-scaled)", flush=True)

    model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)
    config = SolverConfig(seed=seed)

    solvers = [
        ("AnalyticLM",     AnalyticLM(config=config, maxiter_lbfgs=100, maxiter_lm=100)),
        ("AnalyticNewton", AnalyticNewton(config=config, maxiter_lbfgs=100, maxiter_newton=100)),
        ("DENewton",       DENewton(config=config, population_size=de_pop, n_generations=de_gen)),
    ]
    if not skip_dehybrid:
        # DEHybrid is much slower (NEWUOA polish has higher nfev) — only include
        # for set01/set02 by default.
        solvers.append(
            ("DEHybrid",   DEHybrid(config=config, population_size=de_pop, n_generations=de_gen))
        )

    results = []
    for name, solver in solvers:
        print(f"\n=== {name} ===", flush=True)
        t = time.perf_counter()
        try:
            res = solver.fit(model, ds)
            wall = time.perf_counter() - t
            meta = res.solver_metadata
            stage1 = meta.get("wall_lbfgs_s") or meta.get("stage1_wall_s")
            stage2 = (
                meta.get("wall_lm_s")
                or meta.get("wall_newton_s")
                or meta.get("stage2_wall_s")
            )
            phase_str = ""
            if stage1 is not None and stage2 is not None:
                phase_str = f"  (phase1={stage1:.1f}s, phase2={stage2:.1f}s)"
            print(
                f"  RMSE = {res.rmse:.5f}    wall = {wall:.1f}s{phase_str}    "
                f"κ = {res.params.kappa:.4f}    nfev = {res.n_function_evals}",
                flush=True,
            )
            results.append((name, res, wall))
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    # Final summary
    print("\n" + "=" * 78)
    print(
        f"NEEMP set{set_id} — {ds.n_mols} mols, {ds.total_atoms} atoms, "
        f"{ds.n_types} types  (CCD_gen reference: κ = 0.5125)"
    )
    print(
        f"\n{'Solver':<18s} {'RMSE':>10s} {'wall(s)':>10s} {'κ':>10s} {'nfev':>10s}"
    )
    print("-" * 78)
    for name, res, wall in sorted(results, key=lambda kv: kv[1].rmse):
        print(
            f"{name:<18s} {res.rmse:>10.5f} {wall:>10.1f} "
            f"{res.params.kappa:>10.4f} {res.n_function_evals:>10d}"
        )

    # Dump per-element parameters of the best run for sanity inspection
    best = min(results, key=lambda kv: kv[1].rmse)
    name, res, _ = best
    d = res.params.to_dict()
    print(f"\nBest run = {name} (κ = {d['kappa']:.4f}):")
    print(f"  {'type':>10s}  {'α':>10s}  {'β':>10s}")
    for t in res.params.atom_types:
        print(f"  {t:>10s}  {d['alpha'][t]:>10.4f}  {d['beta'][t]:>10.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", type=str, default="01", help="NEEMP set id: 01, 02, 03")
    p.add_argument("--n-mols", type=int, default=500, help="cap on molecules loaded")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-dehybrid", action="store_true", help="omit DEHybrid (slow on large sets)")
    args = p.parse_args()
    main(set_id=args.set, n_mols=args.n_mols, seed=args.seed, skip_dehybrid=args.skip_dehybrid)
