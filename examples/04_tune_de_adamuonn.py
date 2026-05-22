"""Tune the polish stage of DEAdaMuonn to close the gap with DENewton.

DENewton (DE + Hessian)  : RMSE 0.0557 on NEEMP set01 (30 mol)
DEAdaMuonn defaults       : RMSE 0.114, lr=0.005, n_iter=500, betas=(0.9, 0.1, 0.999)

Strategy
--------
1. Stage 1 — sweep ``(polish_learning_rate, polish_iterations)`` on
   default betas/ns_steps to find the descent budget.
2. Stage 2 — at the best (lr, n_iter), sweep the AdamN ``betas`` triple
   to find the EMA structure that fits this loss landscape.
3. Stage 3 — at the best (lr, n_iter, betas), sweep ``ns_steps`` —
   for a flat parameter vector the NS step is degenerate, so higher
   ``ns_steps`` may not help.

Each run uses the same DE warm-start seed (42), so any difference is
purely from the polish stage. Total runtime ≈ 5 min on 30 mol.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

from powelleem import EEMModel
from powelleem.data import load_neemp
from powelleem.solvers import DEAdaMuonn, DENewton, SolverConfig

NEEMP_EXAMPLES = Path("/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples")


def fit(model, ds, **kwargs):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    res = DEAdaMuonn(config=SolverConfig(seed=42), **kwargs).fit(model, ds)
    return {
        "rmse": res.rmse,
        "kappa": res.params.kappa,
        "wall_s": time.perf_counter() - t0,
        "params": kwargs,
        "stage1_de_rmse": res.solver_metadata.get("stage1_de_rmse"),
        "stage2_rmse": res.solver_metadata.get("stage2_rmse"),
    }


def main(n_mols: int = 30) -> None:
    print(f"Loading NEEMP set01 (limit={n_mols})…", flush=True)
    ds = load_neemp(
        NEEMP_EXAMPLES / "set01.sdf",
        NEEMP_EXAMPLES / "set01.chg",
        NEEMP_EXAMPLES / "set01.typ",
        limit=n_mols,
        name=f"NEEMP-set01-{n_mols}",
    )
    print(
        f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types\n",
        flush=True,
    )
    model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)

    # Reference: DENewton with same DE settings
    print("=== Reference: DENewton ===", flush=True)
    t0 = time.perf_counter()
    ref = DENewton(
        config=SolverConfig(seed=42),
        population_size=50,
        n_generations=20,
        maxiter_lbfgs=100,
        maxiter_newton=100,
    ).fit(model, ds)
    print(
        f"  RMSE = {ref.rmse:.5f}  wall = {time.perf_counter() - t0:.2f}s  κ = {ref.params.kappa:.4f}\n",
        flush=True,
    )

    all_results: list[dict] = []

    # ------------------------------------------------------------------
    # Stage 1: sweep (learning_rate × n_iterations) on default betas
    # ------------------------------------------------------------------
    print("--- Stage 1: lr × n_iter sweep ---", flush=True)
    lr_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
    iter_grid = [200, 500, 1000, 2000]
    print(f"{'lr':>8s} {'n_iter':>8s} {'RMSE':>10s} {'wall(s)':>10s} {'κ':>10s}", flush=True)
    s1_results = []
    for lr, n in product(lr_grid, iter_grid):
        r = fit(model, ds, polish_iterations=n, polish_learning_rate=lr)
        print(f"{lr:>8.3f} {n:>8d} {r['rmse']:>10.5f} {r['wall_s']:>10.2f} {r['kappa']:>10.4f}", flush=True)
        r["stage"] = 1
        all_results.append(r)
        s1_results.append(r)

    best_s1 = min(s1_results, key=lambda r: r["rmse"])
    best_lr = best_s1["params"]["polish_learning_rate"]
    best_n = best_s1["params"]["polish_iterations"]
    print(
        f"\nStage 1 best: lr={best_lr}, n_iter={best_n} → RMSE {best_s1['rmse']:.5f}\n",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 2: sweep betas at best (lr, n_iter)
    # ------------------------------------------------------------------
    print("--- Stage 2: betas sweep at best (lr, n_iter) ---", flush=True)
    beta_grid = [
        (0.9, 0.1, 0.999),   # default
        (0.9, 0.0, 0.999),   # no nested EMA — collapses to AdamW-style
        (0.95, 0.2, 0.99),   # more momentum + faster sq
        (0.5, 0.5, 0.95),    # symmetric / quick adaptation
        (0.99, 0.05, 0.9999),  # very slow grad EMA, very slow sq
    ]
    print(
        f"{'betas':>22s} {'RMSE':>10s} {'wall(s)':>10s} {'κ':>10s}",
        flush=True,
    )
    s2_results = []
    for betas in beta_grid:
        r = fit(
            model, ds,
            polish_iterations=best_n, polish_learning_rate=best_lr,
            polish_betas=betas,
        )
        print(
            f"{str(betas):>22s} {r['rmse']:>10.5f} {r['wall_s']:>10.2f} {r['kappa']:>10.4f}",
            flush=True,
        )
        r["stage"] = 2
        all_results.append(r)
        s2_results.append(r)

    best_s2 = min(s2_results, key=lambda r: r["rmse"])
    best_betas = best_s2["params"]["polish_betas"]
    print(f"\nStage 2 best: betas={best_betas} → RMSE {best_s2['rmse']:.5f}\n", flush=True)

    # ------------------------------------------------------------------
    # Stage 3: sweep ns_steps at best (lr, n_iter, betas)
    # ------------------------------------------------------------------
    print("--- Stage 3: ns_steps sweep ---", flush=True)
    ns_grid = [1, 3, 5, 10, 20]
    s3_results = []
    for ns in ns_grid:
        r = fit(
            model, ds,
            polish_iterations=best_n,
            polish_learning_rate=best_lr,
            polish_betas=best_betas,
            polish_ns_steps=ns,
        )
        print(
            f"  ns_steps={ns:>3d}: RMSE = {r['rmse']:.5f}  wall = {r['wall_s']:.2f}s  κ = {r['kappa']:.4f}",
            flush=True,
        )
        r["stage"] = 3
        all_results.append(r)
        s3_results.append(r)

    best_s3 = min(s3_results, key=lambda r: r["rmse"])
    best_ns = best_s3["params"].get("polish_ns_steps", 5)
    print(f"\nStage 3 best: ns_steps={best_ns} → RMSE {best_s3['rmse']:.5f}\n", flush=True)

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"DENewton reference            : RMSE {ref.rmse:.5f}  κ {ref.params.kappa:.4f}")
    print(f"DEAdaMuonn defaults           : RMSE 0.11377  κ 0.2485   (from prior bench)")
    print(
        f"DEAdaMuonn tuned best         : RMSE {best_s3['rmse']:.5f}  κ {best_s3['kappa']:.4f}",
        flush=True,
    )
    print(
        f"  config: lr={best_lr}, n_iter={best_n}, betas={best_betas}, ns_steps={best_ns}",
        flush=True,
    )

    # Save full sweep results
    out = Path("benchmarks/results/de_adamuonn_tune_30mols.json")
    out.write_text(json.dumps({
        "reference_DENewton": {"rmse": ref.rmse, "kappa": ref.params.kappa},
        "best_config": {
            "polish_learning_rate": best_lr,
            "polish_iterations": best_n,
            "polish_betas": best_betas,
            "polish_ns_steps": best_ns,
            "rmse": best_s3["rmse"],
            "kappa": best_s3["kappa"],
            "wall_s": best_s3["wall_s"],
        },
        "all_runs": all_results,
    }, indent=2, default=str))
    print(f"\nSaved sweep → {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-mols", type=int, default=30)
    args = p.parse_args()
    main(n_mols=args.n_mols)
