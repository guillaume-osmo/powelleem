"""Multi-process parallel-speedup benchmark — Linux production tool.

On macOS this script will exhibit BrokenProcessPool because Apple's
Accelerate framework is not fork-safe. The real intent is to run on a
Linux box where fork is natural and we see the true 8-11× speedup.

Reports, for each ``n_workers``:
  - wall time of forward + Jacobian
  - wall time of analytical Hessian
  - speedup vs serial
  - residual / Jacobian / Hessian max |Δ| vs serial reference

Usage::

    POWELLEEM_NEEMP_DIR=~/neemp-data python examples/07_parallel_bench.py \
        --set 01 --n-mols 500 --workers 16
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from powelleem.data import load_neemp
from powelleem.hessian import loss_grad_hessian
from powelleem.jacobian import residuals_and_jacobian
from powelleem.parallel import (
    loss_grad_hessian_parallel,
    residuals_and_jacobian_parallel,
)


def main(set_id: str, n_mols: int, max_workers: int) -> None:
    neemp_dir = Path(
        os.environ.get(
            "POWELLEEM_NEEMP_DIR",
            "/Volumes/RescueCopy/Downloads/de-uoa-matlab/neemp/examples",
        )
    )
    sdf = neemp_dir / f"set{set_id}.sdf"
    chg = neemp_dir / f"set{set_id}.chg"
    typ = neemp_dir / f"set{set_id}.typ"

    print(f"Loading NEEMP set{set_id} (limit={n_mols})…", flush=True)
    t0 = time.perf_counter()
    ds = load_neemp(sdf, chg, typ, limit=n_mols, name=f"set{set_id}-{n_mols}")
    print(
        f"  {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} types  "
        f"(loaded in {time.perf_counter() - t0:.1f}s)\n",
        flush=True,
    )
    T = ds.n_types

    rng = np.random.default_rng(42)
    x = np.concatenate([[0.5], rng.uniform(1.8, 3.2, T), rng.uniform(0.0, 1.0, T)])

    # ---- serial reference ----
    t0 = time.perf_counter()
    r_s, J_s = residuals_and_jacobian(x, ds, T)
    t_serial_rj = time.perf_counter() - t0
    t0 = time.perf_counter()
    l_s, g_s, H_s = loss_grad_hessian(x, ds, T)
    t_serial_H = time.perf_counter() - t0
    print(f"serial    : rJ {t_serial_rj * 1000:>7.1f} ms    H {t_serial_H * 1000:>7.1f} ms", flush=True)

    backends = ["fork", "spawn", "thread"]
    n_workers_list = [w for w in [2, 4, 8, max_workers] if w >= 2]

    print()
    print(f"{'backend':<8s} {'nw':>3s}  {'rJ (ms)':>10s} {'speedup':>9s}  {'H (ms)':>10s} {'speedup':>9s}  {'ΔJ':>10s} {'ΔH':>10s}")
    print("-" * 90)
    for backend in backends:
        for nw in n_workers_list:
            try:
                t0 = time.perf_counter()
                r_p, J_p = residuals_and_jacobian_parallel(x, ds, T, n_workers=nw, backend=backend)
                t_rj = time.perf_counter() - t0
                t0 = time.perf_counter()
                l_p, g_p, H_p = loss_grad_hessian_parallel(x, ds, T, n_workers=nw, backend=backend)
                t_H = time.perf_counter() - t0
                err_J = float(np.max(np.abs(J_s - J_p)))
                err_H = float(np.max(np.abs(H_s - H_p)))
                print(
                    f"{backend:<8s} {nw:>3d}  "
                    f"{t_rj * 1000:>10.1f} {t_serial_rj / t_rj:>8.2f}×  "
                    f"{t_H * 1000:>10.1f} {t_serial_H / t_H:>8.2f}×  "
                    f"{err_J:>10.1e} {err_H:>10.1e}",
                    flush=True,
                )
            except Exception as exc:
                print(f"{backend:<8s} {nw:>3d}  FAILED: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", type=str, default="01")
    p.add_argument("--n-mols", type=int, default=500)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = p.parse_args()
    main(set_id=args.set, n_mols=args.n_mols, max_workers=args.workers)
