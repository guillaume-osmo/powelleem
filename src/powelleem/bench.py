"""Benchmark harness — compare multiple solvers with statistical replication.

Typical usage::

    from powelleem.bench import compare
    from powelleem.solvers import AnalyticLM, Newuoa, Bobyqa, JaxAdam, JaxLM, DEHybrid

    report = compare(
        model, dataset,
        solvers=[AnalyticLM(), Newuoa(), Bobyqa(), JaxAdam(), JaxLM(), DEHybrid()],
        n_repeats=10, seeds=range(42, 52),
    )
    report.to_markdown("bench.md")
    report.plot_convergence("convergence.png")
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from powelleem.model import EEMModel
    from powelleem.solvers.base import Solver
    from powelleem.types import Dataset, FitResult


@dataclass(slots=True)
class BenchmarkReport:
    """Outcome of a multi-solver × multi-seed benchmark."""

    runs: list[FitResult]  # all individual runs (flat)
    by_solver: dict[str, list[FitResult]] = field(default_factory=dict)
    dataset_name: str = ""
    n_mols: int = 0
    n_atoms: int = 0

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-solver mean/median/stddev of RMSE and wall time."""
        out: dict[str, dict[str, float]] = {}
        for solver_name, runs in self.by_solver.items():
            rmses = [r.rmse for r in runs]
            walls = [r.wall_time_s for r in runs]
            loss_finals = [r.loss_final for r in runs]
            out[solver_name] = {
                "n_runs": len(runs),
                "rmse_mean": float(statistics.mean(rmses)),
                "rmse_median": float(statistics.median(rmses)),
                "rmse_stdev": float(statistics.stdev(rmses)) if len(rmses) > 1 else 0.0,
                "rmse_min": float(min(rmses)),
                "rmse_max": float(max(rmses)),
                "wall_mean_s": float(statistics.mean(walls)),
                "wall_median_s": float(statistics.median(walls)),
                "loss_final_mean": float(statistics.mean(loss_finals)),
            }
        return out

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render the per-solver summary as a Markdown table."""
        s = self.summary()
        header = (
            "| Solver | runs | RMSE mean | RMSE median | RMSE σ | "
            "RMSE min | RMSE max | wall mean (s) |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        lines = [header]
        # Sort by mean RMSE ascending
        for name, row in sorted(s.items(), key=lambda kv: kv[1]["rmse_mean"]):
            lines.append(
                f"| {name} | {int(row['n_runs'])} "
                f"| {row['rmse_mean']:.5f} "
                f"| {row['rmse_median']:.5f} "
                f"| {row['rmse_stdev']:.5f} "
                f"| {row['rmse_min']:.5f} "
                f"| {row['rmse_max']:.5f} "
                f"| {row['wall_mean_s']:.2f} |"
            )
        body = "\n".join(lines)
        title = f"# Benchmark report — {self.dataset_name} ({self.n_mols} mols, {self.n_atoms} atoms)\n\n"
        text = title + body + "\n"
        if path is not None:
            Path(path).write_text(text)
        return text

    def to_json(self, path: str | Path) -> None:
        """Dump per-run details and summary to a JSON file."""
        data = {
            "dataset_name": self.dataset_name,
            "n_mols": self.n_mols,
            "n_atoms": self.n_atoms,
            "summary": self.summary(),
            "runs": [r.to_dict() for r in self.runs],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def plot_convergence(self, path: str | Path) -> None:
        """Plot loss-vs-iteration curves, one panel per solver."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "matplotlib is required for plots. Install with `pip install powelleem[viz]`."
            ) from exc

        n_solvers = len(self.by_solver)
        fig, axes = plt.subplots(
            1, n_solvers, figsize=(5 * n_solvers, 4), sharey=True, squeeze=False
        )
        for ax, (name, runs) in zip(axes[0], self.by_solver.items(), strict=True):
            for r in runs:
                if r.loss_trajectory:
                    ax.semilogy(r.loss_trajectory, alpha=0.5, linewidth=0.8)
            ax.set_title(name)
            ax.set_xlabel("function eval")
            ax.grid(True, alpha=0.3)
        axes[0, 0].set_ylabel("loss (log)")
        fig.suptitle(f"Convergence — {self.dataset_name}")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def compare(
    model: EEMModel,
    dataset: Dataset,
    solvers: list[Solver],
    *,
    n_repeats: int = 5,
    seeds: Iterable[int] | None = None,
    verbose: bool = True,
) -> BenchmarkReport:
    """Run every solver ``n_repeats`` times on the same dataset.

    Parameters
    ----------
    model
        The :class:`EEMModel` (atom-type vocabulary).
    dataset
        Training :class:`Dataset`.
    solvers
        List of solver *instances*. Each is shallow-copied per repetition
        with a fresh seed via ``solver.config.seed = seed``.
    n_repeats
        Number of independent fits per solver. Each draws its own ``x_0``.
    seeds
        Seeds to use; defaults to ``range(42, 42 + n_repeats)``.
    """
    if seeds is None:
        seeds = list(range(42, 42 + n_repeats))
    else:
        seeds = list(seeds)

    if len(seeds) < n_repeats:
        raise ValueError(
            f"need {n_repeats} seeds but only {len(seeds)} provided"
        )

    all_runs: list[FitResult] = []
    by_solver: dict[str, list[FitResult]] = {s.name: [] for s in solvers}

    for solver in solvers:
        for rep in range(n_repeats):
            solver.config.seed = seeds[rep]
            if verbose:
                print(f"[bench] {solver.name} run {rep + 1}/{n_repeats} (seed={seeds[rep]})", flush=True)
            result = solver.fit(model, dataset)
            all_runs.append(result)
            by_solver[solver.name].append(result)

    return BenchmarkReport(
        runs=all_runs,
        by_solver=by_solver,
        dataset_name=dataset.name,
        n_mols=dataset.n_mols,
        n_atoms=dataset.total_atoms,
    )
