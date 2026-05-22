"""Typer-based command-line interface.

Examples::

    powelleem load-chaos /path/to/CHAOS.zip --n-mols 1000 --target apt -o data.npz
    powelleem fit data.npz --solver AnalyticLM --out result.json
    powelleem bench data.npz --solvers AnalyticLM,Newuoa,Bobyqa --repeats 5 -o bench
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="powelleem",
    help="EEM parameter fitting — analytical Jacobian + Powell solvers.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def load_chaos(
    zip_path: Path = typer.Argument(..., help="Path to CHAOS.zip"),
    n_mols: int = typer.Option(100, "--n-mols", "-n", help="Number of molecules to load"),
    max_n_atoms: int = typer.Option(30, "--max-atoms", help="Skip mols bigger than this"),
    target: str = typer.Option("apt", help="Target charges: apt, mulliken, cosmo, ..."),
    output: Path = typer.Option(Path("dataset.pkl"), "--output", "-o", help="Pickled dataset out"),
) -> None:
    """Load a CHAOS subset and pickle it for downstream commands."""
    import pickle

    from powelleem.data import load_chaos as load_chaos_fn

    console.print(f"Loading {n_mols} molecules from {zip_path}…")
    ds = load_chaos_fn(zip_path, n_mols=n_mols, max_n_atoms=max_n_atoms, target=target)  # type: ignore[arg-type]
    console.print(
        f"  → {ds.n_mols} mols, {ds.total_atoms} atoms, {ds.n_types} atom types"
    )
    with output.open("wb") as fh:
        pickle.dump(ds, fh)
    console.print(f"[green]Saved {output}[/green]")


@app.command()
def fit(
    dataset_pkl: Path = typer.Argument(..., help="Pickled dataset (from load-chaos)"),
    solver: str = typer.Option("AnalyticLM", help="Solver name (see `--solvers` list)"),
    seed: int = typer.Option(42, help="Random seed"),
    output: Path = typer.Option(Path("result.json"), "--output", "-o"),
    export_rdkit: Path | None = typer.Option(None, "--export-rdkit", help="Write EEM_params.h"),
) -> None:
    """Fit EEM parameters with one solver."""
    import pickle

    from powelleem import EEMModel
    from powelleem.solvers import (
        AnalyticLM,
        Bobyqa,
        DEHybrid,
        JaxAdam,
        JaxLM,
        Newuoa,
        SolverConfig,
    )

    solver_cls = {
        "AnalyticLM": AnalyticLM,
        "JaxAdam": JaxAdam,
        "JaxLM": JaxLM,
        "Newuoa": Newuoa,
        "Bobyqa": Bobyqa,
        "DEHybrid": DEHybrid,
    }.get(solver)
    if solver_cls is None:
        console.print(f"[red]Unknown solver: {solver}[/red]")
        raise typer.Exit(1)

    with dataset_pkl.open("rb") as fh:
        ds = pickle.load(fh)
    model = EEMModel(atom_types=ds.atom_types)

    config = SolverConfig(seed=seed)
    result = solver_cls(config=config).fit(model, ds)

    console.print(f"RMSE  = {result.rmse:.5f}")
    console.print(f"wall  = {result.wall_time_s:.2f}s")
    console.print(f"loss0 = {result.loss_initial:.5f}  loss_final = {result.loss_final:.5f}")
    result.to_json(output)
    if export_rdkit is not None:
        result.export_rdkit_header(export_rdkit)
        console.print(f"[green]Wrote RDKit header → {export_rdkit}[/green]")
    console.print(f"[green]Saved {output}[/green]")


@app.command()
def bench(
    dataset_pkl: Path = typer.Argument(..., help="Pickled dataset"),
    solvers: str = typer.Option(
        "AnalyticLM,Newuoa,Bobyqa",
        help="Comma-separated solver names",
    ),
    repeats: int = typer.Option(5, "--repeats", "-r", help="Repetitions per solver"),
    output_md: Path = typer.Option(Path("bench.md"), "--md"),
    output_json: Path = typer.Option(Path("bench.json"), "--json"),
    output_png: Path | None = typer.Option(None, "--png", help="Convergence plot output"),
) -> None:
    """Run a multi-solver benchmark with statistical replication."""
    import pickle

    from powelleem import EEMModel
    from powelleem.bench import compare
    from powelleem.solvers import (
        AnalyticLM,
        Bobyqa,
        DEHybrid,
        JaxAdam,
        JaxLM,
        Newuoa,
    )

    available = {
        "AnalyticLM": AnalyticLM,
        "JaxAdam": JaxAdam,
        "JaxLM": JaxLM,
        "Newuoa": Newuoa,
        "Bobyqa": Bobyqa,
        "DEHybrid": DEHybrid,
    }
    solver_instances = []
    for name in solvers.split(","):
        name = name.strip()
        if name not in available:
            console.print(f"[red]Unknown solver: {name}[/red]")
            raise typer.Exit(1)
        solver_instances.append(available[name]())

    with dataset_pkl.open("rb") as fh:
        ds = pickle.load(fh)
    model = EEMModel(atom_types=ds.atom_types)

    report = compare(model, ds, solver_instances, n_repeats=repeats)
    report.to_markdown(output_md)
    report.to_json(output_json)

    # Print a Rich table to the console
    table = Table(title=f"Benchmark — {ds.name}")
    table.add_column("Solver")
    table.add_column("RMSE mean", justify="right")
    table.add_column("RMSE σ", justify="right")
    table.add_column("wall mean (s)", justify="right")
    for name, row in sorted(report.summary().items(), key=lambda kv: kv[1]["rmse_mean"]):
        table.add_row(
            name,
            f"{row['rmse_mean']:.5f}",
            f"{row['rmse_stdev']:.5f}",
            f"{row['wall_mean_s']:.2f}",
        )
    console.print(table)

    if output_png is not None:
        report.plot_convergence(output_png)
        console.print(f"[green]Wrote convergence plot → {output_png}[/green]")


if __name__ == "__main__":
    app()
