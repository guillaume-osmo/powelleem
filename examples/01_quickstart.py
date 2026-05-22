"""Quickstart — fit on a tiny synthetic dataset, no external data needed.

Run with::

    python examples/01_quickstart.py
"""

from __future__ import annotations

import numpy as np

from powelleem import EEMModel
from powelleem.solvers import AnalyticLM, SolverConfig
from powelleem.types import Dataset, MoleculeData


def make_water() -> MoleculeData:
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.0, 0.586], [-0.76, 0.0, 0.586]],
        dtype=np.float64,
    )
    diff = coords[:, None, :] - coords[None, :, :]
    r = np.sqrt((diff * diff).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r = np.where(r > 1e-8, 1.0 / r, 0.0)
    np.fill_diagonal(inv_r, 0.0)
    return MoleculeData(
        smiles="O",
        atom_types=np.array([2, 1, 1], dtype=np.int64),
        inv_r=inv_r,
        target_charges=np.array([-0.67, 0.335, 0.335], dtype=np.float64),
    )


def main() -> None:
    dataset = Dataset(molecules=[make_water()], atom_types=("H", "O"), name="single-water")
    model = EEMModel(atom_types=dataset.atom_types)
    solver = AnalyticLM(config=SolverConfig(seed=42), maxiter_lbfgs=100, maxiter_lm=100)

    result = solver.fit(model, dataset)
    print(f"RMSE             : {result.rmse:.6f}")
    print(f"wall (s)         : {result.wall_time_s:.3f}")
    print(f"params (vector)  : {result.params.to_vector()}")
    print(f"params (dict)    : {result.params.to_dict()}")


if __name__ == "__main__":
    main()
