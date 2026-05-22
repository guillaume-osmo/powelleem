"""Shared pytest fixtures — tiny synthetic datasets for fast tests."""

from __future__ import annotations

import numpy as np
import pytest

from powelleem.model import EEMModel
from powelleem.types import Dataset, MoleculeData


def _build_inv_r(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    r = np.sqrt((diff * diff).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r = np.where(r > 1e-8, 1.0 / r, 0.0)
    np.fill_diagonal(inv_r, 0.0)
    return inv_r


@pytest.fixture
def water_molecule() -> MoleculeData:
    """Idealised water at 0.96 Å O–H, 104.5° HOH angle."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],             # O
            [0.760, 0.0, 0.586],          # H
            [-0.760, 0.0, 0.586],         # H
        ],
        dtype=np.float64,
    )
    return MoleculeData(
        smiles="O",
        atom_types=np.array([2, 1, 1], dtype=np.int64),  # type 1 = H, type 2 = O
        inv_r=_build_inv_r(coords),
        target_charges=np.array([-0.67, 0.335, 0.335], dtype=np.float64),
        formal_charge=0.0,
    )


@pytest.fixture
def methane_molecule() -> MoleculeData:
    """Tetrahedral CH4 — used to stress per-element typing."""
    a = 1.087 / np.sqrt(3.0)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, a, a],
            [-a, -a, a],
            [-a, a, -a],
            [a, -a, -a],
        ],
        dtype=np.float64,
    )
    return MoleculeData(
        smiles="C",
        atom_types=np.array([3, 1, 1, 1, 1], dtype=np.int64),  # type 1 = H, type 3 = C
        inv_r=_build_inv_r(coords),
        target_charges=np.array([-0.47, 0.1175, 0.1175, 0.1175, 0.1175], dtype=np.float64),
        formal_charge=0.0,
    )


@pytest.fixture
def micro_dataset(water_molecule: MoleculeData, methane_molecule: MoleculeData) -> Dataset:
    """Two-molecule toy dataset (H, O, C atom types)."""
    return Dataset(
        molecules=[water_molecule, methane_molecule],
        atom_types=("H", "O", "C"),
        name="micro",
    )


@pytest.fixture
def micro_model(micro_dataset: Dataset) -> EEMModel:
    return EEMModel(atom_types=micro_dataset.atom_types)


@pytest.fixture
def reasonable_x(micro_model: EEMModel) -> np.ndarray:
    """A plausible parameter vector inside default bounds."""
    n = micro_model.n_types
    return np.concatenate([[0.5], np.full(n, 2.5), np.full(n, 0.5)])
