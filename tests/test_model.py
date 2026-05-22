"""Sanity tests for the EEM forward pass."""

from __future__ import annotations

import numpy as np
import pytest

from powelleem.model import _build_A_b_numpy, predict_charges_numpy
from powelleem.types import ParamSet


def test_charge_neutrality(water_molecule, reasonable_x):
    """For Q_total = 0, predicted charges must sum to zero."""
    solved = predict_charges_numpy(reasonable_x, water_molecule, n_types=3)
    assert abs(solved.q.sum()) < 1e-10


def test_charge_conservation_nonzero(water_molecule, reasonable_x):
    """For Q_total = +1, predicted charges sum to +1."""
    charged = water_molecule.__class__(
        smiles=water_molecule.smiles,
        atom_types=water_molecule.atom_types,
        inv_r=water_molecule.inv_r,
        target_charges=water_molecule.target_charges,
        formal_charge=1.0,
    )
    solved = predict_charges_numpy(reasonable_x, charged, n_types=3)
    assert abs(solved.q.sum() - 1.0) < 1e-10


def test_matrix_structure(water_molecule, reasonable_x):
    """A must be symmetric, last col/row = 1, A[n,n] = 0."""
    A, b = _build_A_b_numpy(
        reasonable_x, water_molecule.inv_r, water_molecule.atom_types, 0.0, n_types=3
    )
    n = water_molecule.n_atoms
    # Symmetry of the n×n upper block
    np.testing.assert_allclose(A[:n, :n], A[:n, :n].T, atol=1e-12)
    # Last column/row = 1
    np.testing.assert_array_equal(A[n, :n], 1.0)
    np.testing.assert_array_equal(A[:n, n], 1.0)
    assert A[n, n] == 0.0
    # b has -alpha in the n first positions, formal charge in the last
    expected_alpha = reasonable_x[1 : 1 + 3]
    np.testing.assert_array_equal(b[:n], -expected_alpha[water_molecule.atom_types - 1])
    assert b[n] == 0.0


def test_paramset_roundtrip(reasonable_x):
    atom_types = ("H", "O", "C")
    ps = ParamSet.from_vector(reasonable_x, atom_types)
    assert ps.n_types == 3
    assert ps.n_params == 7
    np.testing.assert_array_equal(ps.to_vector(), reasonable_x)

    d = ps.to_dict()
    ps2 = ParamSet.from_dict(d)
    np.testing.assert_allclose(ps2.to_vector(), reasonable_x)


def test_paramset_size_mismatch():
    with pytest.raises(ValueError):
        ParamSet.from_vector(np.array([0.5, 2.0]), ("H", "O", "C"))


def test_model_rmse_consistency(micro_model, micro_dataset, reasonable_x):
    """rmse should equal sqrt(mean(residuals**2))."""
    params = ParamSet.from_vector(reasonable_x, micro_model.atom_types)
    r = micro_model.residuals_flat(params, micro_dataset)
    expected = float(np.sqrt((r * r).mean()))
    assert micro_model.rmse(params, micro_dataset) == pytest.approx(expected)
