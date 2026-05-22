"""Tests for the analytical Jacobian.

The cardinal test: finite-difference vs. analytic should agree to ~1e-5
relative error. If this fails the entire AnalyticLM solver is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from powelleem.jacobian import check_jacobian, jacobian_one, residuals_and_jacobian
from powelleem.model import predict_charges_numpy


def test_analytic_matches_finite_diff(micro_dataset, reasonable_x):
    """Analytical Jacobian must match central finite differences to ~1e-5."""
    n_types = 3  # H, O, C
    diag = check_jacobian(reasonable_x, micro_dataset, n_types, eps=1e-6)
    assert diag["max_abs_err"] < 1e-5, diag
    assert diag["max_rel_err"] < 1e-3, diag


def test_jacobian_one_shape(water_molecule, reasonable_x):
    n_types = 3
    _, J = jacobian_one(reasonable_x, water_molecule, n_types)
    assert J.shape == (water_molecule.n_atoms, 1 + 2 * n_types)


def test_residuals_and_jacobian_concat(micro_dataset, reasonable_x):
    n_types = 3
    r, J = residuals_and_jacobian(reasonable_x, micro_dataset, n_types)
    assert r.shape == (micro_dataset.total_atoms,)
    assert J.shape == (micro_dataset.total_atoms, 1 + 2 * n_types)


def test_jacobian_kappa_column_sign(micro_dataset, reasonable_x):
    """∂q/∂κ via FD should match the analytical κ column."""
    n_types = 3
    r0, J = residuals_and_jacobian(reasonable_x, micro_dataset, n_types)

    eps = 1e-7
    x_plus = reasonable_x.copy()
    x_plus[0] += eps
    r_plus, _ = residuals_and_jacobian(x_plus, micro_dataset, n_types)
    fd = (r_plus - r0) / eps
    np.testing.assert_allclose(fd, J[:, 0], atol=1e-3)


def test_reusing_lu_factor(water_molecule, reasonable_x):
    """jacobian_one should accept a pre-computed _Solved without recomputing LU."""
    n_types = 3
    solved = predict_charges_numpy(reasonable_x, water_molecule, n_types)
    q, J = jacobian_one(reasonable_x, water_molecule, n_types, solved=solved)
    np.testing.assert_array_equal(q, solved.q)
    assert J.shape == (water_molecule.n_atoms, 1 + 2 * n_types)
