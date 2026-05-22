"""Analytical Hessian vs finite-difference of the gradient.

If this fails the AnalyticNewton solver is wrong.
"""

from __future__ import annotations

import pytest

from powelleem.hessian import check_hessian, loss_grad_hessian


def test_hessian_matches_finite_diff(micro_dataset, reasonable_x):
    n_types = 3  # H, O, C
    diag = check_hessian(reasonable_x, micro_dataset, n_types, eps=1e-5)
    assert diag["atol_satisfied"], diag


def test_hessian_symmetry(micro_dataset, reasonable_x):
    n_types = 3
    _, _, H = loss_grad_hessian(reasonable_x, micro_dataset, n_types)
    import numpy as np
    diff = float(np.max(np.abs(H - H.T)))
    assert diff < 1e-10, f"Hessian not symmetric, max |H − Hᵀ| = {diff:.2e}"


def test_analytic_newton_runs(micro_model, micro_dataset):
    from powelleem.solvers import AnalyticNewton, SolverConfig

    solver = AnalyticNewton(
        config=SolverConfig(seed=42),
        maxiter_lbfgs=20,
        maxiter_newton=20,
    )
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial
