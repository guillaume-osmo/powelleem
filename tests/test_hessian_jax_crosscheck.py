"""Cross-check the analytical Hessian against JAX autodiff.

If both routes agree to ~1e-6 we have an end-to-end implementation
check that is more rigorous than the finite-difference test on
gradients (which is already in tests/test_hessian.py).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.jax
def test_analytical_hessian_matches_jax_autodiff(micro_dataset, reasonable_x):
    pytest.importorskip("jax")
    from powelleem.hessian import loss_grad_hessian, loss_hessian_jax

    n_types = 3  # H, O, C
    _, grad_a, H_a = loss_grad_hessian(reasonable_x, micro_dataset, n_types)
    _, grad_j, H_j = loss_hessian_jax(reasonable_x, micro_dataset, n_types)

    np.testing.assert_allclose(grad_a, grad_j, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(H_a, H_j, atol=1e-5, rtol=1e-3)


@pytest.mark.jax
def test_jax_adamuon_runs(micro_model, micro_dataset):
    pytest.importorskip("jax")
    from powelleem.solvers import JaxAdaMuon, SolverConfig

    solver = JaxAdaMuon(
        config=SolverConfig(seed=42),
        n_iterations=200,
        learning_rate=0.02,
    )
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial * 1.1  # AdaMuon should at least not blow up
