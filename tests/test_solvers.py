"""Smoke tests: every available solver runs on a tiny dataset and reduces loss."""

from __future__ import annotations

import pytest

from powelleem.solvers import AnalyticLM, SolverConfig


def test_analytic_lm_converges(micro_model, micro_dataset):
    """AnalyticLM must always work (no optional deps)."""
    solver = AnalyticLM(
        config=SolverConfig(seed=42),
        maxiter_lbfgs=20,
        maxiter_lm=50,
    )
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial
    assert res.rmse >= 0
    assert res.params.atom_types == micro_model.atom_types


def test_analytic_lm_writes_rdkit_header(tmp_path, micro_model, micro_dataset):
    solver = AnalyticLM(maxiter_lbfgs=5, maxiter_lm=5)
    res = solver.fit(micro_model, micro_dataset)
    out = tmp_path / "EEM_params.h"
    res.export_rdkit_header(out)
    text = out.read_text()
    assert "const double kappa" in text
    assert "const double A1[]" in text
    assert "const double B1[]" in text


@pytest.mark.jax
def test_jax_adam_runs(micro_model, micro_dataset):
    pytest.importorskip("jax")
    from powelleem.solvers import JaxAdam

    solver = JaxAdam(n_iterations=50, learning_rate=0.01)
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial * 1.1  # allow tiny overshoot


@pytest.mark.jax
def test_jax_lm_runs(micro_model, micro_dataset):
    pytest.importorskip("jax")
    from powelleem.solvers import JaxLM

    solver = JaxLM(maxiter_lbfgs=10, maxiter_lm=20)
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial


@pytest.mark.powell
def test_newuoa_runs(micro_model, micro_dataset):
    pytest.importorskip("pdfo")
    from powelleem.solvers import Newuoa

    solver = Newuoa(max_fev=500)
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial


@pytest.mark.powell
def test_bobyqa_runs(micro_model, micro_dataset):
    pytest.importorskip("pdfo")
    from powelleem.solvers import Bobyqa

    solver = Bobyqa(max_fev=500)
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial


@pytest.mark.slow
def test_de_hybrid_smoke(micro_model, micro_dataset):
    """DEHybrid with small population — just confirm it runs."""
    from powelleem.solvers import DEHybrid

    solver = DEHybrid(population_size=10, n_generations=5)
    res = solver.fit(micro_model, micro_dataset)
    assert res.loss_final <= res.loss_initial
