# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Switched `Newuoa`/`Bobyqa`/`DEHybrid` from non-existent `prima.minimize`
  to the actual PDFO API (`from pdfo import pdfo`) with the correct
  options layout (`radius_init`/`radius_final` not `rhobeg`/`rhoend`,
  bounds as list of scalar 2-tuples).
- `pyproject.toml` `[powell]` extra now installs `pdfo>=2.0` (not `prima`).
- Test markers updated: `pytest.importorskip("pdfo")` for Powell solver tests.

### Added
- `load_neemp(sdf, chg, typ)` reads the original NEEMP-format triplet
  (Raček 2016) with element-bond-order atom typing.
- `examples/02_validate_against_matlab.py` — first validation against the
  original MATLAB workflow data (NEEMP set01).
- `benchmarks/results/neemp_set01_30mols_first_validation.txt` — first
  benchmark log: all 5 non-DE solvers converge on real B3LYP/6-311G NPA
  reference charges. AnalyticLM is 5× faster than NEWUOA at comparable RMSE.
- **`powelleem.hessian`** — analytical Hessian of the EEM sum-of-squares
  loss via the implicit-function theorem applied twice. Cost is one extra
  LU back-substitution per (κ–κ, κ–α_k, κ–β_k, α_j–β_k, β_j–β_k) pair
  per molecule, on the matrix already factorised by the forward pass.
- **`AnalyticNewton`** solver — L-BFGS-B warm start + SciPy
  ``trust-exact`` Newton with the full analytical Hessian. On NEEMP set01
  it finds κ ≈ 0.53, matching the published NEEMP reference (κ = 0.5125),
  while AnalyticLM (Gauss-Newton / no curvature) stays trapped in a
  local basin at κ ≈ 1.39. The full Hessian's directional information
  escapes the κ-α coupling that flattens the loss along α-only directions.
- `benchmarks/results/neemp_set01_30mols_7solvers_with_newton.txt` — full
  7-solver benchmark showing the Hessian-aware solver finds the published
  κ at the cost of 3.6 s wall (1.7× AnalyticLM).

## [0.1.0a0] — 2026-05-22

### Added
- Initial scaffold from the MATLAB `de-uoa-matlab` pipeline (Godin 2017–2023),
  with parameter-fitting protocol inspired by **NEEMP** (Raček et al.,
  *J. Cheminform.* 2016, 8, 57, DOI: 10.1186/s13321-016-0171-1) — atom
  typing, default-parameter fallback hierarchy, DE+NEWUOA hybrid strategy.
- `powelleem.model` — NumPy + JAX EEM forward pass.
- `powelleem.jacobian` — analytical Jacobian via the implicit function theorem.
- `powelleem.data` — CHAOS, NEEMP-legacy (`.chg`/`.typ`/`.sdf`) and generic loaders.
- `powelleem.solvers` — six interchangeable solvers under a uniform `Solver` ABC:
  - `AnalyticLM` (NumPy + analytic Jacobian + SciPy L-BFGS-B → TRF/LM).
  - `JaxAdam` (JAX autodiff + Adam, mirrors `pure_jax_gradient_optimizer.py`).
  - `JaxLM` (JAX autodiff + L-BFGS-B → TRF/LM).
  - `Newuoa` (PRIMA's NEWUOA, matches MATLAB reference).
  - `Bobyqa` (PRIMA's BOBYQA, native bound constraints).
  - `DEHybrid` (DE outer + NEWUOA polish, reproduces `DE_UOA_FINAL.m`).
- `powelleem.bench` — repeated-runs benchmark harness with summary tables and plots.
- `powelleem.cli` — Typer-based command-line interface.
- Tests for forward pass, Jacobian (finite-diff vs analytic), solver convergence.
- GitHub Actions CI (pytest + ruff + mypy + build).
