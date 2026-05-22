# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a0] — 2026-05-22

### Added
- Initial scaffold from the MATLAB `de-uoa-matlab` pipeline (Godin 2017–2023).
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
