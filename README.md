# powelleem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

**Modern Python port of the EEM parameter-fitting MATLAB pipeline (Godin 2017–2023),
with analytical Jacobian via the implicit function theorem and a uniform API
over six solvers including Powell's NEWUOA/BOBYQA.**

The Electronegativity Equalization Method (EEM, Mortier 1986) predicts atomic
partial charges from per-element parameters (electronegativity α, hardness β,
screening κ). Fitting those parameters to ab-initio reference charges is a
non-linear least-squares problem that historically used derivative-free solvers
(Powell's NEWUOA / BOBYQA). This package adds:

1. **Analytical Jacobian** via the implicit function theorem — exact gradients
   at 1 LU factorization + (1 + 2T) back-substitutions per molecule.
2. **Six solvers under one API**: NumPy + analytic LM, JAX autodiff + Adam,
   JAX + LM, NEWUOA, BOBYQA, DE + NEWUOA hybrid (reproduces MATLAB
   `DE_UOA_FINAL.m` byte-for-byte).
3. **Statistical benchmark harness**: repeated runs, multiple random seeds,
   convergence plots, summary tables.

## Quick start

```bash
pip install "powelleem[all]"
```

```python
from powelleem import EEMModel, load_chaos
from powelleem.solvers import AnalyticLM

ds = load_chaos("/path/to/CHAOS.zip", n_mols=1000, target="apt")
model = EEMModel(atom_types=ds.atom_types, use_bond_order=True)

result = AnalyticLM(maxiter_lbfgs=200, maxiter_lm=200).fit(model, ds, seed=42)
print(f"RMSE: {result.rmse:.4f}")
print(result.params.to_dict())

# Export to RDKit C++ header
result.export_rdkit_header("EEM_params_v2.h")
```

## Comparing solvers

```python
from powelleem.bench import compare
from powelleem.solvers import AnalyticLM, Newuoa, Bobyqa, JaxAdam, JaxLM, DEHybrid

report = compare(
    model, ds,
    solvers=[AnalyticLM(), Newuoa(), Bobyqa(), JaxAdam(), JaxLM(), DEHybrid()],
    n_repeats=10, seeds=range(42, 52),
)
report.to_markdown("benchmark.md")
report.plot_convergence("convergence.png")
```

## Math model

For a molecule of `n` atoms with parameter vector `x = (κ, α_1..α_T, β_1..β_T)`:

```
( β_{t(i)}    κ/r_{ij}   ...   1 ) ( q_i )    ( -α_{t(i)} )
( κ/r_{ji}   β_{t(j)}    ...   1 ) ( ... )  = ( ...       )
( ...                          1 ) ( q_n )    ( -α_{t(n)} )
( 1           1        ...    0 ) (  λ  )    (  Q_total )
```

The analytical Jacobian of charges with respect to parameters follows from the
implicit function theorem:

```
∂q/∂κ   = − A⁻¹ · (M · y)
∂q/∂α_k =   A⁻¹ · 1[type = k]      (with 0 in λ row)
∂q/∂β_k = − A⁻¹ · diag(1[type = k]) · y
```

See [docs/theory.md](docs/theory.md) for the full derivation.

## Why six solvers

| Solver | Strategy | Best for |
|---|---|---|
| `AnalyticLM` | NumPy + analytic ∂q/∂x + L-BFGS-B → TRF/LM | small-to-medium problems, **fastest + most accurate** |
| `Newuoa` | Powell trust-region, derivative-free | matches MATLAB reference exactly |
| `Bobyqa` | Powell with bound constraints | NEWUOA + native bounds |
| `JaxAdam` | JAX autodiff + Adam | scales to thousands of params |
| `JaxLM` | JAX autodiff + L-BFGS-B → TRF | when analytic Jacobian impractical |
| `DEHybrid` | DE outer + NEWUOA polish | matches `DE_UOA_FINAL.m` byte-for-byte |

## Origin & references

This package is a complete rewrite of the original MATLAB code
(`de-uoa-matlab`, Godin 2017–2023). The mathematical model follows
NEEMP (Račkov & Svobodová 2016) and goes back to Mortier (1986).

**Key references**

- Mortier, W. J.; Ghosh, S. K.; Shankar, S. *Electronegativity-equalization
  method for the calculation of atomic charges in molecules.* J. Am. Chem.
  Soc. **1986**, 108, 4315–4320.
- Račkov, T. *NEEMP: software for parametrization of EEM.* J. Cheminform. **2016**, 8, 57.
- Powell, M. J. D. *The NEWUOA software for unconstrained optimization
  without derivatives.* in *Large-Scale Nonlinear Optimization*, Springer 2006.
- Powell, M. J. D. *The BOBYQA algorithm for bound constrained optimization
  without derivatives.* Cambridge NA Report NA2009/06, 2009.
- Ragonneau, T. M.; Zhang, Z. *PDFO: a cross-platform package for Powell's
  derivative-free optimization solvers.* Math. Prog. Comput. **2024**.

## Citation

If you use `powelleem` in a publication, please cite as in [CITATION.cff](CITATION.cff)
(see also Zenodo DOI badge above when released).

## License

MIT — see [LICENSE](LICENSE).
