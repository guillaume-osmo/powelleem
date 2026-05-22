# powelleem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)]()
[![Status](https://img.shields.io/badge/status-beta-yellow)]()
[![CI](https://github.com/guillaume-osmo/powelleem/actions/workflows/ci.yml/badge.svg)](https://github.com/guillaume-osmo/powelleem/actions)

**Modern Python port of the EEM parameter-fitting MATLAB pipeline (Godin 2017–2023),
with analytical Jacobian & Hessian via the implicit function theorem,
JIT-parallel Numba kernels, and a uniform API over ten solvers including
Powell's NEWUOA/BOBYQA.**

> **Headline result.** Fitting NEEMP set03 (17,769 molecules, 821,418 atoms,
> 15 atom types, B3LYP/6-311G NPA reference charges), `powelleem` produces a
> **mol-RMSD of 0.0574** vs Raček 2016's published **0.0648** — an **11.4 %
> improvement** on the reference NEEMP CCD_gen result, in **146 seconds**
> wall time (vs ~10-20 h for the original MATLAB DE+NEWUOA pipeline).

| Metric (set03, mol-RMSD loss) | `powelleem` | Raček 2016 | Δ |
|---|---:|---:|---:|
| κ                | 0.2627    | 0.5125    | (different basin) |
| **mol-RMSD**     | **0.0574** | 0.0648    | **−11.4 %** |
| R                | 0.9876    | 0.9846    | +0.3 % |
| R²               | 0.9754    | 0.9696    | +0.6 % |
| Sp (Spearman)    | 0.9462    | 0.9472    | ≈     |
| D_avg            | 0.0428    | 0.0449    | −4.7 % |
| D_max            | 0.1714    | 0.2219    | −22.8 % |
| wall (full set03)| 146 s     | ~10-20 h  | ~250–500× |

The improvement comes from replacing NEEMP's derivative-free NEWUOA polish
(quadratic interpolation of the Hessian) with the **exact analytical
Hessian** computed via two applications of the implicit function theorem;
near the basin minimum Newton converges quadratically, dropping into a
strictly better local minimum.

The Electronegativity Equalization Method (EEM, Mortier 1986) predicts atomic
partial charges from per-element parameters (electronegativity α, hardness β,
screening κ). Fitting those parameters to ab-initio reference charges is a
non-linear least-squares problem traditionally tackled with derivative-free
solvers (Powell's NEWUOA / BOBYQA), as exemplified by **NEEMP** (Raček
et al. 2016) — the open-source reference implementation that inspires the
overall fitting protocol used here. This package adds:

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

### Linux deployment (multiprocessing speedup)

On macOS the Accelerate framework is not fork-safe, so multiprocessing is
fragile (see [src/powelleem/parallel.py](src/powelleem/parallel.py) docstring).
For full-scale benches (set03 = 17,769 molecules) deploy to a Linux box:

```bash
# From your laptop:
scp deploy/run_on_linux.sh user@host:~/
scp -r ~/path/to/de-uoa-matlab/neemp/examples user@host:~/neemp-data
ssh user@host 'bash run_on_linux.sh'
```

The script installs miniconda + clones this repo + runs:

- `examples/07_parallel_bench.py` — parallel-speedup measurement
- `examples/06_scale_denewton_only.py` — full set03 DENewton on 17k mol



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

## Why ten solvers

| Solver | Strategy | Best for |
|---|---|---|
| `AnalyticLM` | NumPy + analytic ∂q/∂x + L-BFGS-B → TRF/LM | small-to-medium problems, fastest single-start |
| `AnalyticNewton` | + analytic full Hessian + trust-exact Newton | curvature-aware single-start |
| `Newuoa` | Powell trust-region, derivative-free | matches MATLAB DE_UOA reference exactly |
| `Bobyqa` | Powell with bound constraints | NEWUOA + native bounds |
| `JaxAdam` | JAX autodiff + Adam | scales to thousands of params |
| `JaxAdaMuon` | Adam + Newton-Schulz orthogonalisation | per Liu et al. 2025 |
| `JaxMuonN` | nested AdamN + NS-on-direction | Guillaume's AdaMuonn variant |
| `JaxLM` | JAX autodiff + L-BFGS-B → TRF | when analytic Jacobian impractical |
| `DEHybrid` | DE outer + NEWUOA polish | matches `DE_UOA_FINAL.m` byte-for-byte |
| `DENewton` | DE outer + AnalyticNewton polish | best RMSE × speed trade-off |
| `DEAdaMuonn` | DE outer + JaxMuonN polish | AdaMuonn variant of above |
| **`NumbaDENewton`** | DENewton with @njit(parallel=True) kernels | **production fits at scale** |

## Origin & references

This package is a complete rewrite of the original MATLAB code
(`de-uoa-matlab`, Godin 2017–2023). The mathematical model is the
Electronegativity Equalization Method introduced by Mortier (1986). The
parameter-fitting protocol — atom typing, default-parameter fallback
hierarchy, hybrid global/local optimisation, validation suite — is
directly inspired by **NEEMP** (Raček et al., CEITEC, 2016), which is
the reference open-source implementation of EEM parameterisation. If
you use `powelleem`, please also cite NEEMP.

**Key references**

- Mortier, W. J.; Ghosh, S. K.; Shankar, S. *Electronegativity-equalization
  method for the calculation of atomic charges in molecules.* **J. Am. Chem.
  Soc.** 1986, 108, 4315–4320.
- **Raček, T.; Pazúriková, J.; Svobodová Vařeková, R.; Geidl, S.;
  Křenek, A.; Falginella, F. L.; Horský, V.; Hejret, V.; Koča, J.**
  *NEEMP: software for validation, accurate calculation and fast
  parameterization of EEM charges.* **J. Cheminform.** 2016, 8, 57.
  DOI: [10.1186/s13321-016-0171-1](https://doi.org/10.1186/s13321-016-0171-1)
  • PMC: [PMC5067907](https://pmc.ncbi.nlm.nih.gov/articles/PMC5067907/)
- Powell, M. J. D. *The NEWUOA software for unconstrained optimization
  without derivatives.* In *Large-Scale Nonlinear Optimization*, Springer 2006.
- Powell, M. J. D. *The BOBYQA algorithm for bound constrained optimization
  without derivatives.* Cambridge NA Report NA2009/06, 2009.
- Ragonneau, T. M.; Zhang, Z. *PDFO: a cross-platform package for Powell's
  derivative-free optimization solvers.* **Math. Prog. Comput.** 2024.

## Citation

If you use `powelleem` in a publication, please cite as in [CITATION.cff](CITATION.cff)
(see also Zenodo DOI badge above when released).

## License

MIT — see [LICENSE](LICENSE).
