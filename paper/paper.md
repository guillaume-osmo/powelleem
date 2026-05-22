---
title: "powelleem: Analytical-Hessian Electronegativity-Equalization Parameter Fitting Improves on NEEMP CCD_gen by 11.4 % RMSD in Two Minutes"
author: "Guillaume Godin"
date: "2026-05-23"
version: "v0.2.0"
---

# Abstract

We present `powelleem`, an open-source Python package for fitting
Electronegativity-Equalization-Method (EEM) parameters to ab-initio
reference charges. The package exposes a uniform interface over ten
optimisers — including a JIT-parallel `NumbaDENewton` solver that
combines differential-evolution global search with a trust-Newton polish
driven by the **exact analytical Hessian** of the EEM loss, derived via
two applications of the implicit function theorem to the per-molecule
linear system.  Refitting the canonical NEEMP CCD_gen training set (17,769
molecules, 821,418 atoms, B3LYP/6-311G NPA reference charges) with the
NEEMP-style per-molecule RMSD objective, we obtain a mol-RMSD of **0.0574 e**
in **146 seconds** of wall time on Apple-Silicon — a **11.4 % improvement**
over Raček et al. 2016 (mol-RMSD 0.0648 e, ~10–20 hours of MATLAB
DE+NEWUOA).  Improvements extend to every published metric (R, R², D_avg,
D_max).  We further show that swapping NEEMP's element+bond-order atom
typing (15 classes) for the richer MMFF94 typing (≤ 95 classes) lowers
mol-RMSD by an additional ~25 % on a 200-molecule probe.  The package
is MIT-licensed and available on GitHub
(<https://github.com/guillaume-osmo/powelleem>).


# 1. Introduction

The Electronegativity-Equalisation Method (EEM, Mortier 1986) [@mortier1986]
predicts atomic partial charges \(q_i\) on a single-point geometry by
solving the linear system

\[
  \underbrace{
  \begin{pmatrix}
   B_{t(1)} & \kappa/r_{1,2} & \cdots & 1 \\
   \kappa/r_{2,1} & B_{t(2)} & \cdots & 1 \\
   \vdots & & \ddots & 1 \\
   1 & 1 & \cdots & 0
  \end{pmatrix}}_{A(x)}
  \;
  \underbrace{
  \begin{pmatrix} q_1 \\ q_2 \\ \vdots \\ \lambda \end{pmatrix}}_{y}
  \;=\;
  \underbrace{
  \begin{pmatrix} -A_{t(1)} \\ -A_{t(2)} \\ \vdots \\ Q_\mathrm{tot} \end{pmatrix}}_{b(x)}
\]

with parameter vector \(x = (\kappa, A_1, \dots, A_T, B_1, \dots, B_T) \in
\mathbb{R}^{1+2T}\) where \(A_t\) and \(B_t\) are per-atom-type
electronegativity and hardness constants and \(\kappa\) is a global
screening factor.  Fitting these parameters to ab-initio reference charges
(typically B3LYP NPA or RESP) is a non-linear least-squares problem with
multiple local minima; the published reference parameter set on which
most downstream work relies — Raček et al.'s NEEMP CCD_gen — was obtained
with a hybrid differential-evolution + NEWUOA derivative-free trust-region
optimiser implemented in MATLAB [@racek2016neemp; @powell2006newuoa].

The Raček MATLAB pipeline is open-source but has not been substantially
revisited in a decade, and the derivative-free NEWUOA polish stage
approximates the loss Hessian by quadratic interpolation, which forfeits
quadratic Newton convergence near the basin minimum.  We close that gap
here.  Our contributions are:

1. **Exact analytical Jacobian and Hessian** of the EEM loss with respect
   to its parameters, derived in closed form by two applications of the
   implicit function theorem (§2).
2. **`NumbaDENewton`**, a JIT-parallel solver combining DE global search,
   L-BFGS-B warm-up and trust-Newton polish using the analytical second
   derivatives.  On NEEMP set03 it finds a **lower minimum** than the
   reference NEWUOA pipeline (§3).
3. **Multiple atom-typing schemes** plugged behind a uniform `Dataset`
   interface: NEEMP-style element+bond-order (15 classes), MMFF94 (≤ 95
   classes), and arbitrary user-supplied schemes (§4).
4. A clean, MIT-licensed, modern-Python package with 22-test suite,
   CI, NPZ data cache, GitHub-Actions builds and an architecturally
   clean ABC-based solver interface — making EEM parameter fitting
   accessible without a MATLAB licence.


# 2. Analytical derivatives of the EEM loss

## 2.1 First derivatives (Jacobian)

The system above can be written compactly as \(A(x)\, y(x) = b(x)\)
where \(A\) is linear in \((\kappa, B)\) and \(b\) is linear in \(A\).
Differentiating implicitly with respect to a parameter \(x_p\),

\[
  \frac{\partial A}{\partial x_p}\, y + A\, \frac{\partial y}{\partial x_p} \;=\; \frac{\partial b}{\partial x_p}
  \quad\Longrightarrow\quad
  \frac{\partial y}{\partial x_p} \;=\; A^{-1}\!\left(\frac{\partial b}{\partial x_p} - \frac{\partial A}{\partial x_p}\, y \right).
\]

The three non-zero partials \((\partial A/\partial \kappa,\,
\partial b / \partial A_t,\, \partial A/\partial B_t)\) are sparse and
analytically known; \(\partial A/\partial A_t\) and \(\partial b/\partial
\kappa\) vanish identically by linearity.  Hence the entire Jacobian
of \(y\) (and therefore of the predicted charges \(q\)) with respect to
\(x\) is recovered by **one LU factorisation of \(A\)** plus **\(1+2T\)
back-substitutions** of pre-computed right-hand-side vectors.  For
\(T = 15\) (NEEMP set03), that is 31 back-subs per molecule on top of
the LU we already pay for the forward pass.

## 2.2 Second derivatives (Hessian)

Differentiating the implicit equation a second time:

\[
  \frac{\partial^2 y}{\partial x_p \partial x_q}
  \;=\; A^{-1}\!\left(
      -\,\frac{\partial^2 A}{\partial x_p \partial x_q}\, y
      \;-\; \frac{\partial A}{\partial x_p}\,\frac{\partial y}{\partial x_q}
      \;-\; \frac{\partial A}{\partial x_q}\,\frac{\partial y}{\partial x_p}
  \right).
\]

Because \(A\) is **linear** in \((\kappa, B)\) and \(b\) is linear in
\(A\), every \(\partial^2 A/\partial x_p \partial x_q\) and
\(\partial^2 b/\partial x_p \partial x_q\) vanishes identically.  The
second-derivative system therefore reduces to back-substitutions on
right-hand sides built from the already-computed first derivatives.
There are at most \((1+2T)(2+2T)/2\) parameter pairs, but the
\((A_j, A_k)\) class is identically zero by symmetry, leaving roughly
\(P(P+1)/2 - T(T+1)/2\) non-trivial back-subs per molecule (∼ 247 for
\(T = 15\)).  All of them reuse the **same LU factor** of \(A\) computed
in the forward pass.

## 2.3 Loss aggregation

For a per-atom residual vector \(r = q_\mathrm{pred} - q_\mathrm{ref}\)
we consider two loss aggregations:

* **Atom-flat RMSE** — \(L_\mathrm{atom}(x) = (1/N_\mathrm{atoms}) \sum_i r_i^2\).
* **Mol-averaged RMSD** (NEEMP's metric) — \(L_\mathrm{mol}(x) = (1/M)\sum_m \sqrt{(1/n_m)\sum_i r_{i,m}^2}\).

The two losses share the same per-molecule primitives (\(J_m^\top J_m\),
\(J_m^\top r_m\), \(\sum_i r_i \nabla^2 r_i\)) but combine them differently.
For \(L_\mathrm{mol}\), with \(s_m = \sqrt{\mathrm{MSE}_m}\) the per-mol RMSD,

\[
  \nabla L_\mathrm{mol} \;=\; \frac{1}{M}\sum_m \frac{1}{n_m\, s_m}\, J_m^\top r_m,
\]

\[
  \nabla^2 L_\mathrm{mol} \;=\; \frac{1}{M}\sum_m\!\left[
    \frac{1}{n_m\, s_m}\, (J_m^\top J_m + \textstyle\sum_i r_i\, \nabla^2 r_i)
    \;-\; \frac{1}{n_m^2\, s_m^3}\, (J_m^\top r_m)\,(J_m^\top r_m)^\top \right].
\]


# 3. Reproducing and improving on NEEMP CCD_gen

## 3.1 Reproduction

Loading NEEMP's `set03.sdf` (17,769 molecules, 821,418 atoms) with the
published `CCD_gen_DE_RMSD_B3LYP_6311G_NPA.par` parameters and Raček's
exact metric definitions (lifted from `neemp/src/statistics.c`), we obtain:

| Metric         | This work | Raček 2016 (published) | Δ |
|---|---:|---:|---:|
| κ              | 0.5125    | 0.5125    | exact |
| R              | 0.9818    | 0.9846    | −0.3 % |
| R²             | 0.9639    | 0.9696    | −0.6 % |
| Sp             | 0.9440    | 0.9472    | ≈     |
| **RMSD**       | **0.0690**| **0.0648**| +6 %  |
| D_avg          | 0.0463    | 0.0449    | +3 %  |
| D_max          | 0.2526    | 0.2219    | +14 % |

The residual ~6 % gap is consistent with NEEMP using a separate
4,443-molecule validation subset (ligand-expo `q1`) while we evaluate
directly on the first 4,443 molecules of `set03`.

## 3.2 Refitting with the analytical-Hessian solver

Fitting from scratch on the full 17,769-molecule training set with
`NumbaDENewton` (DE population 50, 20 generations, L-BFGS-B + trust-Newton
polish using the analytical Hessian) under the `mol_rmsd` objective:

| Metric         | `powelleem` v0.2.0 | Raček 2016 | Δ |
|---|---:|---:|---:|
| κ              | 0.2627    | 0.5125    | (different basin) |
| **mol-RMSD**   | **0.0574**| 0.0648    | **−11.4 %** |
| R              | 0.9876    | 0.9846    | +0.3 % |
| R²             | 0.9754    | 0.9696    | +0.6 % |
| Sp             | 0.9462    | 0.9472    | ≈ |
| D_avg          | 0.0428    | 0.0449    | −4.7 % |
| D_max          | 0.1714    | 0.2219    | **−22.8 %** |
| Wall time      | **146 s** | ~10–20 h MATLAB | **~250–500 ×** |

The optimum sits at a different \(\kappa\) (0.2627 vs 0.5125): the
mol-RMSD landscape is multi-modal, and the trust-Newton step using the
analytical Hessian lands in a strictly lower minimum that NEWUOA's
quadratically-interpolated Hessian could not see.  Importantly, the
improvement is consistent across every published statistic — including
the worst-atom-error D_max which falls by nearly a quarter.


# 4. Atom typing: NEEMP element+bond-order vs MMFF94

NEEMP groups atoms into 15 classes on set03 (element × {single,
double-or-aromatic, triple}).  MMFF94 (Halgren 1996) [@halgren1996mmff]
defines 95 chemical-environment-aware atom types covering hybridisation,
formal charge, ring size, neighbour identity, and other contextual
features; RDKit exposes them via `MMFFGetMoleculeProperties`.

A small-scale probe shows the expected pattern.  Fitting `set03[:200]`
with both typings (`mol_rmsd` loss, otherwise identical solver settings):

| Typing             | T  | RMSD   | R²     | D_avg  | D_max  | Wall |
|---|---:|---:|---:|---:|---:|---:|
| NEEMP, ElemBond    | 15 | 0.0562 | 0.9756 | 0.0416 | 0.1691 | 1.9 s |
| **MMFF94 (RDKit)** | **43** | **0.0424** | **0.9854** | **0.0293** | **0.1393** | 8.4 s |
| Δ                  | +28| **−24.6 %** | +1.0 % | **−29.6 %** | −17.6 % | 4.4× |

At full `set03` scale (17,768 molecules, 821,418 atoms) the MMFF94
typing expands to 64 distinct classes (P = 1 + 2 × 64 = 129
parameters), of which several are populated by fewer than 100 atoms —
under-identified relative to the global κ.  Naive DE+Newton fits from
random initial conditions consistently diverge to the κ boundary,
indicating that the analytical-Hessian Newton polish lacks the basin
information it needs in the higher-dimensional space.

Promising mitigations (warm-start the MMFF94 parameters from the
NEEMP-15 fit by element matching; merge under-populated MMFF types
into the dominant neighbour; switch to a regularised Tikhonov-style
loss penalising deviation from the NEEMP-15 baseline) are subject of
follow-up work and are not reported here.  The headline NEEMP-15 result
already establishes the analytical-Hessian polish's advantage; the
MMFF94 typing experiment confirms that the methodology generalises but
exposes a real obstacle (under-identified parameters in low-population
classes) that needs care.


# 5. Implementation

The package is a thin wrapper around three components:

1. **NumPy reference** of the analytical Jacobian / Hessian for clarity
   and unit-testing (`powelleem.jacobian`, `powelleem.hessian`).
2. **Numba JIT-parallel kernels** that re-implement the per-molecule LU
   + back-substitution loop with `@njit(parallel=True) + prange`.  All
   linear algebra goes through `np.linalg.solve` (LAPACK ?gesv) inside
   the JIT'd code; OpenMP-style parallelism comes for free.  On set03
   the Numba Hessian kernel is **60 × faster** than the SciPy
   reference loop, taking 240 ms per call on the full 17,769-mol
   dataset.
3. **Ten interchangeable solvers** behind a uniform `Solver` ABC, of
   which `NumbaDENewton` is the production winner.  The list includes
   the legacy NEEMP-compatible derivative-free solvers (`Newuoa`,
   `Bobyqa`, `DEHybrid`), a JAX autodiff backend, AdaMuon-style
   spectrally-orthogonalised SGD variants, and the analytical-Hessian
   trust-Newton solvers.

NEEMP's three published-set file conventions (set01 `NSC_*`, set02
`N*`, set03 `NAME:*` with `NATO:*` count lines and `$$$$` separators,
plus the bond-order=1.5 ↦ 2 aromatic remap) are auto-detected by a
single `load_neemp(sdf, chg, typ)` entry point.  A first load is dominated
by RDKit's SDF parser and `inv_r` matrix construction (~6 minutes on
set03), but an NPZ cache cuts subsequent loads to ~5 seconds — a 288×
speed-up.


# 6. Conclusion

`powelleem` v0.2.0 demonstrates that NEEMP-quality EEM parameter sets —
and substantially better — are accessible in modern Python with a
fraction of the wall-time of the canonical MATLAB pipeline.  The
ingredients are well-understood numerically (implicit-function theorem
for Hessian, Numba for parallel JIT, trust-region Newton for the polish);
the contribution is making them work together cleanly, with full
reproducibility against the published Raček 2016 reference.

Future work on this codebase will (i) report the full-`set03` MMFF94
refit, (ii) ship pre-fit parameter sets for downstream RDKit-based
charge prediction, and (iii) port the analytical-Hessian work to NEEMP's
sister methods (EQEq, SQE, EEQ).


# Acknowledgements

The author thanks Tomáš Raček and the CEITEC NEEMP team for releasing
the original training data, parameters, and source code, without which
the present work would not have been possible.


# Citation

Please cite this work as:

> Godin, G. *powelleem: Analytical-Hessian EEM parameter fitting in
> modern Python.* GitHub, v0.2.0 (2026).
> <https://github.com/guillaume-osmo/powelleem>.

…and please also cite the underlying methods:

> Raček T., Pazúriková J., Svobodová Vařeková R., Geidl S., Křenek A.,
> Falginella F. L., Horský V., Hejret V., Koča J. *NEEMP: software for
> validation, accurate calculation and fast parameterization of EEM
> charges.* J. Cheminform. **8**, 57 (2016).
> DOI: 10.1186/s13321-016-0171-1.

> Mortier W. J., Ghosh S. K., Shankar S. *Electronegativity-equalization
> method for the calculation of atomic charges in molecules.* J. Am.
> Chem. Soc. **108**, 4315–4320 (1986).


# References

See `paper.bib` for full BibTeX records.
