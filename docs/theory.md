# Mathematical theory

## The EEM linear system

The Electronegativity Equalization Method (Mortier 1986) predicts atomic
partial charges by solving, per molecule of $n$ atoms,

$$
\underbrace{\begin{pmatrix}
\beta_{t(1)} & \kappa/r_{12} & \cdots & 1 \\
\kappa/r_{21} & \beta_{t(2)} & \cdots & 1 \\
\vdots & & \ddots & 1 \\
1 & 1 & \cdots & 0
\end{pmatrix}}_{A(x)}
\,
\underbrace{\begin{pmatrix}
q_1 \\ q_2 \\ \vdots \\ \lambda
\end{pmatrix}}_{y}
=
\underbrace{\begin{pmatrix}
-\alpha_{t(1)} \\ -\alpha_{t(2)} \\ \vdots \\ Q_\text{total}
\end{pmatrix}}_{b(x)}
$$

where $t : \{1..n\} \to \{1..T\}$ maps atoms to one of $T$ types and the
parameter vector is

$$
x = (\kappa,\, \alpha_1, \dots, \alpha_T,\, \beta_1, \dots, \beta_T) \in \mathbb{R}^{1 + 2T}.
$$

## Loss and Jacobian

Given reference charges $q^{\text{ref}}$ across a dataset of $M$ molecules
totalling $N$ atoms, the sum-of-squares loss is

$$
L(x) = \frac{1}{N} \sum_{m=1}^{M} \sum_{i=1}^{n_m} (q_i^{(m)}(x) - q_i^{\text{ref},(m)})^2.
$$

To form Gauss–Newton / Levenberg–Marquardt steps we need the Jacobian
$\partial r / \partial x$ where $r_i = q_i - q_i^{\text{ref}}$. From the
linear system $A(x)\,y(x) = b(x)$ and the **implicit function theorem**:

$$
(\partial A / \partial x_p)\,y + A\,(\partial y / \partial x_p) = \partial b / \partial x_p
\quad\Longrightarrow\quad
\partial y / \partial x_p = A^{-1}\!\left(\partial b / \partial x_p - (\partial A / \partial x_p)\,y\right).
$$

The three partials are sparse and closed-form:

| Parameter | $\partial A / \partial x_p$ | $\partial b / \partial x_p$ |
|---|---|---|
| $\kappa$    | $M_\text{ext}$ (off-diag $1/r$, zeros on last row/col) | $0$ |
| $\alpha_k$  | $0$ | $-\mathbf{1}_{t(i)=k}$ (zeros on $\lambda$ row) |
| $\beta_k$   | $\mathrm{diag}(\mathbf{1}_{t(i)=k})$ | $0$ |

Hence per molecule we need

- **one LU factorisation** of $A$ (already done by the forward pass);
- **$1 + 2T$ back-substitutions** to obtain every Jacobian column.

For $T = 10$ that is $21$ back-subs per molecule — negligible compared to
the $O(n^3)$ LU. The total residual Jacobian
$J \in \mathbb{R}^{N \times (1 + 2T)}$ is then fed directly to
[`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
with `method="trf"` to get quadratic local convergence.

## Comparison with the MATLAB original

`DE_UOA_FINAL.m` left the analytical gradient as a stub (`% derivate of f
to be defined...` in `eemeq.m`) and relied on Powell's derivative-free
NEWUOA solver, restarted from a $400$-point Latin Hypercube sampled
population. The MATLAB pipeline is faithfully reproduced as the
`DEHybrid` solver for direct comparison.

## Why six solvers

| Solver        | Gradient source        | Best for |
|---------------|------------------------|----------|
| `AnalyticLM`  | exact, implicit-fn     | small to medium problems — fastest, most accurate |
| `JaxAdam`     | autodiff               | large parametrisations, optional GPU |
| `JaxLM`       | autodiff               | when analytic Jacobian is impractical |
| `Newuoa`      | derivative-free        | matches MATLAB reference exactly |
| `Bobyqa`      | derivative-free        | NEWUOA + native bounds |
| `DEHybrid`    | DE + NEWUOA polish     | reproduces `DE_UOA_FINAL.m` |

## References

1. Mortier, W. J.; Ghosh, S. K.; Shankar, S. *J. Am. Chem. Soc.* **108**, 4315 (1986).
2. Račkov, T. *J. Cheminform.* **8**, 57 (2016) — NEEMP.
3. Powell, M. J. D. *NEWUOA* (2006); *BOBYQA* (2009).
4. Ragonneau, T. M.; Zhang, Z. *Math. Prog. Comput.* (2024) — PDFO.
5. Halgren, T. A. *J. Comput. Chem.* **17**, 490–641 (1996) — MMFF94 (BCI scheme is the MMFF charge analogue).
