"""EEM forward pass — predict atomic charges given parameters.

The math model (matching MATLAB ``structureB.cpp`` and RDKit ``EEM.cpp``):

For a molecule of ``n`` atoms with parameter vector
``x = (κ, α_1, …, α_T, β_1, …, β_T)`` and atom-type assignment
``t : {1..n} → {1..T}``:

::

    ( β_{t(1)}    κ/r_{12}   …   1 ) ( q_1 )    ( -α_{t(1)} )
    ( κ/r_{21}   β_{t(2)}    …   1 ) (  …  )  = ( -α_{t(2)} )
    (    ⋮                       1 ) ( q_n )    ( -α_{t(n)} )
    ( 1            1         …   0 ) (  λ  )    (  Q_total )

This module provides two backends:

- ``predict_charges_numpy`` — single-molecule NumPy reference, used by the
  analytic-Jacobian solver. Solves with :func:`scipy.linalg.lu_solve`.
- ``predict_charges_jax`` — JIT-compiled JAX kernel used by the autodiff
  solvers. Solves with :func:`jax.numpy.linalg.solve`.

Both share the :class:`EEMModel` adapter which holds the atom-type
vocabulary and exposes a single ``predict(params, dataset)`` entry point
that the solvers consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray

if TYPE_CHECKING:
    from powelleem.types import Dataset, MoleculeData, ParamSet


# ---------------------------------------------------------------------------
# NumPy backend (used by AnalyticLM solver — also returns the LU factor so
# the Jacobian module can reuse it without redoing the factorization).
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Solved:
    """Internal: result of one forward solve, kept for the Jacobian module."""

    q: NDArray[np.float64]  # (n,) predicted charges
    y: NDArray[np.float64]  # (n+1,) full solution incl. Lagrange multiplier
    A: NDArray[np.float64]  # (n+1, n+1) system matrix (kept for ∂A/∂param VJPs)
    lu_piv: tuple[NDArray[np.float64], NDArray[np.int_]]  # cached LU factor


def _build_A_b_numpy(
    x: NDArray[np.float64],
    inv_r: NDArray[np.float64],
    type_idx: NDArray[np.int64],
    q_total: float,
    n_types: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the (n+1)×(n+1) EEM system matrix A and RHS b."""
    n = inv_r.shape[0]
    kappa = float(x[0])
    alpha = x[1 : 1 + n_types]
    beta = x[1 + n_types :]

    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    A[:n, :n] = kappa * inv_r
    A[np.arange(n), np.arange(n)] = beta[type_idx - 1]
    A[n, :n] = 1.0
    A[:n, n] = 1.0
    # A[n, n] = 0 already

    b = np.empty(n + 1, dtype=np.float64)
    b[:n] = -alpha[type_idx - 1]
    b[n] = q_total
    return A, b


def predict_charges_numpy(
    x: NDArray[np.float64],
    mol: MoleculeData,
    n_types: int,
) -> _Solved:
    """Solve the EEM system for one molecule and return charges + LU factor."""
    A, b = _build_A_b_numpy(x, mol.inv_r, mol.atom_types, mol.formal_charge, n_types)
    lu, piv = sla.lu_factor(A)
    y = sla.lu_solve((lu, piv), b)
    n = mol.n_atoms
    return _Solved(q=y[:n], y=y, A=A, lu_piv=(lu, piv))


# ---------------------------------------------------------------------------
# JAX backend (lazy import, used by JaxAdam / JaxLM solvers).
# ---------------------------------------------------------------------------

def predict_charges_jax_factory(n_types: int):  # type: ignore[no-untyped-def]
    """Return a JIT-compiled predict_one(x, inv_r, type_idx, q_total, n).

    Cannot be a top-level decorated function because ``n_types`` parameterises
    the slicing into α/β. Caller is responsible for caching the returned closure.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "JAX is required for the autodiff backend. Install with "
            "`pip install powelleem[jax]`."
        ) from exc

    from functools import partial

    @partial(jax.jit, static_argnames=("n",))
    def predict_one(x, inv_r, type_idx, q_total, n):  # type: ignore[no-untyped-def]
        kappa = x[0]
        alpha = x[1 : 1 + n_types]
        beta = x[1 + n_types :]
        A_core = kappa * inv_r
        A_core = A_core.at[jnp.arange(n), jnp.arange(n)].set(beta[type_idx - 1])
        last_row = jnp.ones((1, n))
        last_col = jnp.ones((n, 1))
        A = jnp.block([[A_core, last_col], [last_row, jnp.zeros((1, 1))]])
        b = jnp.concatenate([-alpha[type_idx - 1], jnp.array([q_total])])
        y = jnp.linalg.solve(A, b)
        return y[:n]

    return predict_one


# ---------------------------------------------------------------------------
# High-level adapter used by the solver layer.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EEMModel:
    """Atom-type vocabulary + forward-pass dispatcher.

    The model itself is stateless w.r.t. parameter values; it only fixes the
    atom-type vocabulary so that input :class:`MoleculeData` objects index
    consistently.

    Parameters
    ----------
    atom_types
        Tuple of atom-type strings. Order defines the index used by α/β.
    use_bond_order
        Reserved for NEEMP-style typing (H/C-1/C-2/C-3/…). When ``True`` the
        loader is expected to have produced atom-type strings with bond-order
        suffix. The model itself just trusts the input vocabulary.
    """

    atom_types: tuple[str, ...]
    use_bond_order: bool = False

    @property
    def n_types(self) -> int:
        return len(self.atom_types)

    @property
    def n_params(self) -> int:
        return 1 + 2 * self.n_types

    def predict_one(
        self,
        params: ParamSet,
        mol: MoleculeData,
    ) -> NDArray[np.float64]:
        """Compute predicted charges for one molecule (NumPy reference)."""
        if params.atom_types != self.atom_types:
            raise ValueError(
                "ParamSet.atom_types disagrees with EEMModel.atom_types: "
                f"{params.atom_types} vs {self.atom_types}"
            )
        solved = predict_charges_numpy(params.to_vector(), mol, self.n_types)
        return solved.q

    def predict(self, params: ParamSet, dataset: Dataset) -> list[NDArray[np.float64]]:
        """Compute predicted charges for every molecule in the dataset."""
        return [self.predict_one(params, m) for m in dataset.molecules]

    def residuals_flat(
        self, params: ParamSet, dataset: Dataset
    ) -> NDArray[np.float64]:
        """Concatenated ``q_pred - q_ref`` over all atoms in the dataset."""
        return np.concatenate(
            [self.predict_one(params, m) - m.target_charges for m in dataset.molecules]
        )

    def rmse(self, params: ParamSet, dataset: Dataset) -> float:
        """Atom-averaged RMSE of predicted vs target charges."""
        r = self.residuals_flat(params, dataset)
        return float(np.sqrt((r * r).mean()))
