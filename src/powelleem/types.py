"""Dataclasses for parameter sets, datasets, and fit results.

The math model has parameter vector

    x = (κ, α_1, …, α_T, β_1, …, β_T)   ∈   ℝ^{1 + 2T}

where T is the number of distinct atom types. We keep both the packed
``x`` array form (used by SciPy/scipy.optimize) and a structured
``ParamSet`` form (human-readable, JSON-serialisable, export to RDKit).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ParamSet:
    """Frozen, structured EEM parameter set.

    Parameters
    ----------
    kappa
        Global screening parameter κ ∈ (0, ∞).
    alpha
        Per-atom-type electronegativity α_k, length T.
    beta
        Per-atom-type hardness β_k, length T.
    atom_types
        Length-T list of atom-type identifiers (e.g. ``["H", "C-1", "C-2", "N-1"]``).
        Order matches indices used by ``alpha``/``beta``.
    """

    kappa: float
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    atom_types: tuple[str, ...]

    @property
    def n_types(self) -> int:
        return len(self.atom_types)

    @property
    def n_params(self) -> int:
        return 1 + 2 * self.n_types

    def to_vector(self) -> NDArray[np.float64]:
        """Pack into the canonical ``x = (κ, α_1..α_T, β_1..β_T)`` vector."""
        return np.concatenate([[self.kappa], self.alpha, self.beta])

    @classmethod
    def from_vector(
        cls, x: NDArray[np.float64], atom_types: tuple[str, ...]
    ) -> ParamSet:
        """Unpack a canonical ``x`` vector into a :class:`ParamSet`."""
        x = np.asarray(x, dtype=np.float64)
        n_types = len(atom_types)
        expected = 1 + 2 * n_types
        if x.size != expected:
            raise ValueError(
                f"x has size {x.size} but {expected} = 1 + 2 × {n_types} expected"
            )
        return cls(
            kappa=float(x[0]),
            alpha=x[1 : 1 + n_types].copy(),
            beta=x[1 + n_types :].copy(),
            atom_types=tuple(atom_types),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kappa": self.kappa,
            "alpha": {t: float(v) for t, v in zip(self.atom_types, self.alpha, strict=True)},
            "beta": {t: float(v) for t, v in zip(self.atom_types, self.beta, strict=True)},
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParamSet:
        atom_types = tuple(d["alpha"].keys())
        alpha = np.asarray([d["alpha"][t] for t in atom_types], dtype=np.float64)
        beta = np.asarray([d["beta"][t] for t in atom_types], dtype=np.float64)
        return cls(kappa=float(d["kappa"]), alpha=alpha, beta=beta, atom_types=atom_types)


@dataclass(slots=True)
class MoleculeData:
    """Per-molecule data needed by the EEM solver.

    All fields are NumPy arrays with consistent dtypes so they can be
    passed into the inner solvers without copy. ``inv_r`` is stored
    explicitly (precomputed once) to avoid recomputing per iteration —
    the trick from ``loaddata.m``.
    """

    smiles: str
    atom_types: NDArray[np.int64]  # shape (n,) — 1-based type indices into ParamSet.atom_types
    inv_r: NDArray[np.float64]  # shape (n, n) — off-diag is 1/r_ij, diag is 0
    target_charges: NDArray[np.float64]  # shape (n,) — reference q_ref
    formal_charge: float = 0.0
    weights: NDArray[np.float64] | None = None  # optional per-atom weights
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.atom_types.shape[0])

    def __post_init__(self) -> None:
        n = self.n_atoms
        if self.inv_r.shape != (n, n):
            raise ValueError(
                f"inv_r has shape {self.inv_r.shape}, expected ({n}, {n})"
            )
        if self.target_charges.shape != (n,):
            raise ValueError(
                f"target_charges has shape {self.target_charges.shape}, expected ({n},)"
            )
        if self.weights is not None and self.weights.shape != (n,):
            raise ValueError(
                f"weights has shape {self.weights.shape}, expected ({n},)"
            )


@dataclass(slots=True)
class Dataset:
    """Collection of molecules + atom-type vocabulary.

    The dataset enforces a *fixed* atom-type vocabulary across molecules so
    every :class:`MoleculeData.atom_types` indexes into the same
    :attr:`atom_types` tuple.
    """

    molecules: list[MoleculeData]
    atom_types: tuple[str, ...]
    name: str = "unnamed"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_mols(self) -> int:
        return len(self.molecules)

    @property
    def n_types(self) -> int:
        return len(self.atom_types)

    @property
    def total_atoms(self) -> int:
        return sum(m.n_atoms for m in self.molecules)

    def subset(self, indices: list[int] | NDArray[np.int_]) -> Dataset:
        """Return a new Dataset containing only the selected molecules."""
        return Dataset(
            molecules=[self.molecules[i] for i in indices],
            atom_types=self.atom_types,
            name=f"{self.name}[subset:{len(indices)}]",
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class FitResult:
    """Outcome of fitting a model to a dataset.

    Holds the final parameter set, RMSE on the training data, optimizer
    metadata, wall-clock breakdown, and per-stage loss trajectory. The
    last two are useful for the benchmark harness.
    """

    params: ParamSet
    rmse: float
    loss_initial: float
    loss_final: float
    solver_name: str
    solver_metadata: dict[str, Any]
    wall_time_s: float
    n_function_evals: int
    n_jacobian_evals: int = 0
    converged: bool = True
    message: str = ""
    loss_trajectory: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict explodes ParamSet too; convert arrays to lists for JSON
        d["params"] = self.params.to_dict()
        return d

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def export_rdkit_header(self, path: str | Path) -> None:
        """Emit a C++ header snippet patching RDKit's ``EEM.cpp``.

        The generated file declares ``const double kappa``, ``A1[]``,
        ``B1[]`` arrays in the exact layout used by
        ``rdkit/Code/GraphMol/Descriptors/EEM.cpp``. Only single-bond
        atom-type level is filled; multi-bond levels (A2, A3, B2, B3) are
        produced separately if the model is configured with
        ``use_bond_order=True``.
        """
        from powelleem.io_rdkit import write_rdkit_eem_header

        write_rdkit_eem_header(self.params, Path(path))
