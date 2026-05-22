"""Solver base class + shared utilities (bounds, init, vectorisation)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from powelleem.model import EEMModel
    from powelleem.types import Dataset, FitResult


# MATLAB DE_UOA_FINAL.m line 48 bounds:
#   kappa ∈ [0.01, 3.0]
#   alpha ∈ [1.8, 3.2]
#   beta  ∈ [0.0, 1.0]
DEFAULT_KAPPA_BOUNDS = (0.01, 3.0)
DEFAULT_ALPHA_BOUNDS = (1.8, 3.2)
DEFAULT_BETA_BOUNDS = (0.0, 1.0)


@dataclass(slots=True)
class SolverConfig:
    """Shared configuration knobs across solvers (bounds, max iterations, seed)."""

    kappa_bounds: tuple[float, float] = DEFAULT_KAPPA_BOUNDS
    alpha_bounds: tuple[float, float] = DEFAULT_ALPHA_BOUNDS
    beta_bounds: tuple[float, float] = DEFAULT_BETA_BOUNDS
    max_iter: int = 200
    seed: int = 42
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    def bounds_array(self, n_types: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(lo, hi)`` arrays of length ``1 + 2T``."""
        lo = np.concatenate(
            [
                [self.kappa_bounds[0]],
                np.full(n_types, self.alpha_bounds[0]),
                np.full(n_types, self.beta_bounds[0]),
            ]
        )
        hi = np.concatenate(
            [
                [self.kappa_bounds[1]],
                np.full(n_types, self.alpha_bounds[1]),
                np.full(n_types, self.beta_bounds[1]),
            ]
        )
        return lo, hi


def random_initial_x(
    n_types: int,
    config: SolverConfig,
) -> NDArray[np.float64]:
    """Draw a random ``x_0`` uniformly within the configured bounds."""
    rng = np.random.default_rng(config.seed)
    lo, hi = config.bounds_array(n_types)
    return rng.uniform(lo, hi)


def latin_hypercube_initial_xs(
    n_types: int,
    config: SolverConfig,
    n_points: int,
) -> NDArray[np.float64]:
    """Latin Hypercube Sampling for population-based solvers (DE).

    Matches MATLAB's ``X = lhsdesign(size(bounds,1),NP)`` in DE_UOA_FINAL.m.
    Returns an ``(n_points, 1 + 2T)`` array.
    """
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy.stats.qmc is required for LHS sampling.") from exc

    rng = np.random.default_rng(config.seed)
    sampler = qmc.LatinHypercube(d=1 + 2 * n_types, seed=rng)
    sample = sampler.random(n=n_points)  # in [0, 1]^d
    lo, hi = config.bounds_array(n_types)
    return lo + sample * (hi - lo)


class Solver(ABC):
    """Abstract base class for EEM parameter solvers."""

    name: str
    config: SolverConfig

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()

    @abstractmethod
    def fit(
        self,
        model: EEMModel,
        dataset: Dataset,
        *,
        x0: NDArray[np.float64] | None = None,
    ) -> FitResult:
        """Optimise EEM parameters on ``dataset`` using ``model``.

        Parameters
        ----------
        model
            Atom-type vocabulary + forward dispatcher.
        dataset
            Training molecules with reference charges.
        x0
            Optional packed initial parameter vector. If ``None``, drawn
            uniformly inside the configured bounds with the configured seed.

        Returns
        -------
        FitResult with the optimised :class:`ParamSet`, RMSE, timing,
        loss trajectory, etc.
        """
