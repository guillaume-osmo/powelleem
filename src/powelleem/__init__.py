"""powelleem — EEM parameter fitting with analytical Jacobian and Powell solvers.

Top-level convenience re-exports.
"""

from powelleem.model import EEMModel
from powelleem.types import Dataset, FitResult, MoleculeData, ParamSet

__version__ = "0.1.0a0"

__all__ = [
    "EEMModel",
    "Dataset",
    "FitResult",
    "MoleculeData",
    "ParamSet",
    "__version__",
]
