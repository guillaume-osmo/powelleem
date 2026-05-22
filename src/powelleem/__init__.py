"""powelleem — EEM parameter fitting with analytical Jacobian and Powell solvers.

The parameter-fitting protocol — atom typing, default-parameter fallback
hierarchy, hybrid global/local optimisation, validation suite — is
directly inspired by **NEEMP** (Raček et al., *J. Cheminform.* 2016, 8, 57,
DOI: 10.1186/s13321-016-0171-1). When citing `powelleem`, please also
cite NEEMP.

Top-level convenience re-exports.
"""

from powelleem.model import EEMModel
from powelleem.types import Dataset, FitResult, MoleculeData, ParamSet

__version__ = "0.2.0"

__all__ = [
    "EEMModel",
    "Dataset",
    "FitResult",
    "MoleculeData",
    "ParamSet",
    "__version__",
]
