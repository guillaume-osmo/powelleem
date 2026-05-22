"""Six interchangeable solvers under a uniform :class:`Solver` ABC.

| Solver          | Strategy                                            |
| --------------- | --------------------------------------------------- |
| ``AnalyticLM``  | NumPy + analytic Jacobian + L-BFGS-B → TRF/LM       |
| ``JaxAdam``     | JAX autodiff + Adam                                 |
| ``JaxLM``       | JAX autodiff + L-BFGS-B → TRF/LM                    |
| ``Newuoa``      | Powell's NEWUOA via PRIMA (derivative-free)         |
| ``Bobyqa``      | Powell's BOBYQA via PRIMA (bound-constrained DFO)   |
| ``DEHybrid``    | DE outer + NEWUOA polish (matches MATLAB DE_UOA)    |
"""

from powelleem.solvers.analytic_lm import AnalyticLM
from powelleem.solvers.analytic_newton import AnalyticNewton
from powelleem.solvers.base import SolverConfig, Solver
from powelleem.solvers.bobyqa import Bobyqa
from powelleem.solvers.de_hybrid import DEHybrid
from powelleem.solvers.de_newton import DENewton
from powelleem.solvers.jax_adam import JaxAdam
from powelleem.solvers.jax_adamuon import JaxAdaMuon
from powelleem.solvers.jax_lm import JaxLM
from powelleem.solvers.jax_muonn import JaxMuonN
from powelleem.solvers.newuoa import Newuoa

__all__ = [
    "Solver",
    "SolverConfig",
    "AnalyticLM",
    "AnalyticNewton",
    "JaxAdam",
    "JaxAdaMuon",
    "JaxLM",
    "JaxMuonN",
    "Newuoa",
    "Bobyqa",
    "DEHybrid",
    "DENewton",
]
