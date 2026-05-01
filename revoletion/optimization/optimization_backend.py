import enum
import logging

from revoletion import scenario as scn
from revoletion import time

from . import optimization_problem
from .oemof_backend import VALID_OEMOF_SOLVERS, OemofOptimizationProblem
from .pypsa_backend import VALID_PYPSA_SOLVERS, PypsaOptimizationProblem


class OptimizationBackend(enum.Enum):
    OEMOF = "oemof"
    PYPSA = "pypsa"

    def __str__(self) -> str:
        return self.value

    def is_compatible_solver(self, solver: optimization_problem.Solver) -> bool:
        if self == OptimizationBackend.OEMOF:
            return solver in VALID_OEMOF_SOLVERS
        if self == OptimizationBackend.PYPSA:
            return solver in VALID_PYPSA_SOLVERS
        else:
            raise ValueError()


def create_optimization_problem(
    backend: OptimizationBackend,
    scenario: scn.Scenario,
    horizon: time.TimeFrame,
    logger: logging.Logger,
    config: optimization_problem.OptimizationProblemConfig | None = None,
) -> optimization_problem.OptimizationProblem:
    match backend:
        case OptimizationBackend.OEMOF:
            return OemofOptimizationProblem.from_revoletion_scenario(
                scenario=scenario, horizon=horizon, logger=logger, config=config
            )
        case OptimizationBackend.PYPSA:
            return PypsaOptimizationProblem.from_revoletion_scenario(
                scenario=scenario, horizon=horizon, logger=logger, config=config
            )
        case _:
            raise ValueError(f"Invalid optimization backend: {backend}")
