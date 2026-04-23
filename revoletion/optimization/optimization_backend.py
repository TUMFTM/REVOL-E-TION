import enum
import logging

from revoletion import scenario as scn
from revoletion import time

from . import optimization_problem
from .oemof_backend import VALID_OEMOF_SOLVERS, OemofOptimizationProblem


class OptimizationBackend(enum.Enum):
    OEMOF = "oemof"

    def __str__(self) -> str:
        return self.value

    def is_compatible_solver(self, solver: optimization_problem.Solver) -> bool:
        if self == OptimizationBackend.OEMOF:
            return solver in VALID_OEMOF_SOLVERS
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
        case _:
            raise ValueError(f"Invalid optimization backend: {backend}")
