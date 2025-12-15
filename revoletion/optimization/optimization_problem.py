import abc
import enum
import logging
from dataclasses import dataclass
from functools import singledispatchmethod

import pandas as pd
from typing_extensions import Self

from revoletion import blocks, utils
from revoletion import scenario as scn


class OptimizationStatus(enum.Enum):
    """
    Exit status of an optimization.
    """

    OPTIMAL = enum.auto()
    """Optimizer found an optimal result."""

    INFEASIBLE = enum.auto()
    """Optimizer demonstrated that the problem is infeasible"""

    UNBOUNDED = enum.auto()
    """Optimizer demonstrated that the problem is unbounded."""

    INFEASIBLE_OR_UNBOUNDED = enum.auto()

    ERROR = enum.auto()
    """Optimizer encountered a generic error, e.g., requested solver is not available."""

    OTHER = enum.auto()
    """Optimizer returned an unkown status."""


FloatOrTimeSeries = float | pd.Series


class OptimizationResult(abc.ABC):
    @singledispatchmethod
    @abc.abstractmethod
    def get_power_flow(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> dict[str, FloatOrTimeSeries]: ...

    @abc.abstractmethod
    def get_stored_energy(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> FloatOrTimeSeries: ...

    @singledispatchmethod
    @abc.abstractmethod
    def get_opex(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> FloatOrTimeSeries: ...

    @singledispatchmethod
    @abc.abstractmethod
    def get_expansion(self, block: blocks.BaseBlock) -> dict[str, float]: ...

    @abc.abstractmethod
    def get_objective(self) -> float: ...


class Solver(enum.Enum):
    CBC = "cbc"
    GUROBI = "gurobi"
    HIGHS = "highs"

    def __str__(self) -> str:
        return self.value


@dataclass
class OptimizationProblemConfig:
    cost_eps: float = 1e-8

    debug: bool = False
    """Whether to enable debugging for the optimization."""

    solver: Solver = Solver.CBC
    """The solver that should be used."""

    invest: bool = True
    """Whether captial investments should be enabled in the optimization."""

    warmstart: bool = False

    enforce_soc_constraints: bool = True


class OptimizationProblem(abc.ABC):
    """
    Wrapper around an optimization problem which can be solved using one of the available solvers.
    """

    def __init__(self, logger: logging.Logger, config: OptimizationProblemConfig | None = None) -> None:
        self._logger = logger
        self._config = config or OptimizationProblemConfig()

    @classmethod
    @abc.abstractmethod
    def from_revoletion_scenario(
        cls,
        scenario: scn.Scenario,
        horizon: utils.TimeSettings,
        logger: logging.Logger,
        config: OptimizationProblemConfig | None = None,
    ) -> Self:
        """
        Construct a new `OptimizationProblem` from a REVOL-E-TION scenario.

        The created optimization model covers the given optimization horizon.
        """
        ...

    @abc.abstractmethod
    def set_input_power_unit(
        self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex, power_unit_buffer: float = 0.0
    ) -> None:
        """
        Directly fix the inflow power of an electric block.

        Useful during rolling horizon optimization, if reconstructing the optimization model would be too expensive.

        :param block: Electric block for which to modify the input power.
        :param power_unit: Percentage of the total possible input power of the block. Must be in [0.0, 1.0].
        :param dti: Time step or horizon for which to set this power value.

        :returns: Nothing.
        :raises ValueError: If the input power of the block cannot be set.
        """
        ...

    @abc.abstractmethod
    def set_output_power_unit(
        self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex, power_unit_buffer: float = 0.0
    ) -> None:
        """
        Directly fix the outflow power of an electric block.

        Useful during rolling horizon optimization, if reconstructing the whole optimization model would be too expensive.

        :param block: Electric block for which to modify the output power.
        :param power_unit: Percentage of the total possible output power of the block. Must be in [0.0, 1.0].
        :param dti: Time step or horizon for which to set this power value.

        :returns: Nothing.
        :raises ValueError: If the output power of the block cannot be set.
        """
        ...

    @abc.abstractmethod
    def solve(self) -> tuple[OptimizationStatus, OptimizationResult | None]:
        """
        Solve the optimization problem over a time horizon.

        The optimization will only cover the horizon for which this optimization problem was constructed.
        For sub-horizon rolling horizon optimization use `optimize_time_step`.

        :returns: The exit status of the solver and the optimization result, if the optimization was successful.
        """
        ...

    @abc.abstractmethod
    def solve_time_step(self, time_step: pd.DatetimeIndex) -> tuple[OptimizationStatus, OptimizationResult | None]:
        """
        Solve the optimization problem at one specific time step.

        Differs from `optimize` as it enables sub-horizon rolling horizon optimizations as it ensures
        that the storages are correctly initialized for every subsequent time step.

        :param time_step: The time step at which to optimize. If it's not the first time step of the horizon, all previous time steps must have been already optimized.
        :returns: The exit status of the solver and the optimization result, if the optimization was successful.
        """
        ...
