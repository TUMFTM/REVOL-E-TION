import abc
import enum
import logging
from dataclasses import dataclass
from functools import singledispatchmethod

import pandas as pd
from typing_extensions import Self

from revoletion import blocks
from revoletion import scenario as scn


class OptimizationStatus(enum.Enum):
    OPTIMAL = enum.auto()  # Optimization was sucessful.

    INFEASIBLE = enum.auto()  # Demonstrated that the problem is unbounded.
    UNBOUNDED = enum.auto()  # Demonstrated that the problem is infeasible.
    INFEASIBLE_OR_UNBOUNDED = enum.auto()

    OTHER = enum.auto()

    ERROR = enum.auto()


class OptimizationResult(abc.ABC):
    @singledispatchmethod
    @abc.abstractmethod
    def get_power_flow(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> dict[str, pd.Series]: ...

    @abc.abstractmethod
    def get_stored_energy(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> float | pd.Series: ...

    @abc.abstractmethod
    def get_opex(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> float: ...

    @singledispatchmethod
    @abc.abstractmethod
    def get_capex(self, block: blocks.BaseBlock) -> dict[str, float]: ...

    @abc.abstractmethod
    def get_objective(self) -> float: ...


class Solver(enum.Enum):
    CBC = "cbc"
    GUROBI = "gurobi"
    HIGHS = "highs"


@dataclass
class OptimizationModelConfig:
    cost_eps: float = 1e-8

    debug: bool = False
    """Whether to enable debugging for the optimization model."""

    solver: Solver = Solver.CBC
    """The solver that should be used."""

    invest: bool = True
    """Whether captial investments should be enabled in the optimization."""


class OptimizationModel(abc.ABC):
    def __init__(self, logger: logging.Logger, config: OptimizationModelConfig | None = None) -> None:
        self._logger = logger
        self._config = config or OptimizationModelConfig()

    @classmethod
    @abc.abstractmethod
    def from_revoletion_scenario(
        cls,
        scenario: scn.Scenario,
        horizon: scn.TimeSettings,
        logger: logging.Logger,
        config: OptimizationModelConfig | None = None,
    ) -> Self: ...

    @abc.abstractmethod
    def set_input_power_unit(self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex) -> None: ...

    @abc.abstractmethod
    def set_output_power(self, block: blocks.ElectricBlock, power: float, dti: pd.DatetimeIndex) -> None: ...

    @abc.abstractmethod
    def optimize(self, dti: pd.DatetimeIndex | None = None) -> tuple[OptimizationStatus, OptimizationResult | None]: ...

    @abc.abstractmethod
    def optimize_time_step(
        self, time_step: pd.DatetimeIndex
    ) -> tuple[OptimizationStatus, OptimizationResult | None]: ...
