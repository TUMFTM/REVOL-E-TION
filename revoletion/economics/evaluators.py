from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .abstractclasses import EcoElement, CapexElement, YearlyElement, BlockElement
from .utils import OccursAt, annuity, discount


class CostEvaluator(EcoElement):
    """
    Base class for all Evaluators that evaluate costs, i.e. Capex, Mntex, Opex and Crev.
    """

    _OCCURS_AT = OccursAt.BEGIN

    def __init__(self, name: str, project_duration: int, discount_rate: float):
        self.name = name
        self._project_duration = project_duration
        self._cashflow = np.zeros(self._project_duration + 1, dtype=float)
        self._discount_rate = discount_rate

        # ToDo: remove
        # random cashflow for testing
        self._cashflow = np.random.rand(self._project_duration + 1) * 1000

    @property
    def cashflow(self) -> npt.NDArray:
        return self._cashflow

    @property
    def cashflow_dis(self) -> npt.NDArray:
        discount_factors = discount(
            future_value=1,
            periods=np.arange(self._project_duration + 1),
            discount_rate=self._discount_rate,
            occurs_at=self._OCCURS_AT,
        )
        return self._cashflow * discount_factors

    @property
    def dis(self) -> float:
        return sum(self.cashflow_dis)

    @property
    def ann(self) -> float:
        return annuity(
            present_value=self.dis,
            observation_horizon=self._project_duration,
            discount_rate=self._discount_rate,
            occurs_at=self._OCCURS_AT,
        )


class YearlyEvaluator(CostEvaluator, YearlyElement, ABC):
    def __init__(self, name: str, project_duration: int, discount_rate: float):
        super().__init__(name, project_duration, discount_rate)

        self._sim = 0
        self._sim2yr_ratio = 1

    @property
    @abstractmethod
    def sim(self) -> float: ...

    @property
    def yrl(self) -> float:
        return self._sim * self._sim2yr_ratio


class CapexEvaluator(CostEvaluator, CapexElement):
    def __init__(self, name: str, project_duration: int, discount_rate: float, size: float):
        super().__init__(name, project_duration, discount_rate)
        self._size = size

        # ToDo: replace
        self._preexisting = 0
        self._expansion = 0

    @property
    def preexisting(self) -> float:
        return self._preexisting

    @property
    def expansion(self) -> float:
        return self._expansion

    @property
    def init(self) -> float:
        return self.preexisting + self.expansion


class MntexEvaluator(YearlyEvaluator):
    def __init__(self, name: str, project_duration: int, discount_rate: float, size: float):
        super().__init__(name, project_duration, discount_rate)
        self._size = size

    @property
    def sim(self) -> float:
        return 0


class OpexEvaluator(YearlyEvaluator):
    @property
    def sim(self) -> float:
        return 0


class CrevEvaluator(YearlyEvaluator):
    @property
    def sim(self) -> float:
        return 0


@dataclass
class EvaluatorBlock(BlockElement):
    """
    EvaluatorBlock is a container for all Evaluators of a single component.
    """

    name: str

    capex: CapexEvaluator
    mntex: MntexEvaluator
    opex: OpexEvaluator
    crev: CrevEvaluator
