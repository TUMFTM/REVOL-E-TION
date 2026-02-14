from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import numpy.typing as npt
import pandas as pd

from .abstractclasses import CapexElement, EcoElement, YearlyElement, BlockElement


@dataclass
class BaseAggregator(EcoElement, ABC):
    """
    Base class for all Aggregators.
    The aggregation logic has to be defined in the _aggregate method, which is called by all properties.
    """

    @abstractmethod
    def _aggregate(self, property_name: str) -> float | npt.NDArray: ...

    @property
    def cashflow(self) -> npt.NDArray:
        return self._aggregate("cashflow")

    @property
    def cashflow_dis(self) -> npt.NDArray:
        return self._aggregate("cashflow_dis")

    @property
    def prj(self) -> float:
        return self._aggregate("prj")

    @property
    def dis(self) -> float:
        return self._aggregate("dis")

    @property
    def ann(self) -> float:
        return self._aggregate("ann")


@dataclass
class CrossLevelAggregator(BaseAggregator):
    """
    Base class for all Aggregators, to aggregate values aggregators or Evaluators from subblocks.

    Attributes
    ----------
    elements : dict[str, EcoElement]
        All EcoObjects that are aggregated by this aggregator, e.g. all CapexElements of the subblocks for a CapexAggregator.
    """

    elements: dict[str, EcoElement] = field(default_factory=dict)

    def _aggregate(self, property_name: str) -> float | npt.NDArray:
        return sum(getattr(poi, property_name) for poi in self.elements.values())


@dataclass
class InLevelAggregator(BaseAggregator, ABC):
    """
    Base class for all Aggregators, which aggregate values from other aggregators on the same level.
    This is used for Totex (Capex + Mntex + Opex) and Value (Totex - Crev) aggregation.
    """

    pass


@dataclass
class YearlyAggregator(CrossLevelAggregator, YearlyElement):
    """
    YearlyAggregator aggregates the values of all given YearlyElements.
    This is used for Mntex, Opex and Crev aggregation.
    """

    elements: dict[str, YearlyElement] = field(default_factory=dict)

    @property
    def yrl(self) -> float:
        return self._aggregate("yrl")


@dataclass
class CapexAggregator(CrossLevelAggregator, CapexElement):
    """
    CapexAggregator aggregates the values of all given elements.
    Given elements have to be of type CapexElement.

    Attributes
    ----------
    elements : dict[str, CapexElement]
        All CapexElements that are aggregated by this aggregator.

    preexisting : float
        Sum of preexisting capex of all aggregated CapexElements.

    expansion : float
        Sum of expansion capex of all aggregated CapexElements.

    init : float
        Sum of initial capex of all aggregated CapexElements (preexisting + expansion).
    """

    elements: dict[str, CapexElement] = field(default_factory=dict)

    @property
    def preexisting(self) -> float:
        return self._aggregate("preexisting")

    @property
    def expansion(self) -> float:
        return self._aggregate("expansion")

    @property
    def init(self) -> float:
        return self._aggregate("init")


@dataclass
class MntexAggregator(YearlyAggregator):
    """
    MntexAggregator aggregates the values of all given MntexElements.
    """

    pass


@dataclass
class PowerBasedAggregator(YearlyAggregator):
    """
    PowerBasedAggregator aggregates the values of all given PowerBasedElements.
    """

    @property
    def eval(self) -> float:
        return self._aggregate("eval")


@dataclass
class OpexAggregator(PowerBasedAggregator):
    """
    OpexAggregator aggregates the values of all given OpexElements.
    """

    pass


@dataclass
class CrevAggregator(PowerBasedAggregator):
    """
    CrevAggregator aggregates the values of all given CrevElements.
    """

    pass


@dataclass
class TotexAggregator(InLevelAggregator):
    """
    TotexAggregator aggregates the values of Capex, Mntex and Opex aggregators on the same level to calculate the total costs.
    """

    capex: CapexAggregator
    mntex: YearlyAggregator
    opex: YearlyAggregator

    def _aggregate(self, property_name: str) -> float | npt.NDArray:
        return (
            getattr(self.capex, property_name) + getattr(self.mntex, property_name) + getattr(self.opex, property_name)
        )


@dataclass
class ValueAggregator(InLevelAggregator):
    """
    ValueAggregator subtracts revenues (Crev) from costs (Totex) on the same level.
    """

    totex: TotexAggregator
    crev: YearlyAggregator

    def _aggregate(self, property_name: str) -> float | npt.NDArray:
        return getattr(self.totex, property_name) - getattr(self.crev, property_name)


@dataclass
class Aggregator(BlockElement):
    """
    EcoBlock holds all economic components of a block.
    """

    name: str

    capex: CapexAggregator
    mntex: MntexAggregator
    opex: OpexAggregator
    crev: CrevAggregator
    totex: TotexAggregator
    value: ValueAggregator

    @classmethod
    def create(cls, name: str) -> Self:
        capex = CapexAggregator(name=name)
        mntex = MntexAggregator(name=name)
        opex = OpexAggregator(name=name)
        crev = CrevAggregator(name=name)
        totex = TotexAggregator(name=name, capex=capex, mntex=mntex, opex=opex)
        value = ValueAggregator(name=name, totex=totex, crev=crev)

        return cls(name=name, capex=capex, mntex=mntex, opex=opex, crev=crev, totex=totex, value=value)

    def add_block(self, block: BlockElement) -> None:
        """
        Add a new subblock to the current block.
        """
        self.capex.elements[block.name] = block.capex
        self.mntex.elements[block.name] = block.mntex
        self.opex.elements[block.name] = block.opex
        self.crev.elements[block.name] = block.crev

    @property
    def result_summary(self) -> pd.Series:
        # ToDo: add capex preexisting, expansion and init to result summary
        # ToDo: make loop based?
        return pd.Series(
            {
                "capex_prj": self.capex.prj,
                "capex_dis": self.capex.dis,
                "capex_ann": self.capex.ann,
                "mntex_yrl": self.mntex.yrl,
                "mntex_prj": self.mntex.prj,
                "mntex_dis": self.mntex.dis,
                "mntex_ann": self.mntex.ann,
                "opex_eval": self.opex.eval,
                "opex_yrl": self.opex.yrl,
                "opex_prj": self.opex.prj,
                "opex_dis": self.opex.dis,
                "opex_ann": self.opex.ann,
                "crev_eval": self.crev.eval,
                "crev_yrl": self.crev.yrl,
                "crev_prj": self.crev.prj,
                "crev_dis": self.crev.dis,
                "crev_ann": self.crev.ann,
                "totex_prj": self.totex.prj,
                "totex_dis": self.totex.dis,
                "totex_ann": self.totex.ann,
                "value_prj": self.value.prj,
                "value_dis": self.value.dis,
                "value_ann": self.value.ann,
            }
        )
