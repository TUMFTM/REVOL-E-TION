from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd

from .abstractclasses import (
    BaseElement,
    BlockElement,
    CapexElement,
    CostType,
    CrevElement,
    MntexElement,
    OpexElement,
    TimeseriesElement,
    YearlyElement,
)


class BaseAggregator(BaseElement, ABC):
    """
    Base class for all Aggregators.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def _aggregate_scalar_attribute(self, property_name: str) -> float: ...

    @abstractmethod
    def _aggregate_vector_attribute(self, property_name: str) -> npt.NDArray: ...

    def _aggregate_cashflow(self) -> npt.NDArray:
        return self._aggregate_vector_attribute("cashflow")

    def _aggregate_cashflow_dis(self) -> npt.NDArray:
        return self._aggregate_vector_attribute("cashflow_dis")

    def _aggregate_prj(self) -> float:
        return self._aggregate_scalar_attribute("prj")

    def _aggregate_dis(self) -> float:
        return self._aggregate_scalar_attribute("dis")

    def _aggregate_ann(self) -> float:
        return self._aggregate_scalar_attribute("ann")

    def aggregate(self) -> None:
        """
        Aggregate the values of the given elements and store them in the corresponding attributes.
        This method has to be called after all elements have been added to the aggregator and results have been
        calculated for all elements.
        """
        self._cashflow = self._aggregate_cashflow()
        self._cashflow_dis = self._aggregate_cashflow_dis()
        self._prj = self._aggregate_prj()
        self._dis = self._aggregate_dis()
        self._ann = self._aggregate_ann()


class CrossLevelAggregator(BaseAggregator, ABC):
    """
    Base class for all Aggregators, aggregating costs from the same type (Capex, Mntex, Opex, Crev).
    These costs may occur in the same block (aggregate Evaluators) or in subblocks (aggregate other Aggregators).
    """

    def __init__(self, name: str, prj_duration_yrs: int, **kwargs):
        super().__init__(name=name, **kwargs)

        self.elements = {}
        self._prj_duration_yrs = prj_duration_yrs

    def _aggregate_scalar_attribute(self, property_name: str) -> float:
        return sum(getattr(poi, property_name) for poi in self.elements.values())

    def _aggregate_vector_attribute(self, property_name: str) -> npt.NDArray:
        values = [getattr(poi, property_name) for poi in self.elements.values()]
        if not values:
            # create empty array if there are no elements to aggregate
            return np.zeros(self._prj_duration_yrs + 1)
        return np.sum(values, axis=0)


class InLevelAggregator(BaseAggregator, ABC):
    """
    Base class for all Aggregators, which aggregate values from other aggregators on the same level.
    This is used for Totex (Capex + Mntex + Opex) and Value (Totex - Crev) aggregation.
    """

    pass


class YearlyAggregator(CrossLevelAggregator, YearlyElement, ABC):
    """
    YearlyAggregator aggregates the values of all given YearlyElements.
    This is used for Mntex, Opex and Crev aggregation.
    """

    def aggregate(self):
        self._yrl = self._aggregate_scalar_attribute("yrl")
        super().aggregate()


class TimeseriesAggregator(YearlyAggregator, TimeseriesElement, ABC):
    """
    PowerBasedAggregator aggregates the values of all given PowerBasedElements.
    """

    def aggregate(self):
        self._eval = self._aggregate_scalar_attribute("eval")
        super().aggregate()


class CapexAggregator(CrossLevelAggregator, CapexElement):
    """
    CapexAggregator aggregates the values of all given capex elements (Evaluators and Aggregators).

    """

    def _aggregate_preexisting(self) -> float:
        return self._aggregate_scalar_attribute("preexisting")

    def _aggregate_expansion(self) -> float:
        return self._aggregate_scalar_attribute("expansion")

    def _aggregate_init(self) -> float:
        return self._aggregate_scalar_attribute("init")

    def aggregate(self):
        self._preexisting = self._aggregate_preexisting()
        self._expansion = self._aggregate_expansion()
        self._init = self._aggregate_init()
        super().aggregate()


class MntexAggregator(YearlyAggregator, MntexElement):
    """
    MntexAggregator aggregates the values of all given mntex elements (Evaluators and Aggregators).
    """

    pass


class OpexAggregator(TimeseriesAggregator, OpexElement):
    """
    OpexAggregator aggregates the values of all given opex elements (Evaluators and Aggregators).
    """

    pass


class CrevAggregator(TimeseriesAggregator, CrevElement):
    """
    CrevAggregator aggregates the values of all given crev elements (Evaluators and Aggregators).
    """

    pass


class TotexAggregator(InLevelAggregator):
    """
    TotexAggregator aggregates the values of Capex, Mntex and Opex aggregators on the same level to calculate the total costs.
    """

    _TYPE = CostType.TOTEX

    def __init__(self, name: str, capex: CapexAggregator, mntex: MntexAggregator, opex: OpexAggregator, **kwargs):
        super().__init__(name=name, **kwargs)

        self.capex = capex
        self.mntex = mntex
        self.opex = opex

    def _aggregate_scalar_attribute(self, property_name: str) -> float:
        return (
            getattr(self.capex, property_name) + getattr(self.mntex, property_name) + getattr(self.opex, property_name)
        )

    def _aggregate_vector_attribute(self, property_name: str) -> npt.NDArray:
        return (
            getattr(self.capex, property_name) + getattr(self.mntex, property_name) + getattr(self.opex, property_name)
        )


class ValueAggregator(InLevelAggregator):
    """
    ValueAggregator subtracts revenues (Crev) from costs (Totex) on the same level.
    """

    _TYPE = CostType.VALUE

    def __init__(self, name: str, totex: TotexAggregator, crev: CrevAggregator, **kwargs):
        super().__init__(name=name, **kwargs)

        self.totex = totex
        self.crev = crev

    def _aggregate_scalar_attribute(self, property_name: str) -> float:
        return getattr(self.crev, property_name) - getattr(self.totex, property_name)

    def _aggregate_vector_attribute(self, property_name: str) -> npt.NDArray:
        return getattr(self.crev, property_name) - getattr(self.totex, property_name)


class Aggregator(BlockElement):
    """
    EcoBlock holds all economic components of a block.
    """

    def __init__(self, name: str, prj_duration_yrs: int, **kwargs):
        super().__init__(name=name, **kwargs)

        self.capex: CapexAggregator = CapexAggregator(name=name, prj_duration_yrs=prj_duration_yrs)
        self.mntex: MntexAggregator = MntexAggregator(name=name, prj_duration_yrs=prj_duration_yrs)
        self.opex: OpexAggregator = OpexAggregator(name=name, prj_duration_yrs=prj_duration_yrs)
        self.crev: CrevAggregator = CrevAggregator(name=name, prj_duration_yrs=prj_duration_yrs)
        self.totex: TotexAggregator = TotexAggregator(name=name, capex=self.capex, mntex=self.mntex, opex=self.opex)
        self.value: ValueAggregator = ValueAggregator(name=name, totex=self.totex, crev=self.crev)

    def add_block(self, block: BlockElement) -> None:
        """
        Add a new subblock to the current block.
        """
        for attr in ("capex", "mntex", "opex", "crev"):
            value = getattr(block, attr)
            if value is not None:
                getattr(self, attr).elements[block.name] = value

    def aggregate(self) -> None:
        """
        Aggregate the values of all given elements and store them in the corresponding attributes.
        This method has to be called after all elements have been added to the aggregator and results have been
        calculated for all elements.
        """
        self.capex.aggregate()
        self.mntex.aggregate()
        self.opex.aggregate()
        self.crev.aggregate()
        self.totex.aggregate()
        self.value.aggregate()

    @property
    def result_summary(self) -> pd.Series:
        return pd.concat(
            [
                super().result_summary,
                self.totex.result_summary,
                self.value.result_summary,
            ],
            axis=0,
        )
