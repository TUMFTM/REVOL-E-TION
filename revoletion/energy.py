from abc import ABC

import pandas as pd

from revoletion.eco import EcoParams
from revoletion.eco.abstractclasses import CalculablePowerBasedElement, PowerBasedElement
from revoletion.eco.utils import OccursAt


class Energy(PowerBasedElement, ABC):
    _TYPE = "energy"
    _OCCURS_AT = OccursAt.END


class EnergyEvaluator(CalculablePowerBasedElement, Energy):
    def __init__(self, name: str, eco: EcoParams):
        super().__init__(name=name, eco=eco)

    def _calc_eval(self, flow: pd.Series) -> float:
        return flow[self.eco.dti_eval].to_numpy().sum() * self.eco.timestep_hours

    def evaluate(self, flow: pd.Series, **kwargs) -> None:
        super().evaluate(flow=flow, **kwargs)


class EnergyAggregator(CalculablePowerBasedElement, Energy):
    def __init__(self, name: str, eco: EcoParams):
        super().__init__(name=name, eco=eco)
        # introduce _eval as cumulative variable
        self._eval = 0.0

    def _calc_eval(self, *args, **kwargs) -> float:
        # just return the cumulative variable to preserve the cumulated value of self._eval
        return self._eval

    def add_energy(self, evaluator: EnergyEvaluator) -> None:
        self._eval += evaluator.eval
