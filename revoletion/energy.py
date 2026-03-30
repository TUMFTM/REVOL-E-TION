from abc import ABC

import pandas as pd

from revoletion.economics import EcoParams
from revoletion.economics.abstractclasses import CalculableTimeseriesElement, CostType, TimeseriesElement


class EnergyElement(TimeseriesElement, ABC):
    _TYPE = CostType.ENERGY


class EnergyEvaluator(CalculableTimeseriesElement, EnergyElement):
    def _calc_eval(self, flow: pd.Series) -> float:
        return flow[self.eco.dti_eval].to_numpy().sum() * self.eco.timestep_hours


class EnergyAggregator(CalculableTimeseriesElement, EnergyElement):
    """
    This class is used to aggregate energies.

    Energies already get aggregated within the implemented energy system model as the model is a flow-based model.
    This class is used to aggregate energies for meta-data purposes, e.g. track all generated renewable energy in the system.
    Even though this class is called aggregator, it works differently than economic aggregators in eco/aggregators.py.
    Its functionality is similar to an Evaluator, but instead of calculating the energy based on a flow for the evaluation timeframe, it sums up the energy values of the added evaluators.
    Additionally, the EnergyAggregator does to "pull" the energy values from the added evaluators, but instead relies on other EnergyBlock instances "pushing" their energy value (_eval) for the evaluation timeframe to it via the add_energy method.
    """

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)
        # introduce _eval as cumulative variable
        self._eval = 0.0

    def _calc_eval(self, *args, **kwargs) -> float:
        # just return the cumulative variable to preserve the cumulated value of self._eval
        return self._eval

    def add_energy(self, evaluator: EnergyEvaluator) -> None:
        self._eval += evaluator.eval

    @property
    def _result_summary_prefix(self) -> str:
        return f"{self._TYPE.value.label}_{self.name}_"
