from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Type

import pandas as pd

from revoletion.eco import EcoParams, Evaluator
from revoletion.eco.abstractclasses import CalculablePowerBasedElement, OpexElement
from revoletion.eco.evaluators import CostEvaluator, OpexEvaluator
from revoletion.eco.peakpower.params import PeakPowerOpexParams


class PeakPowerOpexEvaluator(CostEvaluator, CalculablePowerBasedElement, OpexElement):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec: float,
        n_peak_periods_yr: int,
        n_peak_periods_sim: int,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            **kwargs,
        )

        self.spec = spec
        self.n_peak_periods_yr = n_peak_periods_yr
        self.n_peak_periods_sim = n_peak_periods_sim

    @classmethod
    def _build_kwargs_from_params(cls, params: PeakPowerOpexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(
            spec=params.spec,
            n_peak_periods_yr=params.n_peak_periods_yr,
            n_peak_periods_sim=params.n_peak_periods_sim,
            **kwargs,
        )

    def _calc_spec_ep(self, **kwargs) -> float:
        return self.spec * self.n_peak_periods_yr / self.n_peak_periods_sim if self.eco.compensate_sim_prj else 1

    def _calc_eval(self, power_peak: float | None, **kwargs) -> float:
        return power_peak * self.spec if power_peak is not None else 0.0


@dataclass
class PeakPowerEvaluator(Evaluator):
    """
    EvaluatorBlock for Peak Power of GridConnection, which includes specific methods to evaluate peak power-specific costs and results.
    """

    OPEX_EVALUATOR: ClassVar[Type[OpexEvaluator]] = PeakPowerOpexEvaluator

    def evaluate(
        self,
        size_preexisting: float | None = None,
        size_expansion: float | None = None,
        flow: pd.Series | None = None,
        power_peak: float | None = None,
        **kwargs,
    ) -> None:
        super().evaluate(
            size_preexisting=size_preexisting,
            size_expansion=size_expansion,
            flow=flow,
            power_peak=power_peak,
            **kwargs,
        )
