from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Type, Self

import pandas as pd

from revoletion.eco import EcoParams, Evaluator
from revoletion.eco.evaluators import OpexEvaluator
from revoletion.eco.peakpower.params import PeakPowerOpexParams


class PeakPowerOpexEvaluator(OpexEvaluator):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: PeakPowerOpexParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )

    def _calc_spec_ep(self, **kwargs) -> float:
        return (
            self.spec * self.params.n_peak_periods_yr / self.params.n_peak_periods_sim
            if self.eco.compensate_sim_prj
            else 1
        )

    def _calc_eval(self, power_peak: float, **kwargs) -> float:
        return power_peak * self.params.spec


@dataclass
class PeakPowerEvaluator(Evaluator):
    """
    EvaluatorBlock for Peak Power of GridConnection, which includes specific methods to evaluate peak power-specific costs and results.
    """

    OPEX_EVALUATOR: ClassVar[Type[OpexEvaluator]] = PeakPowerOpexEvaluator

    @classmethod
    def _build_opex_params(
        cls,
        spec: float | Path,
        n_peak_periods_yr: int,
        n_peak_periods_sim: int,
        **kwargs,
    ) -> PeakPowerOpexParams:
        return PeakPowerOpexParams.create_from_plain(
            spec=spec,
            n_peak_periods_yr=n_peak_periods_yr,
            n_peak_periods_sim=n_peak_periods_sim,
            **kwargs,
        )

    @classmethod
    def create_from_plain(
        cls,
        name: str,
        eco: EcoParams,
        data_dir: Path,
        name_size: str = None,
        name_flow: str = None,
        # capex
        capex_spec: float = 0.0,
        capex_fix: float = 0.0,
        capex_ccr: float = 1.0,
        consider_preexisting: bool = True,
        ls: int | None = None,
        age_preexisting: int = 0,
        capex_residual_at_ls: float = 0.0,
        # mntex
        mntex_spec: float = 0.0,
        mntex_fix: float = 0.0,
        # opex
        opex_spec: float | Path = 0.0,
        opex_n_peak_periods_yr: int = 12,
        opex_n_peak_periods_sim: int = 12,
        # crev
        crev_spec: float | Path = 0.0,
        crev_spec_dist: float | Path = 0.0,
        crev_spec_time: float | Path = 0.0,
        crev_fix: float = 0.0,
    ) -> Self:
        params_capex = cls._build_capex_params(
            eco=eco,
            spec=capex_spec,
            fix=capex_fix,
            ccr=capex_ccr,
            consider_preexisting=consider_preexisting,
            ls=ls,
            age_preexisting=age_preexisting,
            residual_at_ls=capex_residual_at_ls,
        )

        params_mntex = cls._build_mntex_params(
            spec=mntex_spec,
            fix=mntex_fix,
        )

        params_opex = cls._build_opex_params(
            spec=opex_spec,
            n_peak_periods_yr=opex_n_peak_periods_yr,
            n_peak_periods_sim=opex_n_peak_periods_sim,
        )

        params_crev = cls._build_crev_params(
            eco=eco,
            data_dir=data_dir,
            spec=crev_spec,
            spec_dist=crev_spec_dist,
            spec_time=crev_spec_time,
            fix=crev_fix,
        )

        return cls.create(
            name=name,
            eco=eco,
            name_size=name_size,
            name_flow=name_flow,
            params_capex=params_capex,
            params_mntex=params_mntex,
            params_opex=params_opex,
            params_crev=params_crev,
        )

    def evaluate(
        self,
        sizes: dict,
        flows: pd.DataFrame,
        **kwargs,
    ) -> None:
        peak_periods = kwargs.pop("peak_periods", None)
        if peak_periods is None:
            raise ValueError("PeakPowerEvaluator requires a 'peak_periods' argument")
        peak_period = peak_periods.get(self.name, None)
        if peak_period is None:
            raise ValueError(f"Period {self.name} not found in 'peak_periods'")
        power_peak = peak_period.max_power

        super().evaluate(sizes, flows, power_peak=power_peak, **kwargs)
