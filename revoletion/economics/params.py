from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np
import pandas as pd
from numpy import typing as npt

from .utils import OccursAt, discount, annuity


@dataclass(frozen=True)
class EcoParams:
    """
    Dataclass to store all parameters being valid for all evaluators needed for cost evaluation.
    """

    prj_duration_yrs: int
    discount_rate: float
    compensate_sim_prj: bool
    eval_yr_rat: float
    sim_yr_rat: float
    eval_prj_rat: float
    sim_prj_rat: float
    dti_sim: pd.DatetimeIndex
    dti_eval: pd.DatetimeIndex
    timestep_hours: float

    @classmethod
    def from_simulation_times(
        cls,
        prj_duration_yrs: int,
        discount_rate: float,
        compensate_sim_prj: bool,
        times: "SimulationTimes",
        timestep: "Timestep",
    ) -> Self:
        return cls(
            prj_duration_yrs=prj_duration_yrs,
            discount_rate=discount_rate,
            compensate_sim_prj=compensate_sim_prj,
            eval_yr_rat=pd.Timedelta(days=365) / times.eval.duration,  # no leap years
            eval_prj_rat=times.prj.duration / times.eval.duration,
            sim_yr_rat=pd.Timedelta(days=365) / times.sim.duration,  # no leap years
            sim_prj_rat=times.prj.duration / times.sim.duration,
            dti_sim=times.sim.dti,
            dti_eval=times.eval.dti,
            timestep_hours=timestep.hours,
        )

    @classmethod
    def from_parameters(
        cls,
        prj_duration_yrs: int,
        discount_rate: float,
        compensate_sim_prj: bool,
        dti_sim: pd.DatetimeIndex,
        dti_eval: pd.DatetimeIndex,
    ) -> Self:
        td_eval = dti_eval[-1] - dti_eval[0]
        td_sim = dti_sim[-1] - dti_sim[0]
        return cls(
            prj_duration_yrs=prj_duration_yrs,
            discount_rate=discount_rate,
            compensate_sim_prj=compensate_sim_prj,
            eval_yr_rat=pd.Timedelta(days=365) / td_eval,  # no leap years
            eval_prj_rat=((dti_sim[0] + pd.DateOffset(years=prj_duration_yrs)) - dti_sim[0]) / td_eval,
            sim_yr_rat=pd.Timedelta(days=365) / td_eval,  # no leap years
            sim_prj_rat=((dti_sim[0] + pd.DateOffset(years=prj_duration_yrs)) - dti_sim[0]) / td_sim,
            dti_sim=dti_sim,
            dti_eval=dti_eval,
            timestep_hours=pd.Timedelta(dti_sim.inferred_freq).total_seconds() / 3600,
        )

    @cached_property
    def _discount_factors(self) -> dict[OccursAt, npt.NDArray]:
        periods = np.arange(1, self.prj_duration_yrs + 2)  # 1st year equals to "1"
        return {
            occurs_at: discount(
                future_value=1,
                periods=periods,
                discount_rate=self.discount_rate,
                occurs_at=occurs_at,
            )
            for occurs_at in OccursAt
        }

    def discount_factors(self, occurs_at: OccursAt) -> npt.NDArray:
        return self._discount_factors[occurs_at]

    @cached_property
    def _annuity_factors(self) -> dict[OccursAt, float]:
        return {
            occurs_at: annuity(
                present_value=1,
                observation_horizon=self.prj_duration_yrs,
                discount_rate=self.discount_rate,
                occurs_at=occurs_at,
            )
            for occurs_at in OccursAt
        }

    def annuity_factor(self, occurs_at: OccursAt) -> float:
        return self._annuity_factors[occurs_at]

    def annuity_factor_apriori(self, occurs_at: OccursAt) -> float:
        return self.annuity_factor(occurs_at) if self.compensate_sim_prj else 1.0


@dataclass(frozen=True)
class CostParams(ABC):
    """
    Base dataclass to store all evaluator-specific parameters needed for cost evaluation.
    """


@dataclass(frozen=True)
class CapexParams(CostParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Capex.
    """

    spec: float = 0.0
    fix: float = 0.0
    consider_preexisting: bool = True
    ls: int = None
    age_preexisting: int = 0
    ccr: float = 1.0
    residual_at_ls: float = 0.0


@dataclass(frozen=True)
class MntexParams(CostParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Mntex.
    """

    spec: float = 0.0
    fix: float = 0.0


@dataclass(frozen=True)
class TimeseriesParams(CostParams):
    """
    Base dataclass to store all parameters needed for cost evaluation of a power flow.
    """

    spec_power: str | float | int = 0.0
    spec_dist: str | float | int = 0.0
    spec_time: str | float | int = 0.0
    fix: float = 0.0


@dataclass(frozen=True)
class OpexParams(TimeseriesParams):
    """
    Dataclass to store all parameters needed for opex evaluation.
    """

    spec_peak: float | int = 0.0
    n_peak_periods_yr: int = 1
    n_peak_periods_sim: int = 1


@dataclass(frozen=True)
class CrevParams(TimeseriesParams):
    """
    Dataclass to store all parameters needed for crev evaluation.
    """

    pass
