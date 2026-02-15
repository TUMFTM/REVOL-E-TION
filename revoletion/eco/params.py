from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy import typing as npt

from .utils import OccursAt, discount, annuity, transform_scalar_var


@dataclass(frozen=True)
class EcoParams:
    """
    Dataclass to store all parameters being valid for all evaluators needed for cost evaluation.
    """

    prj_duration_yrs: int
    discount_rate: float
    compensate_sim_prj: bool
    eval_yr_rat: float
    eval_prj_rat: float
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
        return cls(
            prj_duration_yrs=prj_duration_yrs,
            discount_rate=discount_rate,
            compensate_sim_prj=compensate_sim_prj,
            eval_yr_rat=pd.Timedelta(days=365) / td_eval,  # no leap years
            eval_prj_rat=((dti_sim[0] + pd.DateOffset(years=prj_duration_yrs)) - dti_sim[0]) / td_eval,
            dti_sim=dti_sim,
            dti_eval=dti_eval,
            timestep_hours=pd.Timedelta(dti_sim.inferred_freq).total_seconds() / 3600,
        )

    @cached_property
    def _discount_factors(self) -> dict[OccursAt, npt.NDArray]:
        periods = np.arange(self.prj_duration_yrs + 1)
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

    spec: float | pd.Series
    fix: float

    @classmethod
    @abstractmethod
    def create_from_plain(cls, *args, **kwargs) -> Self: ...


@dataclass(frozen=True)
class CapexParams(CostParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Capex.
    """

    spec: float
    fix: float
    consider_preexisting: bool
    ls: int
    age_preexisting: int
    ccr: float
    residual_at_ls: float

    def __post_init__(self):
        if self.consider_preexisting and self.age_preexisting != 0:
            raise ValueError(f"If consider_preexisting is True, age_preexisting must be 0, got {self.age_preexisting}")

    @classmethod
    def create_from_plain(
        cls,
        spec: float,
        fix: float,
        consider_preexisting: bool = False,
        ls: int = 0,
        age_preexisting: int = 0,
        ccr: float = 1.0,
        residual_at_ls: float = 0.0,
    ) -> Self:
        return cls(
            spec=spec,
            fix=fix,
            consider_preexisting=consider_preexisting,
            ls=ls,
            age_preexisting=age_preexisting,
            ccr=ccr,
            residual_at_ls=residual_at_ls,
        )


@dataclass(frozen=True)
class MntexParams(CostParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Mntex.
    """

    spec: float
    fix: float

    @classmethod
    def create_from_plain(cls, spec: float, fix: float) -> Self:
        return cls(spec=spec, fix=fix)


@dataclass(frozen=True)
class PowerBasedParams(CostParams):
    """
    Base dataclass to store all parameters needed for cost evaluation of flow-related costs (Opex and Crev).
    """

    spec: pd.Series
    fix: float

    @classmethod
    def create_from_plain(cls, spec: str | float | int, fix: float, dti_sim: pd.DatetimeIndex, data_dir: Path) -> Self:
        spec_series = transform_scalar_var(value=spec, dti=dti_sim, data_dir=data_dir)
        return cls(spec=spec_series, fix=fix)


@dataclass(frozen=True)
class OpexParams(PowerBasedParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Opex.
    """

    pass


@dataclass(frozen=True)
class CrevParams(PowerBasedParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Crev.
    """

    pass
