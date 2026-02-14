from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy import typing as npt

from .abstractclasses import CalculableEcoElement, CapexElement, YearlyElement, PowerBasedElement, BlockElement
from .utils import (
    OccursAt,
    annuity,
    discount,
    transform_scalar_var,
    calc_residual_value,
    calc_lifetime_remaining,
    DEPRECIATION,
)


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
class CostParams:
    """
    Base dataclass to store all evaluator-specific parameters needed for cost evaluation.
    """

    spec: float | pd.Series
    fix: float


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


@dataclass(frozen=True)
class MntexParams(CostParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Mntex.
    """

    spec: float
    fix: float


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


class CostEvaluator(CalculableEcoElement, ABC):
    """
    Base class for all Evaluators that evaluate costs, i.e. Capex, Mntex, Opex and Crev.
    """

    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CostParams,
    ):
        super().__init__(name=name)
        self.eco = eco
        self.params = params

    @abstractmethod
    def _calc_spec_ep(self) -> float | pd.Series: ...

    def spec(self) -> float | pd.Series:
        return self.params.spec

    def fix(self) -> float:
        return self.params.fix


class YearlyEvaluator(CostEvaluator, YearlyElement, ABC):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CostParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

    @abstractmethod
    def _calc_yrl(self, *args, **kwargs) -> float: ...

    def _calc_cashflow(self, *args, **kwargs) -> npt.NDArray:
        cashflow = np.full(self.eco.prj_duration_yrs + 1, self.yrl)
        cashflow[-1] = 0.0
        return cashflow

    def evaluate(self, *args, **kwargs):
        self._yrl = self._calc_yrl()
        super().evaluate(*args, **kwargs)


class PowerBasedEvaluator(YearlyEvaluator, PowerBasedElement, ABC):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: PowerBasedParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

    def _calc_eval(self, flow) -> float:
        return (
            np.dot(self.params.spec.to_numpy(), flow[self.eco.dti_eval].to_numpy()) * self.eco.timestep_hours
            + self.params.fix
        )

    def _calc_yrl(self) -> float:
        return self.eval * self.eco.eval_yr_rat

    def evaluate(self, flow: pd.Series, *args, **kwargs):
        self._eval = self._calc_eval(flow=flow)
        super().evaluate(*args, **kwargs)

    def _calc_spec_ep(self) -> pd.Series:
        # calculate annuity due factor to compensate operation costs for difference between simulation and project time
        factor_operation_ep = (1 / self.eco.eval_yr_rat) if self.eco.compensate_sim_prj else 1

        return self.params.spec * factor_operation_ep


class CapexEvaluator(CostEvaluator, CapexElement):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CapexParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

    def _calc_cashflow_factors(self, invest_first: int) -> npt.NDArray:
        invest_periods = np.arange(invest_first, self.eco.prj_duration_yrs, self.params.ls)

        capex = np.zeros(self.eco.prj_duration_yrs + 1, dtype=float)
        capex[invest_periods] = 1

        if self.params.residual_at_ls > 0:
            residual_periods = invest_periods[1:] if self.params.consider_preexisting else invest_periods
            capex[residual_periods] -= self.params.residual_at_ls

        capex[-1] = -1 * calc_residual_value(
            lifetime_remaining_frac=calc_lifetime_remaining(
                project_duration=self.eco.prj_duration_yrs,
                ls=self.params.ls,
                init_age=invest_first % self.params.ls,
            )
            / float(self.params.ls),
            depreciation=DEPRECIATION.LINEAR,
            residual_at_ls=self.params.residual_at_ls,
        )

        # apply capex cost change ratio
        capex *= self.params.ccr ** np.arange(self.eco.prj_duration_yrs + 1)

        return capex

    def _calc_cashflow_factor_preexisting(self) -> npt.NDArray:
        invest_first = 0 if self.params.consider_preexisting else self.params.ls - self.params.age_preexisting
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_factor_expansion(self) -> npt.NDArray:
        invest_first = 0
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_preexisting(self, size_preexisting: float) -> npt.NDArray:
        return self._calc_cashflow_factor_preexisting() * (self.params.spec * size_preexisting + self.params.fix)

    def _calc_cashflow_expansion(self, size_expansion: float) -> npt.NDArray:
        return self._calc_cashflow_factor_expansion() * (self.params.spec * size_expansion + self.params.fix)

    def _calc_cashflow(self, size_preexisting: float, size_expansion: float) -> npt.NDArray:
        return self._calc_cashflow_preexisting(size_preexisting) + self._calc_cashflow_expansion(size_expansion)

    def evaluate(self, size_preexisting: float, size_expansion: float, *args, **kwargs) -> None:
        self._cashflow = self._calc_cashflow(size_preexisting=size_preexisting, size_expansion=size_expansion)
        super().evaluate(*args, **kwargs)

    def _calc_spec_ep(self) -> float:
        return np.dot(
            self._calc_cashflow_factor_expansion() * self.params.spec,
            self.eco.discount_factors(self._OCCURS_AT),
        ) * self.eco.annuity_factor_apriori(self._OCCURS_AT)

    # ToDo: move this to methods
    @cached_property
    def preexisting(self) -> float:
        # ToDo: remove this
        return (self.results.size_preexisting * self.params.spec + self.params.fix) * int(
            self.params.consider_preexisting
        )

    @cached_property
    def expansion(self) -> float:
        # Todo: remove this
        return self.results.size_expansion * self.params.spec + self.params.fix

    @cached_property
    def init(self) -> float:
        return self.preexisting + self.expansion


class MntexEvaluator(YearlyEvaluator):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: MntexParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

    def _calc_yrl(self, size_preexisting: float, size_expansion: float, *args, **kwargs) -> float:
        return self.params.spec * (size_preexisting + size_expansion) + self.params.fix

    def evaluate(self, size_preexisting: float, size_expansion: float, *args, **kwargs) -> None:
        self._yrl = self._calc_yrl(size_preexisting=size_preexisting, size_expansion=size_expansion)
        super().evaluate(*args, **kwargs)

    def _calc_spec_ep(self) -> float:
        return np.dot(
            np.full(self.eco.prj_duration_yrs + 1, self.params.spec),
            self.eco.discount_factors(self._OCCURS_AT),
        ) * self.eco.annuity_factor_apriori(self._OCCURS_AT)


class OpexEvaluator(PowerBasedEvaluator):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: OpexParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )


class CrevEvaluator(PowerBasedEvaluator):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CrevParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )


@dataclass(slots=True)
class Evaluator(BlockElement):
    """
    EvaluatorBlock is a container for all Evaluators of a single component.
    """

    name: str

    name_size: str
    name_flow: str

    capex: CapexEvaluator
    mntex: MntexEvaluator
    opex: OpexEvaluator
    crev: CrevEvaluator

    @classmethod
    def create(
        cls,
        name: str,
        eco: EcoParams,
        data_dir: Path,
        capex_spec: float = 0.0,
        capex_fix: float = 0.0,
        capex_ccr: float = 1.0,
        consider_preexisting: bool = True,
        ls: int = None,
        age_preexisting: int = 0,
        capex_residual_at_ls: float = 0.0,
        mntex_spec: float = 0.0,
        mntex_fix: float = 0.0,
        opex_spec: float | Path = 0.0,
        opex_fix: float = 0.0,
        crev_spec: float | Path = 0.0,
        crev_fix: float = 0.0,
        size_name: str = None,
        flow_name: str = None,
    ) -> Self:
        ls = eco.prj_duration_yrs if ls is None else ls

        capex = CapexEvaluator.create_from_plain(
            name=f"{name}_capex",
            eco=eco,
            spec=capex_spec,
            fix=capex_fix,
            ccr=capex_ccr,
            ls=ls,
            consider_preexisting=consider_preexisting,
            age_preexisting=age_preexisting,
            residual_at_ls=capex_residual_at_ls,
        )

        mntex = MntexEvaluator.create_from_plain(
            name=f"{name}_mntex",
            eco=eco,
            spec=mntex_spec,
            fix=mntex_fix,
        )

        opex = OpexEvaluator.create_from_plain(
            name=f"{name}_opex",
            eco=eco,
            spec=opex_spec,
            fix=opex_fix,
            data_dir=data_dir,
        )

        crev = CrevEvaluator.create_from_plain(
            name=f"{name}_crev",
            eco=eco,
            spec=crev_spec,
            fix=crev_fix,
            data_dir=data_dir,
        )

        size_name = size_name if size_name else name
        flow_name = flow_name if flow_name else name

        return cls(
            name=name,
            name_size=size_name,
            name_flow=flow_name,
            capex=capex,
            mntex=mntex,
            opex=opex,
            crev=crev,
        )

    @property
    def spec_ep_invest(self):
        """
        Equivalent present specific costs for investments (cost per size)
        """
        return self.capex.spec_ep + self.mntex.spec_ep

    @property
    def spec_ep_operation(self):
        """
        Equivalent present specific costs for operation (cost per energy)
        """
        return self.opex.spec_ep + self.crev.spec_ep

    def calc_results(
        self,
        sizes: dict = None,
        flows: pd.DataFrame = None,
    ) -> None:
        # ToDo: check whether sizes and flows can be None or if this always is a dict or a DataFrame
        if sizes is not None and self.name_size in sizes:
            size_preexisting = sizes[self.name_size].preexisting
            size_expansion = sizes[self.name_size].expansion
            self.capex.calc_results(size_preexisting=size_preexisting, size_expansion=size_expansion)
            self.mntex.calc_results(size_preexisting=size_preexisting, size_expansion=size_expansion)

        if flows is not None and self.name_flow in flows.columns:
            flow = flows[self.name_flow]
            self.opex.calc_results(flow=flow)
            self.crev.calc_results(flow=flow)


class Energy:
    _OCCURS_AT = OccursAt.END

    def __init__(self, name: str, eco: EcoParams):
        self.name = name
        self._eco = eco

        self._eval = None
        self._yrl = None
        self._prj = None
        self._dis = None
        self._ann = None

    def _calc_derived_results(self) -> None:
        self._yrl = self._eval * self._eco.eval_yr_rat
        self._prj = self._yrl * self._eco.prj_duration_yrs
        # ToDo: check these calculations
        self._dis = self._prj * discount(
            future_value=1,
            periods=self._eco.prj_duration_yrs,
            discount_rate=self._eco.discount_rate,
            occurs_at=self._OCCURS_AT,
        )
        self._ann = annuity(
            present_value=self._dis,
            observation_horizon=self._eco.prj_duration_yrs,
            discount_rate=self._eco.discount_rate,
            occurs_at=self._OCCURS_AT,
        )

    @property
    def result_summary(self) -> pd.Series:
        return pd.Series(
            data={
                f"energy_{self.name}_eval": self.eval,
                f"energy_{self.name}_yrl": self.yrl,
                f"energy_{self.name}_prj": self.prj,
                f"energy_{self.name}_dis": self.dis,
                f"energy_{self.name}_ann": self.ann,
            },
            name=self.name,
        )

    @property
    def eval(self) -> float:
        if self._eval is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._eval

    @property
    def yrl(self) -> float:
        if self._yrl is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._yrl

    @property
    def prj(self) -> float:
        if self._prj is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._prj

    @property
    def dis(self) -> float:
        if self._dis is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._dis

    @property
    def ann(self) -> float:
        if self._ann is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._ann


class EnergyEvaluator(Energy):
    def calc_results(self, flow: pd.Series) -> None:
        self._eval = flow[self._eco.dti_eval].to_numpy().sum() * self._eco.timestep_hours
        self._calc_derived_results()


class EnergyAggregator(Energy):
    def __init__(self, name: str, eco: EcoParams):
        super().__init__(name=name, eco=eco)
        self._eval = 0.0

    def add_energy(self, evaluator: Energy) -> None:
        self._eval += evaluator._eval

    def calc_results(self) -> None:
        self._calc_derived_results()
