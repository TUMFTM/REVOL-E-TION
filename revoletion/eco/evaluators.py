from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol, Self

import numpy as np
import pandas as pd

from .abstractclasses import EcoElement, CapexElement, YearlyElement, BlockElement
from .size import Size
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
    def _discount_factors(self) -> dict[OccursAt, pd.Series]:
        periods = np.arange(self.prj_duration_yrs + 1)
        return {
            occurs_at: pd.Series(
                index=periods,
                data=discount(
                    future_value=1,
                    periods=periods,
                    discount_rate=self.discount_rate,
                    occurs_at=occurs_at,
                ),
            )
            for occurs_at in OccursAt
        }

    def discount_factors(self, occurs_at: OccursAt) -> pd.Series:
        return self._discount_factors[occurs_at]


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


class CostEvaluator(EcoElement, ABC):
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
        self.name = name
        self._eco = eco
        self._params = params

        self._cashflow = np.zeros(self._eco.prj_duration_yrs + 1, dtype=float)

    def __init_subclass__(cls):
        # cache the names of all cached properties in the subclass to invalidate them when any attribute is set
        super().__init_subclass__()
        cls._cached_props = {name for name, obj in cls.__dict__.items() if isinstance(obj, cached_property)}

    def __setattr__(self, name, value):
        # flag to detect object initialization to avoid invalidating cached properties during initialization
        initializing = name not in self.__dict__
        super().__setattr__(name, value)

        # don't invalidate while __init__ is running
        if initializing:
            return

        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        # invalidate all cached properties when any attribute is set
        for prop in type(self)._cached_props:
            self.__dict__.pop(prop, None)

    @classmethod
    @abstractmethod
    def create_from_plain(cls, *_) -> Self: ...

    def calc_results(self, *_) -> None: ...

    @cached_property
    def spec(self) -> float | pd.Series:
        return self._params.spec

    @cached_property
    def fix(self) -> float:
        return self._params.fix

    @cached_property
    def cashflow_dis(self) -> pd.Series:
        return self.cashflow * self._eco.discount_factors(self._OCCURS_AT)

    @cached_property
    def dis(self) -> float:
        return sum(self.cashflow_dis)

    @cached_property
    def ann(self) -> float:
        # ToDo: use cached discount factors here?
        return annuity(
            present_value=self.dis,
            observation_horizon=self._eco.prj_duration_yrs,
            discount_rate=self._eco.discount_rate,
            occurs_at=self._OCCURS_AT,
        )


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

    @cached_property
    @abstractmethod
    def yrl(self) -> float: ...

    @cached_property
    def cashflow(self) -> pd.Series:
        cashflow = pd.Series(np.full(self._eco.prj_duration_yrs + 1, self.yrl))
        cashflow.iloc[-1] = 0.0
        return cashflow


class PowerBasedEvaluator(YearlyEvaluator, ABC):
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

        self._eval = None

    @classmethod
    def create_from_plain(cls, name: str, eco: EcoParams, spec: Path | float, fix: float, data_dir: Path) -> Self:
        params = PowerBasedParams.create_from_plain(spec=spec, fix=fix, dti_sim=eco.dti_sim, data_dir=data_dir)
        return cls(name=name, eco=eco, params=params)

    def calc_results(self, flow: pd.Series) -> None:
        self.invalidate_cache()
        self._eval = (
            float(np.dot(self._params.spec[self._eco.dti_eval].to_numpy(), flow[self._eco.dti_eval].to_numpy()))
            * self._eco.timestep_hours
        )

    @cached_property
    def spec_ep(self):
        # calculate annuity due factor to compensate operation costs for difference between simulation and project time
        factor_operation_ep = (1 / self._eco.eval_yr_rat) if self._eco.compensate_sim_prj else 1

        return self._params.spec * factor_operation_ep

    @cached_property
    def eval(self) -> float:
        if self._eval is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._eval

    @cached_property
    def yrl(self) -> float:
        return self.eval * self._eco.eval_yr_rat + self._params.fix


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

        self._cashflow = None

    def __post_init__(self):
        if self._params.consider_preexisting and self._params.age_preexisting != 0:
            raise ValueError(
                f"If consider_preexisting is True, init_age_years must be 0, got {self._params.age_preexisting}"
            )

    @classmethod
    def create_from_plain(
        cls,
        name: str,
        eco: EcoParams,
        spec: float,
        fix: float,
        consider_preexisting: bool,
        ls: int,
        age_preexisting: int,
        ccr: float,
        residual_at_ls: float,
    ) -> Self:
        params = CapexParams(
            spec=spec,
            fix=fix,
            consider_preexisting=consider_preexisting,
            ls=ls,
            age_preexisting=age_preexisting,
            ccr=ccr,
            residual_at_ls=residual_at_ls,
        )
        return cls(name=name, eco=eco, params=params)

    def calc_results(self, size_preexisting: float, size_expansion: float) -> None:
        self.invalidate_cache()
        self._cashflow = self.cashflow_factor_preexisting * (
            self._params.spec * size_preexisting + self._params.fix
        ) + self.cashflow_factor_expansion * (self._params.spec * size_expansion + self._params.fix)

    @cached_property
    def spec_ep(self) -> float:
        # ToDo: use cached discount values here?
        factor_annuity = (
            annuity(
                present_value=1,
                observation_horizon=self._eco.prj_duration_yrs,
                discount_rate=self._eco.discount_rate,
                occurs_at=self._OCCURS_AT,
            )
            if self._eco.compensate_sim_prj
            else 1
        )
        factor_spec_ep = float(
            np.dot(
                self.cashflow_factor_expansion.to_numpy() * self._params.spec,
                self._eco.discount_factors(self._OCCURS_AT).to_numpy(),
            )
        )

        return factor_annuity * factor_spec_ep

    @cached_property
    def preexisting(self) -> float:
        # ToDo: remove this
        return (self.results.size_preexisting * self._params.spec + self._params.fix) * int(
            self._params.consider_preexisting
        )

    @cached_property
    def expansion(self) -> float:
        # Todo: remove this
        return self.results.size_expansion * self._params.spec + self._params.fix

    @cached_property
    def init(self) -> float:
        return self.preexisting + self.expansion

    def calc_capex_cashflow(self, invest_first: int) -> pd.Series:
        invest_periods = np.arange(invest_first, self._eco.prj_duration_yrs, self._params.ls)

        capex = np.zeros(self._eco.prj_duration_yrs + 1, dtype=float)
        capex[invest_periods] = 1

        if self._params.residual_at_ls > 0:
            residual_periods = invest_periods[1:] if self._params.consider_preexisting else invest_periods
            capex[residual_periods] -= self._params.residual_at_ls

        capex[-1] = -1 * calc_residual_value(
            lifetime_remaining_frac=calc_lifetime_remaining(
                project_duration=self._eco.prj_duration_yrs,
                ls=self._params.ls,
                init_age=invest_first % self._params.ls,
            )
            / float(self._params.ls),
            depreciation=DEPRECIATION.LINEAR,
            residual_at_ls=self._params.residual_at_ls,
        )

        # apply capex cost change ratio
        capex *= self._params.ccr ** np.arange(self._eco.prj_duration_yrs + 1)

        return pd.Series(capex)

    @cached_property
    def cashflow_factor_preexisting(self) -> pd.Series:
        invest_first = 0 if self._params.consider_preexisting else self._params.ls - self._params.age_preexisting
        return self.calc_capex_cashflow(invest_first)

    @cached_property
    def cashflow_factor_expansion(self) -> pd.Series:
        invest_first = 0
        return self.calc_capex_cashflow(invest_first)

    @cached_property
    def cashflow(self) -> pd.Series:
        if self._cashflow is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._cashflow


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

        self._yrl = None

    @classmethod
    def create_from_plain(cls, name: str, eco: EcoParams, spec: float, fix: float) -> Self:
        params = MntexParams(spec=spec, fix=fix)
        return cls(name=name, eco=eco, params=params)

    def calc_results(self, size_preexisting: float, size_expansion: float) -> None:
        self.invalidate_cache()
        self._yrl = (size_preexisting + size_expansion) * self._params.spec + self._params.fix

    @cached_property
    def spec_ep(self) -> float:
        # ToDo: use cached discount values here?
        factor_annuity = (
            annuity(
                present_value=1,
                observation_horizon=self._eco.prj_duration_yrs,
                discount_rate=self._eco.discount_rate,
                occurs_at=self._OCCURS_AT,
            )
            if self._eco.compensate_sim_prj
            else 1
        )
        factor_spec_ep = float(
            np.dot(
                np.full(self._eco.prj_duration_yrs + 1, self._params.spec),
                self._eco.discount_factors(self._OCCURS_AT).to_numpy(),
            )
        )

        return factor_annuity * factor_spec_ep

    @cached_property
    def yrl(self) -> float:
        if self._yrl is None:
            raise ValueError("Results need to be calculated before they can be accessed.")
        return self._yrl


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
