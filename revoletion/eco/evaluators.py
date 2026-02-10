from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from .abstractclasses import EcoElement, CapexElement, YearlyElement, BlockElement
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

    project_duration: int
    discount_rate: float
    compensate_sim_prj: bool
    eval2yr_ratio: float
    dti_sim: pd.DatetimeIndex
    dti_eval: pd.DatetimeIndex
    timestep: float

    @classmethod
    def from_parameters(
        cls,
        project_duration: int,
        discount_rate: float,
        compensate_sim_prj: bool,
        dti_sim: pd.DatetimeIndex,
        dti_eval: pd.DatetimeIndex,
    ) -> Self:
        return cls(
            project_duration=project_duration,
            discount_rate=discount_rate,
            compensate_sim_prj=compensate_sim_prj,
            eval2yr_ratio=pd.Timedelta(days=365) / (dti_eval[-1] - dti_eval[0]),
            dti_sim=dti_sim,
            dti_eval=dti_eval,
            timestep=pd.Timedelta(dti_sim.inferred_freq).total_seconds() / 3600,
        )


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
class FlowParams(CostParams):
    """
    Base dataclass to store all parameters needed for cost evaluation of flow-related costs (Opex and Crev).
    """

    spec: pd.Series
    fix: float

    @classmethod
    def create_from_plain(cls, spec: Path | float | int, fix: float, dti_sim: pd.DatetimeIndex) -> Self:
        spec_series = transform_scalar_var(value=spec, dti=dti_sim)
        return cls(spec=spec_series, fix=fix)


@dataclass(frozen=True)
class OpexParams(FlowParams):
    """
    Dataclass to store all parameters needed for cost evaluation of Opex.
    """

    pass


@dataclass(frozen=True)
class CrevParams(FlowParams):
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

        self._cashflow = np.zeros(self._eco.project_duration + 1, dtype=float)

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

        # invalidate all cached properties when any attribute is set
        for prop in type(self)._cached_props:
            self.__dict__.pop(prop, None)

    @cached_property
    def discount_factors(self) -> pd.Series:
        return pd.Series(
            discount(
                future_value=1,
                periods=np.arange(self._eco.project_duration + 1),
                discount_rate=self._eco.discount_rate,
                occurs_at=self._OCCURS_AT,
            )
        )

    @cached_property
    def cashflow_dis(self) -> pd.Series:
        return self.cashflow * self.discount_factors

    @cached_property
    def dis(self) -> float:
        return sum(self.cashflow_dis)

    @cached_property
    def ann(self) -> float:
        return annuity(
            present_value=self.dis,
            observation_horizon=self._eco.project_duration,
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
        cashflow = pd.Series(np.full(self._eco.project_duration + 1, self.yrl))
        cashflow.iloc[-1] = 0.0
        return cashflow


class FlowEvaluator(YearlyEvaluator, ABC):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: FlowParams,
        flow: pd.Series,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

        self.flow = flow

    @classmethod
    def create_from_plain(cls, name: str, eco: EcoParams, spec: Path | float, fix: float, flow: pd.Series) -> Self:
        params = FlowParams.create_from_plain(spec=spec, fix=fix, dti_sim=eco.dti_sim)
        return cls(name=name, eco=eco, params=params, flow=flow)

    @cached_property
    def spec_ep(self):
        # calculate annuity due factor to compensate operation costs for difference between simulation and project time
        factor_operation_ep = (1 / self._eco.eval2yr_ratio) if self._eco.compensate_sim_prj else 1

        return self._params.spec * factor_operation_ep

    @cached_property
    def eval(self) -> float:
        return (
            float(np.dot(self._params.spec[self._eco.dti_eval].to_numpy(), self.flow[self._eco.dti_eval].to_numpy()))
            * self._eco.timestep
        )

    @cached_property
    def yrl(self) -> float:
        return self.eval * self._eco.eval2yr_ratio + self._params.fix


class CapexEvaluator(CostEvaluator, CapexElement):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CapexParams,
        size_preexisting: float,
        size_expansion: float,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

        self.size_preexisting = size_preexisting
        self.size_expansion = size_expansion

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
        size_preexisting: float,
        size_expansion: float,
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
        return cls(name=name, eco=eco, params=params, size_preexisting=size_preexisting, size_expansion=size_expansion)

    @cached_property
    def spec_ep(self) -> float:
        factor_annuity = (
            annuity(
                present_value=1,
                observation_horizon=self._eco.project_duration,
                discount_rate=self._eco.discount_rate,
                occurs_at=self._OCCURS_AT,
            )
            if self._eco.compensate_sim_prj
            else 1
        )
        factor_spec_ep = float(
            np.dot(
                self.cashflow_factor_expansion.to_numpy() * self._params.spec.to_numpy(),
                self.discount_factors.to_numpy(),
            )
        )

        return factor_annuity * factor_spec_ep

    @cached_property
    def preexisting(self) -> float:
        return (self.size_preexisting * self._params.spec + self._params.fix) * int(self._params.consider_preexisting)

    @cached_property
    def expansion(self) -> float:
        return self.size_expansion * self._params.spec + self._params.fix

    @cached_property
    def init(self) -> float:
        return self.preexisting + self.expansion

    def calc_capex_cashflow(self, invest_first: int) -> pd.Series:
        invest_periods = np.arange(invest_first, self._eco.project_duration, self._params.ls)

        capex = np.zeros(self._eco.project_duration + 1, dtype=float)
        capex[invest_periods] = 1

        if self._params.residual_at_ls > 0:
            residual_periods = invest_periods[1:] if self._params.consider_preexisting else invest_periods
            capex[residual_periods] -= self._params.residual_at_ls

        capex[-1] = -1 * calc_residual_value(
            lifetime_remaining_frac=calc_lifetime_remaining(
                project_duration=self._eco.project_duration,
                ls=self._params.ls,
                init_age=invest_first % self._params.ls,
            )
            / float(self._params.ls),
            depreciation=DEPRECIATION.LINEAR,
            residual_at_ls=self._params.residual_at_ls,
        )

        # apply capex cost change ratio
        capex *= self._params.ccr ** np.arange(self._eco.project_duration + 1)

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
        capex = self.cashflow_factor_preexisting * (
            self._params.spec * self.size_preexisting + self._params.fix
        ) + self.cashflow_factor_expansion * (self._params.spec * self.size_expansion + self._params.fix)

        return capex


class MntexEvaluator(YearlyEvaluator):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: MntexParams,
        size_preexisting: float,
        size_expansion: float,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

        self.size_preexisting = size_preexisting
        self.size_expansion = size_expansion

    @classmethod
    def create_from_plain(
        cls, name: str, eco: EcoParams, spec: float, fix: float, size_preexisting: float, size_expansion: float
    ) -> Self:
        params = MntexParams(spec=spec, fix=fix)
        return cls(name=name, eco=eco, params=params, size_preexisting=size_preexisting, size_expansion=size_expansion)

    @cached_property
    def spec_ep(self) -> float:
        factor_annuity = (
            annuity(
                present_value=1,
                observation_horizon=self._eco.project_duration,
                discount_rate=self._eco.discount_rate,
                occurs_at=self._OCCURS_AT,
            )
            if self._eco.compensate_sim_prj
            else 1
        )
        factor_spec_ep = float(
            np.dot(np.full(self._eco.project_duration + 1, self._params.spec), self.discount_factors.to_numpy())
        )

        return factor_annuity * factor_spec_ep

    @cached_property
    def yrl(self) -> float:
        return (self.size_preexisting + self.size_expansion) * self._params.spec + self._params.fix


class OpexEvaluator(FlowEvaluator):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: OpexParams,
        flow: pd.Series,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            flow=flow,
        )


class CrevEvaluator(FlowEvaluator):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CrevParams,
        flow: pd.Series,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            flow=flow,
        )


@dataclass
class Evaluator(BlockElement):
    """
    EvaluatorBlock is a container for all Evaluators of a single component.
    """

    name: str

    capex: CapexEvaluator
    mntex: MntexEvaluator
    opex: OpexEvaluator
    crev: CrevEvaluator

    @classmethod
    def create_from_plain(
        cls,
        name: str,
        eco: EcoParams,
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
        size_preexisting: float = 0.0,
        size_expansion: float = 0.0,
        flow: pd.Series = None,
    ) -> Self:
        ls = eco.project_duration if ls is None else ls

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
            size_preexisting=size_preexisting,
            size_expansion=size_expansion,
        )

        mntex = MntexEvaluator.create_from_plain(
            name=f"{name}_mntex",
            eco=eco,
            spec=mntex_spec,
            fix=mntex_fix,
            size_preexisting=size_preexisting,
            size_expansion=size_expansion,
        )

        opex = OpexEvaluator.create_from_plain(
            name=f"{name}_opex",
            eco=eco,
            spec=opex_spec,
            fix=opex_fix,
            flow=flow,
        )

        crev = CrevEvaluator.create_from_plain(name=f"{name}_crev", eco=eco, spec=crev_spec, fix=crev_fix, flow=flow)

        return cls(
            name=name,
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

    def set_results(
        self,
        size_preexisting: float = None,
        size_expansion: float = None,
        flow: pd.Series = None,
    ) -> None:
        """
        This method feeds the simulation results to the evaluators for further calculation
        """
        if size_preexisting is not None:
            self.capex.size_preexisting = size_preexisting
            self.mntex.size_preexisting = size_preexisting
        if size_expansion is not None:
            self.capex.size_preexisting = size_expansion
            self.mntex.size_preexisting = size_expansion
        if flow is not None:
            self.opex.flow = flow
            self.crev.flow = flow
