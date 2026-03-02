from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .abstractclasses import (
    BlockElement,
    CalculableBaseElement,
    CalculableTimeseriesElement,
    CalculableYearlyElement,
    CapexElement,
    CrevElement,
    MntexElement,
    OpexElement,
)
from .params import (
    CapexParams,
    CostParams,
    CrevParams,
    EcoParams,
    MntexParams,
    OpexParams,
    TimeseriesParams,
)
from .utils import (
    Depreciation,
    OccursAt,
    calc_lifetime_remaining,
    calc_residual_value,
    transform_scalar_var,
)


class CostEvaluator(CalculableBaseElement, ABC):
    """
    Base class for all Evaluators that evaluate costs, i.e. Capex, Mntex, Opex and Crev.
    """

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        **kwargs,
    ):
        super().__init__(name=name, eco=eco, **kwargs)
        self.eco = eco

    @classmethod
    @abstractmethod
    def _build_kwargs_from_params(cls, params: CostParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict: ...

    @classmethod
    def create_from_params(cls, name: str, eco: EcoParams, params: CostParams, data_dir: Path, **kwargs) -> Self:
        kwargs = cls._build_kwargs_from_params(params=params, eco=eco, data_dir=data_dir, **kwargs)
        return cls(name=name, eco=eco, **kwargs)

    def _calc_ep_factor(self, cashflow_factors: npt.NDArray, **kwargs) -> float:
        return (
            np.dot(cashflow_factors, self.eco.discount_factors(self._TYPE.value.occurs_at))
            * self.eco.annuity_factor_apriori(OccursAt.BEGIN)
            if self.eco.compensate_sim_prj
            else 1.0
        )

    @abstractmethod
    def _calc_spec_ep(self, **kwargs) -> float | pd.Series: ...

    @property
    def spec_ep(self) -> float | pd.Series:
        return self._calc_spec_ep()


class TimeseriesEvaluator(CostEvaluator, CalculableTimeseriesElement, ABC):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec_power: pd.Series,
        spec_dist: pd.Series,
        spec_time: pd.Series,
        fix: float,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            **kwargs,
        )

        self.spec_power = spec_power
        self.spec_dist = spec_dist
        self.spec_time = spec_time
        self.fix = fix

    @classmethod
    def _build_kwargs_from_params(cls, params: TimeseriesParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(
            spec_power=transform_scalar_var(value=params.spec_power, dti=eco.dti_sim, data_dir=data_dir),
            spec_dist=transform_scalar_var(value=params.spec_dist, dti=eco.dti_sim, data_dir=data_dir),
            spec_time=transform_scalar_var(value=params.spec_time, dti=eco.dti_sim, data_dir=data_dir),
            fix=params.fix,
            **kwargs,
        )

    def _calc_eval(self, power: pd.Series | None, dist: pd.Series | None, time: pd.Series | None, **kwargs) -> float:
        # ToDo: check this out for reusability and avoid transforming a scalar to an array if not necessary
        #  make sure to also fix the spec_ep calculation in this case -> can also be scalar for oemof input
        """
        if flow is None:
            cost_flow = 0.0
        else:
            # Extract flow values as a NumPy array
            flow_values = flow[self.eco.dti_eval].to_numpy()

            # Handle spec being an array or a scalar
            if isinstance(self.spec, npt.NDArray):
                cost_flow = np.dot(self.spec.to_numpy(), flow_values)
            else:
                cost_flow = flow_values.sum() * self.spec

            # Scale by timestep
            cost_flow *= self.eco.timestep_hours

        # Add fixed cost
        return cost_flow + self.fix
        """

        cost_power = (
            np.dot(self.spec_power.to_numpy(), power[self.eco.dti_eval].to_numpy()) * self.eco.timestep_hours
            if power is not None
            else 0.0
        )

        cost_dist = np.dot(self.spec_dist.to_numpy(), dist[self.eco.dti_eval].to_numpy()) if dist is not None else 0.0

        cost_time = (
            np.dot(self.spec_time.to_numpy(), time[self.eco.dti_eval].to_numpy()) * self.eco.timestep_hours
            if time is not None
            else 0.0
        )

        return cost_power + cost_dist + cost_time + self.fix

    def evaluate(self, power: pd.Series | None, dist: pd.Series | None = None, time: pd.Series | None = None, **kwargs):
        super().evaluate(power=power, dist=dist, time=time, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> pd.Series:
        factor_ep = self._calc_ep_factor(cashflow_factors=self.cashflow_factors / self.eco.sim_yr_rat)

        return self.spec_power * factor_ep


class CapexEvaluator(CostEvaluator, CapexElement):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec: float,
        fix: float,
        consider_preexisting: bool,
        ls: int,
        age_preexisting: int,
        ccr: float,
        residual_at_ls: float,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            **kwargs,
        )

        self.spec = spec
        self.fix = fix
        self.consider_preexisting = consider_preexisting
        self.ls = ls
        self.age_preexisting = age_preexisting
        self.ccr = ccr
        self.residual_at_ls = residual_at_ls

        if self.consider_preexisting and self.age_preexisting != 0:
            raise ValueError(f"If consider_preexisting is True, age_preexisting must be 0, got {self.age_preexisting}")
        if self.age_preexisting >= self.ls:
            raise ValueError(
                f"age_preexisting must be smaller than ls, got age_preexisting={self.age_preexisting} and ls={self.ls}"
            )

    @classmethod
    def _build_kwargs_from_params(cls, params: CapexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(
            spec=params.spec,
            fix=params.fix,
            consider_preexisting=params.consider_preexisting,
            ls=params.ls if params.ls is not None else eco.prj_duration_yrs,
            age_preexisting=params.age_preexisting,
            ccr=params.ccr,
            residual_at_ls=params.residual_at_ls,
            **kwargs,
        )

    def _calc_preexisting(self, size_preexisting: float | None) -> float:
        cost_size = size_preexisting * self.spec if size_preexisting else 0.0
        return (cost_size + self.fix) * int(self.consider_preexisting)

    def _calc_expansion(self, size_expansion: float | None) -> float:
        return size_expansion * self.spec if size_expansion else 0.0

    def _calc_init(self) -> float:
        return self.preexisting + self.expansion

    def _calc_cashflow_factors(self, invest_first: int, **kwargs) -> npt.NDArray:
        invest_periods = np.arange(invest_first, self.eco.prj_duration_yrs, self.ls)

        capex = np.zeros(self.eco.prj_duration_yrs + 1, dtype=float)
        capex[invest_periods] = 1

        if self.residual_at_ls > 0:
            residual_periods = invest_periods[1:] if self.consider_preexisting else invest_periods
            capex[residual_periods] -= self.residual_at_ls

        capex[-1] = -1 * calc_residual_value(
            lifetime_remaining_frac=calc_lifetime_remaining(
                project_duration=self.eco.prj_duration_yrs,
                ls=self.ls,
                init_age=invest_first % self.ls,
            )
            / float(self.ls),
            depreciation=Depreciation.LINEAR,
            residual_at_ls=self.residual_at_ls,
        )

        # apply capex cost change ratio
        capex *= self.ccr ** np.arange(self.eco.prj_duration_yrs + 1)

        return capex

    def _calc_cashflow_factor_preexisting(self) -> npt.NDArray:
        invest_first = 0 if self.consider_preexisting else self.ls - self.age_preexisting
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_factor_expansion(self) -> npt.NDArray:
        invest_first = 0
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_preexisting(self, size_preexisting: float | None) -> npt.NDArray:
        cost_size = size_preexisting * self.spec if size_preexisting else 0.0
        return self._calc_cashflow_factor_preexisting() * (cost_size + self.fix)

    def _calc_cashflow_expansion(self, size_expansion: float | None) -> npt.NDArray:
        cost_size = size_expansion * self.spec if size_expansion else 0.0
        return self._calc_cashflow_factor_expansion() * cost_size

    def _calc_cashflow(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> npt.NDArray:
        return self._calc_cashflow_preexisting(size_preexisting) + self._calc_cashflow_expansion(size_expansion)

    def evaluate(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> None:
        self._preexisting = self._calc_preexisting(size_preexisting=size_preexisting)
        self._expansion = self._calc_expansion(size_expansion=size_expansion)
        self._init = self._calc_init()
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> float:
        factor_ep = self._calc_ep_factor(cashflow_factors=self._calc_cashflow_factor_expansion())

        return self.spec * factor_ep

    def get_preexisting(self, size_preexisting: float | None) -> float:
        return self._calc_preexisting(size_preexisting=size_preexisting)


class MntexEvaluator(CostEvaluator, CalculableYearlyElement, MntexElement):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec: float,
        fix: float,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            **kwargs,
        )

        self.spec = spec
        self.fix = fix

    @classmethod
    def _build_kwargs_from_params(cls, params: MntexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(spec=params.spec, fix=params.fix, **kwargs)

    def _calc_yrl(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> float:
        size_preexisting = size_preexisting if size_preexisting else 0.0
        size_expansion = size_expansion if size_expansion else 0.0
        return self.spec * (size_preexisting + size_expansion) + self.fix

    def evaluate(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> None:
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> float:
        factor_ep = self._calc_ep_factor(cashflow_factors=self.cashflow_factors)

        return self.spec * factor_ep


class OpexEvaluator(TimeseriesEvaluator, OpexElement):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec_power: pd.Series,
        spec_dist: pd.Series,
        spec_time: pd.Series,
        fix: float,
        spec_peak: float,
        frac_peak: float,
        n_peak_periods_yr: int,
        n_peak_periods_sim: int,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            spec_power=spec_power,
            spec_dist=spec_dist,
            spec_time=spec_time,
            fix=fix,
            **kwargs,
        )

        self.spec_peak = spec_peak
        self.frac_peak = frac_peak
        self.n_peak_periods_yr = n_peak_periods_yr
        self.n_peak_periods_sim = n_peak_periods_sim

    @classmethod
    def _build_kwargs_from_params(cls, params: OpexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(
            **super()._build_kwargs_from_params(params=params, eco=eco, data_dir=data_dir, **kwargs),
            spec_peak=params.spec_peak,
            frac_peak=params.frac_peak,
            n_peak_periods_yr=params.n_peak_periods_yr,
            n_peak_periods_sim=params.n_peak_periods_sim,
        )

    def _calc_spec_ep_peak(self, **kwargs) -> float:
        factor_ep = self._calc_ep_factor(cashflow_factors=self.cashflow_factors / self.eco.sim_yr_rat)

        return self.spec_peak * self.frac_peak * factor_ep

    @property
    def spec_ep_peak(self) -> float:
        return self._calc_spec_ep_peak()

    def _calc_eval(
        self,
        power: pd.Series | None,
        dist: pd.Series | None,
        time: pd.Series | None,
        power_peak: float | None = None,
        period_frac: float | None = None,
        **kwargs,
    ) -> float:
        cost = super()._calc_eval(power=power, dist=dist, time=time, **kwargs)
        cost_peak = power_peak * self.spec_peak * period_frac if power_peak is not None else 0.0

        return cost + cost_peak

    def evaluate(
        self,
        power: pd.Series | None,
        dist: pd.Series | None = None,
        time: pd.Series | None = None,
        power_peak: float | None = None,
        period_frac: float | None = None,
        **kwargs,
    ) -> None:
        super().evaluate(power=power, dist=dist, time=time, power_peak=power_peak, period_frac=period_frac, **kwargs)


class CrevEvaluator(TimeseriesEvaluator, CrevElement):
    pass


@dataclass
class POI(BlockElement):
    """
    EvaluatorBlock is a container for all Evaluators of a single component.
    """

    name: str

    eco: EcoParams

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
        name_size: str | None = None,
        name_flow: str | None = None,
        capex: CapexParams | None = None,
        mntex: MntexParams | None = None,
        opex: OpexParams | None = None,
        crev: CrevParams | None = None,
    ) -> Self:
        return cls(
            name=name,
            eco=eco,
            name_size=name_size,
            name_flow=name_flow,
            capex=CapexEvaluator.create_from_params(name, eco, capex, data_dir) if capex else None,
            mntex=MntexEvaluator.create_from_params(name, eco, mntex, data_dir) if mntex else None,
            opex=OpexEvaluator.create_from_params(name, eco, opex, data_dir) if opex else None,
            crev=CrevEvaluator.create_from_params(name, eco, crev, data_dir) if crev else None,
        )

    @property
    def spec_ep_invest(self):
        """
        Equivalent present specific costs for investments (cost per size)
        """
        return sum(component.spec_ep for component in (self.capex, self.mntex) if component is not None)

    @property
    def spec_ep_operation(self):
        """
        Equivalent present specific costs for operation (cost per energy)
        """
        return self.opex.spec_ep if self.opex is not None else 0.0

    @property
    def spec_ep_peak(self) -> float:
        """
        Equivalent present specific costs for peak power (cost per power)
        """
        return self.opex.spec_ep_peak if self.opex is not None else 0.0

    def evaluate(
        self,
        size_preexisting: float | None = None,
        size_expansion: float | None = None,
        power: pd.Series | None = None,
        dist: pd.Series | None = None,
        time: pd.Series | None = None,
        power_peak: float | None = None,
        **kwargs,
    ) -> None:
        if time is not None:
            time = time.astype(bool).astype(int)  # ensure that time is a binary indicator (1 > 0, else 0)

        for attr in (self.capex, self.mntex, self.opex, self.crev):
            if attr:
                attr.evaluate(
                    size_preexisting=size_preexisting,
                    size_expansion=size_expansion,
                    power=power,
                    dist=dist,
                    time=time,
                    power_peak=power_peak,
                    **kwargs,
                )

    def get_invest_preexisting(self, size_preexisting: float | None) -> float:
        return self.capex.get_preexisting(size_preexisting=size_preexisting) if self.capex is not None else 0.0
