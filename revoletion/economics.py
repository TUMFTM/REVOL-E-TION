#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd

from . import utils

if TYPE_CHECKING:
    from . import blocks, simulation


class EcoTools:
    @staticmethod
    def discount(future_value: float, periods: int, discount_rate: float, occurs_at: str) -> float:
        """
        calculate the present value of a future value in some periods at a discount rate per period
        """
        q = 1 + discount_rate
        exp = {"beginning": 1, "start": 1, "bop": 1, "middle": 0.5, "mid": 0.5, "mop": 0.5, "end": 0, "eop": 0}.get(
            occurs_at, 0
        )
        present_value = future_value / (q ** (periods - exp))
        return present_value

    @staticmethod
    def acc_discount(
        nominal_value: float | pd.Series, observation_horizon: int | pd.Series, discount_rate: float, occurs_at: str
    ) -> float:
        """
        calculate the accumulated present value of a periodical, nominally repeating cashflow in the future
        (from present to the observation horizon) at a discount rate per period
        """
        q = 1 + discount_rate
        exp = {"beginning": 1, "start": 1, "bop": 1, "middle": 0.5, "mid": 0.5, "mop": 0.5, "end": 0, "eop": 0}.get(
            occurs_at, 0
        )
        discount_factor = (q**exp) * (1 - (q**-observation_horizon)) / discount_rate
        return nominal_value * discount_factor

    @staticmethod
    def annuity(present_value: float, observation_horizon: int, discount_rate: float, occurs_at: str) -> float:
        """
        calculate the annuity (the equivalent periodical, nominally recurring value to generate the same
        NPV) of a present value pv over an observation horizon at a discount rate per period. occurs_at denotes whether
        the expense or value occurs at the beginning (making the annuity an annuity due) or end of the period.
        """
        q = 1 + discount_rate
        exp = {"beginning": 1, "start": 1, "bop": 1, "middle": 0.5, "mid": 0.5, "mop": 0.5, "end": 0, "eop": 0}.get(
            occurs_at, 0
        )
        try:
            annuity = present_value * discount_rate / ((1 - (q**-observation_horizon)) * (q**exp))
        except ZeroDivisionError:  # observation_horizon = 0
            annuity = present_value / observation_horizon
        return annuity

    @staticmethod
    def reinvest_periods(lifespan: int, observation_horizon: int, include_init: bool = False) -> list:
        """
        return a list of period numbers to reinvest into a component (i.e. replace it),
        given its lifespan and the observation horizon. Initial investment is removed by default.
        """
        reinvest_periods = [period for period in range(observation_horizon) if period % lifespan == 0]
        if not include_init:
            reinvest_periods.remove(0)
        return reinvest_periods

    @staticmethod
    def calc_wacc(
        share_equity: float,  # share of equity in capital structure
        rate_debt: float,  # interest rate on debt
        rate_market: float = 0.07,  # expected return on market
        rate_riskfree: float = 0.03,  # risk-free return rate
        rate_tax: float = 0.25,  # corporate tax rate
        rate_inflation: float = 0.02,  # expected inflation rate
        volatility_relative: float = 1,  # volatility of stock price relative to market
    ) -> (float, float):
        """
        This function calculates the nominal (including inflation) weighted average cost of capital (WACC) using the
        Capital Asset Pricing Model (CAPM) for equity cost.
        """
        share_debt = 1 - share_equity
        cost_equity = rate_riskfree + volatility_relative * (rate_market - rate_riskfree)  # CAPM
        wacc_nominal = share_debt * rate_debt * (1 - rate_tax) + share_equity * cost_equity
        wacc_real = wacc_nominal * ((1 + wacc_nominal) / (1 + rate_inflation))  # fisher formula
        return wacc_nominal, wacc_real

    @staticmethod
    def transform_scalar_var(
        value: str | float | pd.Series, scenario: simulation.Scenario, block: Optional[blocks.BaseBlock] = None
    ):
        """
        Transform a value holding either the filename of a csv file containing a timeseries or a scalar
        to a pandas Series with the same DatetimeIndex as the simulation.
        """
        if isinstance(value, str):  # value contains filename
            filename = utils.set_extension(filename=value, default_extension=".csv")

            try:
                df = utils.read_timeseries_csv(
                    path_input_file=scenario.paths.input / filename,
                    scenario=scenario,
                    multiheader=False,
                    resampling=True,
                )
            except IndexError as exc:
                raise IndexError(f"Failed to load timeseries data for block {block.name}: {exc}")

            if df.shape[1] != 1:
                scenario.logger.warning(
                    f'Block "{block.name}": Input data in {filename} contains more than one column - '
                    f"only first column is used."
                )

            return df.iloc[:, 0]  # return only first column

        else:  # value is given as scalar
            return pd.Series(data=value, index=scenario.times.sim.dti)

    @staticmethod
    def calc_frac_remaining_ls(ls: int, project_duration: int) -> float:
        """
        Calculate the fraction of the remaining lifespan of a component after the project duration.
        A remaining lifespan fraction of 1 is considered as 0 as the component is not replaced anymore.
        """

        frac_remaining_ls = 1 - (project_duration % ls) / ls
        if frac_remaining_ls == 1:
            frac_remaining_ls = 0

        return frac_remaining_ls


@dataclass
class Size:
    name: Optional[str] = None
    block: Optional[blocks.BaseBlock] = None
    unit: str = ""

    # parameters that are set in __post_init__
    _preexisting: float = field(init=False, repr=False, default=0.0)
    _invest: bool = field(init=False, repr=False, default=False)

    _total_max: Optional[float] = field(init=False, repr=False, default=None)  # "Optional" is equal to "float | None"

    # parameters that are set after optimization
    _expansion: float = field(default=0.0, init=False, repr=False)  # set after optimization, initialized with 0.0

    def __post_init__(self):
        if self.name and self.block:  # name and block are None for default size object
            self.preexisting = self._get_param(param="size_preexisting", default=self.preexisting)

            self.invest = self._get_param(param="invest", default=self.invest)

            self.total_max = self._get_param(param="size_max", default=self.total_max)

    def _get_param(self, param: str, default: Any) -> Any:
        """
        Get a parameter from the block's input data.
        """
        name_param = f"{param}_{self.name}"
        value = getattr(self.block, name_param, default)
        # ToDo: implement removal of parameters from block (also ls and ccr)
        # if hasattr(self.block, name_param):
        #     delattr(self.block, name_param)
        return value

    @property
    def preexisting(self) -> float:
        return self._preexisting

    @preexisting.setter
    def preexisting(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError(f"preexisting must be numeric (int or float), got {type(value).__name__}")
        if value < 0:
            raise ValueError("preexisting cannot be negative")
        self._preexisting = float(value)

    @property
    def invest(self) -> bool:
        return self._invest

    @invest.setter
    def invest(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"invest must be a boolean, got {type(value).__name__}")
        self._invest = value

    @property
    def total_max(self) -> Optional[float]:
        return self._total_max

    @total_max.setter
    def total_max(self, value: Optional[float]):
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError(f"size_max must be numeric (int or float) or None, got {type(value).__name__}")
        if isinstance(value, (int, float)) and (value < self.preexisting):
            raise ValueError("total_max cannot be smaller than preexisting size")
        self._total_max = float(value) if value is not None else None

    @property
    def expansion_max(self) -> Optional[float]:
        if not self.invest:  # no investment allowed -> expansion_max is 0
            return 0
        elif self.total_max is not None:  # invest == True and limit for total size is set -> calculate expansion_max
            return self.total_max - self.preexisting
        else:  # invest == True and no limit for total size is set -> expansion_max is None (=unlimited)
            return None

    @property
    def expansion(self) -> float:
        return self._expansion

    @expansion.setter
    def expansion(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError(f"expansion must be numeric (int or float), got {type(value).__name__}")
        if value < 0:
            raise ValueError("expansion cannot be negative")
        self._expansion = float(value)

    @property
    def total(self) -> float:
        return self.preexisting + self.expansion

    @property
    def result_summary(self) -> pd.Series:
        """
        Create a pd.Series with the size's attributes for result_summary.
        """
        if self.name and self.block:  # name and block are None for default size object
            return pd.Series(
                {
                    f"size_{self.name}_preexisting": self.preexisting,
                    f"size_{self.name}_invest": self.invest,
                    f"size_{self.name}_total_max": self.total_max,
                    f"size_{self.name}_expansion_max": self.expansion_max,
                    f"size_{self.name}_expansion": self.expansion,
                    f"size_{self.name}_total": self.total,
                }
            )
        else:
            return pd.Series()

    @property
    def result_msg(self) -> str:
        """
        Create a message string for result_messages.
        """
        return (
            f'Optimized size of component "{self.name}" in block "{self.block.name}": '
            f"{self.total / 1e3:.1f} {self.unit} "
            f"(existing: {self.preexisting / 1e3:.1f} {self.unit} - "
            f"expansion: {self.expansion / 1e3:.1f} {self.unit})"
            if self.invest
            else ""
        )


@dataclass
class OptimizationConverter:
    poi: EcoEvaluator

    def __post_init__(self):
        # calculate annuity due factor to compensate investment costs for difference between simulation and project time
        self.factor_ep_invest: float = (
            EcoTools.annuity(
                present_value=1,
                observation_horizon=self.poi.scenario.prj_duration_yrs,
                discount_rate=self.poi.scenario.wacc,
                occurs_at="beginning",
            )
            if self.poi.scenario.compensate_sim_prj
            else 1
        )

        # calculate specific present value of investment (addition of capex and mntex) cost
        # join maintenance and capex specific present values for the project duration
        self.spec_ep_invest: float = (
            self._get_spec_prj_ep_capex() + self._get_spec_prj_ep_mntex()
        ) * self.factor_ep_invest

        # calculate annuity due factor to compensate operation costs for difference between simulation and project time
        self.factor_ep_operation: float = (
            (1 / self.poi.scenario.sim_yr_rat) if self.poi.scenario.compensate_sim_prj else 1
        )

        # calculate specific present value of operation cost
        self.spec_ep_operation: float = (
            self.poi.opex.spec * self.factor_ep_operation if self.poi.opex else 0.0
        )  # default value if opex is not defined

    def _get_spec_prj_ep_capex(self) -> float:
        """
        Calculate the specific present value of capex for the project duration.
        """

        # default value if capex is not defined
        if not self.poi.capex:
            return 0.0

        spec_prj_ep = np.array([0.0] * len(self.poi.discount_factors.index), dtype=float)

        # apply specific capex for replacement periods
        spec_prj_ep[
            EcoTools.reinvest_periods(
                lifespan=self.poi.ls, observation_horizon=self.poi.scenario.prj_duration_yrs, include_init=True
            )
        ] = self.poi.capex.spec

        # apply specific salvage value after project duration considering the remaining lifespan
        # salvage values occur at the end of the last year of the project duration but are modeled at the beginning of
        # the next year to use the same discount factor ('beginning') and avoid issues when a replacement occurs at the
        # beginning of the last project year
        spec_prj_ep[self.poi.scenario.prj_duration_yrs] = (
            -1
            * self.poi.capex.spec
            * EcoTools.calc_frac_remaining_ls(ls=self.poi.ls, project_duration=self.poi.scenario.prj_duration_yrs)
        )

        # adjust specific capex by appropriate cost change ratio
        spec_prj_ep *= self.poi.ccr**self.poi.discount_factors.index

        # sum up all specific discounted capex for the project duration
        spec_prj_ep = spec_prj_ep @ self.poi.discount_factors["beginning"]

        return spec_prj_ep

    def _get_spec_prj_ep_mntex(self) -> float:
        # calculate specific present value of mntex for the project duration
        return (
            EcoTools.acc_discount(
                nominal_value=self.poi.mntex.spec,
                observation_horizon=self.poi.scenario.prj_duration_yrs,
                discount_rate=self.poi.scenario.wacc,
                occurs_at="beginning",
            )
            if self.poi.mntex
            else 0.0
        )  # default value if mntex is not defined


@dataclass
class PeakOptimizationConverter(OptimizationConverter):
    def __post_init__(self):
        super().__post_init__()

        # calculate annuity due factor to compensate peak opex for difference between simulation and project time
        self.factor_ep_peak = (
            self.poi.block.n_peak_periods_yr / self.poi.block.peak_periods.shape[0]
            if self.poi.scenario.compensate_sim_prj
            else 1
        )

        # calculate specific present value of peak power operation cost
        self.spec_ep_peak = self.poi.opex_peak.spec_peak * self.factor_ep_peak


@dataclass
class CostAggregator(ABC):
    poi: EcoPOI

    prj: float = field(init=False, repr=False, default=0)

    dis: float = field(init=False, repr=False, default=0)

    ann: float = field(init=False, repr=False, default=0)

    _cashflows: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.cashflows = np.array([0.0] * len(self.poi.discount_factors.index), dtype=float)

    @property
    @abstractmethod
    def cost_type(self) -> str:
        # Return the type of cost (e.g. 'capex', 'mntex', 'opex', 'crev', 'totex', 'value')
        # This is used to identify the cost type for aggregation and result_series generation
        ...

    @property
    @abstractmethod
    def attr_list_expansion(self) -> list[str]:
        # Return a list of attributes expanding the default list for aggregation and result_series generation
        ...

    @property
    def attr_list(self) -> list[str]:
        # List of attributes to be aggregated and returned in result_series
        return ["prj", "dis", "ann", "cashflows"] + self.attr_list_expansion

    @property
    def aggregator(self) -> CostAggregator:
        # Get the aggregator based on the cost type
        return getattr(self.poi.aggregator, self.cost_type) if self.poi.aggregator else None

    @property
    def cashflows(self) -> np.ndarray:
        return self._cashflows

    @cashflows.setter
    def cashflows(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"cashflows must be a numpy array, got {type(value).__name__}")
        if value.dtype != float:
            raise ValueError(f"cashflows must be of type float, got {value.dtype}")
        self._cashflows = value

    def aggregate(self):
        if self.aggregator:
            for attr in self.attr_list:
                setattr(
                    self.aggregator,
                    attr,
                    getattr(self.aggregator, attr) + getattr(self, attr),
                )

    @property
    def result_series(self) -> pd.Series:
        return pd.Series(
            {
                f"{self.cost_type}_{attr}": getattr(self, attr)
                for attr in self.attr_list
                if isinstance(getattr(self, attr), (str, int, float, bool))
            }
        )


@dataclass
class CostEvaluator(CostAggregator):
    @property
    @abstractmethod
    def occurs_at(self) -> str:
        """
        Return the occurrence of the cost (e.g. 'beginning', 'end', 'middle', etc.)
        This is used to determine how the cashflows are discounted.
        """
        ...

    @property
    def prj(self) -> float:
        return self.cashflows.sum()

    @property
    def dis(self) -> float:
        return self.cashflows @ self.poi.discount_factors[self.occurs_at]

    @property
    def ann(self) -> float:
        return EcoTools.annuity(
            present_value=self.dis,
            observation_horizon=self.poi.scenario.prj_duration_yrs,
            discount_rate=self.poi.scenario.wacc,
            occurs_at=self.occurs_at,
        )


@dataclass
class CapexAggregator(CostAggregator):
    poi: EcoPOI

    # all attributes have to be initialized with 0 and calculated
    preexisting: float = field(init=False, repr=False, default=0.0)

    expansion: float = field(init=False, repr=False, default=0.0)

    init: float = field(init=False, repr=False, default=0.0)

    replacement: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return "capex"

    @property
    def attr_list_expansion(self) -> list[str]:
        return ["preexisting", "expansion", "init", "replacement"]


@dataclass
class CapexEvaluator(CapexAggregator, CostEvaluator):
    poi: EcoEvaluator

    consider_preexisting: bool = field(init=True, repr=False, default=True)

    spec: float = field(init=True, repr=False, default=0.0)

    fix: float = field(init=True, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()

        self.poi.scenario.capex_preexisting_considered += self.preexisting

    @property
    def occurs_at(self) -> str:
        return "beginning"

    @property
    def preexisting(self) -> float:
        return int(self.consider_preexisting) * self.poi.size.preexisting * self.spec + self.fix

    @property
    def expansion(self) -> float:
        return self.poi.size.expansion * self.spec

    @property
    def init(self) -> float:
        return self.preexisting + self.expansion

    @property
    def replacement(self) -> float:
        return self.poi.size.total * self.spec + self.fix

    @CapexAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index), dtype=float)

        cashflows[0] += self.init

        for period in EcoTools.reinvest_periods(
            lifespan=self.poi.ls, observation_horizon=self.poi.scenario.prj_duration_yrs, include_init=False
        ):
            cashflows[period] += self.replacement * (self.poi.ccr**period)

        # Subtract salvage value capex (negative capex)
        cashflows[self.poi.scenario.prj_duration_yrs] -= (
            self.replacement
            * (self.poi.ccr**self.poi.scenario.prj_duration_yrs)
            * EcoTools.calc_frac_remaining_ls(ls=self.poi.ls, project_duration=self.poi.scenario.prj_duration_yrs)
        )

        return cashflows


@dataclass
class MntexAggregator(CostAggregator):
    poi: EcoPOI

    sim: float = field(init=False, repr=False, default=0.0)

    yrl: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return "mntex"

    @property
    def attr_list_expansion(self) -> list[str]:
        return ["sim", "yrl"]


@dataclass
class MntexEvaluator(MntexAggregator, CostEvaluator):
    poi: EcoEvaluator

    spec: float = field(init=True, repr=False, default=0.0)

    fix: float = field(init=True, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def occurs_at(self) -> str:
        return "beginning"

    @property
    def yrl(self) -> float:
        return self.poi.size.total * self.spec + self.fix

    @property
    def sim(self) -> float:
        return self.yrl * self.poi.scenario.sim_yr_rat

    @MntexAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index), dtype=float)
        cashflows[self.poi.scenario.periods_prj] = self.yrl
        return cashflows


@dataclass
class OpexAggregator(CostAggregator):
    poi: EcoPOI

    sim: float = field(init=False, repr=False, default=0.0)

    yrl: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return "opex"

    @property
    def attr_list_expansion(self) -> list[str]:
        return ["sim", "yrl"]


@dataclass
class OpexEvaluator(OpexAggregator, CostEvaluator):
    poi: EcoEvaluator

    spec: str | float | pd.Series = field(init=True, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()
        self.spec = EcoTools.transform_scalar_var(value=self.spec, scenario=self.poi.scenario, block=self.poi.block)

    @property
    def occurs_at(self) -> str:
        return "end"

    @property
    def sim(self) -> float:
        return self.poi.flow @ self.spec[self.poi.scenario.times.eval.dti] * self.poi.scenario.timestep.hours

    @property
    def yrl(self) -> float:
        return self.sim / self.poi.scenario.sim_yr_rat

    @OpexAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index), dtype=float)
        cashflows[self.poi.scenario.periods_prj] = self.yrl
        return cashflows


@dataclass
class FleetUnitOpexEvaluator(OpexEvaluator):
    dist: str | float | pd.Series = field(init=True, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()
        self.dist = EcoTools.transform_scalar_var(value=self.dist, scenario=self.poi.scenario, block=self.poi.block)

    @property
    def sim(self) -> float:
        sim = super().sim
        sim += (
            self.poi.block.log.loc[self.poi.scenario.times.eval.dti, "dist"]
            @ self.dist[self.poi.scenario.times.eval.dti]
        )
        return sim


@dataclass
class PeakOpexEvaluator(OpexEvaluator):
    spec_peak: float = field(init=True, default=0.0)

    @property
    def sim(self) -> float:
        sim = super().sim
        sim += (
            self.poi.block.peak_periods.loc[self.poi.name, "power"]
            * self.spec_peak
            * self.poi.block.peak_periods.loc[self.poi.name, "period_fraction"]
        )
        return sim


@dataclass
class CrevAggregator(CostAggregator):
    poi: EcoPOI

    sim: float = field(init=False, repr=False, default=0.0)

    yrl: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return "crev"

    @property
    def attr_list_expansion(self) -> list[str]:
        return ["sim", "yrl"]


@dataclass
class CrevEvaluator(CrevAggregator, CostEvaluator):
    poi: EcoEvaluator

    spec: str | float | pd.Series = field(init=True, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()
        self.spec = EcoTools.transform_scalar_var(value=self.spec, scenario=self.poi.scenario, block=self.poi.block)

    @property
    def occurs_at(self) -> str:
        return "end"

    @property
    def sim(self) -> float:
        return self.poi.flow @ self.spec[self.poi.scenario.times.eval.dti] * self.poi.scenario.timestep.hours

    @property
    def yrl(self) -> float:
        return self.sim / self.poi.scenario.sim_yr_rat

    @CrevAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index), dtype=float)
        cashflows[self.poi.scenario.periods_prj] = self.yrl
        return cashflows


@dataclass
class FleetUnitCrevEvaluator(CrevEvaluator):
    dist: str | float | pd.Series = field(init=True, repr=False, default=0.0)

    time: str | float | pd.Series = field(init=True, repr=False, default=0.0)

    def __post_init__(self):
        super().__post_init__()
        self.dist = EcoTools.transform_scalar_var(value=self.dist, scenario=self.poi.scenario, block=self.poi.block)

        self.time = EcoTools.transform_scalar_var(value=self.time, scenario=self.poi.scenario, block=self.poi.block)

    @property
    def sim(self) -> float:
        sim = super().sim
        sim += (
            self.poi.block.log.loc[self.poi.scenario.times.eval.dti, "dist"]
            @ self.dist[self.poi.scenario.times.eval.dti]
        )

        sim += (
            ~self.poi.block.log.loc[self.poi.scenario.times.eval.dti, "atbase"]
            @ self.time[self.poi.scenario.times.eval.dti]
        ) * self.poi.scenario.timestep.hours

        return sim


@dataclass
class TotexAggregator(CostAggregator):
    poi: EcoAggregator

    @property
    def cost_type(self) -> str:
        return "totex"

    @property
    def attr_list_expansion(self) -> list[str]:
        # no additional attributes to aggregate
        return []

    def aggregate(self):
        for attr_name in ["cashflows", "prj", "dis", "ann"]:
            setattr(
                self,
                attr_name,
                # totex = capex + mntex + opex (if available)
                sum(
                    [
                        getattr(getattr(self.poi, attr), attr_name)
                        for attr in ["capex", "mntex", "opex"]
                        if getattr(self.poi, attr)
                    ]
                ),  # Start with the current value (self.<attr_name>)
            )

        super().aggregate()


@dataclass
class ValueAggregator(CostAggregator):
    poi: EcoAggregator

    @property
    def cost_type(self) -> str:
        return "value"

    @property
    def attr_list_expansion(self) -> list[str]:
        # no additional attributes to aggregate
        return []

    def aggregate(self):
        for attr_name in ["cashflows", "prj", "dis", "ann"]:
            setattr(
                self,
                attr_name,
                # value = crev - totex (if crev is available)
                (
                    getattr(self.poi.crev, attr_name) - getattr(self.poi.totex, attr_name)
                    if self.poi.crev
                    else -1 * getattr(self.poi.totex, attr_name)
                ),  #  0 as default value is not possible due to cashflows being a numpy array
            )

        super().aggregate()


@dataclass
class EcoPOI(ABC):
    name: str
    scenario: simulation.Scenario
    block: Optional[blocks.BaseBlock] = None

    # Initialize in __post_init__()
    capex: Optional[CapexAggregator | CapexEvaluator] = field(
        init=False,
    )
    mntex: Optional[MntexAggregator | MntexEvaluator] = field(
        init=False,
    )
    opex: Optional[OpexAggregator | OpexEvaluator] = field(
        init=False,
    )
    crev: Optional[CrevAggregator | CrevEvaluator] = field(
        init=False,
    )

    @abstractmethod
    def __post_init__(self):
        # Initialize capex, mntex, opex, crev (totex, value) with correct class (Aggregator or Evaluator) here
        ...

    @property
    def discount_factors(self) -> pd.DataFrame:
        return self.scenario.discount_factors

    @property
    @abstractmethod
    def attr2agg_additional(self) -> list[str]:
        # get a list of names of additional attributes which are to be aggregated or written in result_series
        ...

    @property
    def attr2agg(self) -> list[str]:
        # get a list of names of all attributes which are to be aggregated or written in result_series
        return ["capex", "mntex", "opex", "crev"] + self.attr2agg_additional

    @property
    @abstractmethod
    def aggregator(self) -> EcoAggregator:
        # get aggregator based on the type of EcoPOI (EcoAggregator, EcoEvaluator)
        ...

    def aggregate(self):
        # aggregate all specified attributes if they are set (not None), which may not be the case for EcoEvaluators
        for attr in self.attr2agg:
            if getattr(self, attr):
                getattr(self, attr).aggregate()


@dataclass
class EcoAggregator(EcoPOI):
    # add totex and value as additional attributes for EcoAggregator
    totex: TotexAggregator = field(
        init=False,
    )
    value: ValueAggregator = field(
        init=False,
    )

    def __post_init__(self):
        # specify EcoPOI attributes
        self.capex = CapexAggregator(poi=self)
        self.mntex = MntexAggregator(poi=self)
        self.opex = OpexAggregator(poi=self)
        self.crev = CrevAggregator(poi=self)

        # specify additional attributes for EcoAggregator
        self.totex = TotexAggregator(poi=self)
        self.value = ValueAggregator(poi=self)

    @property
    def attr2agg_additional(self) -> List[str]:
        # additional attributes to be aggregated and written in result_series
        return ["totex", "value"]

    @property
    def aggregator(self) -> EcoAggregator:
        # EcoAggregators aggregate their results in the aggregator of the block's parent
        # The class Scenario does not have an attribute parent -> no aggregator is returned
        return self.block.parent.aggregator if hasattr(self.block, "parent") else None

    def write_result_summary(self) -> pd.Series:
        # concat the result_series of all relevant attributes (type hint required for IDE to recognize the type)
        series_list: List[pd.Series] = [getattr(self, attr).result_series for attr in self.attr2agg]
        return pd.concat(series_list)


@dataclass
class EcoEvaluator(EcoPOI):
    block: blocks.BaseBlock

    size_name: Optional[str] = field(init=True, repr=False, default=None)
    size_unit: str = field(init=True, repr=False, default="kW")
    flow_name: Optional[str] = field(init=True, repr=False, default=None)
    ls: Optional[int] = field(init=True, repr=False, default=None)
    ccr: Optional[float] = field(init=True, repr=False, default=1.0)

    capex_config: InitVar[dict] = None
    mntex_config: InitVar[dict] = None
    opex_config: InitVar[dict] = None
    crev_config: InitVar[dict] = None

    opex_config_fleetunit: InitVar[dict] = None
    crev_config_fleetunit: InitVar[dict] = None
    opex_config_peak: InitVar[dict] = None

    opt: OptimizationConverter = field(init=False, repr=False)

    def __post_init__(
        self,
        capex_config,
        mntex_config,
        opex_config,
        crev_config,
        opex_config_fleetunit,
        crev_config_fleetunit,
        opex_config_peak,
    ):
        if not self.ls:  # set default value for lifespan from scenario -> not possible in init definition
            self.ls = self.scenario.prj_duration_yrs

        if self.size_name is not None:
            if self.size_name in self.block.sizes:
                raise ValueError(f"Size with name '{self.size_name}' already exists in block '{self.block.name}'.")

            self.block.sizes[self.name] = Size(name=self.name, block=self.block, unit=self.size_unit)

        # Only add flow_name to block's flow_names if block.flow_names exist -> only ElectricBlock instances
        if self.flow_name:
            if hasattr(self.block, "flow_names"):
                self.block.flow_names.add(self.flow_name)
            else:
                raise AttributeError(
                    f"Attribute 'flow_name' was specified for EcoPOI '{self.name}' in block "
                    f"'{self.block.name}'. This block does not have an attribute 'flow_names'."
                )

        # Define Evaluators if the corresponding config is provided
        self.capex = CapexEvaluator(poi=self, **capex_config) if capex_config else None
        self.mntex = MntexEvaluator(poi=self, **mntex_config) if mntex_config else None
        self.opex = OpexEvaluator(poi=self, **opex_config) if opex_config else None
        self.crev = CrevEvaluator(poi=self, **crev_config) if crev_config else None

        # Define additional Evaluators for FleetUnit if required
        self.opex_fleetunit = (
            FleetUnitOpexEvaluator(poi=self, **opex_config_fleetunit) if opex_config_fleetunit else None
        )
        self.crev_fleetunit = (
            FleetUnitCrevEvaluator(poi=self, **crev_config_fleetunit) if crev_config_fleetunit else None
        )

        # Define additional Evaluators for PeakPower if required
        self.opex_peak = PeakOpexEvaluator(poi=self, **opex_config_peak) if opex_config_peak else None

        self.opt = OptimizationConverter(poi=self) if not opex_config_peak else PeakOptimizationConverter(poi=self)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, block={self.block!r})"

    @property
    def attr2agg_additional(self) -> list[str]:
        # additional attributes to be aggregated
        return ["opex_fleetunit", "crev_fleetunit", "opex_peak"]

    @property
    def aggregator(self) -> EcoAggregator:
        # EcoEvaluators aggregate their results in the block's aggregator
        return self.block.aggregator

    @property
    def size(self) -> Size:
        if self.size_name is None:  # no size_name given -> create a default size
            return Size()
        elif self.size_name in self.block.sizes:  # size_name given and exists in block -> return size object
            return self.block.sizes[self.size_name]
        else:  # size_name given but does not exist in block -> raise error
            raise ValueError(f"Size with name '{self.size_name}' does not exist in block '{self.block.name}'.")

    @property
    def flow(self) -> np.ndarray:
        # if flow with name self.flow_name exists in block return this flow as numpy array
        if hasattr(self.block, "flows") and self.flow_name in self.block.flows.columns:
            return self.block.flows.loc[self.scenario.times.eval.dti, self.flow_name].values
        # else return default flow array with zeros
        else:
            return np.array([0.0] * len(self.scenario.times.eval.dti), dtype=float)
