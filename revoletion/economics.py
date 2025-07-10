#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Optional, Any
from abc import ABC, abstractmethod
from . import utils


if TYPE_CHECKING:
    from . import blocks
    from . import simulation

def discount(future_value: float,
             periods: int,
             discount_rate: float,
             occurs_at: str) -> float:
    """
    calculate the present value of a future value in some periods at a discount rate per period
    """
    q = 1 + discount_rate
    exp = {'beginning': 1,
           'start': 1,
           'bop': 1,
           'middle': 0.5,
           'mid': 0.5,
           'mop': 0.5,
           'end': 0,
           'eop': 0}.get(occurs_at, 0)
    present_value = future_value / (q ** (periods - exp))
    return present_value


def acc_discount(nominal_value: float | pd.Series,
                 observation_horizon: int | pd.Series,
                 discount_rate: float,
                 occurs_at: str) -> float:
    """
    calculate the accumulated present value of a periodical, nominally repeating cashflow in the future
    (from present to the observation horizon) at a discount rate per period
    """
    q = 1 + discount_rate
    exp = {'beginning': 1,
           'start': 1,
           'bop': 1,
           'middle': 0.5,
           'mid': 0.5,
           'mop': 0.5,
           'end': 0,
           'eop': 0}.get(occurs_at, 0)
    discount_factor = (q ** exp) * (1 - (q ** -observation_horizon)) / discount_rate
    return nominal_value * discount_factor


def annuity(present_value: float,
            observation_horizon: int,
            discount_rate: float,
            occurs_at: str) -> float:
    """
    calculate the annuity (the equivalent periodical, nominally recurring value to generate the same
    NPV) of a present value pv over an observation horizon at a discount rate per period. occurs_at denotes whether
    the expense or value occurs at the beginning (making the annuity an annuity due) or end of the period.
    """
    q = 1 + discount_rate
    exp = {'beginning': 1,
           'start': 1,
           'bop': 1,
           'middle': 0.5,
           'mid': 0.5,
           'mop': 0.5,
           'end': 0,
           'eop': 0}.get(occurs_at, 0)
    try:
        annuity = present_value * discount_rate / ((1 - (q ** -observation_horizon)) * (q ** exp))
    except ZeroDivisionError:  # observation_horizon = 0
        annuity = present_value / observation_horizon
    return annuity


def reinvest_periods(lifespan: int,
                     observation_horizon: int,
                     include_init: bool = False) -> list:
    """
    return a list of period numbers to reinvest into a component (i.e. replace it),
    given its lifespan and the observation horizon. Initial investment is removed by default.
    """
    reinvest_periods = [period for period in range(observation_horizon) if period % lifespan == 0]
    if not include_init:
        reinvest_periods.remove(0)
    return reinvest_periods


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
    wacc_real = (1 + wacc_nominal) / (1 + rate_inflation)  # fisher formula
    return wacc_nominal, wacc_real


def transform_scalar_var(value: str |  float | pd.Series,
                         scenario: simulation.Scenario,
                         block=None):
    """
    Transform a value holding either the filename of a csv file containing a timeseries or a scalar
    to a pandas Series with the same DatetimeIndex as the simulation.
    """
    if isinstance(value, str):  # value contains filename
        filename = utils.set_extension(filename=value,
                                       default_extension='.csv')

        df = utils.read_timeseries_csv(path_input_file=scenario.paths.input / filename,
                                       block=block,
                                       scenario=scenario,
                                       multiheader=False,
                                       resampling=True)
        if df.shape[1] != 1:
            scenario.logger.warning(f'Block "{block.name}": Input data in {filename} contains more than one column - '
                                    f'only first column is used.')

        return df.iloc[:, 0]  # return only first column

    else:  # value is given as scalar
        return pd.Series(data=value,
                         index=scenario.dti_sim)


def calc_frac_remaining_ls(ls: int,
                           project_duration: int) -> float:
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
    name: str
    block: blocks.BaseBlock
    unit: str = 'kW'  # ToDo: pass upon initialization from block and remove workaround in __post_init__

    # parameters that are set in __post_init__
    _preexisting: float = field(init=False,
                                repr=False,
                                default=0.0)
    _invest: bool = field(init=False,
                          repr=False,
                          default=False)

    _total_max: Optional[float] = field(init=False,
                                        repr=False,
                                        default=None)  # "Optional" is equal to "float | None"

    # parameters that are set after optimization
    expansion: float = field(default=0.0,
                             init=False,
                             repr=False)  # set after optimization, initialized with 0.0

    def __post_init__(self):
        self.preexisting = self._get_param(param='size_preexisting',
                                           default=self.preexisting)

        self.invest = self._get_param(param='invest',
                                      default=self.invest)

        self.total_max = self._get_param(param='size_max',
                                         default=self.total_max)

    def _get_param(self,
                   param: str,
                   default: Any) -> Any:
        """
        Get a parameter from the block's input data.
        """
        name_param = f'{param}_{self.name}'
        value = getattr(self.block, name_param, default)
        # ToDo: remove deletion of attribute in block, if parameters are kept in pydantic model
        if hasattr(self.block, name_param):
            delattr(self.block, name_param)
        return value

    @property
    def preexisting(self) -> float:
        return self._preexisting

    @preexisting.setter
    def preexisting(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError(f"preexisting must be numeric (int or float), got {type(value).__name__}")
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
        self._total_max = float(value) if value is not None else None

    @property
    def expansion_max(self) -> Optional[float]:
        """
        Get the maximum additional investment size.
        0:      no investment           -> invest == False or size_max == size_preexisting
        float:  limited investment      -> invest == True and size_max is not None
        None:   unlimited investment    -> invest == True and size_max is None
        """
        if not self.invest:
            return 0
        else:
            if self.total_max is not None:
                return self.total_max - self.preexisting
            else:
                return None

    @property
    def total(self) -> float:
        return self.preexisting + self.expansion

    @property
    def result_summary(self) -> pd.Series:
        """
        Create a pd.Series with the size's attributes for result_summary.
        """
        return pd.Series({f'size_{self.name}_preexisting': self.preexisting,
                          f'size_{self.name}_invest': self.invest,
                          f'size_{self.name}_total_max': self.total_max,
                          f'size_{self.name}_expansion_max': self.expansion_max,
                          f'size_{self.name}_expansion': self.expansion,
                          f'size_{self.name}_total': self.total})

    @property
    def result_msg(self) -> str:
        """
        Create a message string for result_messages.
        """
        return (f'Optimized size of component "{self.name}" in block "{self.block.name}": '
                f'{self.total / 1e3:.1f} {self.unit} '
                f'(existing: {self.preexisting / 1e3:.1f} {self.unit} - '
                f'expansion: {self.expansion / 1e3:.1f} {self.unit})'
                if self.invest else '')


@dataclass
class OptimizationConverter:
    poi: EcoEvaluator
    
    # ToDo: Optimize this structure to avoid repeated calculations
    @property
    def _spec_prj_ep_capex(self) -> float:
        """
        Calculate the specific present value of capex for the project duration.
        """

        spec_prj_ep = np.array([0.0] * len(self.poi.discount_factors.index),
                               dtype=float)

        # apply specific capex for replacement periods
        spec_prj_ep[reinvest_periods(lifespan=self.poi.aux['ls'],
                                     observation_horizon=self.poi.scenario.prj_duration_yrs,
                                     include_init=True)] = self.poi.capex.spec

        # apply specific salvage value after project duration considering the remaining lifespan
        # salvage values occur at the end of the last year of the project duration but are modeled at the beginning of
        # the next year to use the same discount factor ('beginning') and avoid issues when a replacement occurs at the
        # beginning of the last project year
        spec_prj_ep[self.poi.scenario.prj_duration_yrs] = (
                -1 * self.poi.capex.spec * calc_frac_remaining_ls(ls=self.poi.aux['ls'],
                                                                  project_duration=self.poi.scenario.prj_duration_yrs)
        )

        # adjust specific capex by appropriate cost change ratio
        spec_prj_ep *= self.poi.aux['ccr'] ** self.poi.discount_factors.index

        # sum up all specific discounted capex for the project duration
        spec_prj_ep = spec_prj_ep @ self.poi.discount_factors['beginning']

        return spec_prj_ep

    @property
    def _spec_prj_ep_mntex(self) -> float:
        # calculate specific present value of mntex for the project duration
        return acc_discount(nominal_value=self.poi.mntex.spec,
                            observation_horizon=self.poi.scenario.prj_duration_yrs,
                            discount_rate=self.poi.scenario.wacc,
                            occurs_at='beginning')

    @property
    def _spec_prj_ep_invest(self) -> float:
        # join maintenance and capex specific present values for the project duration
        return self._spec_prj_ep_capex + self._spec_prj_ep_mntex

    @property
    def factor_ep_invest(self) -> float:
        # calculate annuity due factor to compensate for difference between simulation and project time
        return annuity(present_value=1,
                       observation_horizon=self.poi.scenario.prj_duration_yrs,
                       discount_rate=self.poi.scenario.wacc,
                       occurs_at='beginning') if self.poi.scenario.compensate_sim_prj else 1

    @property
    def spec_ep_invest(self) -> float:
        # calculate specific capex/mntex value used for the optimization problem
        return self._spec_prj_ep_invest * self.factor_ep_invest

    @property
    def factor_ep_operation(self) -> float:
        # calculate annuity due factor to compensate for difference between simulation and project time
        return (1 / self.poi.scenario.sim_yr_rat) if self.poi.scenario.compensate_sim_prj else 1

    @property
    def spec_ep_operation(self) -> float:
        # calculate specific capex/mntex value used for the optimization problem
        return self.poi.opex.spec * self.factor_ep_operation


@dataclass
class CostAggregator(ABC):
    poi: EcoPOI

    prj: float = field(init=False,
                       repr=False,
                       default=0)

    dis: float = field(init=False,
                       repr=False,
                       default=0)

    ann: float = field(init=False,
                       repr=False,
                       default=0)

    _cashflows: np.ndarray = field(init=False,
                                   repr=False)

    def __post_init__(self):
        self.cashflows = np.array([0.0] * len(self.poi.discount_factors.index),
                                  dtype=float)

    @property
    @abstractmethod
    def cost_type(self) -> str:
        # Return the type of cost (e.g. 'capex', 'mntex', 'opex', 'crev', 'totex', 'value')
        ...

    @property
    def aggregator(self) -> CostAggregator:
        # Get the aggregator based on the cost type
        return getattr(self.poi.aggregator, self.cost_type, None) if self.poi.aggregator else None

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
            self.aggregator.prj += self.prj
            self.aggregator.dis += self.dis
            self.aggregator.ann += self.ann
            self.aggregator.cashflows += self.cashflows


@dataclass
class CapexAggregator(CostAggregator):
    poi: EcoPOI

    # all attributes have to be initialized with 0 and calculated
    preexisting: float = field(init=False,
                               repr=False,
                               default=0)

    expansion: float = field(init=False,
                             repr=False,
                             default=0)

    init: float = field(init=False,
                        repr=False,
                        default=0)

    replacement: float = field(init=False,
                               repr=False,
                               default=0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return 'capex'

    def aggregate(self):
        super().aggregate()
        if self.aggregator:
            self.aggregator.preexisting += self.preexisting
            self.aggregator.expansion += self.expansion
            self.aggregator.init += self.init
            self.aggregator.replacement += self.replacement


@dataclass
class CapexEvaluator(CapexAggregator):
    poi: EcoEvaluator

    consider_preexisting: bool = field(init=False,
                                       repr=False,
                                       default=True)

    spec: float = field(init=False,
                        repr=False,
                        default=0.0)

    fix: float = field(init=False,
                       repr=False,
                       default=0.0)

    def __post_init__(self):
        super().__post_init__()

        if ('capex', 'preexisting') in self.poi.params:
            self.consider_preexisting = getattr(self.poi.block,
                                                self.poi.params[('capex', 'preexisting')],
                                                )
        else:
            self.consider_preexisting = True

        if ('capex', 'spec') in self.poi.params:
            self.spec = getattr(self.poi.block,
                                self.poi.params[('capex', 'spec')],
                                )
        else:
            self.spec = 0.0

        if ('capex', 'fix') in self.poi.params:
            self.fix = getattr(self.poi.block,
                               self.poi.params[('capex', 'fix')],
                               )
        else:
            self.fix = 0.0

        self.poi.scenario.capex_preexisting_considered += self.preexisting

    @property
    def size(self) -> Size:
        return self.poi.size  # ToDo: remove and get from poi directly

    @property
    def preexisting(self) -> float:
        return int(self.consider_preexisting) * self.size.preexisting * self.spec + self.fix

    @property
    def expansion(self) -> float:
        return self.size.expansion * self.spec

    @property
    def init(self) -> float:
        return self.preexisting + self.expansion

    @property
    def replacement(self) -> float:
        return self.size.total * self.spec + self.fix

    @CapexAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index),
                             dtype=float)

        cashflows[0] -= self.init

        for period in reinvest_periods(lifespan=self.poi.aux['ls'],
                                       observation_horizon=self.poi.scenario.prj_duration_yrs,
                                       include_init=False):
            cashflows[period] -= self.replacement * (self.poi.aux['ccr'] ** period)

        # Add salvage value capex (positive cashflow)
        cashflows[self.poi.scenario.prj_duration_yrs] += (
                self.replacement * (self.poi.aux['ccr'] ** self.poi.scenario.prj_duration_yrs) *
                calc_frac_remaining_ls(ls=self.poi.aux['ls'],
                                       project_duration=self.poi.scenario.prj_duration_yrs)
        )

        return cashflows

    @property
    def prj(self) -> float:
        return -1 * self.cashflows.sum()

    @property
    def dis(self) -> float:
        return -1 * self.cashflows @ self.poi.discount_factors['beginning']

    @property
    def ann(self) -> float:
        return annuity(present_value=self.dis,
                       observation_horizon=self.poi.scenario.prj_duration_yrs,
                       discount_rate=self.poi.scenario.wacc,
                       occurs_at='beginning')


@dataclass
class MntexAggregator(CostAggregator):
    poi: EcoPOI

    sim: float = field(init=False,
                       repr=False,
                       default=0)

    yrl: float = field(init=False,
                       repr=False,
                       default=0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return 'mntex'

    def aggregate(self):
        super().aggregate()

        if self.aggregator:
            self.aggregator.sim += self.sim
            self.aggregator.yrl += self.yrl


@dataclass
class MntexEvaluator(MntexAggregator):
    poi: EcoEvaluator

    spec: float = field(init=False,
                        repr=False,
                        default=0.0)

    fix: float = field(init=False,
                       repr=False,
                       default=0.0)

    def __post_init__(self):
        super().__post_init__()

        if ('mntex', 'spec') in self.poi.params:
            self.spec = getattr(self.poi.block,
                                self.poi.params[('mntex', 'spec')],
                                )
        else:
            self.spec = 0.0

        if ('mntex', 'fix') in self.poi.params:
            self.fix = getattr(self.poi.block,
                                self.poi.params[('mntex', 'fix')],
                                )
        else:
            self.fix = 0.0

    @property
    def size(self) -> Size:
        return self.poi.size  # ToDo: remove and get from poi directly

    @property
    def yrl(self) -> float:
        return self.size.total * self.spec + self.fix

    @property
    def sim(self) -> float:
        return self.yrl * self.poi.scenario.sim_yr_rat

    @MntexAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index),
                             dtype=float)
        cashflows [self.poi.scenario.periods_prj] = -1 * self.yrl  # ToDo: check if this is correct (indexing)
        return cashflows

    @property
    def prj(self) -> float:
        return -1 * self.cashflows.sum()

    @property
    def dis(self) -> float:
        return -1 * self.cashflows @ self.poi.discount_factors['beginning']

    @property
    def ann(self) -> float:
        return annuity(present_value=self.dis,
                       observation_horizon=self.poi.scenario.prj_duration_yrs,
                       discount_rate=self.poi.scenario.wacc,
                       occurs_at='beginning')


@dataclass
class OpexAggregator(CostAggregator):
    poi: EcoPOI

    sim: float = field(init=False,
                       repr=False,
                       default=0)

    yrl: float = field(init=False,
                       repr=False,
                       default=0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return 'opex'

    def aggregate(self):
        super().aggregate()

        if self.aggregator:
            self.aggregator.sim += self.sim
            self.aggregator.yrl += self.yrl


@dataclass
class OpexEvaluator(OpexAggregator):
    poi: EcoEvaluator

    _spec: pd.Series = field(init=False,
                             repr=False,
                             default=0.0)  # set default value of 0

    def __post_init__(self):
        super().__post_init__()

        if ('opex', 'spec') in self.poi.params:
            self.spec = getattr(self.poi.block,
                                self.poi.params[('opex', 'spec')],
                                )
        else:
            self.spec = 0.0

    @property
    def spec(self) -> pd.Series:
        return self._spec

    @spec.setter
    def spec(self, value: str | float | pd.Series):
        self._spec = transform_scalar_var(value=value,
                                          scenario=self.poi.scenario,
                                          block=self.poi.block)

    @property
    def flow(self) -> np.ndarray:
        return self.poi.flow  # ToDo: remove and get from poi directly

    @property
    def sim(self) -> float:
        # ToDo: add calc_opex_sim_additional (e.g. for PeakEvaluator or FleetUnitEvaluator)
        return self.flow @ self.spec[self.poi.scenario.dti_eval] * self.poi.scenario.timestep_hours

    @property
    def yrl(self) -> float:
        return self.sim / self.poi.scenario.sim_yr_rat

    @OpexAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index),
                             dtype=float)
        cashflows[self.poi.scenario.periods_prj] = -1 * self.yrl  # ToDo: check if this is correct (indexing)
        return cashflows

    @property
    def prj(self) -> float:
        return -1 * self.cashflows.sum()

    @property
    def dis(self) -> float:
        return -1 * self.cashflows @ self.poi.discount_factors['end']

    @property
    def ann(self) -> float:
        return annuity(present_value=self.dis,
                       observation_horizon=self.poi.scenario.prj_duration_yrs,
                       discount_rate=self.poi.scenario.wacc,
                       occurs_at='end')


@dataclass
class CrevAggregator(CostAggregator):
    poi: EcoPOI

    sim: float = field(init=False,
                       repr=False,
                       default=0)

    yrl: float = field(init=False,
                       repr=False,
                       default=0)

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return 'crev'

    def aggregate(self):
        super().aggregate()

        if self.aggregator:
            self.aggregator.sim += self.sim
            self.aggregator.yrl += self.yrl


@dataclass
class CrevEvaluator(OpexAggregator):
    poi: EcoEvaluator

    flow_name: Optional[str] = field(init=False,
                                     repr=False,
                                     default=None)  # ToDo: define flow

    _spec: pd.Series = field(init=False,
                             repr=False,
                             default=0.0)  # set default value of 0

    def __post_init__(self):
        super().__post_init__()

        if ('crev', 'spec') in self.poi.params:
            self.spec = getattr(self.poi.block,
                                self.poi.params[('crev', 'spec')],
                                )
        else:
            self.spec = 0.0

    @property
    def spec(self) -> pd.Series:
        return self._spec

    @spec.setter
    def spec(self, value: str | float | pd.Series):
        self._spec = transform_scalar_var(value=value,
                                          scenario=self.poi.scenario,
                                          block=self.poi.block)

    @property
    def flow(self) -> np.ndarray:
        return self.poi.flow  # ToDo: remove and get from poi directly

    @property
    def sim(self) -> float:
        # ToDo: add calc_crev_sim_additional (e.g. for PeakEvaluator or FleetUnitEvaluator)
        return (self.flow @ self.spec[self.poi.scenario.dti_eval] * self.poi.scenario.timestep_hours
                if self.flow_name is not None else 0)

    @property
    def yrl(self) -> float:
        return self.sim / self.poi.scenario.sim_yr_rat

    @CrevAggregator.cashflows.getter  # Only override the getter as otherwise (@property) also the setter is overridden
    def cashflows(self) -> np.ndarray:
        cashflows = np.array([0.0] * len(self.poi.discount_factors.index),
                             dtype=float)
        cashflows[self.poi.scenario.periods_prj] = self.yrl  # ToDo: check if this is correct (indexing)
        return cashflows

    @property
    def prj(self) -> float:
        return self.cashflows.sum()

    @property
    def dis(self) -> float:
        return self.cashflows @ self.poi.discount_factors['end']

    @property
    def ann(self) -> float:
        return annuity(present_value=self.dis,
                       observation_horizon=self.poi.scenario.prj_duration_yrs,
                       discount_rate=self.poi.scenario.wacc,
                       occurs_at='end')


@dataclass
class TotexAggregator(CostAggregator):
    poi: EcoAggregator

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return 'totex'

    def aggregate(self):
        self.cashflows = self.poi.capex.cashflows + self.poi.mntex.cashflows + self.poi.opex.cashflows

        self.prj = self.poi.capex.prj + self.poi.mntex.prj + self.poi.opex.prj
        self.dis = self.poi.capex.dis + self.poi.mntex.dis + self.poi.opex.dis
        self.ann = self.poi.capex.ann + self.poi.mntex.ann + self.poi.opex.ann

        super().aggregate()


@dataclass
class ValueAggregator(CostAggregator):
    poi: EcoAggregator

    def __post_init__(self):
        super().__post_init__()

    @property
    def cost_type(self) -> str:
        return 'value'

    def aggregate(self):
        self.cashflows = self.poi.crev.cashflows - self.poi.totex.cashflows  # Check whether this is correct

        self.prj = self.poi.crev.prj - self.poi.totex.prj
        self.dis = self.poi.crev.dis - self.poi.totex.dis
        self.ann = self.poi.crev.ann - self.poi.totex.ann

        super().aggregate()


@dataclass
class EcoPOI(ABC):
    name: str
    scenario: simulation.Scenario
    block: Optional[blocks.BaseBlock] = None

    # Initialize in __post_init__()
    capex: CapexAggregator | CapexEvaluator = field(init=False,)
    mntex: MntexAggregator | MntexEvaluator = field(init=False,)
    opex: OpexAggregator | OpexEvaluator = field(init=False,)
    crev: CrevAggregator | CrevEvaluator = field(init=False,)

    @abstractmethod
    def __post_init__(self):
        # Initialize capex, mntex, opex, crev (totex, value) with correct class (Aggregator or Evaluator) here
        ...

    @property
    def discount_factors(self) -> pd.DataFrame:
        return self.scenario.discount_factors

    @property
    @abstractmethod
    def aggregator(self) -> EcoAggregator:
        # Get aggregator based on the type of EcoPOI (EcoAggregator, EcoEvaluator)
        ...

    def aggregate(self):
        self.capex.aggregate()
        self.mntex.aggregate()
        self.opex.aggregate()
        self.crev.aggregate()


@dataclass
class EcoAggregator(EcoPOI):
    totex: TotexAggregator = field(init=False,)
    value: ValueAggregator = field(init=False,)

    def __post_init__(self):
        self.capex = CapexAggregator(poi=self)
        self.mntex = MntexAggregator(poi=self)
        self.opex = OpexAggregator(poi=self)
        self.crev = CrevAggregator(poi=self)

        self.totex = TotexAggregator(poi=self)
        self.value = ValueAggregator(poi=self)

    @property
    def aggregator(self) -> EcoAggregator:
        return self.block.parent.aggregator if hasattr(self.block, 'parent') else None

    def aggregate(self):
        super().aggregate()

        self.totex.aggregate()
        self.value.aggregate()

    def write_result_summary(self) -> pd.Series:
        # ToDo: implement a method to write a summary of the economic results
        return pd.Series()


@dataclass
class EcoEvaluator(EcoPOI):
    block: blocks.BaseBlock
    params: dict = None

    aux: dict = field(init=False,
                      repr=False,
                      )

    opt: OptimizationConverter = field(init=False,
                                       repr=False)

    def __post_init__(self):
        self.aux = dict()

        if ('aux', 'ls') in self.params:
            self.aux['ls'] = getattr(self.block,
                                     self.params[('aux', 'ls')])
        else:
            self.aux['ls'] = self.scenario.prj_duration_yrs

        if ('aux', 'ccr') in self.params:
            self.aux['ccr'] = getattr(self.block,
                                     self.params[('aux', 'ccr')])
        else:
            self.aux['ccr'] = 1.0

        self.block.sizes[self.name] = Size(name=self.name,
                                           block=self.block,
                                           unit=self.params.get(('size', 'unit'), 'kW')
                                           )

        if ('flow', 'name') in self.params:
            # ToDo: flow_names not available for BaseBlock, but ElectricBlock only
            self.block.flow_names.add(self.params[('flow', 'name')])

        self.capex = CapexEvaluator(poi=self)
        self.mntex = MntexEvaluator(poi=self)
        self.opex = OpexEvaluator(poi=self)
        self.crev = CrevEvaluator(poi=self)

        self.opt = OptimizationConverter(poi=self)
        pass

    @property
    def aggregator(self) -> EcoAggregator:
        return self.block.aggregator

    @property
    def size(self) -> Size:
        return self.block.sizes[self.name]

    @property
    def flow(self) -> np.ndarray:
        if hasattr(self.block, 'flows') and self.name in self.block.flows.columns:
            return self.block.flows.loc[self.scenario.dti_eval, self.name].values
        else:
            return np.array([0.0] * len(self.scenario.dti_eval),
                            dtype=float)


class EconomicPointOfInterest:
    """
    abstractclass
    """

    def __init__(self,
                 name: str,
                 block: blocks.BaseBlock,
                 scenario: simulation.Scenario = None):

        self.name = name
        self.block = block

        if self.block is None and scenario is None:
            raise ValueError('At least one of the parameters "block" or "scenario" has to be provided to initialize'
                             'an EconomicPointOfInterest instance')

        self.scenario = self.block.scenario if scenario is None else scenario
        self.discount_factors = self.scenario.discount_factors
        self.cashflows = pd.DataFrame(index=self.discount_factors.index,
                                      columns=['capex', 'mntex', 'opex', 'crev'],
                                      data=0.0,
                                      dtype='float64')

        self.capex = CapexAggregator
        self.mntex = MntexAggregator
        self.opex = OpexAggregator
        self.crev = CrevAggregator
        self.totex = TotexAggregator
        self.value = ValueAggregator

    def aggregate_pre_scenario(self,
                               target: 'EconomicAggregator'):
        """
        aggregate capex preexisting one level up
        """
        # ToDo: create aggregate_pre_scenario and aggregate_post_scenario methods for Aggregator classes
        # ToDo: trigger aggregation of capex, mntex, opex, crev, totex, value
        pass

    def aggregate_post_scenario(self,
                                target: 'EconomicAggregator'):
        """
        aggregate all economic values (except capex preexisting) one level up
        """

        # ToDo: create aggregate_pre_scenario and aggregate_post_scenario methods for Aggregator classes
        # ToDo: trigger aggregation of capex, mntex, opex, crev, totex, value
        pass



class EconomicAggregator(EconomicPointOfInterest):

    def __init__(self,
                 name: str,
                 block: blocks.BaseBlock,
                 scenario: simulation.Scenario = None):

        super().__init__(name=name,
                         block=block,
                         scenario=scenario)

        self.totex = {'prj': 0.0,
                      'dis': 0.0,
                      'ann': 0.0}
        self.value = {'prj': 0.0,
                      'dis': 0.0,
                      'ann': 0.0}

    def pre_scenario(self):
        if self.block is not None:
            self.aggregate_pre_scenario(target=self.block.parent.aggregator)

    def post_scenario(self):
        """
        aggregate all economic values (except capex preexisting) one level up
        """
        if self.block is not None:
            self.aggregate_post_scenario(target=self.block.parent.aggregator)

        for key in self.totex:
            self.totex[key] = self.capex[key] + self.mntex[key] + self.opex[key]
            self.value[key] = self.crev[key] - self.totex[key]

    def write_result_summary(self):
        # combine all dicts in a series
        result_series = pd.Series({f'{dict_name}_{key}': value
                                   for dict_name in ['capex', 'mntex', 'opex', 'crev', 'totex', 'value']
                                   for key, value in getattr(self, dict_name).items()})

        return result_series


class EconomicEvaluator(EconomicPointOfInterest):
    """
    Point of ts result interest or economic influence
    """

    def __init__(self,
                 name: str,
                 block: blocks.BaseBlock,
                 params: dict):

        super().__init__(name=name,
                         block=block)

        # ToDo: create corresponding Size object in block.sizes
        self.block.sizes[name] = Size(name=name,
                                      block=block,
                                      unit=params['unit'])

        # region set default values
        self.capex = CapexEvaluator(poi=self,
                                    target=self.block.aggregator.capex,
                                    )
        self.mntex = MntexEvaluator(poi=self,
                                    target=self.block.aggregator.mntex,
                                    )

        self.opex = OpexEvaluator(poi=self,
                                  target=self.block.aggregator.opex,
                                  )

        self.crev = CrevEvaluator(poi=self,
                                  target=self.block.aggregator.opex,
                                  )

        self.opex.update({'spec': transform_scalar_var(value=0.0,
                                                       scenario=self.scenario,
                                                       block=block)})
        self.crev.update({'spec': transform_scalar_var(value=0.0,
                                                       scenario=self.scenario,
                                                       block=block)})

        self.aux = {'ls': self.scenario.prj_duration_yrs,
                    'ccr': 1.0}
        self.size_name = None
        self.flow_name = None
        # endregion

        # region set values from block
        for param_tuple, param_name in params.items():
            dict_name, dict_key = param_tuple
            if param_tuple == ('size', 'name'):
                self.size_name = param_name
            elif param_tuple == ('flow', 'name'):
                self.flow_name = param_name
            elif dict_name in ['opex', 'crev']:
                getattr(self, dict_name)[dict_key] = transform_scalar_var(value=getattr(self.block, param_name, 0.0),
                                                                          scenario=self.scenario,
                                                                          block=self.block)
            elif dict_name == 'capex':
                setattr(getattr(self, dict_name),
                        dict_key,
                        getattr(self.block, param_name, 0.0))
            else:  # mntex, aux
                getattr(self, dict_name)[dict_key] = getattr(self.block, param_name, 0.0)
        # endregion

        self.conv_opt = OptimizationConverter(poi=self)

        self.pre_scenario()
        self.aggregate_pre_scenario(target=self.block.aggregator)

    def post_scenario(self):

        super().aggregate_post_scenario(target=self.block.aggregator)

    def get_size(self,
                 size_name: str,
                 scope_name: str,
                 default_value: float = 0) -> float:
        """
        get a value from the block's size df
        """

        if size_name in self.block.sizes.keys():
            value = getattr(self.block.sizes[size_name], scope_name)
            if pd.isna(value):  # sizes in GridMarkets may be None (inherit limit of GridConnection)
                value = default_value
            return value
        elif size_name is None:
            return default_value
        else:
            raise ValueError(f'Block "{self.block.name}": Size "{size_name}" not found in size dataframe.')


    def calc_opex_sim_additional(self):
        """
        dummy method
        """
        pass

    def calc_crev_sim_additional(self):
        """
        dummy method
        """
        pass


class FleetUnitEvaluator(EconomicEvaluator):

    def calc_opex_sim_additional(self):

        if self.block.classname in ['ElectricVehicle', 'CombustionVehicle']:
            self.opex['sim'] += (self.block.log.loc[self.scenario.dti_eval, 'dist'] @
                                 self.opex['dist'][self.scenario.dti_eval])

    def calc_crev_sim_additional(self):

        if self.block.classname in ['ElectricVehicle', 'CombustionVehicle']:
            self.crev['sim'] += (self.block.log.loc[self.scenario.dti_eval, 'dist'] @
                                 self.crev['dist'][self.scenario.dti_eval])

        self.crev['sim'] += ((~self.block.log.loc[self.scenario.dti_eval, 'atbase'] @
                             self.crev['time'][self.scenario.dti_eval]) *
                             self.scenario.timestep_hours)


class PeakEvaluator(EconomicEvaluator):

    def pre_scenario(self):

        # get and set the opex_spec at the first timestep of the peakshaving period
        self.opex['spec'] = self.opex['spec'][self.block.peak_periods.loc[self.name, 'start']]

        self.opex['factor_ep'] = (self.block.n_peak_periods_yr / self.block.peak_periods.shape[0]
                                  if self.scenario.compensate_sim_prj else 1)

        self.opex['spec_ep'] = self.opex['spec'] * self.opex['factor_ep']

    def calc_opex_sim_additional(self):
        self.opex['sim'] += self.block.peak_periods.loc[self.name, 'power'] * self.opex['spec'] * \
                            self.block.peak_periods.loc[self.name, 'period_fraction']
