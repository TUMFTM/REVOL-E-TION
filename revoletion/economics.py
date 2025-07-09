#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Optional

from . import utils


if TYPE_CHECKING:
    from .blocks import Size

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
) -> float:
    """
    This function calculates the nominal (inluding inflation) weighted average cost of capital (WACC) using the
    Capital Asset Pricing Model (CAPM) for equity cost.
    """
    share_debt = 1 - share_equity
    cost_equity = rate_riskfree + volatility_relative * (rate_market - rate_riskfree)  # CAPM
    wacc_nominal = share_debt * rate_debt * (1 - rate_tax) + share_equity * cost_equity
    wacc_real = (1 + wacc_nominal) / (1 + rate_inflation)  # fisher formula
    return wacc_nominal, wacc_real


def transform_scalar_var(value, scenario, block=None):
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
class OptimizationConverter:
    poi: EconomicEvaluator

    @property
    def spec_prj_ep_capex(self) -> float:
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
    def spec_prj_ep_mntex(self) -> float:
        # calculate specific present value of mntex for the project duration
        return acc_discount(nominal_value=self.poi.mntex.spec,
                            observation_horizon=self.poi.scenario.prj_duration_yrs,
                            discount_rate=self.poi.scenario.wacc,
                            occurs_at='beginning')

    @property
    def spec_prj_ep_invest(self) -> float:
        # join maintenance and capex specific present values for the project duration
        return self.spec_prj_ep_capex + self.spec_prj_ep_mntex

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
        return self.spec_prj_ep_invest * self.factor_ep_invest

    @property
    def factor_ep_operation(self) -> float:
        # calculate annuity due factor to compensate for difference between simulation and project time
        return (1 / self.poi.scenario.sim_yr_rat) if self.poi.scenario.compensate_sim_prj else 1

    @property
    def spec_ep_operation(self) -> float:
        # calculate specific capex/mntex value used for the optimization problem
        return self.spec_prj_ep_invest * self.factor_ep_operation


@dataclass
class Aggregator:
    poi: EconomicPointOfInterest
    target: Optional[Aggregator]

    prj: float = field(init=False,
                       repr=False,
                       default=0)

    dis: float = field(init=False,
                       repr=False,
                       default=0)

    ann: float = field(init=False,
                       repr=False,
                       default=0)

    cashflows: np.array = field(init=False,
                                repr=False)

    def __post_init__(self):
        self.cashflows = np.array([0.0] * len(self.poi.discount_factors.index),
                                  dtype=float)

    def aggregate(self):
        if self.target:
            self.target.prj += self.prj
            self.target.dis += self.dis
            self.target.ann += self.ann
            self.target.cashflows += self.cashflows


@dataclass
class CapExAggregator(Aggregator):
    poi: EconomicPointOfInterest
    target: Optional[CapExAggregator]

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

    def aggregate(self):
        super().aggregate()
        if self.target:
            self.target.preexisting += self.preexisting
            self.target.expansion += self.expansion
            self.target.init += self.init
            self.target.replacement += self.replacement


@dataclass
class CapExEvaluator(CapExAggregator):
    poi: EconomicEvaluator
    target: Optional[CapExAggregator]

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

        # ToDo: set consider_preexisting
        # ToDo: set spec
        # ToDo: set fix

    @property
    def size(self) -> Size:
        return self.poi.block.sizes[self.poi.name]

    @property
    def preexisting(self) -> float:
        # ToDo: add preexisting to constraint limit
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

    @property
    def cashflows(self) -> np.array:
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
class MntExAggregator(Aggregator):
    poi: EconomicPointOfInterest
    target: MntExAggregator

    sim: float = field(init=False,
                       repr=False,
                       default=0)

    yrl: float = field(init=False,
                       repr=False,
                       default=0)

    def __post_init__(self):
        super().__post_init__()

    def aggregate(self):
        super().aggregate()

        if self.target:
            self.target.sim += self.sim
            self.target.yrl += self.yrl


@dataclass
class MntExEvaluator(MntExAggregator):
    poi: EconomicEvaluator
    target: MntExAggregator

    spec: float = field(init=False,
                        repr=False,
                        default=0.0)

    fix: float = field(init=False,
                       repr=False,
                       default=0.0)

    def __post_init__(self):
        super().__post_init__()

        # ToDo: set spec
        # ToDo: set fix

    @property
    def size(self) -> Size:
        return self.poi.block.sizes[self.poi.name]

    @property
    def yrl(self) -> float:
        return self.size.total * self.spec + self.fix

    @property
    def sim(self) -> float:
        return self.yrl * self.poi.scenario.sim_yr_rat


    @property
    def cashflows(self) -> np.array:
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
class OpExAggregator(Aggregator):
    poi: EconomicPointOfInterest
    target: Optional[OpExAggregator]

    sim: float = field(init=False,
                       repr=False,
                       default=0)

    yrl: float = field(init=False,
                       repr=False,
                       default=0)

    def __post_init__(self):
        super().__post_init__()

    def aggregate(self):
        super().aggregate()

        if self.target:
            self.target.sim += self.sim
            self.target.yrl += self.yrl


@dataclass
class OpExEvaluator(OpExAggregator):
    poi: EconomicEvaluator
    target: OpExAggregator

    spec: pd.Series = field(init=False,
                            repr=False,
                            default=0.0)  # set default value of 0

    def __post_init__(self):
        super().__post_init__()

        # ToDo: set spec -> use property setter to transform scalar variable to pandas Series

    @property
    def flow(self) -> np.array:
        if self.poi.name in self.poi.block.flow.columns:
            return self.poi.block.flows.loc[self.poi.scenario.dti_eval, self.poi.name].values
        else:
            return np.array([0.0] * len(self.poi.scenario.dti_eval),
                            dtype=float)

    @property
    def sim(self) -> float:
        # ToDo: add calc_opex_sim_additional (e.g. for PeakEvaluator or FleetUnitEvaluator)
        return self.flow @ self.spec[self.poi.scenario.dti_eval] * self.poi.scenario.timestep_hours

    @property
    def yrl(self) -> float:
        return self.sim / self.poi.scenario.sim_yr_rat

    @property
    def cashflows(self) -> np.array:
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
class CRevAggregator(Aggregator):
    poi: EconomicPointOfInterest
    target: Optional[CRevAggregator]

    sim: float = field(init=False,
                       repr=False,
                       default=0)

    yrl: float = field(init=False,
                       repr=False,
                       default=0)

    def __post_init__(self):
        super().__post_init__()

    def aggregate(self):
        super().aggregate()

        if self.target:
            self.target.sim += self.sim
            self.target.yrl += self.yrl


@dataclass
class CRevEvaluator(OpExAggregator):
    poi: EconomicEvaluator
    target: CRevAggregator

    flow_name: Optional[str] = field(init=False,
                                     repr=False,
                                     default=None)  # ToDo: define flow

    spec: pd.Series = field(init=False,
                            repr=False,
                            default=0.0)  # set default value of 0

    def __post_init__(self):
        super().__post_init__()

        # ToDo: set spec -> use property setter to transform scalar variable to pandas Series

    @property
    def flow(self) -> np.array:
        if self.poi.name in self.poi.block.flow.columns:
            return self.poi.block.flows.loc[self.poi.scenario.dti_eval, self.poi.name].values
        else:
            return np.array([0.0] * len(self.poi.scenario.dti_eval),
                            dtype=float)

    @property
    def sim(self) -> float:
        # ToDo: add calc_crev_sim_additional (e.g. for PeakEvaluator or FleetUnitEvaluator)
        return (self.poi.block.flows.loc[self.poi.scenario.dti_eval, self.flow_name]
                @ self.spec[self.poi.scenario.dti_eval]
                * self.poi.scenario.timestep_hours) if self.flow_name is not None else 0

    @property
    def yrl(self) -> float:
        return self.sim / self.poi.scenario.sim_yr_rat

    @property
    def cashflows(self) -> np.array:
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
class TotExAggregator(Aggregator):
    poi: EconomicPointOfInterest
    target: Optional[TotExAggregator]

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ValueAggregator(Aggregator):
    poi: EconomicPointOfInterest
    target: Optional[ValueAggregator]

    def __post_init__(self):
        super().__post_init__()


class EconomicPointOfInterest:
    """
    abstractclass
    """

    def __init__(self,
                 name: str,
                 block: 'blocks.BaseBlock',
                 scenario: 'simulation.Scenario' = None):

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

        self.capex = CapExAggregator
        self.mntex = MntExAggregator
        self.opex = OpExAggregator
        self.crev = CRevAggregator

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
                 block: 'blocks.BaseBlock',
                 scenario: 'simulation.Scenario' = None):

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
                 block: 'blocks.BaseBlock',
                 params: dict):

        super().__init__(name=name,
                         block=block)

        # region set default values
        self.capex = CapExEvaluator(poi=self,
                                    target=self.block.aggregator.capex,
                                    )
        self.mntex = MntExEvaluator(poi=self,
                                    target=self.block.aggregator.mntex,
                                    )

        self.opex = OpExEvaluator(poi=self,
                                  target=self.block.aggregator.opex,
                                  )

        self.crev = CRevEvaluator(poi=self,
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
