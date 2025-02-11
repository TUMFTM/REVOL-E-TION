#!/usr/bin/env python3

import os
import pandas as pd

from revoletion import utils


def discount(future_value: float,
             periods: int,
             discount_rate: float) -> float:
    """
    This function calculates the present value of a future value in some periods at a discount rate per period
    """
    q = 1 + discount_rate
    present_value = future_value / (q ** periods)
    return present_value


def acc_discount(nominal_value: float | pd.Series,
                 observation_horizon: int,
                 discount_rate: float,
                 occurs_at='beginning') -> float:
    """
    This function calculates the accumulated present value of a periodical, nominally repeating cashflow in the future
    (from present to the observation horizon) at a discount rate per period
    """
    p_delta = {'beginning': 0, 'middle': 0.5, 'end': 1}[occurs_at]

    if isinstance(nominal_value, pd.Series):
        return nominal_value.apply(lambda x: acc_discount(x, observation_horizon, discount_rate, occurs_at))

    present_value = sum([discount(nominal_value, period + p_delta, discount_rate)
                         for period in range(observation_horizon)])
    return present_value


def join_capex_mntex(capex: float,
                     mntex: float,
                     lifespan: int,
                     discount_rate: float) -> float:
    """
    This function adjusts a component's capex to include accumulated present maintenance cost for time based maintenance
    """
    capex_adjusted = capex + acc_discount(nominal_value=mntex,
                                          observation_horizon=lifespan,
                                          discount_rate=discount_rate,
                                          occurs_at='beginning')
    return capex_adjusted


def annuity(present_value: float,
            observation_horizon: int,
            discount_rate: float,
            occurs_at='beginning') -> float:
    """
    This function calculates the annuity (the equivalent periodical, nominally recurring value to generate the same
    NPV) of a present value pv over an observation horizon at a discount rate per period. occurs_at denotes whether
    the expense or value occurs at the beginning (making the annuity an annuity due) or end of the period.
    """
    q = 1 + discount_rate
    due_factor = {'beginning': q, 'end': 1}[occurs_at]
    try:
        annuity = present_value * discount_rate / ((1 - (q ** -observation_horizon)) * due_factor)
    except ZeroDivisionError:
        annuity = present_value / observation_horizon
    return annuity


def reinvest_periods(lifespan: int,
                     observation_horizon: int) -> list:
    """
    This function returns a list of period numbers to reinvest into a component (i.e. replace it),
    given its lifespan and the observation horizon. Initial investment is removed.
    """
    reinvest_periods = [period for period in range(observation_horizon) if period % lifespan == 0]
    reinvest_periods.remove(0)
    return reinvest_periods


def capex_sum(capex_init: float,
              capex_replacement: float,
              cost_change_ratio: float,
              lifespan: int,
              observation_horizon: int) -> float:
    """
    This function calculates the total (non-discounted) capital expenses for a component that has to be replaced
    after its lifespan during the observation horizon and changes in price at a cost change ratio every period.
    """
    capex_total = capex_init + sum([capex_replacement * (cost_change_ratio ** period)
                                    for period in reinvest_periods(lifespan, observation_horizon)])
    return capex_total


def capex_present(capex_init: float,
                  capex_replacement: float,
                  cost_change_ratio: float,
                  discount_rate: float,
                  lifespan: int,
                  observation_horizon: int) -> float:
    """
    This function calculates the present (discounted) capital expenses for a component that has to be replaced
    after its lifespan during the observation horizon and changes in price at a cost change ratio every period.
    """
    capex_present = capex_init + sum([discount(capex_replacement * (cost_change_ratio ** period), period, discount_rate)
                                      for period in reinvest_periods(lifespan, observation_horizon)])
    return capex_present


def annuity_due_capex(capex_init: float,
                      capex_replacement: float,
                      lifespan: int,
                      observation_horizon: int,
                      discount_rate: float,
                      cost_change_ratio: float) -> float:
    """
    This function calculates the annuity due of a recurring (every ls years) and price changing (annual ratio ccr)
    investment (the equivalent yearly sum to generate the same NPV) over a horizon of hor years
    """
    present_value = capex_present(capex_init=capex_init,
                                  capex_replacement=capex_replacement,
                                  cost_change_ratio=cost_change_ratio,
                                  discount_rate=discount_rate,
                                  lifespan=lifespan,
                                  observation_horizon=observation_horizon)
    annuity_due = annuity(present_value=present_value,
                          observation_horizon=observation_horizon,
                          discount_rate=discount_rate,
                          occurs_at='beginning')
    return annuity_due


def annuity_due_recur(nominal_value: float,
                      observation_horizon: float,
                      discount_rate: float):
    """
    Calculate the annuity due of a yearly recurring (lifespan=1) and nonchanging (cost_change_ratio=1)
    maintenance expense (the equivalent yearly sum to generate the same NPV) over an observation horizon.
    """
    annuity_due_recur = annuity_due_capex(capex_init=nominal_value,
                                          capex_replacement=nominal_value,
                                          lifespan=1,
                                          observation_horizon=observation_horizon,
                                          discount_rate=discount_rate,
                                          cost_change_ratio=1)
    return annuity_due_recur


def annuity_recur(nominal_value: float,
                  observation_horizon: float,
                  discount_rate: float):
    """
    Calculate the annuity of a periodically recurring  and nonchanging (cost_change_ratio=1)
    expense (the equivalent yearly sum to generate the same NPV) over an observation horizon.
    """
    present_value = acc_discount(nominal_value=nominal_value,
                                 observation_horizon=observation_horizon,
                                 discount_rate=discount_rate,
                                 occurs_at='end')
    annuity_recur = annuity(present_value=present_value,
                            observation_horizon=observation_horizon,
                            discount_rate=discount_rate,
                            occurs_at='end')
    return annuity_recur


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


class EconomicPointOfInterest:
    """
    abstractclass
    """

    def __init__(self,
                 name: str,
                 block: 'blocks.Block'):

        self.name = name
        self.block = block

        self.capex = {'fix': 0,
                      'preexisting': 0,
                      'expansion': 0,
                      'init': 0,
                      'replacement': 0,
                      'prj': 0,
                      'dis': 0,
                      'ann': 0}
        self.mntex = {'fix': 0,
                      'yrl': 0,
                      'sim': 0,
                      'prj': 0,
                      'dis': 0,
                      'ann': 0}
        self.opex = {'sim': 0,
                     'yrl': 0,
                     'prj': 0,
                     'dis': 0,
                     'ann': 0}
        self.crev = {'sim': 0,
                     'yrl': 0,
                     'prj': 0,
                     'dis': 0,
                     'ann': 0}

    def aggregate_pre_scenario(self,
                               target: 'EconomicAggregator'):
        """
        aggregate capex preexisting one level up
        """
        target.capex['preexisting'] += self.capex['preexisting']

    def aggregate_post_scenario(self,
                                target: 'EconomicAggregator'):
        """
        aggregate all economic values (except capex preexisting) one level up
        """
        for key in ['expansion', 'init', 'replacement', 'prj', 'dis', 'ann']:
            target.capex[key] += self.capex[key]
        for key in ['sim', 'yrl', 'prj', 'dis', 'ann']:
            target.mntex[key] += self.mntex[key]
            target.opex[key] += self.opex[key]
            target.crev[key] += self.crev[key]


class EconomicAggregator(EconomicPointOfInterest):

    def __init__(self,
                 name: str,
                 block: 'blocks.Block',
                 scenario: 'simulation.Scenario'=None):

        super().__init__(name=name,
                         block=block)

        # set scenario to write values to scenario summary
        if self.block is None and scenario is None:
            raise ValueError('At least one of the arguments "block" or "scenario" has to be provided to initialize'
                             'an object of type "EconomicAggregator".')
        self.scenario = self.block.scenario if scenario is None else scenario

        self.totex = {'prj': 0,
                      'dis': 0,
                      'ann': 0}
        self.value = {'prj': 0,
                      'dis': 0,
                      'ann': 0}

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
            self.value[key] = self.totex[key] - self.crev[key]

        self.write_results()

    def write_results(self):
        # ToDo: fasten calculation / use pd.concat instead of many .loc operations?
        block_name = self.block.name if self.block is not None else 'scenario'
        for dict_name in ['capex', 'mntex', 'opex', 'crev', 'totex', 'value']:
            for key, value in getattr(self, dict_name).items():
                self.scenario.result_summary.loc[(block_name, f'{dict_name}_{key}'), self.scenario.name] = value


class EconomicEvaluator(EconomicPointOfInterest):
    """
    Point of ts result interest or economic influence
    """

    def __init__(self,
                 name: str,
                 block: 'blocks.Block',
                 params: dict):

        super().__init__(name=name,
                         block=block)

        # region set default values
        self.capex.update({'spec': 0})
        self.mntex.update({'spec': 0})
        self.opex.update({'spec': utils.transform_scalar_var(value=0,
                                                             scenario=self.block.scenario,
                                                             block=block),
                          'dist': utils.transform_scalar_var(value=0,
                                                             scenario=self.block.scenario,
                                                             block=self.block)})
        self.crev.update({'spec': utils.transform_scalar_var(value=0,
                                                             scenario=self.block.scenario,
                                                             block=block),
                          'dist': utils.transform_scalar_var(value=0,
                                                             scenario=self.block.scenario,
                                                             block=self.block)})
        self.aux = {'ls': self.block.scenario.prj_duration_yrs,
                    'ccr': 1}
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
                getattr(self, dict_name)[dict_key] = utils.transform_scalar_var(value=getattr(self.block, param_name),
                                                                                scenario=self.block.scenario,
                                                                                block=self.block)
            else:  # capex, mntex, aux
                getattr(self, dict_name)[dict_key] = getattr(self.block, param_name)
        # endregion

        self.pre_scenario()
        self.aggregate_pre_scenario(target=self.block.aggregator)

    def pre_scenario(self):

        # region calculate equivalent present specific capex and opex for optimizer
        # include (time-based) maintenance expenses in capex calculation as equivalent present cost
        self.capex['spec_joined'] = join_capex_mntex(capex=self.capex['spec'],
                                                     mntex=self.mntex['spec'],
                                                     lifespan=self.aux['ls'],
                                                     discount_rate=self.block.scenario.wacc)

        # annuity due factor (incl. replacements) to compensate for difference between simulation and project time in
        # component sizing; ep = equivalent present (i.e. specific values prediscounted)
        self.capex['factor_ep'] = annuity_due_capex(capex_init=1,
                                                    capex_replacement=1,
                                                    lifespan=self.aux['ls'],
                                                    observation_horizon=self.block.scenario.prj_duration_yrs,
                                                    discount_rate=self.block.scenario.wacc,
                                                    cost_change_ratio=self.aux['ccr']) \
            if self.block.scenario.compensate_sim_prj else 1

        self.capex['spec_ep'] = self.capex['spec_joined'] * self.capex['factor_ep']

        # runtime factor to compensate for difference between simulation and project timeframe
        # opex is uprated in importance for short simulations
        self.opex['factor_ep'] = annuity_recur(nominal_value=utils.scale_sim2year(1, self.block.scenario),
                                               observation_horizon=self.block.scenario.prj_duration_yrs,
                                               discount_rate=self.block.scenario.wacc) \
            if self.block.scenario.compensate_sim_prj else 1
        self.opex['spec_ep'] = self.opex['spec'] * self.opex['factor_ep']
        # endregion

        # region calculate capital expenses for preexisting block size
        self.capex['preexisting'] = (self.capex['preexisting'] *  # boolean so far - will be overwritten
                                     (self.get_size(self.size_name, 'preexisting') * self.capex['spec'] +
                                      self.capex['fix']))
        # endregion

    def post_scenario(self):

        # region capex
        self.capex['expansion'] = self.capex['spec'] * self.get_size(self.size_name, 'expansion')
        self.capex['init'] = self.capex['preexisting'] + self.capex['expansion']
        self.capex['replacement'] = (self.capex['spec'] * self.get_size(self.size_name, 'total') +
                                     self.capex['fix'])
        self.capex['prj'] = capex_sum(capex_init=self.capex['init'],
                                      capex_replacement=self.capex['replacement'],
                                      cost_change_ratio=self.aux['ccr'],
                                      lifespan=self.aux['ls'],
                                      observation_horizon=self.block.scenario.prj_duration_yrs)
        self.capex['dis'] = capex_present(capex_init=self.capex['init'],
                                          capex_replacement=self.capex['replacement'],
                                          cost_change_ratio=self.aux['ccr'],
                                          discount_rate=self.block.scenario.wacc,
                                          lifespan=self.aux['ls'],
                                          observation_horizon=self.block.scenario.prj_duration_yrs)
        self.capex['ann'] = annuity_due_capex(capex_init=self.capex['init'],
                                              capex_replacement=self.capex['replacement'],
                                              lifespan=self.aux['ls'],
                                              observation_horizon=self.block.scenario.prj_duration_yrs,
                                              discount_rate=self.block.scenario.wacc,
                                              cost_change_ratio=self.aux['ccr'])
        # endregion

        # region mntex
        self.mntex['yrl'] = self.get_size(self.size_name, 'total') * self.mntex['spec']
        self.mntex['sim'] = self.mntex['yrl'] * self.block.scenario.sim_yr_rat
        self.mntex['prj'] = utils.scale_year2prj(self.mntex['yrl'], self.block.scenario)
        self.mntex['dis'] = acc_discount(nominal_value=self.mntex['yrl'],
                                         observation_horizon=self.block.scenario.prj_duration_yrs,
                                         discount_rate=self.block.scenario.wacc,
                                         occurs_at='beginning')
        self.mntex['ann'] = annuity_due_recur(nominal_value=self.mntex['yrl'],
                                              observation_horizon=self.block.scenario.prj_duration_yrs,
                                              discount_rate=self.block.scenario.wacc)
        # endregion

        # region opex
        self.opex['sim'] = (self.block.flows[self.flow_name] @ self.opex['spec'][self.block.scenario.dti_sim] *
                            self.block.scenario.timestep_hours) if self.flow_name is not None else 0
        self.opex['yrl'] = utils.scale_sim2year(self.opex['sim'], self.block.scenario)
        self.opex['prj'] = utils.scale_year2prj(self.opex['yrl'], self.block.scenario)
        self.opex['dis'] = acc_discount(nominal_value=self.opex['yrl'],
                                        observation_horizon=self.block.scenario.prj_duration_yrs,
                                        discount_rate=self.block.scenario.wacc,
                                        occurs_at='end')
        self.opex['ann'] = annuity_recur(nominal_value=self.opex['yrl'],
                                         observation_horizon=self.block.scenario.prj_duration_yrs,
                                         discount_rate=self.block.scenario.wacc)
        # endregion

        # region crev
        self.crev['sim'] = (self.block.flows[self.flow_name] @ self.crev['spec'][self.block.scenario.dti_sim] *
                            self.block.scenario.timestep_hours) if self.flow_name is not None else 0
        self.crev['yrl'] = utils.scale_sim2year(self.crev['sim'], self.block.scenario)
        self.crev['prj'] = utils.scale_year2prj(self.crev['yrl'], self.block.scenario)
        self.crev['dis'] = acc_discount(nominal_value=self.crev['yrl'],
                                        observation_horizon=self.block.scenario.prj_duration_yrs,
                                        discount_rate=self.block.scenario.wacc,
                                        occurs_at='end')
        self.crev['ann'] = annuity_recur(nominal_value=self.crev['yrl'],
                                         observation_horizon=self.block.scenario.prj_duration_yrs,
                                         discount_rate=self.block.scenario.wacc)
        # endregion

        super().aggregate_post_scenario(target=self.block.aggregator)

    def get_size(self,
                 size_name: str,
                 scope_name: str,
                 default_value: float = 0) -> float:
        """
        get a value from the block's size df
        """
        return self.block.sizes.loc[size_name, scope_name] if size_name in self.block.sizes.index else default_value


class FleetUnitEvaluator(EconomicEvaluator):
    """
    dist and time shit
    """
    pass


class PeakEvaluator(EconomicEvaluator):

    def pre_scenario(self):

        # get and set the opex_spec at the first timestep of the peakshaving period
        self.opex['spec'] = self.opex['spec'][self.block.peakshaving_periods.loc[self.name, 'start']]

        self.opex['factor_ep'] = (self.block.n_peakshaving_periods_yr / self.block.peakshaving_periods.shape[0]
                                  if self.block.scenario.compensate_sim_prj else 1)

        self.opex['spec_ep'] = self.opex['spec'] * self.opex['factor_ep']

    def post_scenario(self):

        self.opex['sim'] = self.block.peakshaving_periods.loc[self.name, 'power'] * self.opex['spec'] *\
                           self.block.peakshaving_periods.loc[self.name, 'period_fraction']
        self.opex['yrl'] = utils.scale_sim2year(self.opex['sim'], self.block.scenario)
        self.opex['prj'] = utils.scale_year2prj(self.opex['yrl'], self.block.scenario)
        self.opex['dis'] = acc_discount(nominal_value=self.opex['yrl'],
                                        observation_horizon=self.block.scenario.prj_duration_yrs,
                                        discount_rate=self.block.scenario.wacc,
                                        occurs_at='end')
        self.opex['ann'] = annuity_recur(nominal_value=self.opex['yrl'],
                                         observation_horizon=self.block.scenario.prj_duration_yrs,
                                         discount_rate=self.block.scenario.wacc)

        self.aggregate_post_scenario(target=self.block.aggregator)


