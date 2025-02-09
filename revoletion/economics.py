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
    mainenance expense (the equivalent yearly sum to generate the same NPV) over an observation horizon.
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


@abstractclass
class EconomicPointOfInterest:

    def __init__(self,
                 name: str,
                 parent: 'EconomicAggregator'):

        self.name = name
        self.parent = parent


class EconomicAggregator(EconomicPointOfInterest):

    def __init__(self,
                 name: str,
                 parent: 'EconomicAggregator',
                 scenario,
                 block=None):

        super().__init__(name=name,
                         parent=parent)

        self.scenario = scenario  # might be needed as block is not mandatory for PoAs
        self.block = block

        self.capex = getattr(self, 'capex', dict())
        self.capex.update({'fix': 0,
                           'preexisting': 0,
                           'expansion': 0,
                           'init': 0,
                           'replacement': 0,
                           'prj': 0,
                           'dis': 0,
                           'ann': 0})
        self.mntex = getattr(self, 'mntex', dict())
        self.mntex.update({'fix': 0,
                           'yrl': 0,
                           'sim': 0,
                           'prj': 0,
                           'dis': 0,
                           'ann': 0})
        self.opex = getattr(self, 'opex', dict())
        self.opex.update({'yrl': 0,
                          'sim': 0,
                          'prj': 0,
                          'dis': 0,
                          'ann': 0})
        self.crev = dict()
        self.aux_params = dict()

        self.get_block_params()
        self.pre_scenario()

        if self.parent is not None:
            self.parent.capex['preexisting'] += self.capex['preexisting']

    def get_block_params(self):
        """
        default method
        """
        pass

    def pre_scenario(self):
        """
        default method
        """
        pass

    def post_scenario(self):
        """
        Aggregate economic results one level up
        """
        if self.parent is not None:
            for key in ['expansion', 'init', 'replacement', 'prj', 'dis', 'ann']:
                self.parent.capex[key] += self.capex[key]
            for key in ['yrl', 'sim', 'prj', 'dis', 'ann']:
                self.parent.mntex[key] += self.mntex[key]
                self.parent.opex[key] += self.opex[key]


@ abstractclass
class EconomicEvaluator(EconomicPointOfInterest):

    def __init__(self,
                 name: str,
                 parent,
                 scenario,
                 block,
                 opex_flow_name: str = None):

        self.scenario = scenario

        self.capex = {'spec': 0}
        self.mntex = {'spec': 0}
        self.opex = {'spec': utils.transform_scalar_var(value=0,
                                                        scenario=self.scenario,
                                                        block=block)}

        super().__init__(name=name,
                         parent=parent,
                         scenario=scenario,
                         block=block)

        self.opex_flow_name = opex_flow_name


    def get_block_params(self):
        """
        Get the block's parameters for the specific point of evaluation and combine them in the respective dictionaries
        """

        self.aux_params['ls'] = getattr(self.block, 'ls', self.block.scenario.prj_duration_yrs)
        self.aux_params['ccr'] = getattr(self.block, 'ccr', 1)

        for key in [key for key in self.block.__dict__.keys()
                    if key.startswith(('capex', 'mntex', 'opex', 'crev'))]:
            param_split = key.split('_')
            if len(param_split) == 2:  # single cost occurrence location in block
                param_split.append('block')
            value_type, cause, poe = param_split  # examples: capex, spec, g2s

            if poe == self.name:
                if value_type in ['capex', 'mntex']:
                    getattr(self, value_type)[cause] = getattr(self.block, key)
                else:  # opex, crev
                    getattr(self, value_type)[cause] = utils.transform_scalar_var(value=getattr(self.block, key),
                                                                                  scenario=self.scenario,
                                                                                  block=self.block)
                delattr(self.block, key)

    def pre_scenario(self):

        # region calculate equivalent present specific capex and opex for optimizer
        # include (time-based) maintenance expenses in capex calculation as equivalent present cost
        self.capex['spec_joined'] = join_capex_mntex(capex=self.capex.get('spec', 0),
                                                     mntex=self.mntex.get('spec', 0),
                                                     lifespan=self.aux_params['ls'],
                                                     discount_rate=self.block.scenario.wacc)

        # annuity due factor (incl. replacements) to compensate for difference between simulation and project time in
        # component sizing; ep = equivalent present (i.e. specific values prediscounted)
        self.capex['factor_ep'] = annuity_due_capex(capex_init=1,
                                                    capex_replacement=1,
                                                    lifespan=self.aux_params['ls'],
                                                    observation_horizon=self.block.scenario.prj_duration_yrs,
                                                    discount_rate=self.block.scenario.wacc,
                                                    cost_change_ratio=self.aux_params['ccr'])\
            if self.block.scenario.compensate_sim_prj else 1

        self.capex['spec_ep'] = self.capex['spec_joined'] * self.capex['factor_ep']


        # runtime factor to compensate for difference between simulation and project timeframe
        # opex is uprated in importance for short simulations
        self.calc_opex_factor_ep()

        self.opex['spec_ep'] = self.opex['spec'] * self.opex['factor_ep']
        # endregion


        # region calculate capital expenses for preexisting block size
        size_preexisting = self.block.sizes.loc[self.name, 'preexisting'] if self.name in self.block.sizes.index else 0
        self.capex['preexisting'] = (self.capex['preexisting'] *  # boolean so far - will be overwritten
                                     (size_preexisting * self.capex['spec'] + self.capex['fix']))
        # endregion

    def get_block_params(self):

        self.aux_params['ls'] = getattr(self.block, 'ls', self.block.scenario.prj_duration_yrs)
        self.aux_params['ccr'] = getattr(self.block, 'ccr', 1)

        for key in [key for key in self.block.__dict__.keys()
                    if key.startswith(('capex', 'mntex', 'opex', 'crev'))]:
            param_split = key.split('_')
            if len(param_split) == 2:  # single cost occurrence location in block
                param_split.append('block')
            value_type, cause, poe = param_split  # examples: capex, spec, g2s

            if poe == self.name:
                if value_type in ['capex', 'mntex']:
                    getattr(self, value_type)[cause] = getattr(self.block, key)
                else:  # opex, crev
                    getattr(self, value_type)[cause] = utils.transform_scalar_var(value=getattr(self.block, key),
                                                                                  scenario=self.scenario,
                                                                                  block=self.block)
                delattr(self.block, key)

    def calc_opex_factor_ep(self):
        self.opex['factor_ep'] = annuity_recur(nominal_value=utils.scale_sim2year(1, self.block.scenario),
                                               observation_horizon=self.block.scenario.prj_duration_yrs,
                                               discount_rate=self.block.scenario.wacc)\
            if self.block.scenario.compensate_sim_prj else 1

    def post_scenario(self):

        # region capex
        self.capex['expansion'] = self.capex['spec'] * self.block.sizes.loc[self.name, 'expansion']
        self.capex['init'] = self.capex['preexisting'] + self.capex['expansion']
        self.capex['replacement'] = self.capex['spec'] * self.block.sizes.loc[self.name, 'total'] + self.capex['fix']
        self.capex['prj'] = capex_sum(capex_init=self.capex['init'],
                                      capex_replacement=self.capex['replacement'],
                                      cost_change_ratio=self.aux_params['ccr'],
                                      lifespan=self.aux_params['ls'],
                                      observation_horizon=self.scenario.prj_duration_yrs)
        self.capex['dis'] = capex_present(capex_init=self.capex['init'],
                                          capex_replacement=self.capex['replacement'],
                                          cost_change_ratio=self.aux_params['ccr'],
                                          discount_rate=self.scenario.wacc,
                                          lifespan=self.aux_params['ls'],
                                          observation_horizon=self.scenario.prj_duration_yrs)
        self.capex['ann'] = annuity_due_capex(capex_init=self.capex['init'],
                                              capex_replacement=self.capex['replacement'],
                                              lifespan=self.aux_params['ls'],
                                              observation_horizon=self.scenario.prj_duration_yrs,
                                              discount_rate=self.scenario.wacc,
                                              cost_change_ratio=self.aux_params['ccr'])
        # endregion

        # region mntex
        self.mntex['yrl'] = self.block.sizes.loc[self.name, 'total'] * self.mntex['spec']
        self.mntex['sim'] = self.mntex['yrl'] * self.scenario.sim_yr_rat
        self.mntex['prj'] = utils.scale_year2prj(self.mntex['yrl'], self.scenario)
        self.mntex['dis'] = acc_discount(nominal_value=self.mntex['yrl'],
                                         observation_horizon=self.scenario.prj_duration_yrs,
                                         discount_rate=self.scenario.wacc,
                                         occurs_at='beginning')
        self.mntex['ann'] = annuity_due_recur(nominal_value=self.mntex['yrl'],
                                              observation_horizon=self.scenario.prj_duration_yrs,
                                              discount_rate=self.scenario.wacc)
        # endregion

        # region opex
        self.opex['sim'] = self.block.flows[self.opex_flow_name] @ self.opex['spec'][self.scenario.dti_sim] \
            if self.opex_flow_name is not None else 0
        self.opex['yrl'] = utils.scale_sim2year(self.opex['sim'], self.scenario)
        self.opex['prj'] = utils.scale_year2prj(self.opex['yrl'], self.scenario)
        self.opex['dis'] = acc_discount(nominal_value=self.opex['yrl'],
                                        observation_horizon=self.scenario.prj_duration_yrs,
                                        discount_rate=self.scenario.wacc,
                                        occurs_at='end')
        self.opex['ann'] = annuity_recur(nominal_value=self.opex['yrl'],
                                         observation_horizon=self.scenario.prj_duration_yrs,
                                         discount_rate=self.scenario.wacc)
        # endregion

        super().post_scenario()


# generally: blocks get a dict like this self.epois = {self.name: EconomicAggregator, 'storage': CapexEvaluator, 'inflow': OpexEvaluator, 'outflow': OpexEvaluator}
class CapexEvaluator(EconomicEvaluator):
    # covers size_names / investable parameter
    # this one gets the functions join_capex_mntex, annuity_due_capex, reinvest_periods etc. as methods
    pass


class OpexEvaluator(EconomicEvaluator):
    # covers flow_names parameter
    pass


class PeakEvaluator(EconomicEvaluator):

    def get_block_params(self):

        self.aux_params['ls'] = getattr(self.block, 'ls', self.block.scenario.prj_duration_yrs)
        self.aux_params['ccr'] = getattr(self.block, 'ccr', 1)

        # get opex spec timeseries
        opex_spec_ts = utils.transform_scalar_var(value=self.block.opex_spec_peak,
                                                  scenario=self.scenario,
                                                  block=self.block)

        # get and set the opex_spec at the first timestep of the peakshaving period
        self.opex['spec'] = opex_spec_ts[self.block.peakshaving_periods.loc[self.name, 'start']]

    def calc_opex_factor_ep(self):
        self.opex['factor_ep'] = (self.block.n_peakshaving_periods_yr / self.block.peakshaving_periods.shape[0]
                                  if self.block.scenario.compensate_sim_prj else 1)

    def post_scenario(self):

        self.opex['sim'] = self.block.flows[self.opex_flow_name] @ self.opex['spec'][self.scenario.dti_sim] \
            if self.opex_flow_name is not None else 0
        self.opex['yrl'] = utils.scale_sim2year(self.opex['sim'], self.scenario)
        self.opex['prj'] = utils.scale_year2prj(self.opex['yrl'], self.scenario)
        self.opex['dis'] = acc_discount(nominal_value=self.opex['yrl'],
                                        observation_horizon=self.scenario.prj_duration_yrs,
                                        discount_rate=self.scenario.wacc,
                                        occurs_at='end')
        self.opex['ann'] = annuity_recur(nominal_value=self.opex['yrl'],
                                         observation_horizon=self.scenario.prj_duration_yrs,
                                         discount_rate=self.scenario.wacc)

        super(EconomicEvaluator, self).post_scenario()  # call EconomicAggregator.post_scenario(self)


