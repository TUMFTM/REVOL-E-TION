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


def acc_discount(nominal_value: float,
                 observation_horizon: int,
                 discount_rate: float,
                 occurs_at='beginning') -> float:
    """
    This function calculates the accumulated present value of a periodical, nominally repeating cashflow in the future
    (from present to the observation horizon) at a discount rate per period
    """
    p_delta = {'beginning': 0, 'middle': 0.5, 'end': 1}[occurs_at]
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


class EconomicResults:
    def __init__(self):
        self._capex = pd.DataFrame(columns=['init', 'yrl', 'prj', 'dis'])
        # ToDo: include
        #  capex_replacement     ->  cost for replacement
        #  capex_init_existing   ->  cost for existing size
        #  capex_init_additional ->  cost for additional size
        #  capex_joined_spec     ->  capex + mntex

        self._mntex = pd.DataFrame(columns=['sim', 'yrl', 'prj', 'dis'])
        self._opex = pd.DataFrame(columns=['sim', 'yrl', 'prj', 'dis'])
        self._crev = pd.DataFrame(columns=['sim', 'yrl', 'prj', 'dis'])
        self._cashflows = None
        self._totex = None

        # todo cashflows
        pass


class PointOfEvaluation:

    def __init__(self,
                 name: str,
                 block,
                 aggregator: bool = False,
                 ):

        self.name = name
        self.block = block
        self.aggregator = aggregator
        self.ls = getattr(self.block, 'ls', self.block.scenario.prj_duration_yrs)
        self.ccr = getattr(self.block, 'ccr', 1)

        value_types = ['capex', 'mntex', 'opex', 'crev']

        self.capex = {'fix': 0, 'preexisting': 0, 'spec': 0}
        self.mntex = {'fix': 0, 'spec': 0}
        self.opex = {'spec': self.scalar_to_ts(value=0)}
        self.crev = dict()

        # ToDo: also get ls and ccr if specified for this specific poe

        # region get all values from block
        for key in [key for key in self.block.__dict__.keys()
                    if key.startswith(tuple(value_types))]:
            param_split = key.split('_')
            if len(param_split) == 2:  # single cost occurrence location in block
                param_split.append('block')
            value_type, cause, poe = param_split  # examples: capex, spec, g2s

            if poe == self.name:
                if value_type in ['capex', 'mntex']:
                    getattr(self, value_type)[cause] = getattr(self.block, key)
                else:  # opex, crev
                    getattr(self, value_type)[cause] = self.scalar_to_ts(value=getattr(self.block, key))
                delattr(self.block, key)
        # endregion

        # region calculate equivalent present specific capex and opex for optimizer
        # include (time-based) maintenance expenses in capex calculation as equivalent present cost
        self.capex['spec_joined'] = join_capex_mntex(capex=self.capex.get('spec', 0),
                                                     mntex=self.mntex.get('spec', 0),
                                                     lifespan=self.ls,
                                                     discount_rate=self.block.scenario.wacc)

        # annuity due factor (incl. replacements) to compensate for difference between simulation and project time in
        # component sizing; ep = equivalent present (i.e. specific values prediscounted)
        self.capex['factor_ep'] = annuity_due_capex(capex_init=1,
                                                    capex_replacement=1,
                                                    lifespan=self.ls,
                                                    observation_horizon=self.block.scenario.prj_duration_yrs,
                                                    discount_rate=self.block.scenario.wacc,
                                                    cost_change_ratio=self.ccr)\
            if self.block.scenario.compensate_sim_prj else 1

        self.capex['spec_ep'] = self.capex['spec_joined'] * self.capex['factor_ep']

        # runtime factor to compensate for difference between simulation and project timeframe
        # opex is uprated in importance for short simulations
        self.opex['factor_ep'] = annuity_recur(nominal_value=utils.scale_sim2year(1, self.block.scenario),
                                               observation_horizon=self.block.scenario.prj_duration_yrs,
                                               discount_rate=self.block.scenario.wacc)\
            if self.block.scenario.compensate_sim_prj else 1

        self.opex['spec_ep'] = self.opex['spec'] * self.opex['factor_ep']
        # endregion

        # region calculate capital expenses for preexisting block size
        size_preexisting = self.block.sizes.loc[self.name, 'preexisting'] if self.name in self.block.sizes.index else 0
        self.capex['preexisting'] = (self.capex['preexisting'] *  # boolean so far - will be overwritten
                                     (size_preexisting * self.capex['spec'] + self.capex['fix']))
        if not aggregator:
            self.block.poes['total'].capex['preexisting'] += self.capex['preexisting']
        # endregion

    def scalar_to_ts(self,
                     value: str | float):
        """
        Transform scalar value or filename (contents) to timeseries to be able to calculate the economic results using
        dot product with the block's flows
        """

        if isinstance(value, str):
            ts = utils.read_input_csv(block=self.block,
                                      path_input_file=os.path.join(self.block.scenario.run.path_input_data,
                                                                   utils.set_extension(value)),
                                      scenario=self.block.scenario)
            if ts.shape[1] != 1:  # todo: fix this. value_type, cause, component are not available anymore
                raise ValueError(f'Input file "{utils.set_extension(value)}" for parameter '
                                 f'"{value_type}_{cause}{f"_{component}" if component != "block" else ""}" '
                                 f'in block "{self.block.name}" has more than one column')

            ts = ts.loc[self.block.scenario.dti_sim_extd, ts.columns[0]]  # convert to series and slice to sim timeframe
        else:  # scalar value
            ts = pd.Series(index=self.block.scenario.dti_sim_extd,
                           data=value)

        return ts

