#!/usr/bin/env python3

import pandas as pd

from revoletion import utils


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

        self.capex = {'preexisting': 0.0,
                      'expansion': 0.0,
                      'init': 0.0,
                      'replacement': 0.0,
                      'prj': 0.0,
                      'dis': 0.0,
                      'ann': 0.0}
        self.mntex = {'yrl': 0.0,
                      'sim': 0.0,
                      'prj': 0.0,
                      'dis': 0.0,
                      'ann': 0.0}
        self.opex = {'sim': 0.0,
                     'yrl': 0.0,
                     'prj': 0.0,
                     'dis': 0.0,
                     'ann': 0.0}
        self.crev = {'sim': 0.0,
                     'yrl': 0.0,
                     'prj': 0.0,
                     'dis': 0.0,
                     'ann': 0.0}

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

        # ToDo: aggregate cashflow dataframes

        for key in ['expansion', 'init', 'replacement', 'prj', 'dis', 'ann']:
            target.capex[key] += self.capex[key]
        for key in ['sim', 'yrl', 'prj', 'dis', 'ann']:
            target.mntex[key] += self.mntex[key]
            target.opex[key] += self.opex[key]
            target.crev[key] += self.crev[key]


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
        self.capex.update({'spec': 0.0,
                           'fix': 0.0,})
        self.mntex.update({'spec': 0.0,
                           'fix': 0.0})
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
            else:  # capex, mntex, aux
                getattr(self, dict_name)[dict_key] = getattr(self.block, param_name, 0.0)
        # endregion

        self.pre_scenario()
        self.aggregate_pre_scenario(target=self.block.aggregator)

    def pre_scenario(self):

        # region calculate equivalent present specific capex and opex for optimizer

        # calculate specific present value of capex for the project duration
        # initialize Series for all years of the project duration
        self.capex['spec_prj_ep'] = pd.Series(index=self.discount_factors.index,
                                              data=0.0)

        # apply specific capex for replacement periods
        self.capex['spec_prj_ep'].loc[reinvest_periods(lifespan=self.aux['ls'],
                                                       observation_horizon=self.scenario.prj_duration_yrs,
                                                       include_init=True)] = self.capex['spec']

        # apply specific salvage value after project duration considering the remaining lifespan
        # salvage values occur at the end of the last year of the project duration but are modeled at the beginning of
        # the next year to use the same discount factor ('beginning') and avoid issues when a replacement occurs at the
        # beginning of the last project year
        self.capex['spec_prj_ep'].loc[self.scenario.prj_duration_yrs] = (
                -1 * self.capex['spec'] * calc_frac_remaining_ls(ls=self.aux['ls'],
                                                                 project_duration=self.scenario.prj_duration_yrs)
        )

        # adjust specific capex by appropriate cost change ratio
        self.capex['spec_prj_ep'] *= pd.Series(index=self.discount_factors.index,
                                               data=self.aux['ccr'] ** self.discount_factors.index)

        # sum up all specific discounted capex for the project duration
        self.capex['spec_prj_ep'] = self.capex['spec_prj_ep'] @ self.discount_factors['beginning']

        # calculate specific present value of mntex for the project duration
        self.mntex['spec_prj_ep'] = acc_discount(nominal_value=self.mntex['spec'],
                                                 observation_horizon=self.scenario.prj_duration_yrs,
                                                 discount_rate=self.scenario.wacc,
                                                 occurs_at='beginning')

        # join maintenance and capex specific present values for the project duration
        self.capex['spec_prj_ep_joined'] = self.capex['spec_prj_ep'] + self.mntex['spec_prj_ep']

        # calculate annuity due factor to compensate for difference between simulation and project time
        self.capex['factor_ep'] = annuity(present_value=1,
                                          observation_horizon=self.scenario.prj_duration_yrs,
                                          discount_rate=self.scenario.wacc,
                                          occurs_at='beginning')\
            if self.scenario.compensate_sim_prj else 1

        # calculate specific capex/mntex value used for the optimization problem
        self.capex['spec_opt'] = self.capex['spec_prj_ep_joined'] * self.capex['factor_ep']

        # runtime factor to compensate for difference between simulation and project timeframe
        # annuity is equal to the already yearly recurring expense
        self.opex['factor_ep'] = (1 / self.scenario.sim_yr_rat) if self.scenario.compensate_sim_prj else 1
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
        self.cashflows.loc[0, 'capex'] -= self.capex['init']

        # Add replacement capex
        self.capex['replacement'] = (self.capex['spec'] * self.get_size(self.size_name, 'total') +
                                     self.capex['fix'])
        for period in reinvest_periods(lifespan=self.aux['ls'],
                                       observation_horizon=self.scenario.prj_duration_yrs,
                                       include_init=False):
            self.cashflows.loc[period, 'capex'] -= self.capex['replacement'] * (self.aux['ccr'] ** period)

        # Add salvage value capex (positive cashflow)
        self.cashflows.loc[self.scenario.prj_duration_yrs, 'capex'] += (
                self.capex['replacement'] * (self.aux['ccr'] ** self.scenario.prj_duration_yrs) *
                calc_frac_remaining_ls(ls=self.aux['ls'],
                                       project_duration=self.scenario.prj_duration_yrs)
        )

        self.capex['prj'] = -1 * self.cashflows['capex'].sum()
        self.capex['dis'] = -1 * self.cashflows['capex'] @ self.discount_factors['beginning']
        self.capex['ann'] = annuity(present_value=self.capex['dis'],
                                    observation_horizon=self.scenario.prj_duration_yrs,
                                    discount_rate=self.scenario.wacc,
                                    occurs_at='beginning')
        # endregion

        # region mntex
        self.mntex['yrl'] = self.get_size(self.size_name, 'total') * self.mntex['spec'] + self.mntex['fix']
        self.cashflows.loc[self.scenario.periods_prj, 'mntex'] = -1 * self.mntex['yrl']
        self.mntex['sim'] = self.mntex['yrl'] * self.scenario.sim_yr_rat

        self.mntex['prj'] = -1 * self.cashflows['mntex'].sum()
        self.mntex['dis'] = -1 * self.cashflows['mntex'] @ self.discount_factors['beginning']
        self.mntex['ann'] = annuity(present_value=self.mntex['dis'],
                                    observation_horizon=self.scenario.prj_duration_yrs,
                                    discount_rate=self.scenario.wacc,
                                    occurs_at='beginning')
        # endregion

        # region opex
        self.opex['sim'] = (self.block.flows.loc[self.scenario.dti_eval, self.flow_name] @ self.opex['spec'][self.scenario.dti_eval] *
                            self.scenario.timestep_hours) if self.flow_name is not None else 0
        self.calc_opex_sim_additional()

        self.opex['yrl'] = self.opex['sim'] / self.scenario.sim_yr_rat
        self.cashflows.loc[self.scenario.periods_prj, 'opex'] = -1 * self.opex['yrl']

        self.opex['prj'] = -1 * self.cashflows['opex'].sum()
        self.opex['dis'] = -1 * self.cashflows['opex'] @ self.discount_factors['end']
        self.opex['ann'] = annuity(present_value=self.opex['dis'],
                                   observation_horizon=self.scenario.prj_duration_yrs,
                                   discount_rate=self.scenario.wacc,
                                   occurs_at='end')
        # endregion

        # region crev
        self.crev['sim'] = (self.block.flows.loc[self.scenario.dti_eval, self.flow_name] @ self.crev['spec'][self.scenario.dti_eval] *
                            self.scenario.timestep_hours) if self.flow_name is not None else 0
        self.calc_crev_sim_additional()

        self.crev['yrl'] = self.crev['sim'] / self.scenario.sim_yr_rat
        self.cashflows.loc[self.scenario.periods_prj, 'crev'] = self.crev['yrl']

        self.crev['prj'] = self.cashflows['crev'].sum()
        self.crev['dis'] = self.cashflows['crev'] @ self.discount_factors['end']
        self.crev['ann'] = annuity(present_value=self.crev['dis'],
                                   observation_horizon=self.scenario.prj_duration_yrs,
                                   discount_rate=self.scenario.wacc,
                                   occurs_at='end')
        # endregion

        super().aggregate_post_scenario(target=self.block.aggregator)

    def get_size(self,
                 size_name: str,
                 scope_name: str,
                 default_value: float = 0) -> float:
        """
        get a value from the block's size df
        """

        if size_name in self.block.sizes.index:
            value = self.block.sizes.loc[size_name, scope_name]
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
