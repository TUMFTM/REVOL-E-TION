#!/usr/bin/env python3

import ast
import io
import numpy as np
import oemof.solph as solph
import os
import pandas as pd
import plotly.graph_objects as go
import pvlib
import requests
import statistics
import windpowerlib

from revoletion import battery as bat
from revoletion import economics as eco
from revoletion import mobility
from revoletion import utils


class Block:

    def __init__(self, name, scenario, flow_names=['total', 'in', 'out']):
        self.name = name
        self.scenario = scenario
        self.scenario.blocks[self.name] = self

        self.parameters = self.scenario.parameters.loc[self.name]
        for key, value in self.parameters.items():
            setattr(self, key, value)  # this sets all the parameters defined in the scenario file

        self.subblocks = {}

        time_var_params = [var for var in vars(self) if ('opex_spec' in var) or ('crev_spec' in var)]
        # Don't transform variables for GridConnections, as the GridMarket opex defined specifically
        if not isinstance(self, GridConnection):
            for var in time_var_params:
                utils.transform_scalar_var(self, var)

        # Empty result series
        # flow direction is specified with respect to the block -> "in" is from energy system into block
        self.flows = pd.DataFrame(index=self.scenario.dti_sim,
                                  columns=flow_names,
                                  data=0,
                                  dtype='float64')

        # Placeholder for storage timeseries results
        self.storage_timeseries = None

        # Empty result scalar variables
        self.energies = pd.DataFrame(index=['total', 'in', 'out', 'del', 'pro'],
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,
                                     dtype=float)

        self.expenditures = utils.create_expenditures_dataframe()
        self.capex_fix = 0  # size agnostic cost component (e.g. for vehicle glider and charger)

        self.cashflows = pd.DataFrame()

        self.apriori_data = None

    def accumulate_crev(self):
        """
        crev_sim is calculated beforehand for the individual blocks
        """

        self.expenditures.loc['crev', 'yrl'] = utils.scale_sim2year(self.expenditures.loc['crev', 'sim'], self.scenario)
        self.expenditures.loc['crev', 'prj'] = utils.scale_year2prj(self.expenditures.loc['crev', 'yrl'], self.scenario)
        self.expenditures.loc['crev', 'dis'] = eco.acc_discount(self.expenditures.loc['crev', 'yrl'],
                                                                self.scenario.prj_duration_yrs,
                                                                self.scenario.wacc,
                                                                occurs_at='end')

        self.scenario.expenditures.loc['crev', :] += self.expenditures.loc['crev', :]  # sim, yrl, prj, dis

    def accumulate_expenses(self):
        # add all expenditures (capex, mntex, opex) for the current block
        self.expenditures.loc['totex', ['sim', 'prj', 'dis', 'ann']] = self.expenditures.loc[['capex', 'mntex', 'opex'], ['sim', 'prj', 'dis', 'ann']].sum()

        self.scenario.expenditures.loc['totex', :] += self.expenditures.loc['totex', :]  # sim, prj, dis, ann

    def add_power_trace(self):
        legentry = self.get_legend_entry()
        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=self.flows['total'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None)),
                                       secondary_y=False)

        for subblock in self.subblocks.values():
            subblock.add_power_trace()

    def calc_cashflows(self):
        """
        Collect nominal cashflows for the block for each year in the project.
        """

        capex = pd.Series(dtype='float64', index=range(self.scenario.prj_duration_yrs), data=0)
        capex[0] = self.expenditures.loc['capex', 'sim']
        if hasattr(self, 'ls'):
            for year in eco.reinvest_periods(lifespan=self.ls,
                                           observation_horizon=self.scenario.prj_duration_yrs):
                capex[year] = self.capex_replacement * (self.ccr ** year)
        self.cashflows[f'capex_{self.name}'] = -1 * capex  # expenses are negative cashflows (outgoing)
        self.cashflows[f'mntex_{self.name}'] = -1 * self.expenditures.loc['mntex', 'yrl']
        self.cashflows[f'opex_{self.name}'] = -1 * self.expenditures.loc['opex', 'yrl']
        self.cashflows[f'crev_{self.name}'] = self.expenditures.loc['crev', 'yrl']

        self.scenario.cashflows = pd.concat([self.scenario.cashflows, self.cashflows], axis=1)

    def calc_energy_storage_block(self):
        """
        Calculate the energy results for bidirectional blocks (CommoditySystems and StationaryEnergyStorages).
        Bidirectional blocks can be either counted towards energy production or delivery, depending on their balance.
        """
        self.calc_energy_common(flow_names=['in', 'out'])

        if self.energies.loc['in', 'sim'] > self.energies.loc['out', 'sim']:
            mode = 'del'
            self.energies.loc['del', 'sim'] = self.energies.loc['in', 'sim'] - self.energies.loc['out', 'sim']
        else:  # storage was emptied
            mode = 'pro'
            self.energies.loc['pro', 'sim'] = self.energies.loc['out', 'sim'] - self.energies.loc['in', 'sim']

        self.energies.loc[mode, 'yrl'] = utils.scale_sim2year(self.energies.loc[mode, 'sim'], self.scenario)
        self.energies.loc[mode, 'prj'] = utils.scale_year2prj(self.energies.loc[mode, 'yrl'], self.scenario)
        self.energies.loc[mode, 'dis'] = eco.acc_discount(self.energies.loc[mode, 'yrl'],
                                                           self.scenario.prj_duration_yrs,
                                                           self.scenario.wacc,
                                                           occurs_at='end')

        self.scenario.energies.loc[mode, :] += self.energies.loc[mode, :]

    def calc_energy_source_sink(self):
        """
        Accumulating results for sources and sinks
        """
        modes = []
        flow_names = []
        if 'in' in self.flows.columns:
            modes.append('del')
            flow_names.append('in')
        if 'out' in self.flows.columns:
            modes.append('pro')
            flow_names.append('out')

        self.calc_energy_common(flow_names)

        for mode, source_row in zip(modes, flow_names):
            for col in ['sim', 'yrl', 'prj', 'dis']:
                self.energies.loc[mode, col] = self.energies.loc[source_row, col]
                self.scenario.energies.loc[mode, col] += self.energies.loc[mode, col]

    def calc_energy_common(self, flow_names):
        if 'in' in flow_names and 'out' in flow_names:
            self.flows['total'] = self.flows['out'] - self.flows['in']

            if any(~(self.flows['in'] == 0) & ~(self.flows['out'] == 0)):
                self.scenario.logger.warning(f'Block {self.name} - '
                                             f'simultaneous in- and outflow detected!')
        elif 'in' in flow_names:
            self.flows['total'] = -1 * self.flows['in']
        elif 'out' in flow_names:
            self.flows['total'] = self.flows['out']
        else:
            raise ValueError(f'Block {self.name} - no flow names specified')

        for flow_name in flow_names:
            self.energies.loc[flow_name, 'sim'] = self.flows[flow_name].sum() * self.scenario.timestep_hours  # flow values are powers in W --> conversion to Wh
            self.energies.loc[flow_name, 'yrl'] = utils.scale_sim2year(self.energies.loc[flow_name, 'sim'], self.scenario)
            self.energies.loc[flow_name, 'prj'] = utils.scale_year2prj(self.energies.loc[flow_name, 'yrl'], self.scenario)
            self.energies.loc[flow_name, 'dis'] = eco.acc_discount(self.energies.loc[flow_name, 'yrl'],
                                                                   self.scenario.prj_duration_yrs,
                                                                   self.scenario.wacc,
                                                                   occurs_at='end')

    def calc_expenses(self):
        """
        dummy method for code structure simplification.
        Method is called for all block, but only InvestBlocks and ICEVSystems have expenses.
        """
        pass

    def calc_revenue(self):
        """
        dummy function for code structure simplification
        Method is called for all blocks, but only ICEVSystems, BatteryCommoditySystems, VehicleCommoditySystems
        and FixedDemands actually generate revenue
        """
        pass

    def extrapolate_capex(self):
        """
        Extrapolate initial capital investment including replacements to project timeframe and calculate annuity.
        Method is called for all InvestBlocks and ICEVSystems.
        """
        self.expenditures.loc['capex', 'prj'] = eco.capex_sum(capex_init=self.expenditures.loc['capex', 'sim'],
                                                              capex_replacement=self.capex_replacement,
                                                              cost_change_ratio=self.ccr,
                                                              lifespan=self.ls,
                                                              observation_horizon=self.scenario.prj_duration_yrs)
        self.expenditures.loc['capex', 'dis'] = eco.capex_present(capex_init=self.expenditures.loc['capex', 'sim'],
                                                                  capex_replacement=self.capex_replacement,
                                                                  cost_change_ratio=self.ccr,
                                                                  discount_rate=self.scenario.wacc,
                                                                  lifespan=self.ls,
                                                                  observation_horizon=self.scenario.prj_duration_yrs)
        self.expenditures.loc['capex', 'ann'] = eco.annuity_due_capex(capex_init=self.expenditures.loc['capex', 'sim'],
                                                                      capex_replacement=self.capex_replacement,
                                                                      lifespan=self.ls,
                                                                      observation_horizon=self.scenario.prj_duration_yrs,
                                                                      discount_rate=self.scenario.wacc,
                                                                      cost_change_ratio=self.ccr)
        self.scenario.expenditures.loc['capex', :] += self.expenditures.loc['capex', :]  # init, prj, dis, ann

    def extrapolate_mntex(self):
        """
        Extrapolate yearly maintenance expenses to project timeframe and calculate annuity.
        Method is called for all InvestBlocks and ICEVSystems.
        """
        self.expenditures.loc['mntex', 'sim'] = self.expenditures.loc['mntex', 'yrl'] * self.scenario.sim_yr_rat
        self.expenditures.loc['mntex', 'prj'] = utils.scale_year2prj(self.expenditures.loc['mntex', 'yrl'], self.scenario)
        self.expenditures.loc['mntex', 'dis'] = eco.acc_discount(nominal_value=self.expenditures.loc['mntex', 'yrl'],
                                                                 observation_horizon=self.scenario.prj_duration_yrs,
                                                                 discount_rate=self.scenario.wacc,
                                                                 occurs_at='beginning')
        self.expenditures.loc['mntex', 'ann'] = eco.annuity_due_recur(nominal_value=self.expenditures.loc['mntex', 'yrl'],
                                                                      observation_horizon=self.scenario.prj_duration_yrs,
                                                                      discount_rate=self.scenario.wacc,)
        self.scenario.expenditures.loc['mntex',:] += self.expenditures.loc['mntex', :]  # yrl, prj, dis, ann

    def extrapolate_opex(self):
        """
        Extrapolate operational expenses in simulation timeframe to project timeframe and calculate annuity.
        Method is called for all InvestBlocks and ICEVSystems.
        """
        self.expenditures.loc['opex', 'yrl'] = utils.scale_sim2year(self.expenditures.loc['opex', 'sim'], self.scenario)
        self.expenditures.loc['opex', 'prj'] = utils.scale_year2prj(self.expenditures.loc['opex', 'yrl'], self.scenario)
        self.expenditures.loc['opex', 'dis'] = eco.acc_discount(nominal_value=self.expenditures.loc['opex', 'yrl'],
                                                                observation_horizon=self.scenario.prj_duration_yrs,
                                                                discount_rate=self.scenario.wacc,
                                                                occurs_at='end')
        self.expenditures.loc['opex', 'ann'] = eco.annuity_recur(nominal_value=self.expenditures.loc['opex', 'yrl'],
                                                                 observation_horizon=self.scenario.prj_duration_yrs,
                                                                 discount_rate=self.scenario.wacc)
        self.scenario.expenditures.loc['opex', :] += self.expenditures.loc['opex', :]  # sim, yrl, prj, dis, ann

    def get_legend_entry(self):
        """
        Standard legend entry for simple blocks using power as their size
        """
        return f'{self.name} power (max. {self.size.loc["block", "total"] / 1e3:.1f} kW)'

    def get_timeseries_results(self):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        block_ts_results_flows = pd.DataFrame({(self.name, col): self.flows[col] for col in self.flows.columns})
        self.scenario.result_timeseries = pd.concat([self.scenario.result_timeseries, block_ts_results_flows], axis=1)

        if self.storage_timeseries is not None:
            block_ts_results_storage = pd.DataFrame({(self.name, col): self.storage_timeseries[col] for col in self.storage_timeseries.columns})

            self.scenario.result_timeseries = pd.concat([self.scenario.result_timeseries,
                                                         block_ts_results_storage], axis=1)

    def print_results(self):
        # Dummy function; has to be available for every block, but is implemented for InvestBlocks and Subblocks only
        pass


class InvestBlock(Block):
    """
    An InvestBlock is a block that can be optimized in size. It has therefore incurs expenses.
    """

    def __init__(self, name, scenario, flow_names=['total', 'in', 'out'], size_names=[]):
        self.invest = False  # not every block has example parameter invest without extension -> default: False
        super().__init__(name, scenario, flow_names)

        self.size = pd.DataFrame()
        self.set_init_size(size_names=size_names)

        self.capex_init_existing = self.capex_init_additional = self.capex_replacement = 0

        if self.invest and self.scenario.strategy != 'go':
            raise ValueError(f'Block "{self.name}" component size optimization '
                             f'not implemented for any other strategy than "GO"')

        # Include (time-based) maintenance expenses in capex calculation for optimizer
        # results are disaggregated anyway through separate postprocessing
        self.capex_joined_spec = eco.join_capex_mntex(capex=self.capex_spec,
                                                      mntex=self.mntex_spec,
                                                      lifespan=self.ls,
                                                      discount_rate=self.scenario.wacc)

        # annuity due factor (incl. replacements) to compensate for difference between simulation and project time in
        # component sizing; ep = equivalent present (i.e. specific values prediscounted)
        self.factor_capex = eco.annuity_due_capex(capex_init=1,
                                                  capex_replacement=1,
                                                  lifespan=self.ls,
                                                  observation_horizon=self.scenario.prj_duration_yrs,
                                                  discount_rate=self.scenario.wacc,
                                                  cost_change_ratio=self.ccr) if scenario.compensate_sim_prj else 1
        self.capex_ep_spec = self.capex_joined_spec * self.factor_capex

        # runtime factor to compensate for difference between simulation and project timeframe
        # opex is uprated in importance for short simulations
        self.factor_opex = eco.annuity_recur(nominal_value=utils.scale_sim2year(1, self.scenario),
                                             observation_horizon=self.scenario.prj_duration_yrs,
                                             discount_rate=self.scenario.wacc) if scenario.compensate_sim_prj else 1
        self.opex_ep_spec = None  # initial value
        self.calc_opex_ep_spec()  # uprate opex values for short simulations, exact process depends on class

    def calc_capex(self):
        """
        Calculate capital expenses over simulation timeframe and extrapolate to other timeframes.
        """
        self.calc_capex_init()  # initial investment references to different parameters depending on block type
        self.extrapolate_capex()

    def calc_capex_init(self):
        """
        Default function for blocks with a single size value.
        GridConnections, SystemCore and CommoditySystems are more complex and have their own functions
        """
        self.capex_init_additional = self.size.loc['block', 'additional'] * self.capex_spec
        self.expenditures.loc['capex', 'sim'] = self.capex_init_existing + self.capex_init_additional

        # replacements are full cost irrespective of existing size
        self.capex_replacement = self.size.loc['block', 'total'] * self.capex_spec

    def calc_capex_init_existing(self):
        """
        Calculate the initial capital expenses for existing block size.
        This value is required as these expenses have to be taken into account for the scenarios maximum investment.
        """

        # do not use 'block' size if there are other sizes in the block
        for sub_size in (self.size.index.difference(['block']) if len(self.size.index) > 1 else ['block']):
            capex_bool = 'capex_existing' if sub_size == 'block' else f'capex_existing_{sub_size}'
            self.capex_init_existing += self.size.loc[sub_size, 'existing'] * self.capex_spec * getattr(self, capex_bool)

        self.scenario.capex_init_existing += self.capex_init_existing

    def calc_expenses(self):
        """
        Calculate all expenses of an InvestBlock over simulation timeframe and extrapolate to project timeframe.
        """
        self.calc_capex()
        self.calc_mntex()
        self.calc_opex()
        self.accumulate_expenses()

    def calc_mntex(self):
        """
        Calculate maintenance expenses over simulation timeframe and convert to other timeframes.
        Maintenance expenses are solely time-based. Throughput-based maintenance should be included in opex.
        """
        self.calc_mntex_yrl()
        self.extrapolate_mntex()

    def calc_mntex_yrl(self):
        """
        Calculate yearly maintenance expenses
        """
        self.expenditures.loc['mntex', 'yrl'] = self.size.loc['block', 'total'] * self.mntex_spec

    def calc_opex(self):
        """
        Calculate operational expenses over simulation timeframe and convert to other timeframes.
        """
        self.calc_opex_sim()  # opex is defined differently depending on the block type
        self.extrapolate_opex()

    def calc_opex_ep_spec(self):
        """
        Default opex precompensation method for blocks with a single size value.
        GridConnection (g2s/s2g) and CommoditySystem (sys/ext/opex)
        are more complex and have their own methods.
        """
        self.opex_ep_spec = self.opex_spec * self.factor_opex

    def print_results(self):
        if self.invest:
            self.scenario.logger.info(f'Optimized size of component "{self.name}": {self.size.loc["block", "total"] / 1e3:.1f} kW'
                                      f' (existing: {self.size.loc["block", "existing"] / 1e3:.1f} kW'
                                      f' - additional: {self.size.loc["block", "additional"] / 1e3:.1f} kW)')
            for subblock in self.subblocks.values():
                subblock.print_results()

    def set_init_size(self, size_names=[]):
        """
        Set the initial size dataframe based on a given list of sizes.
        """

        sizes = []

        size_names = ['block'] if len(size_names) == 0 else size_names

        for size in size_names:
            size_str = '' if size == 'block' else f'_{size}'
            # size_additional_max: 0=no investment, None=unlimited investment, float=limited investment
            if getattr(self, f'invest{size_str}') and getattr(self, f'size{size_str}_max') is not None:
                size_additional_max = getattr(self, f'size{size_str}_max') - getattr(self, f'size{size_str}_existing')
            elif getattr(self, f'invest{size_str}') and getattr(self, f'size{size_str}_max') is None:
                size_additional_max = None
            else:
                size_additional_max = 0.0

            sizes.append({'total': 0.0,  # total size - determined after optimization
                          'existing': getattr(self, f'size{size_str}_existing'),  # existing size - given parameter
                          'additional': 0.0,  # additional size - determined by optimization
                          'total_max': getattr(self, f'size{size_str}_max'),  # maximum total size - given parameter
                          'additional_max': size_additional_max  # maximum additional size - calculate above
                          })

            delattr(self, f'size{size_str}_existing')
            delattr(self, f'size{size_str}_max')
            delattr(self, f'invest{size_str}')

        self.size = pd.DataFrame(index=size_names,
                                 columns=['total', 'existing', 'additional', 'total_max', 'additional_max'],
                                 data=sizes,
                                 dtype='float64')

        # skipna=False to preserve NaN values; necessary for column additional_max to determine invest status
        self.size.loc['block', :] = self.size.sum(axis=0, skipna=False)

        # set invest
        self.invest = True if any(self.size.loc[:, 'additional_max'] != 0) else False


class SubBlock:

    def __init__(self, name, parent, params, flow_names=['total', 'in', 'out']):
        self.name = name
        self.parent = parent
        self.scenario = self.parent.scenario

        for param, value in params.items():
            setattr(self, param, value)

        self.energies = pd.DataFrame(index=['in', 'out'],
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,
                                     dtype='float64')

        # timeseries result initialization
        self.flows = pd.DataFrame(index=self.scenario.dti_sim,
                                  columns=flow_names,
                                  data=0,
                                  dtype='float64')

    def calc_results(self, flows=[]):
        # energy result calculation does not count towards delivered/produced energy (already done at the system level)
        # calculate energy results for bidi charging at site and external chargers
        for flow in flows:
            self.energies.loc[flow, 'sim'] = self.flows[flow].sum() * self.scenario.timestep_hours  # flow values are powers --> conversion to Wh
            self.energies.loc[flow, 'yrl'] = utils.scale_sim2year(self.energies.loc[flow, 'sim'], self.scenario)
            self.energies.loc[flow, 'prj'] = utils.scale_year2prj(self.energies.loc[flow, 'yrl'], self.scenario)
            self.energies.loc[flow, 'dis'] = eco.acc_discount(self.energies.loc[flow, 'yrl'],
                                                                   self.scenario.prj_duration_yrs,
                                                                   self.scenario.wacc,
                                                                   occurs_at='end')


class RenewableInvestBlock(InvestBlock):

    def __init__(self, name, scenario):

        super().__init__(name, scenario, flow_names=['total', 'out', 'pot', 'curt'])

        self.bus = self.bus_connected = self.exc = self.src = None  # initialization of oemof-solph components

        self.data = self.data_ph = self.input_file_name = self.path_input_file = None  # placeholders, are filled later

        self.e_pot = self.e_curt = 0

        self.get_timeseries_data()

    def add_curtailment_trace(self):
        legentry = f'{self.name} curtailed power'
        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=-1 * self.flows['curt'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=False)

        legentry = f'{self.name} potential power'
        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=self.flows['pot'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=False)

    def calc_energy(self):
        self.calc_energy_source_sink()

        self.e_pot = self.flows['pot'].sum() * self.scenario.timestep_hours  # convert flow powers in W to energy in Wh
        self.e_curt = self.flows['curt'].sum() * self.scenario.timestep_hours

        self.scenario.e_renewable_act += self.energies.loc['out', 'sim']
        self.scenario.e_renewable_pot += self.e_pot
        self.scenario.e_renewable_curt += self.e_curt

    def calc_opex_sim(self):
        self.expenditures.loc['opex', 'sim'] = self.flows['out'] @ self.opex_spec[self.scenario.dti_sim] * self.scenario.timestep_hours

    def get_ch_results(self, horizon):
        # flow values are powers
        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.outflow, self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'pot'] = horizon.results[(self.src, self.bus)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'curt'] = horizon.results[(self.bus, self.exc)]['sequences']['flow'][horizon.dti_ch]

    def get_invest_size(self, horizon):
        self.size.loc['block', 'additional'] = horizon.results[(self.src, self.bus)]['scalars']['invest']
        self.size['total'] = self.size['existing'] + self.size['additional']

    def get_legend_entry(self):
        return f'{self.name} power (nom. {self.size.loc["block", "total"] / 1e3:.1f} kW)'

    def update_input_components(self, horizon):
        """
        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        xc_bus             self.bus
          |                   |
          |<--x----self_out---|<--self_src
          |                   |
          |                   |-->self_exc
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        horizon.components.append(self.bus)

        self.outflow = solph.components.Converter(label=f'{self.name}_out',
                                                  inputs={self.bus: solph.Flow()},
                                                  outputs={self.bus_connected: solph.Flow()},
                                                  conversion_factors={self.bus_connected: self.eff})
        horizon.components.append(self.outflow)

        # Curtailment has to be disincentivized in the optimization to force optimizer to charge storage or commodities
        # instead of curtailment. 2x cost_eps is required as SystemCore also has ccost_eps in charging direction.
        # All other components such as converters and storages only have cost_eps in the output direction.
        self.exc = solph.components.Sink(label=f'{self.name}_exc',
                                         inputs={self.bus: solph.Flow(variable_costs=2 * self.scenario.cost_eps)})
        horizon.components.append(self.exc)

        self.src = solph.components.Source(label=f'{self.name}_src',
                                           outputs={self.bus: solph.Flow(
                                               nominal_value=solph.Investment(ep_costs=self.capex_ep_spec,
                                                                              existing=self.size.loc['block', 'existing'],
                                                                              maximum=utils.conv_add_max(self.size.loc['block', 'additional_max'])),
                                               fix=self.data_ph['power_spec'],
                                               variable_costs=self.opex_ep_spec[horizon.dti_ph])})
        horizon.components.append(self.src)

        horizon.constraints.add_invest_costs(invest=(self.src, self.bus),
                                             capex_spec=self.capex_spec,
                                             invest_type='flow')


class CommoditySystem(InvestBlock):

    def __init__(self, name, scenario):

        super().__init__(name, scenario)

        self.bus = self.bus_connected = self.inflow = self.outflow = None  # initialization of oemof-solph components

        self.pwr_chg_max_observed = self.pwr_dis_max_observed = None
        self.loss_rate = utils.convert_sdr(self.sdr, pd.Timedelta(hours=1))

        self.mode_dispatch = None  # apriori_unlimited, apriori_static, apriori_dynamic, opt_myopic, opt_global
        self.get_dispatch_mode()

        # commodity names might be rewritten in case of imported log data with differing name
        self.com_names = [f'{self.name}_{i}' for i in range(self.num)]
        self.data = self.data_ph = None

        if self.system == 'ac':
            self.eff_chg = self.eff_chg_ac
            self.eff_dis = self.eff_dis_ac
        elif self.system == 'dc':
            self.eff_chg = self.eff_chg_dc
            self.eff_dis = self.eff_dis_dc
        else:
            raise ValueError(f'Block "{self.name}": invalid system type')

        if self.data_source == 'usecases':
            self.usecases = utils.read_usecase_file(self)
            self.demand = self.demand.sample()
        elif self.data_source == 'demand':
            self.demand = utils.read_demand_file(self)
        elif self.data_source == 'log':
            self.data = utils.read_input_log(self)
            self.com_names = self.data.columns.get_level_values(0).unique()[:self.num].tolist()
        else:
            raise ValueError(f'Block "{self.name}": invalid data source')

        if self.data_source in ['usecases', 'demand']:  # dispatch simulation will run
            # estimate maximum power drawn by self discharge
            self.pwr_loss_max = (utils.convert_sdr(self.sdr, self.scenario.timestep_td) *
                                 self.size.loc['block', 'existing'] *
                                 self.scenario.timestep_hours)
            # downrate power for a priori dispatch simulation
            self.pwr_chg_des = (self.pwr_chg * self.eff_chg - self.pwr_loss_max) * self.factor_pwr_des

        self.opex_sys = self.opex_commodities = self.opex_commodities_ext = 0
        self.energies.loc['ext', :] = 0  # results of external charging

        # Get commodity-specific parameters defined on commodity system level
        params_to_inherit = ['invest', 'aging', 'dsoc_buffer', 'mode_dispatch', 'soc_init',
                          'chemistry', 'q_loss_cal_init', 'q_loss_cyc_init',
                          'pwr_chg', 'pwr_dis', 'eff_chg', 'eff_dis', 'eff_storage_roundtrip', 'temp_battery']
        params = {key: getattr(self, key) for key in params_to_inherit}
        # Generate individual commodity instances
        self.subblocks = {com_name: MobileCommodity(com_name, self, params) for com_name in self.com_names}

    def add_soc_trace(self):
        for commodity in self.subblocks.values():
            commodity.add_soc_trace()

    def calc_aging(self, horizon):
        for commodity in self.subblocks.values():
            commodity.calc_aging(horizon)

    def calc_capex_init_existing(self):
        """ Preprocessing method """
        for commodity in self.subblocks.values():
            commodity.calc_capex_init_existing()
            self.capex_init_existing += commodity.capex_init_existing

        self.scenario.capex_init_existing += self.capex_init_existing

    def calc_energy(self):

        self.calc_pwr_max_observed()

        # Aggregate energy results for external charging for all MobileCommodities within the CommoditySystem
        for commodity in self.subblocks.values():
            commodity.calc_results()
            self.energies.loc['ext', ['sim', 'yrl', 'prj', 'dis']] += commodity.energies.loc[['ext_ac', 'ext_dc'], ['sim', 'yrl', 'prj', 'dis']].sum()

        self.calc_energy_storage_block()  # bidirectional block

    def calc_opex_ep_spec(self):
        """
        Uprate opex in importance for short simulations.
        For VehicleCommoditySystems, opex is not considered in optimization (opex_ep_spec = 0 is default initialization)
        as opex is not dependent on battery sizing.
        """
        self.opex_ep_spec = self.opex_spec * self.factor_opex
        self.opex_ep_spec_sys_chg = self.opex_spec_sys_chg * self.factor_opex
        self.opex_ep_spec_sys_dis = self.opex_spec_sys_dis * self.factor_opex
        self.opex_ep_spec_ext_ac = self.opex_spec_ext_ac * self.factor_opex
        self.opex_ep_spec_ext_dc = self.opex_spec_ext_dc * self.factor_opex

    def calc_opex_sim(self):

        self.opex_sys = (self.flows['in'] @ self.opex_spec_sys_chg[self.scenario.dti_sim] +
                         self.flows['out'] @ self.opex_spec_sys_dis[self.scenario.dti_sim])

        for commodity in self.subblocks.values():
            commodity.calc_opex_sim()
            self.opex_commodities += commodity.opex_sim
            self.opex_commodities_ext += commodity.opex_sim_ext

        self.expenditures.loc['opex', 'sim'] = self.opex_sys + self.opex_commodities
        self.expenditures.loc['opex_ext', 'sim'] = self.opex_commodities_ext

        # Calc opex for external charging
        self.expenditures.loc['opex_ext', 'yrl'] = utils.scale_sim2year(self.expenditures.loc['opex_ext', 'sim'], self.scenario)
        self.expenditures.loc['opex_ext', 'prj'] = utils.scale_year2prj(self.expenditures.loc['opex_ext', 'yrl'], self.scenario)
        self.expenditures.loc['opex_ext', 'dis'] = eco.acc_discount(self.expenditures.loc['opex_ext', 'yrl'],
                                                                    self.scenario.prj_duration_yrs,
                                                                    self.scenario.wacc,
                                                                    occurs_at='end')
        self.expenditures.loc['opex_ext', 'ann'] = eco.annuity_recur(nominal_value=self.expenditures.loc['opex_ext', 'yrl'],
                                                                     observation_horizon=self.scenario.prj_duration_yrs,
                                                                     discount_rate=self.scenario.wacc)
        self.scenario.expenditures.loc['opex_ext', :] += self.expenditures.loc['opex_ext', :]  # sim, yrl, prj, dis, ann

    def calc_pwr_max_observed(self):
        """
        Calculate maximum power drawn by the system for external charging and discharging
        """
        self.pwr_chg_max_observed = self.flows['in'].max()
        self.pwr_dis_max_observed = self.flows['out'].max()

    def calc_revenue(self):
        for commodity in self.subblocks.values():
            commodity.calc_revenue()
            self.expenditures.loc['crev', 'sim'] += commodity.crev_sim

        self.accumulate_crev()

    def get_ch_results(self, horizon):
        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[
            (self.outflow, self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[
            (self.bus_connected, self.inflow)]['sequences']['flow'][horizon.dti_ch]

        for commodity in self.subblocks.values():
            commodity.get_ch_results(horizon)

    def get_dispatch_mode(self):
        if self.mode_scheduling in self.scenario.run.apriori_lvls:  # uc, equal, soc, fcfs
            if self.mode_scheduling == 'uc':
                self.mode_dispatch = 'apriori_unlimited'
            elif isinstance(self.power_lim_static, (int, float)):
                self.mode_dispatch = 'apriori_static'
            elif self.power_lim_static is None:
                self.mode_dispatch = 'apriori_dynamic'

        elif self.scenario.strategy == 'rh':
            self.mode_dispatch = 'opt_myopic'
        elif self.scenario.strategy == 'go':
            self.mode_dispatch = 'opt_global'

        # static load management is deactivated for 'uc' mode
        if self.power_lim_static and self.mode_scheduling == 'uc':
            self.scenario.logger.warning(f'CommoditySystem "{self.name}": static load management is not implemented'
                                         f' for scheduling mode "uc". deactivating static load management')
            self.power_lim_static = None

        # ToDo: move to checker.py
        if self.invest and self.mode_scheduling in self.scenario.run.apriori_lvls:
            raise ValueError(f'CommoditySystem "{self.name}": commodity size optimization not '
                             f'implemented for a priori integration levels: {self.scenario.run.apriori_lvls}')

    def get_invest_size(self, horizon):
        """
        Size for the commodity system is the sum of all commodity sizes in results
        """
        for commodity in self.subblocks.values():
            commodity.get_invest_size(horizon)
            self.size.loc['block', :] += commodity.size.loc['block', :]

    def get_legend_entry(self):
        return (f'{self.name} total power'
                f'{f" (static load management {self.power_lim_static / 1e3:.1f} kW)" if self.power_lim_static else ""}')

    def get_timeseries_results(self):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        super().get_timeseries_results()  # this goes up to the Block class
        for commodity in self.subblocks.values():
            commodity.get_timeseries_results()

    def print_results(self):
        # No invest happens for CommoditySystems, only for subblocks
        if self.invest:
            for commodity in self.subblocks.values():
                commodity.print_results()

    def set_init_size(self, size_names):
        super().set_init_size()

        if self.invest and self.data_source in ['usecases', 'demand']:
            self.scenario.logger.warning(f'CommoditySystem "{self.name}": Specified input (active invest and data'
                                         f' source DES is not possible. Deactivated invest.')
            self.invest = False

        # define sizes per commodity
        self.size.loc['pc', :] = self.size.loc['block', :]
        # define sizes for whole commodity system
        self.size.loc['block', :] = self.size.loc['block', :] * self.num

    def update_input_components(self, horizon):
        """
        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        xc_bus         self.bus
          |<-x---cs_xc---|---(MobileCommodity Instance)
          |              |
          |-x----xc_cs-->|---(MobileCommodity Instance)
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        horizon.components.append(self.bus)
        self.bus_connected = self.scenario.blocks['core'].ac_bus if self.system == 'ac' else self.scenario.blocks['core'].dc_bus

        self.inflow = solph.components.Converter(label=f'xc_{self.name}',
                                                 inputs={self.bus_connected: solph.Flow(
                                                     variable_costs=self.opex_ep_spec_sys_chg[horizon.dti_ph],
                                                     nominal_value=self.power_lim_static,
                                                     max=1 if self.power_lim_static else None
                                                 )},
                                                 outputs={self.bus: solph.Flow()},
                                                 conversion_factors={self.bus: 1})

        self.outflow = solph.components.Converter(label=f'{self.name}_xc',
                                                  inputs={self.bus: solph.Flow(
                                                      nominal_value=(self.power_lim_static if self.lvl_cap in ['v2s'] else 0),
                                                      max=(1 if self.power_lim_static is not None else None),
                                                      variable_costs=self.opex_ep_spec_sys_dis[horizon.dti_ph])},
                                                  outputs={self.bus_connected: solph.Flow(
                                                      variable_costs=self.scenario.cost_eps)},
                                                  conversion_factors={self.bus_connected: 1})

        horizon.components.append(self.inflow)
        horizon.components.append(self.outflow)

        for commodity in self.subblocks.values():
            commodity.update_input_components(horizon)


class BatteryCommoditySystem(CommoditySystem):
    """
    Dummy class to keep track of the different commodity system types in the energy system
    """

    def __init__(self, name, scenario):

        self.dsoc_buffer = 0  # necessary as only VehicleCommoditySystem has this as example parameter
        self.demand = mobility.BatteryCommodityDemand(scenario, self)
        super().__init__(name, scenario)

        # only a single target value is set for BatteryCommoditySystems, as these are assumed to always be charged
        # to one SOC before rental
        self.soc_target_high = self.soc_target
        self.soc_target_low = self.soc_target


class ControllableSource(InvestBlock):

    def __init__(self, name, scenario):

        super().__init__(name, scenario, flow_names=['total', 'out'], size_names=[])

        self.bus_connected = self.src = None  # initialize oemof-solph components

    def calc_energy(self):
        self.calc_energy_source_sink()

    def calc_opex_sim(self):
        self.expenditures.loc['opex', 'sim'] = self.flows['out'] @ self.opex_spec[self.scenario.dti_sim] * self.scenario.timestep_hours

    def get_ch_results(self, horizon, *_):
        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.src, self.bus_connected)]['sequences']['flow'][horizon.dti_ch]

    def get_invest_size(self, horizon):
        self.size.loc['block', 'additional'] = horizon.results[(self.src, self.bus_connected)]['scalars']['invest']
        self.size['total'] = self.size['existing'] + self.size['additional']

    def update_input_components(self, horizon):
        """
        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        xc_bus
          |
          |<-x-gen
          |
        """

        self.bus_connected = self.scenario.blocks['core'].ac_bus if self.system == 'ac' else self.scenario.blocks['core'].dc_bus

        self.src = solph.components.Source(label=f'{self.name}_src',
                                           outputs={self.bus_connected: solph.Flow(
                                               nominal_value=solph.Investment(ep_costs=self.capex_ep_spec,
                                                                              existing=self.size.loc['block', 'existing'],
                                                                              maximum=utils.conv_add_max(self.size.loc['block', 'additional_max'])),
                                               variable_costs=self.opex_ep_spec[horizon.dti_ph])})

        horizon.components.append(self.src)

        horizon.constraints.add_invest_costs(invest=(self.src, self.bus_connected),
                                             capex_spec=self.capex_spec,
                                             invest_type='flow')


class GridConnection(InvestBlock):
    def __init__(self, name, scenario):
        self.equal = None

        self.capex_init_existing_s2g = self.capex_init_existing_g2s = 0
        self.capex_init_additional_s2g = self.capex_init_additional_g2s = 0

        super().__init__(name, scenario, size_names=['s2g', 'g2s'])

        self.bus = self.bus_connected = self.inflow = self.outflow = None  # initialization of oemof-solph components

        self.opex_sim_power = self.opex_sim_energy = 0

        self.factor_opex_peak = self.opex_ep_spec_peak = 0

        self.initialize_peakshaving()
        self.initialize_markets()

    def calc_energy(self):
        # Aggregate energy results for all GridMarkets
        for market in self.subblocks.values():
            market.calc_results()

        self.calc_energy_source_sink()

    def calc_opex_ep_spec(self):
        # Method has to be callable from InvestBlock.__init__, but energy based opex is in GridMarket
        pass

    def calc_opex_sim(self):
        # Calculate costs for grid peak power
        self.peakshaving_ints['opex'] = self.peakshaving_ints[['power', 'period_fraction', 'opex_spec']].prod(axis=1)
        self.opex_sim_power = self.peakshaving_ints['opex'].sum()

        # Calculate costs of different markets
        for market in self.subblocks.values():
            market.opex_sim = market.flows['out'] @ market.opex_spec_g2s[self.scenario.dti_sim] * self.scenario.timestep_hours + \
                              market.flows['in'] @ market.opex_spec_s2g[self.scenario.dti_sim] * self.scenario.timestep_hours

            self.opex_sim_energy += market.opex_sim

        self.expenditures.loc['opex', 'sim'] = self.opex_sim_power + self.opex_sim_energy

    def get_ch_results(self, horizon, *_):
        self.flows.loc[horizon.dti_ch, 'in'] = sum([horizon.results[(inflow, self.bus)]['sequences']['flow'][horizon.dti_ch]
                                            for inflow in self.inflow.values()])
        self.flows.loc[horizon.dti_ch, 'out'] = sum([horizon.results[(self.bus, outflow)]['sequences']['flow'][horizon.dti_ch]
                                                     for outflow in self.outflow.values()])

        for market in self.subblocks.values():
            market.get_ch_results(horizon)

        if self.peakshaving:
            self.get_peak_powers(horizon)

    def get_invest_size(self, horizon):
        # Get optimized sizes of the grid connection. Select first size, as they all have to be the same
        self.size.loc['g2s', 'additional'] = horizon.results[(self.bus, list(self.outflow.values())[0])]['scalars']['invest']
        self.size.loc['s2g', 'additional'] = horizon.results[(list(self.inflow.values())[0]), self.bus]['scalars']['invest']
        self.size.loc['block', :] = self.size.loc['g2s', :] + self.size.loc['s2g', :]

        self.size['total'] = self.size['existing'] + self.size['additional']

        for market in self.subblocks.values():
            market.set_size('g2s')
            market.set_size('s2g')

    def get_legend_entry(self):
        return (f'{self.name} power (max. {self.size.loc["g2s", "total"] / 1e3:.1f} kW from / '
                f'{self.size.loc["s2g", "total"] / 1e3:.1f} kW to grid)')

    def get_peak_powers(self, horizon):
        # Peakshaving happens between converter and bus_connected -> select this flow to get peak values
        for interval in self.peakshaving_ints.index:
            converter = self.outflow[f'{self.name}_xc_{interval}']
            self.peakshaving_ints.loc[interval, 'power'] = max(self.peakshaving_ints.loc[interval, 'power'],
                                                               horizon.results[(converter, self.bus_connected)]['sequences']['flow'][horizon.dti_ch].max())

    def get_timeseries_results(self):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        super().get_timeseries_results()  # this goes up to the Block class
        for market in self.subblocks.values():
            market.get_timeseries_results()

    def initialize_markets(self):
        # get information about GridMarkets specified in the scenario file
        if self.filename_markets:
            markets = pd.read_csv(os.path.join(self.scenario.run.path_input_data,
                                               utils.set_extension(self.filename_markets)),
                                  index_col=[0])
            markets = markets.map(utils.infer_dtype)
        else:
            markets = pd.DataFrame(index=['res_only', 'opex_spec_g2s', 'opex_spec_s2g', 'pwr_g2s', 'pwr_s2g'],
                                   columns=[f'{self.name}_market'],
                                   data=[self.res_only, self.opex_spec_g2s, self.opex_spec_s2g, None, None])
        # Generate individual GridMarkets instances
        self.subblocks = {market: GridMarket(market, self, markets.loc[:, market])
                          for market in markets.columns}

    def initialize_peakshaving(self):
        # Create functions to extract relevant property of datetimeindex for peakshaving intervals
        periods_func = {'day': lambda x: x.strftime('%Y-%m-%d'),
                        'week': lambda x: x.strftime('%Y-CW%W'),
                        'month': lambda x: x.strftime('%Y-%m'),
                        'quarter': lambda x: f"{x.year}-Q{(x.month - 1) // 3 + 1}",
                        'year': lambda x: x.strftime('%Y')}

        if self.peakshaving is None:
            peakshaving_ints = ['sim_duration']
            n_peakshaving_ints = 0
        else:
            # Assign the corresponding interval to each timestep
            periods = self.scenario.dti_sim_extd.to_series().apply(periods_func[self.peakshaving])
            peakshaving_ints = periods.unique()
            n_peakshaving_ints = len(peakshaving_ints)

            # Activate the corresponding bus for each period
            self.bus_activation = pd.DataFrame({period_label: (periods == period_label).astype(int)
                                                for period_label in peakshaving_ints}, index=self.scenario.dti_sim_extd)

        # Create a series to store peak power values
        self.peakshaving_ints = pd.DataFrame(index=peakshaving_ints,
                                             columns=['power', 'period_fraction', 'start', 'end', 'opex_spec'])
        # initialize power to 0 as for rh this value will be used to initialize the existing peak power
        self.peakshaving_ints.loc[:, 'power'] = 0

        self.peakshaving_ints['opex_spec'] = self.opex_spec_peak  # can be adapted for multiple cost levels over time

        if self.peakshaving is not None:
            # calculate the fraction of each period that is covered by the sim time (NOT sim_extd!)
            for interval in self.peakshaving_ints.index:
                self.peakshaving_ints.loc[interval, 'period_fraction'] = utils.get_period_fraction(
                    dti=self.bus_activation.loc[self.scenario.dti_sim][
                        self.bus_activation.loc[self.scenario.dti_sim, interval] == 1].index,
                    period=self.peakshaving,
                    freq=self.scenario.timestep)

                # Get first and last timestep of each peakshaving interval -> used for rh calculation later on
                self.peakshaving_ints.loc[interval, 'start'] = \
                self.bus_activation[self.bus_activation[interval] == 1].index[0]
                self.peakshaving_ints.loc[interval, 'end'] = self.bus_activation[self.bus_activation[interval] == 1].index[
                    -1]

            # Count number of "actual" peakshaving intervals
            # (i.e. not entered as 'sim duration', which happens when self.peakshaving is None)
            n_peakshaving_ints_yr = (pd.date_range(start=self.scenario.starttime,
                                                   end=self.scenario.starttime + pd.DateOffset(years=1),
                                                   freq=self.scenario.timestep,
                                                   inclusive='left')
                                     .to_series().apply(periods_func[self.peakshaving])).unique().size
            self.factor_opex_peak = n_peakshaving_ints_yr / n_peakshaving_ints
            self.opex_ep_spec_peak = self.opex_spec_peak * self.factor_opex_peak

    def print_results(self):
        if self.invest:
            self.scenario.logger.info(f'Optimized size of g2s power in component "{self.name}":'
                                      f' {self.size.loc["g2s", "total"] / 1e3:.1f} kW' + \
                                      f' (existing: {self.size.loc["g2s", "existing"] / 1e3:.1f} kW'
                                      f' - additional: {self.size.loc["g2s", "additional"] / 1e3:.1f} kW)')
            self.scenario.logger.info(f'Optimized size of s2g power in component "{self.name}":'
                                      f' {self.size.loc["s2g", "total"] / 1e3:.1f} kW'
                                      f' (existing: {self.size.loc["s2g", "existing"] / 1e3:.1f} kW'
                                      f' - additional: {self.size.loc["s2g", "additional"] / 1e3:.1f} kW)')

        if self.peakshaving:
            for interval in self.peakshaving_ints.index:
                if self.peakshaving_ints.loc[interval, 'start'] <= self.dti_sim[-1]:
                    self.scenario.logger.info(f'Optimized peak power in component "{self.name}" for interval'
                                     f' {interval}: {self.peakshaving_ints.loc[interval, "power"] / 1e3:.1f} kW'
                                     f' - OPEX: {self.opex_spec_peak * self.peakshaving_ints.loc[interval, ["period_fraction", "power"]].prod():.2f} {self.currency}')

    def set_init_size(self, size_names):
        self.equal = True if self.invest_g2s == 'equal' or self.invest_s2g == 'equal' else False

        utils.init_equalizable_variables(self, ['invest_s2g', 'invest_g2s'])
        utils.init_equalizable_variables(self, ['size_g2s_existing', 'size_s2g_existing'])
        utils.init_equalizable_variables(self, ['size_g2s_max', 'size_s2g_max'])

        super().set_init_size(size_names)

    def update_input_components(self, horizon):
        """
        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        xc_bus             self.bus
          |                   |
          |---xc_grid_1--x--->|
          |<--grid_xc_1--x----|
          |                   |---(GridMarket Instance)
          |---xc_grid_2--x--->|
          |<--grid_xc_2--x----|
          |                   |---(GridMarket Instance)
          |         .         |
                    .
          |         .         |
          |---xc_grid_n--x--->|
          |<--grid_xc_n--x----|
          |                   |
        """

        self.bus_connected = self.scenario.blocks['core'].ac_bus if self.system == 'ac' else self.scenario.blocks['core'].dc_bus
        self.bus = solph.Bus(label=f'{self.name}_bus')
        horizon.components.append(self.bus)

        self.inflow = {f'xc_{self.name}': solph.components.Converter(
            label=f'xc_{self.name}',
            # Peakshaving not implemented for feed-in into grid
            inputs={self.bus_connected: solph.Flow()},
            # Size optimization
            outputs={self.bus: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.capex_ep_spec,
                                               existing=self.size.loc['s2g', 'existing'],
                                               maximum=utils.conv_add_max(self.size.loc['s2g', 'additional_max'])),
                variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.bus: 1})}

        self.outflow = {f'{self.name}_xc_{intv}': solph.components.Converter(
            label=f'{self.name}_xc_{intv}',
            # Size optimization: investment costs are assigned to first peakshaving interval only. The application of
            # constraints ensures that the optimized grid connection sizes of all peakshaving intervals are equal
            inputs={self.bus: solph.Flow(
                nominal_value=solph.Investment(ep_costs=(self.capex_ep_spec if intv == self.peakshaving_ints.index[0] else 0),
                                               existing=self.size.loc['g2s', 'existing'],
                                               maximum=utils.conv_add_max(self.size.loc['g2s', 'additional_max']))
            )},
            # Peakshaving
            outputs={self.bus_connected: solph.Flow(nominal_value=(solph.Investment(ep_costs=self.opex_ep_spec_peak,
                                                                                    existing=self.peakshaving_ints.loc[intv, 'power'],)
                                                                   if self.peakshaving else None),
                                                    max=(self.bus_activation.loc[horizon.dti_ph, intv] if self.peakshaving else None))},
            conversion_factors={self.bus_connected: 1}) for intv in self.peakshaving_ints.index}

        horizon.components.extend(self.inflow.values())
        horizon.components.extend(self.outflow.values())

        horizon.constraints.add_invest_costs(invest=(self.inflow[f'xc_{self.name}'], self.bus),
                                             capex_spec=self.capex_spec,
                                             invest_type='flow')
        horizon.constraints.add_invest_costs(invest=(self.bus,
                                                     self.outflow[f'{self.name}_xc_{self.peakshaving_ints.index[0]}']),
                                             capex_spec=self.capex_spec,
                                             invest_type='flow')

        # The optimized sizes of the buses of all peakshaving intervals have to be the same as they technically
        # represent the same grid connection
        equal_investments = [{'in': self.bus, 'out': outflow} for outflow in self.outflow.values()]

        # If size of in- and outflow from and to the grid have to be the same size, add outflow investment(s)
        if self.equal:
            equal_investments.extend([{'in': inflow, 'out': self.bus} for inflow in self.inflow.values()])

        # add list of variables to the scenario constraints if list contains more than one element
        # lists with one element occur, if peakshaving is deactivated and grid sizes don't have to be equal
        if len(equal_investments) > 1:
            horizon.constraints.add_equal_invests(equal_investments)

        for market in self.subblocks.values():
            market.update_input_components(horizon)


class GridMarket(SubBlock):
    def __init__(self, name, parent, params):

        super().__init__(name, parent, params, flow_names=['total', 'in', 'out'])

        # initialize oemof-solph components
        self.src = self.snk = None

        self.set_init_size()

        # opex_spec_g2s has always to be specified as a scalar or a filename containing a timeseries
        if self.opex_spec_g2s == 'equal':
            raise ValueError(f'GridMarket "{self.name}": opex_spec_g2s cannot be set to "equal".'
                             f' When using the same cost for g2s and s2g specifiy the cost in  opex_spec_g2s and set'
                             f' opex_spec_s2g to "equal".')
        utils.transform_scalar_var(self, 'opex_spec_g2s')

        # opex_spec_s2g can be specified as a scalar, a filename containing a timeseries or as 'equal' to opex_spec_g2s
        if self.opex_spec_s2g == 'equal':
            self.equal_prices = True
            self.opex_spec_s2g = -1 * self.opex_spec_g2s
        else:
            self.equal_prices = False
            utils.transform_scalar_var(self, 'opex_spec_s2g')

        self.calc_opex_ep_spec()

    def add_power_trace(self):
        if self.parent.filename_markets is None:
            return

        legentry = (f'{self.name} power (max.'
                    f' {(self.parent.size.loc["g2s", "total"] if pd.isna(self.pwr_g2s) else self.pwr_g2s) / 1e3:.1f} kW from /'
                    f' {(self.parent.size.loc["s2g", "total"] if pd.isna(self.pwr_s2g) else self.pwr_s2g) / 1e3:.1f} kW to grid)')

        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=self.flows['total'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=False)

    def calc_opex_ep_spec(self):
        self.opex_ep_spec_g2s = self.opex_spec_g2s * self.parent.factor_opex
        self.opex_ep_spec_s2g = self.opex_spec_s2g * self.parent.factor_opex

    def calc_results(self):
        # energy result calculation does not count towards delivered/produced energy (already done at the system level)
        super().calc_results(flows=['in', 'out'])

        self.flows['total'] = self.flows['out'] - self.flows['in']  # for plotting

    def get_ch_results(self, horizon):
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.parent.bus, self.snk)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.src, self.parent.bus)]['sequences']['flow'][horizon.dti_ch]

    def get_timeseries_results(self):
        """
        Collect timeseries results of the commodity in a scenario wide dataframe to be saved
        """
        market_ts_results = pd.DataFrame({(self.name, col): self.flows[col] for col in self.flows.columns})
        self.scenario.result_timeseries = pd.concat([self.scenario.result_timeseries, market_ts_results], axis=1)

    def set_init_size(self):
        # power of market cannot exceed GridConnection limit due to constraints
        # GridMarket limits are already set in __init__() by given parameters
        pass

    def set_size(self, dir):
        # if no limit is passed for the grid market's direction, use max power of the grid connection
        # always use sum of existing and additional power as this function also is called after optimization
        if pd.isna(getattr(self, f'pwr_{dir}')):
            setattr(self,
                    f'pwr_{dir}',
                    self.parent.size.loc[dir, 'total']
                    )
        # otherwise use the minimum of the grid market's and the grid connection's maximum power
        else:
            setattr(self,
                    f'pwr_{dir}',
                    min(getattr(self, f'pwr_{dir}'),
                        self.parent.size.loc[dir, 'total'])
                    )

    def update_input_components(self, horizon):
        """
         grid_bus
          |<---x----grid_src
          |
          |----x--->grid_snk
          |
        """

        self.src = solph.components.Source(label=f'{self.name}_src',
                                           outputs={self.parent.bus: solph.Flow(
                                               nominal_value=(self.pwr_g2s if not pd.isna(self.pwr_g2s) else None),
                                               variable_costs=self.opex_ep_spec_g2s[horizon.dti_ph])
                                           })

        self.snk = solph.components.Sink(label=f'{self.name}_snk',
                                         inputs={self.parent.bus: solph.Flow(
                                             nominal_value=(self.pwr_s2g if not pd.isna(self.pwr_s2g) else None),
                                             variable_costs=self.opex_ep_spec_s2g[horizon.dti_ph] +
                                                            self.scenario.cost_eps * self.equal_prices)
                                         })

        horizon.components.append(self.src)
        horizon.components.append(self.snk)


class ICEVSystem(Block):
    """
    Dummy class to include ICEV fleets into economic result calculation, even though they do not have any energy system
    impact and/or need any preprocessing.
    """

    def __init__(self, name, scenario):

        super().__init__(name, scenario)

        # commodity names might be rewritten in case of imported log data with differing name
        self.com_names = [f'{self.name}{str(i)}' for i in range(self.num)]
        self.data = None

        if self.data_source in ['usecases', 'demand']:
            raise NotImplementedError(f'Block "{self.name}": '
                                      f'dispatch_source "{self.data_source}" is not yet implemented for ICEVSystem')
        elif self.data_source == 'log':
            self.data = utils.read_input_log(self)
            # rewrite commodity names to match modified ones from imported log file
            self.com_names = self.data.columns.get_level_values(0).unique()[:self.num].tolist()
        else:
            raise ValueError(f'Block "{self.name}": invalid data source ("{self.data_source}")')

        self.crev_time = self.crev_usage = None  # intermediary variables

    def add_power_trace(self):
        pass  # function has to be callable, but ICEVSystem does not have a power trace

    def add_soc_trace(self):
        pass  # function has to be callable, but ICEVSystem does not have a SOC trace

    def calc_energy(self):
        pass  # function has to be callable, but ICEVSystem does not impose energy transfer

    def calc_capex_init_existing(self):
        self.expenditures.loc['capex', 'sim'] = self.capex_glider * self.num if self.capex_existing else 0
        self.scenario.capex_init_existing += self.expenditures.loc['capex', 'sim']

    def calc_expenses(self):
        self.capex_replacement = self.capex_glider * self.num
        self.extrapolate_capex()  # Method defined in Block class

        self.expenditures.loc['opex', 'sim'] = sum([self.data.loc[self.scenario.dti_sim, (com, 'tour_dist')] @ self.opex_spec_dist
                                                    for com in self.com_names])
        self.extrapolate_opex()  # Method defined in Block class

        self.expenditures.loc['mntex', 'yrl'] = self.mntex_glider * self.num
        self.extrapolate_mntex()  # Method defined in Block class

        self.accumulate_expenses()

    def calc_revenue(self):

        self.crev_time = {commodity: ~self.data.loc[self.scenario.dti_sim, (commodity, 'atbase')]
                                     @ self.crev_spec_time[self.scenario.dti_sim]
                                     * self.scenario.timestep_hours
                          for commodity in self.com_names}
        self.crev_usage = {commodity: self.data.loc[self.scenario.dti_sim, (commodity, 'tour_dist')]
                                      @ self.crev_spec_dist[self.scenario.dti_sim]
                           for commodity in self.com_names}
        self.expenditures.loc['crev', 'sim'] = sum(self.crev_time.values()) + sum(self.crev_usage.values())

        self.accumulate_crev()

    def get_ch_results(self, horizon):
        pass  # function has to be callable, but ICEVSystem does not have timeseries results from optimization

    def get_timeseries_results(self):
        pass  # function has to be callable, but ICEVSystem does not have timeseries results

    def update_input_components(self, *_):
        pass  # function has to be callable, but ICEVSystem does not have energy system components


class FixedDemand(Block):

    def __init__(self, name, scenario):
        super().__init__(name, scenario, flow_names=['total', 'in'])

        self.bus_connected = self.snk = None  # initialize oemof-solph component

        self.data = None

        if self.load_profile.upper() in ['H0', 'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'L0', 'L1', 'L2']:
            self.generate_timeseries_from_slp()
        elif self.load_profile == 'const':
            self.data = pd.Series(index=self.scenario.dti_sim_extd, data=self.consumption_yrl / (365 * 24))
        else:
            utils.transform_scalar_var(self, 'load_profile')
            self.data = self.load_profile

        self.data_ph = None  # placeholder

    def calc_energy(self):
        self.calc_energy_source_sink()

    def calc_capex_init_existing(self):
        pass  # function has to be callable, but FixedDemand does not have any capital expenses

    def calc_revenue(self):
        self.expenditures.loc['crev', 'sim'] = (self.flows['in'] @ self.crev_spec[self.scenario.dti_sim]) * self.scenario.timestep_hours  # @ is dot product (Skalarprodukt)
        self.accumulate_crev()

    def generate_timeseries_from_slp(self):
        def get_timeframe(date):
            month = date.month
            day = date.day
            if ((month, day) >= (11, 1)) or ((month, day) <= (3, 20)):
                return 'Winter'
            elif (5, 15) <= (month, day) <= (9, 14):
                return 'Summer'
            else:  # Transition months
                return 'Transition'

        def get_daytype(date, holidays):
            if date.date() in holidays or date.weekday() == 6:
                return 'Sunday'
            # Treat Christmas Eve and New Year's Eve as Saturdays if they are not Sundays
            elif (date.weekday() == 5) or ((date.month, date.day) in [(12, 24), (12, 31)]):
                return 'Saturday'
            else:
                return 'Workday'

        # Read BDEW SLP profiles
        slp = pd.read_csv(os.path.join(self.scenario.run.path_data_immut, 'slp_bdew.csv'),
                          skiprows=[0],
                          header=[0, 1, 2],
                          index_col=0)

        slp.index = pd.to_datetime(slp.index, format='%H:%M').time

        # use a fixed frequency of 15 minutes for the timeseries generation as the SLPs are given with that frequency
        freq_slp = '15min'
        dti_slp = pd.DatetimeIndex(pd.date_range(start=self.scenario.starttime.floor(freq_slp),
                                                 end=max(self.scenario.dti_sim_extd).ceil(freq_slp),
                                                 freq=freq_slp))

        self.data = pd.Series(index=dti_slp, data=0, dtype='float64')

        self.data = self.data.index.to_series().apply(
        lambda x: slp.loc[x.time(), (self.load_profile.upper(), get_timeframe(x), get_daytype(x, self.scenario.holiday_dates))])

        # apply dynamic correction for household profiles
        if self.load_profile.upper() == 'H0':
            # for private households use dynamic correction as stated in VDEW manual -> round to 1/10 Watt
            num_day = self.data.index.dayofyear.astype('int64')
            self.data = round(self.data * (-3.92e-10 * num_day ** 4 + 3.2e-7 * num_day ** 3 -
                                           7.02e-5 * num_day ** 2 + 2.1e-3 * num_day ** 1 + 1.24),
                              ndigits=1)

        # scale load profile (given for consumption of 1MWh per year) to specified yearly consumption
        # This calculation leads to small deviations from the specified yearly consumption due to varying holidays and
        # leap years, but is the correct way as stated by the VDEW manual
        self.data *= (self.consumption_yrl / 1e6)

        # resample to simulation time step
        self.data = self.data.resample(self.scenario.timestep).mean().ffill().bfill()

    def get_ch_results(self, horizon, *_):
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.bus_connected, self.snk)]['sequences']['flow'][horizon.dti_ch]

    def get_legend_entry(self):
        return f'{self.name} power'

    def update_input_components(self, horizon):
        # new ph data slice is created during initialization of the PredictionHorizon
        """
        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        xc_bus
          |
          |-x->dem_snk
          |
        """

        self.bus_connected = self.scenario.blocks['core'].ac_bus if self.system == 'ac' else self.scenario.blocks['core'].dc_bus

        self.snk = solph.components.Sink(label='dem_snk',
                                         inputs={self.bus_connected: solph.Flow(nominal_value=1,
                                                                                fix=self.data_ph)})
        horizon.components.append(self.snk)


class MobileCommodity(SubBlock):

    def __init__(self, name, parent, params):

        super().__init__(name, parent, params, flow_names=['total', 'in', 'out', 'bat_in', 'bat_out', 'ext_ac', 'ext_dc'])

        # Add block specific energies
        self.energies.loc['ext_ac', :] = 0.0
        self.energies.loc['ext_dc', :] = 0.0

        # initialize oemof-solph components
        self.bus = self.inflow = self.outflow = self.ess = None
        self.bus_ext_ac = self.conv_ext_ac = self.src_ext_ac = None
        self.bus_ext_dc = self.conv_ext_dc = self.src_ext_dc = None
        self.pwr_chg_max_observed = self.pwr_dis_max_observed = None

        self.size = pd.DataFrame()
        self.set_init_size()

        self.dsoc_buffer += (self.q_loss_cal_init + self.q_loss_cyc_init) / 2

        # Data Source has been checked in the parent class to be either 'des' or 'log'
        if self.parent.data_source in ['usecases', 'demand']:
            self.data = None  # parent data does not exist yet, filtering is done later
        elif self.parent.data_source == 'log':  # predetermined log file
            self.data = self.parent.data.loc[:, (self.name, slice(None))].droplevel(0, axis=1)

        self.apriori_data = None

        self.data_ph = None  # placeholder, is filled in update_input_components

        self.capex_fix = self.capex_init_existing = 0
        self.crev_time = self.crev_usage = self.crev_sim = self.crev_yrl = self.crev_prj = self.crev_dis = 0
        self.opex_sim = self.opex_sim_int = self.opex_sim_ext = 0

        # timeseries result initialization
        self.storage_timeseries = pd.DataFrame(index=utils.extend_dti(self.scenario.dti_sim),
                                               columns=['soc', 'soh', 'q_loss_cal', 'q_loss_cyc'],
                                               dtype='float64')

        self.storage_timeseries.loc[self.scenario.starttime, 'soc'] = self.soc_init

        self.aging_model = bat.BatteryPackModel(self)
        self.soc_min = (1 - self.storage_timeseries.loc[self.scenario.starttime, 'soh']) / 2
        self.soc_max = 1 - ((1 - self.storage_timeseries.loc[self.scenario.starttime, 'soh']) / 2)

    def add_power_trace(self):
        legentry = (f'{self.name} power (max. {self.pwr_chg / 1e3:.1f} kW charge / '
                    f'{(self.pwr_dis * self.eff_dis if self.parent.lvl_cap != "ud" else 0) / 1e3:.1f} kW discharge)')
        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=self.flows['total'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=False)

        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=self.flows['ext_ac'] + self.flows['ext_dc'],
                                                  mode='lines',
                                                  name=f'{self.name} external charging power'
                                                       f' (AC max. {self.parent.pwr_ext_ac / 1e3:.1f} kW &'
                                                       f' DC max. {self.parent.pwr_ext_dc / 1e3:.1f} kW)',
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=False)

    def add_soc_trace(self):
        legentry = f'{self.name} SOC ({self.size.loc["block", "total"]/1e3:.1f} kWh)'
        self.scenario.figure.add_trace(go.Scatter(x=self.storage_timeseries.index,
                                                  y=self.storage_timeseries['soc'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=True)

        legentry = f'{self.name} SOH'
        data = self.storage_timeseries['soh'].dropna()
        self.scenario.figure.add_trace(go.Scatter(x=data.index,
                                                  y=data,
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=True)

    def calc_aging(self, horizon):
        self.aging_model.age(horizon)

    def calc_capex_init_existing(self):
        """ Preprocessing method """
        self.capex_fix = self.capex_glider + self.capex_charger
        if self.parent.capex_existing:
            self.capex_init_existing = (self.size.loc['block', 'existing'] * self.capex_spec +
                                        self.capex_fix)

    def calc_opex_sim(self):
        """
        Postprocessing method
        Calculate internal and external opex of individual commodities.
        Vehicle opex is distance based, battery opex is energy throughput based.
        """

        if isinstance(self.parent, VehicleCommoditySystem):
            self.opex_sim_int = (self.data.loc[self.scenario.dti_sim, 'tour_dist'] @
                                 self.parent.opex_spec_dist[self.scenario.dti_sim])
        elif isinstance(self.parent, BatteryCommoditySystem):
            self.opex_sim_int = (self.flows['in'] @ self.parent.opex_spec[self.scenario.dti_sim] *
                                 self.scenario.timestep_hours)
        else:
            raise ValueError

        self.opex_sim_ext = (((self.flows['ext_ac'] @ self.parent.opex_spec_ext_ac[self.scenario.dti_sim]) +
                             (self.flows['ext_dc'] @ self.parent.opex_spec_ext_dc[self.scenario.dti_sim])) *
                             self.scenario.timestep_hours)

        self.opex_sim += (self.opex_sim_int + self.opex_sim_ext)

    def calc_results(self):
        super().calc_results(flows=['in', 'out', 'ext_ac', 'ext_dc'])
        self.flows['total'] = self.flows['in'] - self.flows['out']  # for plotting

        self.pwr_chg_max_observed = self.flows['in'].max()
        self.pwr_dis_max_observed = self.flows['out'].max()

    def calc_revenue(self):

        # rental time based revenue
        self.crev_time = ((~self.data.loc[self.scenario.dti_sim, 'atbase'] @ self.parent.crev_spec_time[self.scenario.dti_sim]) *
                          self.scenario.timestep_hours)

        # usage based revenue
        if isinstance(self.parent, VehicleCommoditySystem):
            self.crev_usage = self.data.loc[self.scenario.dti_sim, 'tour_dist'] @ self.parent.crev_spec_dist[self.scenario.dti_sim]
        else:  # BatteryCommoditySystems have no usage based revenue
            self.crev_usage = 0  # Battery rental is a fixed time based price, irrespective of energy consumption

        self.crev_sim = self.crev_time + self.crev_usage

    def get_ch_results(self, horizon):

        self.flows.loc[horizon.dti_ch, 'bat_out'] = horizon.results[(self.ess, self.bus)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'bat_in'] = horizon.results[(self.bus, self.ess)]['sequences']['flow'][horizon.dti_ch]

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.outflow, self.parent.bus)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.parent.bus, self.inflow)]['sequences']['flow'][horizon.dti_ch]

        # Get results of external chargers
        self.flows.loc[horizon.dti_ch, 'ext_ac'] = horizon.results[(self.src_ext_ac, self.bus_ext_ac)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'ext_dc'] = horizon.results[(self.src_ext_dc, self.bus_ext_dc)]['sequences']['flow'][horizon.dti_ch]

        # storage content during PH (including endtime)
        self.storage_timeseries.loc[utils.extend_dti(horizon.dti_ch), 'soc'] = solph.views.node(
            horizon.results, f'{self.name}_ess')['sequences'][((f'{self.name}_ess', 'None'), 'storage_content')][
                                                         utils.extend_dti(horizon.dti_ch)] / self.size.loc['block', 'total']

    def get_invest_size(self, horizon):
        self.size.loc['block', 'additional'] = horizon.results[(self.ess, None)]['scalars']['invest']
        self.size['total'] = self.size['existing'] + self.size['additional']

    def get_timeseries_results(self):
        """
        Collect timeseries results of the commodity in a scenario wide dataframe to be saved
        """

        commodity_ts_results_flows = pd.DataFrame({(self.name, col): self.flows[col] for col in self.flows.columns})
        commodity_ts_results_storage = pd.DataFrame({(self.name, col): self.storage_timeseries[col] for col in self.storage_timeseries.columns})

        self.scenario.result_timeseries = pd.concat([self.scenario.result_timeseries,
                                                     commodity_ts_results_flows,
                                                     commodity_ts_results_storage], axis=1)

    def print_results(self):
        if self.invest:
            self.scenario.logger.info(f'Optimized size of commodity "{self.name}" in component "{self.parent.name}":'
                                      f' {self.size.loc["block", "total"] / 1e3:.1f} kWh'
                                      f' (existing: {self.size.loc["block", "existing"] / 1e3:.1f} kWh'
                                      f' - additional: {self.size.loc["block", "additional"] / 1e3:.1f} kWh)')

    def set_init_size(self):
        self.size = pd.DataFrame(self.parent.size.loc['pc']).T
        self.size.index = ['block']

    def update_input_components(self, horizon):

        inflow_fix = outflow_fix = ext_ac_fix = ext_dc_fix = None
        inflow_max = outflow_max = ext_ac_max = ext_dc_max = None

        if self.apriori_data is not None:
            # define charging powers (as per uc power calculation)
            inflow_fix = self.apriori_data['p_int_ac'].clip(lower=0) / self.pwr_chg
            outflow_fix = self.apriori_data['p_int_ac'].clip(upper=0) * (-1) / self.pwr_dis

            if self.parent.pwr_ext_ac > 0:
                ext_ac_fix = self.apriori_data['p_ext_ac'] / self.parent.pwr_ext_ac
            if self.parent.pwr_ext_dc > 0:
                ext_dc_fix = self.apriori_data['p_ext_dc'] / self.parent.pwr_ext_dc
        else:
            # enable/disable Converters to mcx_bus depending on whether the commodity is at base
            inflow_max = self.data_ph['atbase'].astype(int)
            outflow_max = self.data_ph['atbase'].astype(int)

            # enable/disable ac and dc charging station dependent on example data
            ext_ac_max = self.data_ph['atac'].astype(int)
            ext_dc_max = self.data_ph['atdc'].astype(int)

        soc_max = pd.Series(data=self.soc_max, index=self.data_ph.index)

        # Adjust min storage levels based on state of health for the upcoming prediction horizon
        # nominal_storage_capacity is retained for accurate state of charge tracking and cycle depth
        # relative to nominal capacity. Disregard minsoc values from DES for any case except myopic optimization.
        if self.mode_dispatch == 'opt_myopic' and isinstance(self.parent, VehicleCommoditySystem):
            # VehicleCommoditySystems operate on the premise of not necessarily renting out at high SOC level
            dsoc_dep_ph = self.data_ph['dsoc'].where(self.data_ph['dsoc'] == 0,
                                                     self.data_ph['dsoc'] + self.dsoc_buffer)
            soc_min = dsoc_dep_ph.clip(lower=self.soc_min, upper=self.soc_max)
        elif self.mode_dispatch == 'opt_myopic' and isinstance(self.parent, BatteryCommoditySystem):
            # BatteryCommoditySystems operate on the premise of renting out at max SOC
            soc_min = self.data_ph['dsoc'].where(
                self.data_ph['dsoc'] == 0,
                self.parent.soc_target_high).clip(lower=self.soc_min, upper=self.soc_max)
        else:  # opt_global or apriori cases
            soc_min = pd.Series(data=self.soc_min, index=self.data_ph.index)

        """
         bus               mc1_bus
          |<--x------mc1_mc---|<->mc1_ess
          |                   |
          |---x--mc_mc1------>|-->mc1_snk
          |                   |
          |                   |<--mc1_ext_ac (external charging AC)
          |                   |
          |                   |<--mc1_ext_dc (external charging DC)
          |
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        horizon.components.append(self.bus)

        self.inflow = solph.components.Converter(label=f'mc_{self.name}',
                                                 inputs={self.parent.bus: solph.Flow(nominal_value=self.pwr_chg,
                                                                                     max=inflow_max,
                                                                                     fix=inflow_fix)},
                                                 outputs={self.bus: solph.Flow()},
                                                 conversion_factors={self.bus: self.eff_chg})
        horizon.components.append(self.inflow)

        self.outflow = solph.components.Converter(label=f'{self.name}_mc',
                                                  inputs={self.bus: solph.Flow()},
                                                  outputs={
                                                      self.parent.bus: solph.Flow(
                                                          nominal_value=(self.pwr_dis * self.eff_dis if self.parent.lvl_cap in ['v2v', 'v2s'] else 0),
                                                          max=outflow_max,
                                                          fix=outflow_fix,
                                                          variable_costs=self.scenario.cost_eps)
                                                  },
                                                  conversion_factors={self.parent.bus: self.eff_dis})
        horizon.components.append(self.outflow)

        self.snk = solph.components.Sink(label=f'{self.name}_snk',
                                         inputs={self.bus: solph.Flow(nominal_value=1,
                                                                      fix=self.data_ph['consumption'])})
        # actual values are set later in update_input_components for each prediction horizon
        horizon.components.append(self.snk)

        self.ess = solph.components.GenericStorage(label=f'{self.name}_ess',
                                                   inputs={self.bus: solph.Flow(variable_costs=self.parent.opex_ep_spec[horizon.dti_ph])},
                                                   # cost_eps are needed to prevent storage from being emptied in RH
                                                   outputs={self.bus: solph.Flow(variable_costs=self.scenario.cost_eps)},
                                                   loss_rate=self.parent.loss_rate,
                                                   balanced=False,
                                                   initial_storage_level=statistics.median(
                                                       [soc_min[horizon.starttime],
                                                        self.storage_timeseries.loc[horizon.starttime, 'soc'],
                                                        soc_max[horizon.starttime]]),
                                                   inflow_conversion_factor=np.sqrt(self.eff_storage_roundtrip),
                                                   outflow_conversion_factor=np.sqrt(self.eff_storage_roundtrip),
                                                   nominal_storage_capacity=solph.Investment(ep_costs=self.parent.capex_ep_spec,
                                                                                             existing=self.size.loc['block', 'existing'],
                                                                                             maximum=utils.conv_add_max(self.size.loc['block', 'additional_max'])),
                                                   min_storage_level=soc_min,
                                                   max_storage_level=soc_max
                                                   )

        horizon.components.append(self.ess)

        # always add charger -> reduce different paths of result calculations; no chargers -> power is set to 0 kW
        # add external AC charger as new energy source
        self.bus_ext_ac = solph.Bus(label=f'{self.name}_bus_ext_ac')

        self.src_ext_ac = solph.components.Source(label=f'{self.name}_src_ext_ac',
                                                  outputs={self.bus_ext_ac: solph.Flow(
                                                      nominal_value=self.parent.pwr_ext_ac,
                                                      max=ext_ac_max,
                                                      fix=ext_ac_fix,
                                                      variable_costs=self.parent.opex_ep_spec_ext_ac[horizon.dti_ph])}
                                                  )

        self.conv_ext_ac = solph.components.Converter(label=f'{self.name}_conv_ext_ac',
                                                      inputs={self.bus_ext_ac: solph.Flow()},
                                                      outputs={self.bus: solph.Flow()},
                                                      conversion_factors={self.bus: self.parent.eff_chg_ac}
                                                      )

        horizon.components.append(self.bus_ext_ac)
        horizon.components.append(self.src_ext_ac)
        horizon.components.append(self.conv_ext_ac)

        # add external DC charger as new energy source
        self.bus_ext_dc = solph.Bus(label=f'{self.name}_bus_ext_dc')

        self.src_ext_dc = solph.components.Source(label=f'{self.name}_src_ext_dc',
                                                  outputs={self.bus_ext_dc: solph.Flow(
                                                      nominal_value=self.parent.pwr_ext_dc,
                                                      max=ext_dc_max,
                                                      fix=ext_dc_fix,
                                                      variable_costs=self.parent.opex_ep_spec_ext_dc[horizon.dti_ph])}
                                                  )

        self.conv_ext_dc = solph.components.Converter(label=f'{self.name}_conv_ext_dc',
                                                      inputs={self.bus_ext_dc: solph.Flow()},
                                                      outputs={self.bus: solph.Flow()},
                                                      conversion_factors={self.bus: 1}
                                                      )

        horizon.components.append(self.bus_ext_dc)
        horizon.components.append(self.src_ext_dc)
        horizon.components.append(self.conv_ext_dc)

        horizon.constraints.add_invest_costs(invest=(self.ess,),
                                             capex_spec=self.parent.capex_spec,
                                             invest_type='storage')


class PVSource(RenewableInvestBlock):

    def __init__(self, name, scenario):

        self.api_startyear = self.api_endyear = self.api_shift = self.api_length = self.api_params = self.meta = None

        super().__init__(name, scenario)

    def calc_power_solcast(self):

        u0 = 26.9  # W/(˚C.m2) - cSi Free standing
        u1 = 6.2  # W.s/(˚C.m3) - cSi Free standing
        mod_temp = self.data['temp_air'] + (self.data['gti'] / (u0 + (u1 * self.data['wind_speed'])))

        # PVGIS temperature and irradiance coefficients for cSi panels as per Huld T., Friesen G., Skoczek A.,
        # Kenny R.P., Sample T., Field M., Dunlop E.D. A power-rating model for crystalline silicon PV modules
        # Solar Energy Materials & Solar Cells. 2011 95, 3359-3369.
        k1 = -0.017237
        k2 = -0.040465
        k3 = -0.004702
        k4 = 0.000149
        k5 = 0.000170
        k6 = 0.000005
        g = self.data['gti'] / 1000
        t = mod_temp - 25
        lng = np.zeros_like(g)
        lng[g != 0] = np.log(g[g != 0])  # ln(g) ignoring zeros

        # Faiman, D. Assessing the outdoor operating temperature of photovoltaic modules.
        # Prog. Photovolt. Res. Appl.2008, 16, 307–315
        eff_rel = (1 +
                   (k1 * lng) +
                   (k2 * (lng ** 2)) +
                   (k3 * t) +
                   (k4 * t * lng) +
                   (k5 * t * (lng ** 2)) +
                   (k6 * (t ** 2)))
        eff_rel = eff_rel.fillna(0)

        # calculate power of a 1kWp array, limited to 0 (negative values fail calculation)
        self.data['P'] = np.maximum(0, eff_rel * self.data['gti'])

    def get_timeseries_data(self):
        if 'api' in self.data_source.lower():  # PVGIS API or Solcast API example selected
            if self.filename:
                try:
                    self.api_params = pd.read_csv(os.path.join(self.scenario.run.path_input_data,
                                                               utils.set_extension(self.filename)),
                                                  index_col=[0],
                                                  na_filter=False)
                    self.api_params = self.api_params.map(utils.infer_dtype)['value'].to_dict() if self.api_params.index.name == 'parameter' and all(self.api_params.columns == 'value') else {}
                except FileNotFoundError:
                    self.api_params = {}
            else:
                self.api_params = {}

            if self.data_source == 'pvgis api':  # PVGIS API example selected
                self.api_startyear = self.scenario.starttime.tz_convert('utc').year
                self.api_endyear = self.scenario.sim_extd_endtime.tz_convert('utc').year
                self.api_length = self.api_endyear - self.api_startyear
                self.api_shift = pd.to_timedelta('0 days')

                if self.api_length > 15:
                    raise ValueError('PVGIS API only allows a maximum of 15 years of data')
                elif self.api_endyear > 2023:  # PVGIS-SARAH3 only has data up to 2023
                    self.api_shift = (pd.to_datetime('2023-01-01 00:00:00+00:00') -
                                      pd.to_datetime(f'{self.api_endyear}-01-01 00:00:00+00:00'))
                    self.api_endyear = 2023
                    self.api_startyear = 2023 - self.api_length
                elif self.api_startyear < 2005:  # PVGIS-SARAH3 only has data from 2005
                    self.api_shift = (pd.to_datetime('2005-01-01 00:00:00+00:00') -
                                      pd.to_datetime(f'{self.api_startyear}-01-01 00:00:00+00:00'))
                    self.api_startyear = 2005
                    self.api_endyear = 2005 + self.api_length
                # Todo leap years can result in data shifting not landing at the same point in time

                # revert lower() in reading data as pvgis is case-sensitive
                # ToDo: move to checker.py
                self.api_params['raddatabase'] = self.api_params.get('raddatabase', 'PVGIS-SARAH3').upper()
                self.api_params['pvtechchoice'] = {'crystsi': 'crystSi',
                                               'cis': 'CIS',
                                               'cdte': 'CdTe',
                                               'unknown': 'Unknown'}[self.api_params.get('pvtechchoice', 'crystsi')]

                self.data, self.meta, _ = pvlib.iotools.get_pvgis_hourly(
                    self.scenario.latitude,
                    self.scenario.longitude,
                    start=self.api_startyear,
                    end=self.api_endyear,
                    url='https://re.jrc.ec.europa.eu/api/v5_3/',
                    components=False,
                    outputformat='json',
                    pvcalculation=True,
                    peakpower=1,
                    map_variables=True,
                    loss=0,
                    raddatabase=self.api_params['raddatabase'],  # conversion above ensures that the parameter exists
                    pvtechchoice=self.api_params['pvtechchoice'],  # conversion above ensures that the parameter exists
                    mountingplace=self.api_params.get('mountingplace', 'free'),
                    optimalangles=self.api_params.get('optimalangles', True),
                    optimal_surface_tilt=self.api_params.get('optimal_surface_tilt', False),
                    surface_azimuth=self.api_params.get('surface_azimuth', 180),
                    surface_tilt=self.api_params.get('surface_tilt', 0),
                    trackingtype=self.api_params.get('trackingtype', 0),
                    usehorizon=self.api_params.get('usehorizon', True),
                    userhorizon=self.api_params.get('userhorizon', None),
                )
                # PVGIS gives time slots not as full hours - round to full hour
                self.data.index = self.data.index.round('h')
                self.data.index = self.data.index - self.api_shift

            elif self.data_source == 'solcast api':  # solcast API example selected
                # set api key as bearer token
                headers = {'Authorization': f'Bearer {self.scenario.run.key_api_solcast}'}

                params = {**{'latitude': self.scenario.latitude,  # unmetered location for testing 41.89021,
                             'longitude': self.scenario.longitude,  # unmetered location for testing 12.492231,
                             'period': 'PT5M',
                             'output_parameters': ['air_temp', 'gti', 'wind_speed_10m'],
                             'start': self.scenario.starttime,
                             'end': self.scenario.sim_extd_endtime,
                             'format': 'csv',
                             'time_zone': 'utc',
                             },
                          **{parameter: value for parameter, value in self.api_params.items() if value is not None}}

                url = 'https://api.solcast.com.au/data/historic/radiation_and_weather'

                # get data from Solcast API
                response = requests.get(url, headers=headers, params=params)
                # convert to csv
                self.data = pd.read_csv(io.StringIO(response.text))
                # calculate period_start as only period_end is given, set as index and remove unnecessary columns
                self.data['period_start'] = pd.to_datetime(self.data['period_end']) - pd.to_timedelta(self.data['period'])
                self.data.set_index(pd.DatetimeIndex(self.data['period_start']), inplace=True)
                self.data = self.data.tz_convert('Europe/Berlin')
                self.data.drop(columns=['period', 'period_start', 'period_end'], inplace=True)
                # rename columns according to further processing steps
                self.data.rename(columns={'air_temp': 'temp_air', 'wind_speed_10m': 'wind_speed'}, inplace=True)
                # calculate specific pv power
                self.calc_power_solcast()

        else:  # example from file instead of API
            self.path_input_file = os.path.join(self.scenario.run.path_input_data,
                                                utils.set_extension(self.filename))

            if self.data_source == 'pvgis file':  # data example from fixed PVGIS csv file
                self.data, self.meta, _ = pvlib.iotools.read_pvgis_hourly(self.path_input_file, map_variables=True)
                self.scenario.latitude = self.meta['latitude']
                self.scenario.longitude = self.meta['longitude']
                # PVGIS gives time slots as XX:06 - round to full hour
                self.data.index = self.data.index.round('h')
            elif self.data_source == 'solcast file':  # data example from fixed Solcast csv file
                # no lat/lon contained in solcast files
                self.data = pd.read_csv(self.path_input_file)
                self.data.rename(columns={'PeriodStart': 'period_start',
                                          'PeriodEnd': 'period_end',
                                          'AirTemp': 'temp_air',
                                          'GtiFixedTilt': 'gti',
                                          'WindSpeed10m': 'wind_speed'}, inplace=True)
                self.data['period_start'] = pd.to_datetime(self.data['period_start'], utc=True)
                self.data['period_end'] = pd.to_datetime(self.data['period_end'], utc=True)
                self.data.set_index(pd.DatetimeIndex(self.data['period_start']), inplace=True)
                self.data = self.data[['temp_air', 'wind_speed', 'gti']]
                self.calc_power_solcast()

            else:
                raise ValueError(f'Scenario {self.scenario.name} - Block {self.name}: No usable PV data input specified')

        # resample to timestep, fill NaN values with previous ones (or next ones, if not available)
        self.data = self.data.resample(self.scenario.timestep).mean().ffill().bfill()
        # convert to local time
        self.data.index = self.data.index.tz_convert(tz=self.scenario.timezone)
        # data is in W for a 1kWp PV array -> convert to specific power
        self.data['power_spec'] = self.data['P'] / 1e3

        self.data = self.data[['power_spec', 'wind_speed', 'temp_air']]  # only keep relevant columns

    def update_input_components(self, horizon):
        self.bus_connected = self.scenario.blocks['core'].dc_bus if self.system == 'dc' else self.scenario.blocks['core'].ac_bus
        super().update_input_components(horizon)


class StationaryEnergyStorage(InvestBlock):

    def __init__(self, name, scenario):

        super().__init__(name, scenario)

        # initialization of oemof-solph components
        self.bus = self.bus_connected = self.inflow = self.outflow = self.ess = None

        self.apriori_data = None

        self.eff_chg = self.eff_acdc if self.system == 'ac' else 1
        self.eff_dis = self.eff_dcac if self.system == 'ac' else 1
        self.loss_rate = utils.convert_sdr(self.sdr, pd.Timedelta(hours=1))

        self.storage_timeseries = pd.DataFrame(index=utils.extend_dti(self.scenario.dti_sim),
                                               columns=['soc', 'soh', 'q_loss_cal', 'q_loss_cyc'],
                                               dtype='float64')

        self.storage_timeseries.loc[self.scenario.starttime, 'soc'] = self.soc_init

        self.aging_model = bat.BatteryPackModel(self)
        self.soc_min = (1 - self.storage_timeseries.loc[self.scenario.starttime, 'soh']) / 2
        self.soc_max = 1 - ((1 - self.storage_timeseries.loc[self.scenario.starttime, 'soh']) / 2)

    def add_soc_trace(self):
        legentry = f'{self.name} SOC ({self.size.loc["block", "total"]/1e3:.1f} kWh)'
        self.scenario.figure.add_trace(go.Scatter(x=self.storage_timeseries.index,
                                                  y=self.storage_timeseries['soc'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None)),
                                       secondary_y=True)

        legentry = f'{self.name} SOH'
        data = self.storage_timeseries['soh'].dropna()
        self.scenario.figure.add_trace(go.Scatter(x=data.index,
                                                  y=data,
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=True)

    def calc_aging(self, horizon):
        self.aging_model.age(horizon)

    def calc_energy(self):
        self.calc_energy_storage_block()

    def calc_opex_sim(self):
        # opex are assigned to inflow only
        self.expenditures.loc['opex', 'sim'] = self.flows['in'] @ self.opex_spec[self.scenario.dti_sim] * self.scenario.timestep_hours

    def get_ch_results(self, horizon, *_):

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.outflow, self.bus_connected)]['sequences']['flow'][
            horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.bus_connected, self.inflow)]['sequences']['flow'][
            horizon.dti_ch]

        # storage content during PH (including endtime)
        self.storage_timeseries.loc[utils.extend_dti(horizon.dti_ch), 'soc'] = solph.views.node(horizon.results, self.name)['sequences'][
            ((self.name, 'None'), 'storage_content')][utils.extend_dti(horizon.dti_ch)] / self.size.loc['block', 'total']

    def get_invest_size(self, horizon):
        self.size.loc['block', 'additional'] = horizon.results[(self.ess, None)]['scalars']['invest']
        self.size['total'] = self.size['existing'] + self.size['additional']

    def get_legend_entry(self):
        return (f'{self.name} power (max. {self.size.loc["block", "total"] * self.crate_chg * self.eff_chg / 1e3:.1f} kW charge /'
                f' {self.size.loc["block", "total"] * self.crate_dis * self.eff_dis / 1e3:.1f} kW discharge)')

    def print_results(self):
        if self.invest:
            self.scenario.logger.info(
                f'Optimized size of component "{self.name}": {self.size.loc["block", "total"] / 1e3:.1f} kWh'
                f' (existing: {self.size.loc["block", "existing"] / 1e3:.1f} kWh'
                f' - additional: {self.size.loc["block", "additional"] / 1e3:.1f} kWh)')

    def update_input_components(self, horizon):
        """
        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        xc_bus      self.bus
          |            |
          |<-x-ess_xc--|
          |            |<--->ess
          |-x-xc_ess-->|
          |            |

        """

        self.bus_connected = self.scenario.blocks['core'].dc_bus if self.system == 'dc' else self.scenario.blocks['core'].ac_bus

        self.bus = solph.Bus(label=f'{self.name}_bus')

        self.inflow = solph.components.Converter(label=f'xc_{self.name}',
                                                 inputs={self.bus_connected: solph.Flow(
                                                     variable_costs=self.opex_ep_spec[horizon.dti_ph]
                                                 )},
                                                 outputs={self.bus: solph.Flow()},
                                                 conversion_factors={self.bus: self.eff_chg})

        self.outflow = solph.components.Converter(label=f'{self.name}_xc',
                                                  inputs={self.bus: solph.Flow()},
                                                  # cost_eps are needed to prevent storage from being emptied in RH
                                                  outputs={self.bus_connected: solph.Flow(
                                                      variable_costs=self.scenario.cost_eps
                                                  )},
                                                  conversion_factors={self.bus_connected: self.eff_dis})

        self.ess = solph.components.GenericStorage(label='ess',
                                                   inputs={self.bus: solph.Flow()},
                                                   outputs={self.bus: solph.Flow(variable_costs=self.scenario.cost_eps)},
                                                   loss_rate=self.loss_rate,
                                                   balanced={'go': True, 'rh': False}[self.scenario.strategy],
                                                   initial_storage_level=statistics.median(
                                                       [self.soc_min, self.storage_timeseries.loc[horizon.starttime, 'soc'], self.soc_max]),
                                                   invest_relation_input_capacity=self.crate_chg,
                                                   # crate measured "outside" of conversion factor (efficiency)
                                                   # p_max at outputs is size * crate_dis (not incl. eff)
                                                   invest_relation_output_capacity=self.crate_dis,
                                                   inflow_conversion_factor=np.sqrt(self.eff_roundtrip),
                                                   outflow_conversion_factor=np.sqrt(self.eff_roundtrip),
                                                   nominal_storage_capacity=solph.Investment(ep_costs=self.capex_ep_spec,
                                                                                             existing=self.size.loc['block', 'existing'],
                                                                                             maximum=utils.conv_add_max(self.size.loc['block', 'additional_max'])),
                                                   max_storage_level=pd.Series(data=self.soc_max,
                                                                               index=utils.extend_dti(horizon.dti_ph)),
                                                   min_storage_level=pd.Series(data=self.soc_min,
                                                                               index=utils.extend_dti(horizon.dti_ph))
                                                   )

        horizon.components.append(self.bus)
        horizon.components.append(self.inflow)
        horizon.components.append(self.outflow)
        horizon.components.append(self.ess)

        horizon.constraints.add_invest_costs(invest=(self.ess,),
                                             capex_spec=self.capex_spec,
                                             invest_type='storage')


class SystemCore(InvestBlock):

    def __init__(self, name, scenario):
        self.equal = None

        self.capex_init_existing_acdc = self.capex_init_existing_dcac = 0

        super().__init__(name, scenario, flow_names=['acdc', 'dcac'], size_names=['acdc', 'dcac'])
        self.ac_bus = self.dc_bus = self.ac_dc = self.dc_ac = None  # initialize oemof-solph components

        self.energies.loc['acdc', :] = 0
        self.energies.loc['dcac', :] = 0

    def add_power_trace(self):
        legentry = f'{self.name} DC-AC power (max. {self.size.loc["dcac", "total"]/1e3:.1f} kW)'
        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=self.flows['dcac'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=False)

        legentry = f'{self.name} AC-DC power (max. {self.size.loc["acdc", "total"]/1e3:.1f} kW)'
        self.scenario.figure.add_trace(go.Scatter(x=self.flows.index,
                                                  y=self.flows['acdc'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=False)

    def calc_energy(self):
        """
        energy result calculation is different from any other block as there is no in/out definition of flow,
        but rather acdc/dcac
        """
        if any(~(self.flows['acdc'] == 0) & ~(self.flows['dcac'] == 0)):
            self.scenario.logger.warning(f'Block {self.name} - '
                                         f'simultaneous flow for "acdc" and "dcac" detected!')

        for flow in ['acdc', 'dcac']:
            self.energies.loc[flow, 'sim'] = self.flows[flow].sum() * self.scenario.timestep_hours
            self.energies.loc[flow, 'yrl'] = utils.scale_sim2year(self.energies.loc[flow, 'sim'], self.scenario)
            self.energies.loc[flow, 'prj'] = utils.scale_year2prj(self.energies.loc[flow, 'yrl'], self.scenario)
            self.energies.loc[flow, 'dis'] = eco.acc_discount(self.energies.loc[flow, 'yrl'],
                                                              self.scenario.prj_duration_yrs,
                                                              self.scenario.wacc,
                                                              occurs_at='end')

    def calc_opex_sim(self):
        # opex are assigned to both flows (AC/DC and DC/AC)
        self.expenditures.loc['opex', 'sim'] = (self.flows['acdc'] + self.flows['dcac']) @ self.opex_spec[self.scenario.dti_sim] * self.scenario.timestep_hours

    def get_ch_results(self, horizon):
        self.flows.loc[horizon.dti_ch, 'acdc'] = horizon.results[(self.scenario.blocks['core'].ac_bus, self.ac_dc)]['sequences']['flow'][
            horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'dcac'] = horizon.results[(self.scenario.blocks['core'].dc_bus, self.dc_ac)]['sequences']['flow'][
            horizon.dti_ch]

    def get_invest_size(self, horizon):
        self.size.loc['acdc', 'additional'] = horizon.results[(self.ac_bus, self.ac_dc)]['scalars']['invest']
        self.size.loc['dcac', 'additional'] = horizon.results[(self.dc_bus, self.dc_ac)]['scalars']['invest']

        self.size.loc['block', :] = self.size.loc['acdc', :] + self.size.loc['dcac', :]
        self.size['total'] = self.size['existing'] + self.size['additional']

    def print_results(self):
        if self.invest:
            self.scenario.logger.info(f'Optimized size of AC/DC power in component "{self.name}":'
                                      f' {self.size.loc["acdc", "total"] / 1e3:.1f} kW'
                                      f' (existing: {self.size.loc["acdc", "existing"] / 1e3:.1f} kW'
                                      f' - additional: {self.size.loc["acdc", "additional"] / 1e3:.1f} kW)')
            self.scenario.logger.info(f'Optimized size of DC/AC power in component "{self.name}":'
                                      f' {self.size.loc["dcac", "total"] / 1e3:.1f} kW'
                                      f' (existing: {self.size.loc["dcac", "existing"] / 1e3:.1f} kW'
                                      f' - additional: {self.size.loc["dcac", "additional"] / 1e3:.1f} kW)')

    def set_init_size(self, size_names):
        self.equal = True if self.invest_acdc =='equal' or self.invest_dcac == 'equal' else False

        utils.init_equalizable_variables(block=self, name_vars=['invest_acdc', 'invest_dcac'])
        utils.init_equalizable_variables(block=self, name_vars=['size_acdc_existing', 'size_dcac_existing'])
        utils.init_equalizable_variables(block=self, name_vars=['size_acdc_max', 'size_dcac_max'])

        super().set_init_size(size_names)

    def update_input_components(self, horizon):
        """
        x denotes the flow measurement point in results

        dc_bus       ac_bus
          |-x--dc_ac-->|
          |            |
          |<---ac_dc-x-|
        """

        self.ac_bus = solph.Bus(label='ac_bus')
        horizon.components.append(self.ac_bus)

        self.dc_bus = solph.Bus(label='dc_bus')
        horizon.components.append(self.dc_bus)

        self.ac_dc = solph.components.Converter(label='ac_dc',
                                                inputs={self.ac_bus: solph.Flow(
                                                    nominal_value=solph.Investment(ep_costs=self.capex_ep_spec,
                                                                                   existing=self.size.loc['acdc', 'existing'],
                                                                                   maximum=utils.conv_add_max(self.size.loc['acdc', 'additional_max'])),
                                                    variable_costs=self.opex_ep_spec[horizon.dti_ph])},
                                                outputs={self.dc_bus: solph.Flow(
                                                    variable_costs=self.scenario.cost_eps)},
                                                conversion_factors={self.dc_bus: self.eff_acdc})

        self.dc_ac = solph.components.Converter(label='dc_ac',
                                                inputs={self.dc_bus: solph.Flow(
                                                    nominal_value=solph.Investment(ep_costs=self.capex_ep_spec,
                                                                                   existing=self.size.loc['dcac', 'existing'],
                                                                                   maximum=utils.conv_add_max(self.size.loc['dcac', 'additional_max'])),
                                                    variable_costs=self.opex_ep_spec[horizon.dti_ph])},
                                                outputs={self.ac_bus: solph.Flow(
                                                    variable_costs=self.scenario.cost_eps)},
                                                conversion_factors={self.ac_bus: self.eff_dcac})

        horizon.components.append(self.ac_dc)
        horizon.components.append(self.dc_ac)

        horizon.constraints.add_invest_costs(invest=(self.ac_bus, self.ac_dc),
                                             capex_spec=self.capex_spec,
                                             invest_type='flow')
        horizon.constraints.add_invest_costs(invest=(self.dc_bus, self.dc_ac),
                                             capex_spec=self.capex_spec,
                                             invest_type='flow')

        if self.equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            horizon.constraints.add_equal_invests([{'in': self.dc_bus, 'out': self.dc_ac},
                                                   {'in': self.ac_bus, 'out': self.ac_dc}])

class VehicleCommoditySystem(CommoditySystem):
    """
    VehicleCommoditySystem is a CommoditySystem operating vehicles in a service system generating revenues from rentals
    """

    def __init__(self, name, scenario):
        self.opex_spec = self.mntex_spec = 0  # ensures common methods for Battery and VehicleCommoditySystems
        self.demand = mobility.VehicleCommodityDemand(scenario, self)
        super().__init__(name, scenario)


class WindSource(RenewableInvestBlock):

    def __init__(self, name, scenario):

        super().__init__(name, scenario)

        self.path_turbine_data_file = self.turbine_data = self.turbine_type = None

    def get_timeseries_data(self):

        if self.data_source in self.scenario.blocks.keys():  # example from a PV block

            self.data = self.scenario.blocks[self.data_source].data.copy()
            self.data['wind_speed_adj'] = windpowerlib.wind_speed.hellman(self.data['wind_speed'], 10, self.height)

            self.path_turbine_data_file = os.path.join(self.scenario.run.path_data_immut, 'turbine_data.pkl')
            self.turbine_type = 'E-53/800'  # smallest fully filled wind turbine in dataseta as per June 2024
            self.turbine_data = pd.read_pickle(self.path_turbine_data_file)
            self.turbine_data = self.turbine_data.loc[
                self.turbine_data['turbine_type'] == self.turbine_type].reset_index()

            self.data['power_original'] = windpowerlib.power_output.power_curve(
                wind_speed=self.data['wind_speed_adj'],
                power_curve_wind_speeds=ast.literal_eval(self.turbine_data.loc[0, 'power_curve_wind_speeds']),
                power_curve_values=ast.literal_eval(self.turbine_data.loc[0, 'power_curve_values']),
                density_correction=False)
            self.data['power_spec'] = self.data['power_original'] / self.turbine_data.loc[0, 'nominal_power']

        else:  # example from file instead of PV block

            self.path_input_file = os.path.join(self.scenario.run.path_input_data,
                                                utils.set_extension(self.filename))
            self.data = utils.read_input_csv(self, self.path_input_file, self.scenario)

    def update_input_components(self, horizon):
        self.bus_connected = self.scenario.blocks['core'].ac_bus if self.system == 'ac' else self.scenario.blocks['core'].dc_bus
        super().update_input_components(horizon)
