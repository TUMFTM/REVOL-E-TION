#!/usr/bin/env python3

import graphviz
import importlib.metadata
import logging
import logging.handlers
import math
import os
import pathlib
import plotly.subplots
import pprint
import psutil
import pytz
import shutil
import subprocess
import sys
import time
import timezonefinder
import traceback

import multiprocessing as mp
import numpy_financial as npf
import oemof.solph as solph
import pandas as pd
import tkinter as tk
import tkinter.filedialog

from revoletion import blocks
from revoletion import checker
from revoletion import constraints
from revoletion import colors
from revoletion import dispatch
from revoletion import scheduler
from revoletion import utils

class OptimizationSuccessfulFilter(logging.Filter):
    def filter(self, record):
        # Filter out log messages from the root logger
        return not (record.name == 'root' and record.msg == 'Optimization successful...')


class PredictionHorizon:

    def __init__(self, index, scenario, run):

        self.index = index

        # Time and data slicing --------------------------------
        self.starttime = scenario.starttime + (index * scenario.len_ch)  # calc both start times
        self.ch_endtime = self.starttime + scenario.len_ch
        self.ph_endtime = self.starttime + scenario.len_ph
        self.timestep = scenario.timestep

        # Display logger message if PH exceeds simulation end time and has to be truncated
        if self.ph_endtime > scenario.sim_endtime and scenario.truncate_ph:
            scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - ' +
                                 f'Prediction Horizon truncated to simulation end time')

        # Truncate PH and CH to simulation or eval end time
        self.ph_endtime = min(self.ph_endtime, scenario.sim_extd_endtime)
        self.ch_endtime = min(self.ch_endtime, scenario.sim_endtime)

        scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - ' +
                             f'Start: {self.starttime} - ' +
                             f'CH end: {self.ch_endtime} - ' +
                             f'PH end: {self.ph_endtime}')

        # Create datetimeindex for ph and ch; neglect last timestep as this is the first timestep of the next ph / ch
        self.dti_ph = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=scenario.timestep, inclusive='left')
        self.dti_ch = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=scenario.timestep, inclusive='left')

        scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                             f'Initializing model build')

        for block in [block for block in scenario.blocks.values() if hasattr(block, 'data')]:
            block.data_ph = block.data[self.starttime:self.ph_endtime]
            if isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities.values():
                    commodity.data_ph = commodity.data[self.starttime:self.ph_endtime]

        # if apriori power scheduling is necessary, calculate power schedules:
        if scenario.scheduler:
            scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                                 f'Calculating power schedules for commodities with rulebased charging strategies')
            scenario.scheduler.calc_ph_schedule(self)

        for block in scenario.blocks.values():
            block.update_input_components(scenario, self)  # (re)define solph components that need input slices

        self.results = None
        self.meta_results = None

        # Build energy system model --------------------------------

        scenario.logger.debug(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                              f'Building energy system instance')

        self.es = solph.EnergySystem(timeindex=self.dti_ph,
                                     infer_last_interval=True)  # initialize energy system model instance

        for component in scenario.components:
            self.es.add(component)  # add components to this horizon's energy system

        if self.index == 0 and run.save_system_graphs:  # first horizon - create graph of energy system
            self.draw_energy_system(scenario)

        scenario.logger.debug(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                              f'Creating optimization model')

        try:
            # Build the mathematical linear optimization model with pyomo
            self.model = solph.Model(self.es, debug=run.debugmode)
        except IndexError:
            msg = (f'Horizon {self.index + 1} of {scenario.nhorizons} -'
                   f'Input data not matching time index - check input data and time index consistency')
            scenario.logger.error(msg)
            raise IndexError(msg)

        # Apply custom constraints
        scenario.constraints.apply_constraints(model=self.model)

        if run.dump_model and scenario.strategy != 'rh':
            self.model.write(run.path_dump_file, io_options={'symbolic_solver_labels': True})

        scenario.logger.debug(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                              f'Model build completed')

    def draw_energy_system(self, scenario):

        # Creates the Directed-Graph
        dot = graphviz.Digraph(filename=scenario.path_system_graph_file, format='pdf')

        dot.node("Bus", shape='rectangle', fontsize="10", color='red')
        dot.node("Sink", shape='trapezium', fontsize="10")
        dot.node("Source", shape='invtrapezium', fontsize="10")
        dot.node("Transformer", shape='rectangle', fontsize="10")
        dot.node("Storage", shape='rectangle', style='dashed', fontsize="10", color="green")

        busses = []
        # draw a node for each of the network's component. The shape depends on the component's type
        for nd in self.es.nodes:
            if isinstance(nd, solph.Bus):
                dot.node(nd.label,
                         shape='rectangle',
                         fontsize="10",
                         fixedsize='shape',
                         width='2.4',
                         height='0.6',
                         color='red')
                # keep the bus reference for drawing edges later
                busses.append(nd)
            elif isinstance(nd, solph.components.Sink):
                dot.node(nd.label, shape='trapezium', fontsize="10")
            elif isinstance(nd, solph.components.Source):
                dot.node(nd.label, shape='invtrapezium', fontsize="10")
            elif isinstance(nd, solph.components.Converter):
                dot.node(nd.label, shape='rectangle', fontsize="10")
            elif isinstance(nd, solph.components.GenericStorage):
                dot.node(nd.label, shape='rectangle', style='dashed', fontsize="10", color="green")
            else:
                scenario.logger.debug(f'System Node {nd.label} - Type {type(nd)} not recognized')

        # draw the edges between the nodes based on each bus inputs/outputs
        for bus in busses:
            for component in bus.inputs:
                # draw an arrow from the component to the bus
                dot.edge(component.label, bus.label)
            for component in bus.outputs:
                # draw an arrow from the bus to the component
                dot.edge(bus.label, component.label)

        try:
            dot.render()
        except Exception as e:  # inhibiting renderer from stopping model execution
            scenario.logger.warning(f'Graphviz rendering failed - '
                                    f'Error Message: {e}')

    def get_results(self, scenario, run):
        """
        Get result data slice for current CH from results and save in result dataframes for later analysis
        Get (possibly optimized) component sizes from results to handle outputs more easily
        """

        scenario.logger.debug(f'Horizon {self.index + 1} of {scenario.nhorizons} - Getting results')

        self.results = solph.processing.results(self.model)  # Get the results of the solved horizon from the solver

        if run.debugmode:
            self.meta_results = solph.processing.meta_results(self.model)
            pprint.pprint(self.meta_results)

        # free up RAM
        del self.model

        # get optimum component sizes for optimized blocks
        for block in [block for block in scenario.blocks.values()
                      if isinstance(block, blocks.InvestBlock) and block.invest]:
            block.get_opt_size(self)

        for block in scenario.blocks.values():
            block.get_ch_results(self, scenario)

        for block in [block for block in scenario.blocks.values() if hasattr(block, 'aging')]:
            if block.aging:
                block.calc_aging(run, scenario, self)

    def run_optimization(self, scenario, run):
        scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                             f'Model built, starting optimization')
        try:
            self.model.solve(solver=run.solver, solve_kwargs={'tee': run.debugmode})
            scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                                 f'Optimization completed, getting results')
        except UserWarning as exc:
            scenario.logger.warning(f'Scenario failed or infeasible - continue on next scenario')
            scenario.exception = str(exc)


class Scenario:

    def __init__(self, scenario_name, run, logger):

        self.name = scenario_name
        self.run = run
        self.logger = logger
        self.logger.propagate = False

        # General Information --------------------------------

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        self.worker = mp.current_process()

        self.logger.info(f'Scenario initialized on {self.worker.name.ljust(18)}' +
                         (f' - Parent: {self.worker._parent_name}' if hasattr(self.worker, '_parent_name') else ''))

        self.parameters = run.scenario_data[self.name]
        for key, value in self.parameters.loc['scenario', :].items():
            setattr(self, key, value)  # this sets all the parameters defined in the csv file

        # add SystemCore to blocks ensuring SystemCore is the first component to be built
        self.blocks = {**{'core': 'SystemCore'}, **self.blocks}

        self.currency = self.currency.upper()  # all other parameters are .lower()-ed

        self.tzfinder = timezonefinder.TimezoneFinder()
        self.timezone = pytz.timezone(self.tzfinder.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        # convert to datetime and calculate time(delta) values
        # simulation and project timeframe start simultaneously
        # simulation vs. extended simulation: for rh strategy and truncate_ph = False, the extended simulation timeframe
        # is longer than the simulation timeframe defined by the input parameter duration. Otherwise, they are the same.
        self.starttime = pd.to_datetime(self.starttime, format='%d.%m.%Y').tz_localize(self.timezone)
        self.sim_duration = pd.Timedelta(days=self.sim_duration)
        self.sim_extd_duration = self.sim_duration
        self.sim_endtime = self.starttime + self.sim_duration
        self.sim_extd_endtime = self.sim_endtime
        self.prj_duration_yrs = self.prj_duration
        self.prj_endtime = self.starttime + pd.DateOffset(years=self.prj_duration)
        self.prj_duration = self.prj_endtime - self.starttime

        if self.strategy == 'rh':
            self.len_ph = pd.Timedelta(hours=self.len_ph)
            self.len_ch = pd.Timedelta(hours=self.len_ch)
            self.nhorizons = math.ceil(self.sim_duration / self.len_ch)  # number of timeslices to run
            if not self.truncate_ph:
                # if PH is not truncated, the end of the last PH may be later than the end of the evaluation period
                self.sim_extd_duration = self.len_ch * (self.nhorizons - 1) + self.len_ph
                self.sim_extd_endtime = self.starttime + self.sim_extd_duration
        elif self.strategy in ['go']:
            self.len_ph = self.sim_duration
            self.len_ch = self.sim_duration
            self.nhorizons = 1

        # generate a datetimeindex for the energy system model to run on
        self.dti_sim = pd.date_range(start=self.starttime, end=self.sim_endtime, freq=self.timestep, inclusive='left')
        self.dti_sim_extd = pd.date_range(start=self.starttime, end=self.sim_extd_endtime, freq=self.timestep,
                                          inclusive='left')

        # generate variables for calculations
        self.timestep_td = pd.Timedelta(self.dti_sim_extd.freq)
        self.timestep_hours = self.timestep_td.total_seconds() / 3600
        self.sim_yr_rat = self.sim_duration.days / 365  # no leap years
        self.sim_prj_rat = self.sim_duration.days / self.prj_duration.days

        # prepare for system graph saving later on
        self.path_system_graph_file = os.path.join(
            run.path_result_dir,
            f'{run.runtimestamp}_{run.scenario_file_name}_{self.name}_system_graph.pdf')

        # prepare for dispatch plot saving later on
        self.plot_file_path = os.path.join(run.path_result_dir, f'{run.runtimestamp}_'
                                                                   f'{run.scenario_file_name}_'
                                                                   f'{self.name}.html')

        # prepare for cumulative result saving later on
        self.result_summary = pd.DataFrame(columns=['Block', 'Key', self.name])
        self.result_summary = self.result_summary.set_index(['Block', 'Key'])
        self.path_result_summary_tempfile = os.path.join(
            run.path_result_dir,
            f'{self.name}_tempresults.csv')

        self.result_timeseries = pd.DataFrame(index=self.dti_sim_extd)
        self.path_result_file = os.path.join(
            run.path_result_dir,
            f'{run.runtimestamp}_{run.scenario_file_name}_{self.name}_results.csv')

        self.exception = None  # placeholder for possible infeasibility

        # Energy System Blocks --------------------------------

        self.components = []  # placeholder
        self.constraints = constraints.CustomConstraints(scenario=self)

        # create all block objects defined in the scenario DataFrame under "scenario/blocks" as a dict
        self.blocks = self.create_block_objects(self.blocks, run)
        self.commodity_systems = {block.name: block for block in self.blocks.values()
                                  if isinstance(block, blocks.CommoditySystem)}

        # Execute commodity system discrete event simulation
        # can only be started after all blocks have been initialized, as the different systems depend on each other.
        if any([cs.data_source == 'des' for cs in self.commodity_systems.values()]):
            dispatch.execute_des(self, run)

        for cs in [cs for cs in self.commodity_systems.values() if cs.data_source == 'des']:
            for commodity in cs.commodities.values():
                commodity.data = cs.data.loc[:, (commodity.name, slice(None))].droplevel(0, axis=1)

        # ToDo: put into extra function
        # check input parameter configuration for model dump
        if run.dump_model and self.strategy == 'rh':
            self.logger.warning('Model file dump not implemented for RH operating strategy - ' +
                                'File dump deactivated for current scenario')

        # check input parameter configuration of rulebased charging for validity
        if cs_unlim := [cs for cs in self.commodity_systems.values() if
                        (cs.lvl_opt in self.run.apriori_lvls)
                        and cs.lvl_opt != 'uc'
                        and not cs.power_lim_static]:
            if [block for block in self.blocks.values() if getattr(block, 'invest', False)]:
                run.logger.error(f'Scenario {self.name} - Rulebased charging except for uncoordinated charging (uc)'
                                 f' without static load management (lm_static) is not compatible with size optimization')
                exit()  # TODO exit scenario instead of run
            if [block for block in self.blocks.values() if isinstance(block, blocks.StationaryEnergyStorage)]:
                run.logger.error(f'Scenario {self.name} - Rulebased charging except for uncoordinated charging (uc)'
                                 f' without static load management (lm_static) is not implemented for systems with'
                                 f' stationary energy storage')
                exit()  # TODO exit scenario instead of run
            if len(set([cs.lvl_opt for cs in cs_unlim])) > 1:
                run.logger.error(f'Scenario {self.name} - All rulebased CommoditySystems with dynamic load management'
                                 f' have to follow the same strategy. Different strategies are not possible')
                exit()  # TODO exit scenario instead of run
            if cs_unlim[0].lvl_opt == 'equal' and len(set([cs.bus_connected for cs in cs_unlim])) > 1:
                run.logger.error(f'Scenario {self.name} - If strategy "equal" is chosen for CommoditySystems with'
                                 f' dynamic load management, all CommoditySystems with dynamic load management have to'
                                 f' be connected to the same bus')
                exit()  # TODO exit scenario instead of run

        self.scheduler = None
        if any([cs for cs in self.commodity_systems.values() if cs.lvl_opt in self.run.apriori_lvls]):
            self.scheduler = scheduler.AprioriPowerScheduler(scenario=self)

        # Result variables --------------------------------
        self.figure = None  # placeholder for plotting

        self.cashflows = pd.DataFrame()

        # Result variables - Energy
        self.e_sim_del = self.e_yrl_del = self.e_prj_del = self.e_dis_del = 0
        self.e_sim_pro = self.e_yrl_pro = self.e_prj_pro = self.e_dis_pro = 0
        self.e_sim_ext = self.e_yrl_ext = self.e_prj_ext = self.e_dis_ext = 0  # external charging
        self.e_eta = None
        self.renewable_curtailment = self.renewable_share = None
        self.e_renewable_act = self.e_renewable_pot = self.e_renewable_curt = 0

        # Result variables - Cost
        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = 0
        self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = 0
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = 0
        self.opex_sim_ext = self.opex_yrl_ext = self.opex_prj_ext = self.opex_dis_ext = self.opex_ann_ext = 0
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = 0
        self.crev_sim = self.crev_yrl = self.crev_prj = self.crev_dis = 0
        self.lcoe_total = self.lcoe_wocs = None
        self.npv = self.irr = self.mirr = None

        self.logger.debug(f'Scenario initialization completed')

    def calc_meta_results(self):

        # TODO implement commodity v2s usage share
        # TODO implement energy storage usage share

        if self.e_sim_pro == 0:
            self.logger.warning(f'Core efficiency calculation: division by zero')
        else:
            try:
                self.e_eta = self.e_sim_del / self.e_sim_pro
            except ZeroDivisionError:
                self.logger.warning(f'Core efficiency calculation: division by zero')

        if self.e_renewable_pot == 0:
            self.logger.warning(f'Renewable curtailment calculation: division by zero')
        else:
            try:
                self.renewable_curtailment = self.e_renewable_curt / self.e_renewable_pot
            except ZeroDivisionError:
                self.logger.warning(f'Renewable curtailment calculation: division by zero')

        if self.e_sim_pro == 0:
            self.logger.warning(f'Renewable share calculation: division by zero')
        else:
            try:
                self.renewable_share = self.e_renewable_act / self.e_sim_pro
            except ZeroDivisionError:
                self.logger.warning(f'Renewable share calculation: division by zero')

        totex_dis_cs = sum([cs.totex_dis for cs in self.commodity_systems.values()])
        if self.e_dis_del == 0:
            self.logger.warning(f'LCOE calculation: division by zero')
        else:
            try:
                self.lcoe_total = self.totex_dis / self.e_dis_del
                self.lcoe_wocs = (self.totex_dis - totex_dis_cs) / self.e_dis_del
            except ZeroDivisionError:
                self.lcoe_total = self.lcoe_wocs = None
                self.logger.warning(f'LCOE calculation: division by zero')

        self.npv = self.crev_dis - self.totex_dis
        self.irr = npf.irr(self.cashflows.sum(axis=1).to_numpy())
        self.mirr = npf.mirr(self.cashflows.sum(axis=1).to_numpy(), self.wacc, self.wacc)

        # print basic results
        self.logger.info(f'NPC {f"{self.totex_dis:,.2f}" if pd.notna(self.totex_dis) else "-"} {self.currency} -'
                         f' NPV {f"{self.npv:,.2f}" if pd.notna(self.npv) else "-"} {self.currency} -'
                         f' LCOE {f"{self.lcoe_wocs * 1e5:,.1f}" if pd.notna(self.lcoe_wocs) else "-"} {self.currency}-ct/kWh -'
                         f' mIRR {f"{self.mirr * 100:,.2f}" if pd.notna(self.mirr) else "-"} % -'
                         f' Renewable Share:'
                         f' {f"{self.renewable_share * 100:.1f}" if pd.notna(self.renewable_share) else "-"} % -'
                         f' Renewable Curtailment:'
                         f' {f"{self.renewable_curtailment * 100:.1f}" if pd.notna(self.renewable_curtailment) else "-"} %')

    def create_block_objects(self, class_dict, run):
        objects = {}
        for name, class_name in class_dict.items():
            class_obj = getattr(blocks, class_name, None)
            if class_obj is not None and isinstance(class_obj, type):
                objects[name] = class_obj(name, self, run)
            else:
                raise ValueError(f"Class '{class_name}' not found in blocks.py file - Check for typos or add class.")
        return objects

    def end_timing(self):
        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 2)
        self.logger.info(f'Scenario finished - runtime {self.runtime_len} s')

    def generate_plots(self):

        self.figure = plotly.subplots.make_subplots(specs=[[{'secondary_y': True}]])

        for block in self.blocks.values():
            block.add_power_trace(self)
            if hasattr(block, 'add_soc_trace'):  # should affect CommoditySystems and StationaryEnergyStorage
                block.add_soc_trace(self)
            if hasattr(block, 'add_curtailment_trace'):  # should affect PVSource and WindSource
                block.add_curtailment_trace(self)

        self.figure.update_layout(plot_bgcolor=colors.tum_white)
        self.figure.update_xaxes(title='Local Time',
                                 showgrid=True,
                                 linecolor=colors.tum_grey_20,
                                 gridcolor=colors.tum_grey_20, )
        self.figure.update_yaxes(title='Power in W',
                                 showgrid=True,
                                 linecolor=colors.tum_grey_20,
                                 gridcolor=colors.tum_grey_20,
                                 secondary_y=False, )
        self.figure.update_yaxes(title='State of Charge',
                                 showgrid=False,
                                 secondary_y=True)

        if self.strategy == 'go':
            self.figure.update_layout(title=f'Global Optimum Results - '
                                            f'{self.run.scenario_file_name} - '
                                            f'Scenario: {self.name}')
        if self.strategy == 'rh':
            self.figure.update_layout(title=f'Rolling Horizon Results - '
                                            f'{self.run.scenario_file_name} - '
                                            f'Scenario: {self.name} - '
                                            f'PH: {self.len_ph}h - '
                                            f'CH: {self.len_ch}h')

    def get_results(self):
        for block in self.blocks.values():
            block.calc_energy(self)
            block.calc_expenses(self)
            block.calc_revenue(self)
            block.calc_cashflows(self)

    def print_results(self):
        print('#################')
        for block in [block for block in self.blocks.values() if hasattr(block, 'invest') and block.invest]:
            unit = 'kWh' if isinstance(block, (blocks.CommoditySystem, blocks.StationaryEnergyStorage)) else 'kW'
            if isinstance(block, blocks.SystemCore):
                self.logger.info(f'Optimized size of AC/DC power in component \"{block.name}\":'
                                 f' {block.size_acdc / 1e3:.1f} {unit}'
                                 f' (existing: {block.size_acdc_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_acdc_additional / 1e3:.1f} {unit})')
                self.logger.info(f'Optimized size of DC/AC power in component \"{block.name}\":'
                                 f' {block.size_dcac / 1e3:.1f} {unit}'
                                 f' (existing: {block.size_dcac_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_dcac_additional / 1e3:.1f} {unit})')
            elif isinstance(block, blocks.GridConnection):
                self.logger.info(f'Optimized size of g2s power in component \"{block.name}\":'
                                 f' {block.size_g2s / 1e3:.1f} {unit}' + \
                                 f' (existing: {block.size_g2s_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_g2s_additional / 1e3:.1f} {unit})')
                self.logger.info(f'Optimized size of s2g power in component \"{block.name}\":'
                                 f' {block.size_s2g / 1e3:.1f} {unit}'
                                 f' (existing: {block.size_s2g_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_s2g_additional / 1e3:.1f} {unit})')
                if block.peakshaving:
                    for interval in block.peakshaving_ints.index:
                        if block.peakshaving_ints.loc[interval, 'start'] <= self.dti_sim[-1]:
                            self.logger.info(f'Optimized peak power in component \"{block.name}\" for interval'
                                             f' {interval}: {block.peakshaving_ints.loc[interval, "power"] / 1e3:.1f} {unit}'
                                             f' - OPEX: {block.peakshaving_ints.loc[interval, ["period_fraction", "power", "opex_spec"]].prod():.2f} {self.currency}')

            elif isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities.values():
                    self.logger.info(f'Optimized size of commodity \"{commodity.name}\" in component \"{block.name}\":'
                                     f' {commodity.size / 1e3:.1f} {unit}'
                                     f' (existing: {commodity.size_existing / 1e3:.1f} {unit}'
                                     f' - additional: {commodity.size_additional / 1e3:.1f} {unit})')
            else:
                self.logger.info(f'Optimized size of component \"{block.name}\": {block.size / 1e3:.1f} {unit}'
                                 f' (existing: {block.size_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_additional / 1e3:.1f} {unit})')

        # ToDo: state that these results are internal costs of local site only neglecting costs for external charging
        self.logger.info(f'Total simulated cost at local site: {self.totex_sim / 1e6:.2f} million {self.currency}')
        self.logger.info(f'Total simulated cost for external charging: {self.opex_sim_ext:.2f} {self.currency}')
        self.logger.info(
            f'Levelized cost of electricity for local site: {f"{1e5 * self.lcoe_wocs:,.2f}" if pd.notna(self.lcoe_wocs) else "-"} {self.currency}-ct/kWh')
        print('#################')

    def save_plots(self):
        self.figure.write_html(self.plot_file_path)

    def save_result_summary(self):
        """
        Saves all int, float and str attributes of run, scenario (incl. technoeconomic KPIs) and all blocks to the
        results dataframe
        :return: none
        """

        def write_values(name, block):
            for key in [key for key in block.__dict__.keys() if isinstance(block.__dict__[key], result_types)]:
                value = block.__dict__[key]
                if isinstance(value, int):
                    self.result_summary.loc[(name, key), self.name] = float(value)
                else:
                    self.result_summary.loc[(name, key), self.name] = value

        result_types = (int, float, str, bool)
        result_blocks = {'run': self.run, 'scenario': self}
        result_blocks.update(self.blocks)

        for block_name, block_obj in result_blocks.items():
            write_values(block_name, block_obj)
            if isinstance(block_obj, blocks.CommoditySystem):
                for commodity_name, commodity_obj in block_obj.commodities.items():
                    write_values(commodity_name, commodity_obj)
            if hasattr(block_obj, 'peakshaving_ints') and block_obj.peakshaving:
                for interval in block_obj.peakshaving_ints.index:
                    if block_obj.peakshaving_ints.loc[interval, 'start'] <= self.dti_sim[-1]:
                        self.result_summary.loc[(block_name, f'power_peak_{interval}'), self.name] = float(
                            block_obj.peakshaving_ints.loc[interval, 'power'])
                        self.result_summary.loc[(block_name, f'power_period_fraction_{interval}'), self.name] = float(
                            block_obj.peakshaving_ints.loc[interval, 'period_fraction'])
                        self.result_summary.loc[(block_name, f'power_opex_spec_{interval}'), self.name] = float(
                            block_obj.peakshaving_ints.loc[interval, 'opex_spec'])
                        self.result_summary.loc[(block_name, f'power_opex_{interval}'), self.name] = \
                        block_obj.peakshaving_ints.loc[interval, ["period_fraction", "power", "opex_spec"]].prod()

        self.result_summary.reset_index(inplace=True, names=['block', 'key'])
        self.result_summary.to_csv(self.path_result_summary_tempfile, index=False)

    def save_result_timeseries(self):
        for block in self.blocks.values():
            block.get_timeseries_results(self)
        self.result_timeseries.to_csv(self.path_result_file)

    def show_plots(self):
        self.figure.show(renderer='browser')


class SimulationRun:

    def __init__(self):

        self.name = 'run'
        self.cwd = os.getcwd()
        self.process = None

        self.scenarios_file_path = self.settings_file_path = None
        if len(sys.argv) == 1:  # no arguments passed
            self.select_arguments()
        elif len(sys.argv) == 3:  # two arguments passed
            self.read_arguments()
        else:
            raise ValueError('Invalid number of arguments - please provide either none (GUI input) '
                             'or two arguments: scenarios file name or path and settings file name or path')

        self.runtime_start = time.perf_counter()
        self.runtimestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
        self.runtime_end = self.runtime_len = None

        self.version_solph = solph.__version__
        self.version_revoletion = importlib.metadata.version('revoletion')
        self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()[0:6]

        self.scenario_file_name = pathlib.Path(self.scenarios_file_path).stem  # file name without extension
        self.scenario_data = pd.read_csv(self.scenarios_file_path,
                                         index_col=[0, 1],
                                         keep_default_na=False)
        self.scenario_data = self.scenario_data.sort_index(sort_remaining=True).map(utils.infer_dtype)
        self.scenario_names = self.scenario_data.columns  # Get list of column names, each column is one scenario
        self.scenario_num = len(self.scenario_names)

        self.settings = pd.read_csv(self.settings_file_path, index_col=[0])
        self.settings = self.settings.map(utils.infer_dtype)
        for key, value in self.settings['value'].items():
            setattr(self, key, value)  # this sets all the parameters defined in the settings file
        checker.check_settings_complete(self)
        self.define_paths()
        self.get_process_num()
        self.copy_scenario_file()

        self.define_logger()

        # integration levels at which power consumption is determined a priori
        self.result_df = pd.DataFrame  # blank DataFrame for technoeconomic result saving
        self.apriori_lvls = ['uc', 'fcfs', 'equal', 'soc']

    def copy_scenario_file(self):
        shutil.copy2(self.scenarios_file_path, self.path_result_scenario_file)

    def define_logger(self):

        self.logger = logging.getLogger()
        log_formatter = logging.Formatter(f'%(levelname)-{len("WARNING")}s'
                                          f'  %(name)-{max([len(el) for el in list(self.scenario_names) + ["root"]])}s'
                                          f'  %(message)s')
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setFormatter(log_formatter)
        log_file_handler = logging.FileHandler(os.environ.get("LOGFILE", self.path_log_file))
        log_file_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_stream_handler)
        self.logger.addHandler(log_file_handler)

        # Adding the custom filter to prevent root logger messages
        log_stream_handler.addFilter(OptimizationSuccessfulFilter())
        log_file_handler.addFilter(OptimizationSuccessfulFilter())

        if self.parallel:
            log_stream_handler.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)
        else:
            if self.debugmode:
                self.logger.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
                log_stream_handler.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
                log_stream_handler.setLevel(logging.INFO)

        # plural extensions
        pe1 = 's' if self.scenario_num > 1 else ''
        pe2 = 'es' if self.process_num > 1 else ''

        mode = f'parallel mode with {self.process_num} process{pe2}' if self.parallel else 'sequential mode'
        self.logger.info(
            f'Global settings read - running {self.scenario_num} scenario{pe1} in {mode}'
        )

        # make sure that errors are logged to logfile
        sys.excepthook = self.handle_exception

    def define_paths(self):

        if self.path_input_data == 'project':
            self.path_input_data = os.path.join(self.cwd, 'input')
        elif os.path.isdir(self.path_input_data):
            pass  # no modification of path necessary
        else:
            raise NotADirectoryError(f'Input directory not found: {self.path_input_data}')

        if self.path_output_data == 'project':
            self.path_output_data = os.path.join(self.cwd, 'results')
        elif os.path.isdir(self.path_output_data):
            pass  # no modification of path necessary
        else:
            raise NotADirectoryError(f'Output directory not found: {self.path_output_data}')

        self.path_result_dir = os.path.join(self.path_output_data,
                                            f'{self.runtimestamp}_{self.scenario_file_name}')
        os.mkdir(self.path_result_dir)

        self.path_result_scenario_file = os.path.join(self.path_result_dir,
                                                      f'{self.runtimestamp}_{self.scenario_file_name}_scenarios.csv')
        self.path_result_summary_file = os.path.join(self.path_result_dir,
                                                     f'{self.runtimestamp}_{self.scenario_file_name}_summary.csv')
        self.path_dump_file = os.path.join(self.path_result_dir, f'{self.runtimestamp}_{self.scenario_file_name}.lp')
        self.path_log_file = os.path.join(self.path_result_dir, f'{self.runtimestamp}_{self.scenario_file_name}.log')

    def end_timing(self):

        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 1)
        self.logger.info(f'Total runtime for all scenarios: {str(self.runtime_len)} s')

    def get_process_num(self):
        if self.max_process_num == 'max':
            self.max_process_num = os.cpu_count()
        elif self.max_process_num == 'physical':
            self.max_process_num = psutil.cpu_count(logical=False)
        else:
            self.max_process_num = int(self.max_process_num)
        self.process_num = min(self.scenario_num, os.cpu_count(), self.max_process_num)

        if (len(self.scenario_names) == 1 or self.process_num == 1) and self.parallel:
            print('Single scenario or process: Parallel mode not possible - switching to sequential mode')
            self.parallel = False

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error(f"Exception type: {exc_type.__name__}")
        self.logger.error(f"Exception message: {str(exc_value)}")
        self.logger.error("Traceback:")
        self.logger.error(''.join(traceback.format_tb(exc_traceback)))

        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        exit()

    def join_results(self):

        files = [filename for filename in os.listdir(self.path_result_dir) if filename.endswith('_tempresults.csv')]

        scenario_frames = []

        for file in files:
            file_path = os.path.join(self.path_result_dir, file)
            file_results = pd.read_csv(file_path, index_col=[0, 1], header=[0], low_memory=False)
            scenario_frames.append(file_results)

        joined_results = pd.concat(scenario_frames, axis=1)[self.scenario_names]
        joined_results.reset_index(inplace=True, names=['block', 'key'])  # necessary for saving in csv
        joined_results.to_csv(self.path_result_summary_file)
        self.logger.info("Technoeconomic output file created")

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in files:
            file_path = os.path.join(self.path_result_dir, file)
            os.remove(file_path)

    def read_arguments(self):

        if os.path.isfile(sys.argv[1]):
            self.scenarios_file_path = sys.argv[1]
        elif os.path.isfile(os.path.join(self.cwd, 'input', 'scenarios', sys.argv[1])):
            self.scenarios_file_path = os.path.join(self.cwd, 'input', 'scenarios', sys.argv[1])
        else:
            raise FileNotFoundError(f'Scenario file or path not found: {sys.argv[1]}')

        if os.path.isfile(sys.argv[2]):
            self.settings_file_path = sys.argv[2]
        elif os.path.isfile(os.path.join(self.cwd, 'input', 'settings', sys.argv[2])):
            self.settings_file_path = os.path.join(self.cwd, 'input', 'settings', sys.argv[2])
        else:
            raise FileNotFoundError(f'Settings file or pathnot found: {sys.argv[2]} not found')

    def select_arguments(self):

        root = tk.Tk()
        root.withdraw()  # hide small tk-window
        root.lift()  # make sure all tk windows appear in front of other windows

        # get scenarios file
        scenarios_default_dir = os.path.join(self.cwd, 'input', 'scenarios')
        self.scenarios_file_path = tk.filedialog.askopenfilename(initialdir=scenarios_default_dir,
                                                                 title="Select scenario file",
                                                                 filetypes=(("CSV files", "*.csv"),
                                                                            ("All files", "*.*")))
        if not self.scenarios_file_path:
            raise FileNotFoundError('No scenario file selected')

        # get settings file
        settings_default_dir = os.path.join(self.cwd, 'input', 'settings')
        self.settings_file_path = tk.filedialog.askopenfilename(initialdir=settings_default_dir,
                                                                title="Select settings file",
                                                                filetypes=(("CSV files", "*.csv"),
                                                                           ("All files", "*.*")))
        if not self.settings_file_path:
            raise FileNotFoundError('No settings file selected')