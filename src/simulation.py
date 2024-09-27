#!/usr/bin/env python3

import graphviz
import logging
import logging.handlers
import math
import os
import pprint
import psutil
import pytz
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

from pathlib import Path
from plotly.subplots import make_subplots

from src import blocks
from src import commodity_des as des
from src import custom_constraints as cc
from src import tum_colors as col
from src import scheduler
from src import utils


def input_gui(directory):
    # create a Tkinter window to select files and folders
    root = tk.Tk()
    root.withdraw()  # hide small tk-window
    root.lift()  # make sure all tk windows appear in front of other windows

    # get scenario file
    scenarios_default_dir = os.path.join(directory, 'input', 'scenarios')
    scenarios_default_filename = os.path.join(scenarios_default_dir, 'example.csv')
    scenarios_filename = tk.filedialog.askopenfilename(initialdir=scenarios_default_dir, title="Select scenario file",
                                                       filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not scenarios_filename:
        scenarios_filename = scenarios_default_filename

    # get settings file
    settings_default_dir = os.path.join(directory, 'input', 'settings')
    settings_default_filename = os.path.join(settings_default_dir, 'default.csv')
    settings_filename = tk.filedialog.askopenfilename(initialdir=settings_default_dir, title="Select settings file",
                                                      filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not settings_filename:
        settings_filename = settings_default_filename

    # get result folder
    results_default_dir = os.path.join(directory, 'results')
    results_foldername = tk.filedialog.askdirectory(initialdir=results_default_dir, title="Select result directory")
    if not results_foldername:
        results_foldername = results_default_dir

    return scenarios_filename, settings_filename, results_foldername


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
            self.draw_energy_system(scenario, run)

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

    def draw_energy_system(self, scenario, run):

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
                      if isinstance(block, blocks.InvestBlock) and block.opt]:
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
            run.path_result_folder,
            f'{run.runtimestamp}_{run.scenario_file_name}_{self.name}_system_graph.pdf')

        # prepare for dispatch plot saving later on
        self.plot_file_path = os.path.join(run.path_result_folder, f'{run.runtimestamp}_'
                                                                   f'{run.scenario_file_name}_'
                                                                   f'{self.name}.html')

        # prepare for cumulative result saving later on
        self.result_summary = pd.DataFrame(columns=['Block', 'Key', self.name])
        self.result_summary = self.result_summary.set_index(['Block', 'Key'])
        self.path_result_summary_tempfile = os.path.join(
            run.path_result_folder,
            f'{self.name}_tempresults.csv')

        self.result_timeseries = pd.DataFrame(index=self.dti_sim_extd)
        self.path_result_file = os.path.join(
            run.path_result_folder,
            f'{run.runtimestamp}_{run.scenario_file_name}_{self.name}_results.csv')

        self.exception = None  # placeholder for possible infeasibility

        # Energy System Blocks --------------------------------

        self.components = []  # placeholder
        self.constraints = cc.CustomConstraints(scenario=self)

        # create all block objects defined in the scenario DataFrame under "scenario/blocks" as a dict
        self.blocks = self.create_block_objects(self.blocks, run)
        self.commodity_systems = {block.name: block for block in self.blocks.values()
                                  if isinstance(block, blocks.CommoditySystem)}

        # Execute commodity system discrete event simulation
        # can only be started after all blocks have been initialized, as the different systems depend on each other.
        if any([cs.data_source == 'des' for cs in self.commodity_systems.values()]):
            des.execute_des(self, run)

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
            if [block for block in self.blocks.values() if getattr(block, 'opt', False)]:
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

        # TODO implement commodity v2mg usage share
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

        self.figure = make_subplots(specs=[[{'secondary_y': True}]])

        for block in self.blocks.values():
            block.add_power_trace(self)
            if hasattr(block, 'add_soc_trace'):  # should affect CommoditySystems and StationaryEnergyStorage
                block.add_soc_trace(self)
            if hasattr(block, 'add_curtailment_trace'):  # should affect PVSource and WindSource
                block.add_curtailment_trace(self)

        self.figure.update_layout(plot_bgcolor=col.tum_white)
        self.figure.update_xaxes(title='Local Time',
                                 showgrid=True,
                                 linecolor=col.tum_grey_20,
                                 gridcolor=col.tum_grey_20, )
        self.figure.update_yaxes(title='Power in W',
                                 showgrid=True,
                                 linecolor=col.tum_grey_20,
                                 gridcolor=col.tum_grey_20,
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
        for block in [block for block in self.blocks.values() if hasattr(block, 'opt')]:
            unit = 'kWh' if isinstance(block, (blocks.CommoditySystem, blocks.StationaryEnergyStorage)) else 'kW'
            if isinstance(block, blocks.SystemCore):
                if block.opt_acdc:
                    self.logger.info(f'Optimized size of AC/DC power in component \"{block.name}\":'
                                     f' {block.size_acdc / 1e3:.1f} {unit}')
                if block.opt_dcac:
                    self.logger.info(f'Optimized size of DC/AC power in component \"{block.name}\":'
                                     f' {block.size_dcac / 1e3:.1f} {unit}')
            elif isinstance(block, blocks.GridConnection):
                if block.opt_g2mg:
                    self.logger.info(f'Optimized size of g2mg power in component \"{block.name}\":'
                                     f' {block.size_g2mg / 1e3:.1f} {unit}')
                if block.opt_mg2g:
                    self.logger.info(f'Optimized size of mg2g power in component \"{block.name}\":'
                                     f' {block.size_mg2g / 1e3:.1f} {unit}')
                if block.peakshaving:
                    for interval in block.peakshaving_ints.index:
                        if block.peakshaving_ints.loc[interval, 'start'] <= self.dti_sim[-1]:
                            self.logger.info(f'Optimized peak power in component \"{block.name}\" for interval'
                                             f' {interval}: {block.peakshaving_ints.loc[interval, "power"] / 1e3:.2f} {unit}'
                                             f' - OPEX: {block.peakshaving_ints.loc[interval, ["period_fraction", "power", "opex_spec"]].prod():.2f} {self.currency}')

            elif isinstance(block, blocks.CommoditySystem) and block.opt:
                for commodity in block.commodities.values():
                    self.logger.info(f'Optimized size of commodity \"{commodity.name}\" in component \"{block.name}\":'
                                     f' {commodity.size / 1e3:.1f} {unit}')
            elif block.opt:
                self.logger.info(f'Optimized size of component \"{block.name}\": {block.size / 1e3:.1f} {unit}')
        # ToDo: state that these results are internal costs of minigrid only neglecting costs for external charging
        self.logger.info(f'Total simulated cost at local site: {self.totex_sim / 1e6:.2f} million {self.currency}')
        self.logger.info(f'Total simulated cost for external charging: {self.opex_sim_ext:.2f} {self.currency}')
        self.logger.info(f'Levelized cost of electricity for local site: {f"{1e5 * self.lcoe_wocs:,.2f}" if pd.notna(self.lcoe_wocs) else "-"} {self.currency}-ct/kWh')
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
                        self.result_summary.loc[(block_name, f'power_peak_{interval}'), self.name] = float(block_obj.peakshaving_ints.loc[interval, 'power'])
                        self.result_summary.loc[(block_name, f'power_period_fraction_{interval}'), self.name] = float(block_obj.peakshaving_ints.loc[interval, 'period_fraction'])
                        self.result_summary.loc[(block_name, f'power_opex_spec_{interval}'), self.name] = float(block_obj.peakshaving_ints.loc[interval, 'opex_spec'])
                        self.result_summary.loc[(block_name, f'power_opex_{interval}'), self.name] = block_obj.peakshaving_ints.loc[interval, ["period_fraction", "power", "opex_spec"]].prod()

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

        # make sure that errors are logged to logfile
        sys.excepthook = self.handle_exception

        if len(sys.argv) == 1:  # if no arguments have been passed
            self.scenarios_file_path, self.settings_file_path, self.result_path = input_gui(self.cwd)
        elif len(sys.argv) == 2:  # only one argument, default result storage
            self.scenarios_file_path = os.path.join(self.cwd, 'input', 'scenarios', sys.argv[1])
            self.settings_file_path = os.path.join(self.cwd, 'input', 'settings', 'default.csv')
            self.result_path = os.path.join(self.cwd, 'results')
        elif len(sys.argv) == 3:
            self.scenarios_file_path = os.path.join(self.cwd, 'input', 'scenarios', sys.argv[1])
            self.settings_file_path = os.path.join(self.cwd, 'input', 'settings', sys.argv[2])
            self.result_path = os.path.join(self.cwd, 'results')
        else:  # more than three inputs
            self.scenarios_file_path = os.path.join(self.cwd, 'input', 'scenarios', sys.argv[1])
            self.settings_file_path = os.path.join(self.cwd, 'input', 'settings', sys.argv[2])
            self.result_path = sys.argv[3]

        self.scenario_file_name = Path(self.scenarios_file_path).stem  # Gives file name without extension
        self.scenario_data = pd.read_csv(self.scenarios_file_path,
                                         index_col=[0, 1],
                                         keep_default_na=False)
        self.scenario_data = self.scenario_data.sort_index(sort_remaining=True).map(utils.infer_dtype)
        self.scenario_names = self.scenario_data.columns  # Get list of column names, each column is one scenario
        self.scenario_num = len(self.scenario_names)

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder
        self.runtimestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')  # create str of runtime_start

        self.commit_hash = self.get_git_commit_hash()

        settings = pd.read_csv(self.settings_file_path, index_col=[0])
        settings = settings.map(utils.infer_dtype)

        for key, value in settings['value'].items():
            setattr(self, key, value)  # this sets all the parameters defined in the settings file

        self.path_input_data = os.path.join(self.cwd, 'input')
        self.path_result_folder = os.path.join(self.result_path,
                                               f'{self.runtimestamp}_{self.scenario_file_name}')
        self.path_result_summary_file = os.path.join(self.path_result_folder,
                                                     f'{self.runtimestamp}_{self.scenario_file_name}_summary.csv')
        self.path_dump_file = os.path.join(self.path_result_folder,
                                           f'{self.runtimestamp}_{self.scenario_file_name}.lp')
        self.path_log_file = os.path.join(self.path_result_folder,
                                          f'{self.runtimestamp}_{self.scenario_file_name}.log')

        self.result_df = pd.DataFrame  # blank DataFrame for technoeconomic result saving

        if self.save_results or self.save_des_results:
            os.mkdir(self.path_result_folder)

        self.logger = logging.getLogger()
        log_formatter = logging.Formatter(f'%(levelname)-{len("WARNING")}s'
                                          f'  %(name)-{max([len(el) for el in list(self.scenario_names) + ["root"]])}s'
                                          f'  %(message)s')
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setFormatter(log_formatter)
        log_file_handler = logging.FileHandler(os.environ.get("LOGFILE", self.path_log_file))
        log_file_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_stream_handler)
        self.logger.addHandler(log_file_handler)  # TODO global messages not getting through to logs in parallel mode

        # Adding the custom filter to prevent root logger messages
        log_stream_handler.addFilter(OptimizationSuccessfulFilter())
        log_file_handler.addFilter(OptimizationSuccessfulFilter())

        # set number of processes based on specified settings and available CPUs
        self.max_process_num = os.cpu_count() if self.max_process_num == 'max' else psutil.cpu_count(
            logical=False) if self.max_process_num == 'physical' else int(self.max_process_num)
        self.process_num = min(self.scenario_num, os.cpu_count(), self.max_process_num)

        if (len(self.scenario_names) == 1 or self.process_num == 1) and self.parallel:
            self.logger.warning('Single scenario or process: Parallel mode not possible - switching to sequential mode')
            self.parallel = False

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

        self.logger.info(
            f'Global settings read - simulating {self.scenario_num} scenario{"s" if self.scenario_num > 1 else ""} '
            f'{"in parallel mode with " + str(self.process_num) + (" process" + ("es" if self.process_num > 1 else "")) if self.parallel else "in sequential mode"}'
        )

        # integration levels at which power consumption is determined a priori
        self.apriori_lvls = ['uc', 'fcfs', 'equal', 'soc']

    def end_timing(self):

        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 1)
        self.logger.info(f'Total runtime for all scenarios: {str(self.runtime_len)} s')

    @staticmethod
    def get_git_commit_hash():
        """
        Get commit hash of current HEAD. Caution: does not consider work in progress
        """
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)

            if result.returncode == 0:  # success
                return result.stdout.strip()
            else:  # error case
                return result.stderr

        except Exception as e:
            return e

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

        files = [filename for filename in os.listdir(self.path_result_folder) if filename.endswith('_tempresults.csv')]

        scenario_frames = []

        for file in files:
            file_path = os.path.join(self.path_result_folder, file)
            file_results = pd.read_csv(file_path, index_col=[0, 1], header=[0], low_memory=False)
            scenario_frames.append(file_results)

        joined_results = pd.concat(scenario_frames, axis=1)[self.scenario_names]
        joined_results.reset_index(inplace=True, names=['block', 'key'])  # necessary for saving in csv
        joined_results.to_csv(self.path_result_summary_file)
        self.logger.info("Technoeconomic output file created")

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in files:
            file_path = os.path.join(self.path_result_folder, file)
            os.remove(file_path)
