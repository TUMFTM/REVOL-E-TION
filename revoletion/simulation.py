#!/usr/bin/env python3

import geopy
import graphviz
import holidays
import importlib.metadata
import itertools
import logging
import logging.handlers
import math
import os
import pathlib
import plotly.subplots
import pprint
import psutil
import pytz
import re
import shutil
import subprocess
import sys
import threading
import time
import timezonefinder
import traceback
import warnings

import multiprocessing as mp
import numpy_financial as npf
import oemof.solph as solph
import pandas as pd
import pyomo.environ as po

from revoletion import blocks
from revoletion import checker
from revoletion import constraints
from revoletion import colors
from revoletion import dispatch
from revoletion import logger as logger_fcs
from revoletion import scheduler
from revoletion import utils


class OptimizationError(Exception):
    pass


class OptimizationSuccessfulFilter(logging.Filter):
    def filter(self, record):
        # Filter out log messages from the root logger
        return not (record.name == 'root' and record.msg == 'Optimization successful...')


class PredictionHorizon:

    def __init__(self, index, scenario):

        self.index = index
        self.scenario = scenario

        # Time and data slicing --------------------------------
        self.starttime = self.scenario.starttime + (index * self.scenario.len_ch)  # calc both start times
        self.ch_endtime = self.starttime + self.scenario.len_ch
        self.ph_endtime = self.starttime + self.scenario.len_ph
        self.timestep = self.scenario.timestep

        self.components = []  # empty list to store all oemof-solph components
        self.constraints = constraints.CustomConstraints(scenario=self.scenario)
        # add existing capex costs to constraints
        self.constraints.capex_init_existing = self.scenario.capex_init_existing

        # Display logger message if PH exceeds simulation end time and has to be truncated
        if self.ph_endtime > self.scenario.sim_endtime and self.scenario.truncate_ph:
            self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - ' +
                                      f'Prediction Horizon truncated to simulation end time')

        # Truncate PH and CH to simulation or eval end time
        self.ph_endtime = min(self.ph_endtime, self.scenario.sim_extd_endtime)
        self.ch_endtime = min(self.ch_endtime, self.scenario.sim_endtime)

        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - ' +
                                  f'Start: {self.starttime} - ' +
                                  f'CH end: {self.ch_endtime} - ' +
                                  f'PH end: {self.ph_endtime}')

        # Create datetimeindex for ph and ch; neglect last timestep as this is the first timestep of the next ph / ch
        self.dti_ph = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=self.scenario.timestep, inclusive='left')
        self.dti_ch = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=self.scenario.timestep, inclusive='left')

        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Initializing model build')

        for block in [block for block in self.scenario.blocks.values() if hasattr(block, 'data')]:
            block.data_ph = block.data[self.starttime:self.ph_endtime]
            if isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities.values():
                    commodity.data_ph = commodity.data[self.starttime:self.ph_endtime]

        # if apriori power scheduling is necessary, calculate power schedules:
        if self.scenario.scheduler:
            self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                      f'Calculating power schedules for commodities with rulebased charging strategies')
            self.scenario.scheduler.calc_ph_schedule(self)

        for block in self.scenario.blocks.values():
            block.update_input_components(self)  # (re)define solph components that need example slices

        self.results = None
        self.meta_results = None

        # Build energy system model --------------------------------

        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                   f'Building oemof model')

        self.es = solph.EnergySystem(timeindex=self.dti_ph,
                                     infer_last_interval=True)  # initialize energy system model instance

        for component in self.components:
            self.es.add(component)  # add components to this horizon's energy system

        if self.index == 0 and self.scenario.run.save_system_graphs:  # first horizon - create graph of energy system
            self.draw_energy_system()

        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Building optimization problem')

        # Build the mathematical linear optimization model with pyomo
        self.model = solph.Model(self.es, debug=self.scenario.run.debugmode)

        # Apply custom constraints
        self.constraints.apply_constraints(model=self.model)

        if self.scenario.run.dump_model and self.scenario.strategy != 'rh':
            self.model.write(self.scenario.run.path_dump_file, io_options={'symbolic_solver_labels': True})

        self.scenario.logger.debug(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                   f'Model build completed')

    def draw_energy_system(self):
        """
        This method draws a directed graph of the scenario's energy system and saves it as a pdf file
        """
        # Initialize the graph with the filepath without extension
        dot = graphviz.Digraph(filename=os.path.splitext(self.scenario.path_system_graph_file)[0])

        # Define drawing styles for certain components
        dot.node('Bus', shape='rectangle', fontsize='10', color='red')
        dot.node('Sink', shape='trapezium', fontsize='10')
        dot.node('Source', shape='invtrapezium', fontsize='10')
        dot.node('Storage', shape='rectangle', style='dashed', fontsize='10', color='green')

        busses = []
        # draw a node for each of the network's components.
        for nd in self.es.nodes:
            if isinstance(nd, solph.Bus):
                dot.node(nd.label,
                         shape='rectangle',
                         fontsize='10',
                         fixedsize='shape',
                         width='2.4',
                         height='0.6',
                         color='red')
                # keep the bus reference for drawing edges later
                busses.append(nd)
            elif isinstance(nd, solph.components.Sink):
                dot.node(nd.label, shape='trapezium', fontsize='10')
            elif isinstance(nd, solph.components.Source):
                dot.node(nd.label, shape='invtrapezium', fontsize='10')
            elif isinstance(nd, solph.components.Converter):
                dot.node(nd.label, shape='rectangle', fontsize='10')
            elif isinstance(nd, solph.components.GenericStorage):
                dot.node(nd.label, shape='rectangle', style='dashed', fontsize='10', color='green')
            else:
                self.scenario.logger.debug(f'System Node {nd.label} - Type {type(nd)} not recognized')

        # draw the edges between the nodes based on each bus inputs/outputs
        for bus in busses:
            for component in bus.inputs:
                # draw an arrow from the component to the bus
                dot.edge(component.label, bus.label)
            for component in bus.outputs:
                # draw an arrow from the bus to the component
                dot.edge(bus.label, component.label)

        try:
            dot.render(cleanup=True)
        except Exception as e:  # inhibiting failing renderer from stopping model execution
            self.scenario.logger.warning(f'System graph rendering failed - Traceback: {e}')

    def get_results(self):
        """
        Get result data slice for current CH from results and save in result dataframes for later analysis
        Get (possibly optimized) component sizes from results to handle outputs more easily
        """

        self.scenario.logger.debug(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - Getting results')

        self.results = solph.processing.results(self.model)  # Get the results of the solved horizon from the solver

        if self.scenario.run.debugmode:
            self.meta_results = solph.processing.meta_results(self.model)
            pprint.pprint(self.meta_results)

        # free up RAM
        del self.model

        # get optimum component sizes for optimized blocks
        for block in [block for block in self.scenario.blocks.values()
                      if isinstance(block, blocks.InvestBlock) and block.invest]:
            block.get_invest_size(self)

        for block in self.scenario.blocks.values():
            block.get_ch_results(self)

        for block in [block for block in self.scenario.blocks.values() if hasattr(block, 'aging')]:
            block.calc_aging(self)

    def run_optimization(self):
        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Model built, starting optimization')
        results = self.model.solve(solver=self.scenario.run.solver, solve_kwargs={'tee': self.scenario.run.debugmode})
        if (results.solver.status == po.SolverStatus.ok) and \
                (results.solver.termination_condition == po.TerminationCondition.optimal):
            self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                      f'Optimization completed, getting results')
            if self.scenario.nhorizons == 1:  # Don't store objective for multiple horizons in scenario (most RH scenarios)
                self.scenario.objective_opt = self.model.objective()
        elif results.solver.termination_condition == po.TerminationCondition.infeasible:
            raise OptimizationError(
                f'Horizon {self.index + 1} of {self.scenario.nhorizons} - Scenario failed: Infeasible')
        elif results.solver.termination_condition == po.TerminationCondition.unbounded:
            raise OptimizationError(
                f'Horizon {self.index + 1} of {self.scenario.nhorizons} - Scenario failed: Unbounded')
        elif results.solver.termination_condition == po.TerminationCondition.infeasibleOrUnbounded:
            raise OptimizationError(
                f'Horizon {self.index + 1} of {self.scenario.nhorizons} - Scenario failed: Infeasible or Unbounded '
                f'(To solve this error try to set investment limits for blocks or for the scenario)')
        else:
            raise Exception(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                            f'Optimization terminated with unknown status: {results.solver.termination_condition}')


class Scenario:

    def __init__(self, name, run, logger, lock):

        self.name = name
        self.run = run
        self.logger = logger
        self.logger.propagate = False

        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            # Force warnings in custom formatting and ignore warnings about infeasible or unbounded optimizations
            if not 'Optimization ended with status warning and termination condition' in str(message):
                logger.warning(f'{category.__name__}: {message} (in {filename}, line {lineno})')

        warnings.showwarning = custom_warning_handler

        # General Information --------------------------------

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        self.worker = mp.current_process()

        self.logger.info(f'Scenario initialized on {self.worker.name.ljust(18)}' +
                         (f' - Parent: {self.worker._parent_name}' if hasattr(self.worker, '_parent_name') else ''))

        self.parameters = self.run.scenario_data[self.name]
        for key, value in self.parameters.loc['scenario', :].items():
            setattr(self, key, value)  # this sets all the parameters defined in the csv file

        # add SystemCore to blocks ensuring SystemCore is the first component to be built
        self.blocks = {**{'core': 'SystemCore'}, **self.blocks}

        self.currency = self.currency.upper()  # all other parameters are .lower()-ed

        self.tzfinder = timezonefinder.TimezoneFinder()
        self.timezone = pytz.timezone(self.tzfinder.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        geolocator = geopy.geocoders.Nominatim(user_agent=f'location_finder')
        self.country = 'DE'  # set default country
        self.state = 'BY'  # set default state
        try:
            if lock is None:  # sequential
                location = geolocator.reverse((self.latitude, self.longitude), language="en", exactly_one=True)
            else:  # parallel
                with lock:
                    time.sleep(2)  # max 1 request per second --> wait for 2 seconds to make sure to not avoid the limit
                    location = geolocator.reverse((self.latitude, self.longitude), language="en", exactly_one=True)
            if location:
                self.country, self.state = location.raw['address']['ISO3166-2-lvl4'].split('-')
        except geopy.exc.GeocoderUnavailable:
            self.logger.warning(f'Connection to Geocoder failed.'
                                f' Using default country ({self.country}) and state ({self.state}).')

        # convert to datetime and calculate time(delta) values
        # simulation and project timeframe start simultaneously
        # simulation vs. extended simulation: for rh strategy and truncate_ph = False, the extended simulation timeframe
        # is longer than the simulation timeframe defined by the example parameter duration. Otherwise, they are the same.
        # ToDo: check for format not only len of string
        self.starttime = self.starttime if len(self.starttime) > 10 else self.starttime + ' 00:00'
        self.starttime = pd.to_datetime(self.starttime, format='%d.%m.%Y %H:%M').floor(self.timestep).tz_localize(self.timezone)

        self.sim_duration = (pd.Timedelta(days=self.sim_duration) if isinstance(self.sim_duration, (float, int))
                             else pd.Timedelta(self.sim_duration)).floor(self.timestep)
        self.sim_extd_duration = self.sim_duration
        self.sim_endtime = self.starttime + self.sim_duration
        self.sim_extd_endtime = self.sim_endtime
        self.prj_duration_yrs = self.prj_duration
        self.prj_endtime = self.starttime + pd.DateOffset(years=self.prj_duration)
        self.prj_duration = self.prj_endtime - self.starttime  # takes leap years into account

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
        # extended index covers PHs that are not truncated after simulation end time
        self.dti_sim_extd = pd.date_range(start=self.starttime, end=self.sim_extd_endtime, freq=self.timestep,
                                          inclusive='left')

        # generate variables for calculations
        self.timestep_td = pd.Timedelta(self.dti_sim_extd.freq)
        self.timestep_hours = self.timestep_td.total_seconds() / 3600
        self.sim_yr_rat = self.sim_duration / pd.Timedelta(days=365)  # no leap years
        self.sim_prj_rat = self.sim_duration / self.prj_duration

        # get holidays during simulation timeframe
        years = range(min(self.dti_sim_extd).year, max(self.dti_sim_extd).year + 1)
        try:
            self.holiday_dates = sorted(
                getattr(holidays, self.country)(years=years,
                                                state=self.state))
        except NotImplementedError:  # not for all countries the states are available (e.g. France)
            try:
                self.holiday_dates = sorted(
                    getattr(holidays, self.country)(years=years))
                self.logger.warning(f'Holidays for state {self.state} not available.'
                                    f' Country-wide holidays for {self.country} are used instead.')
            except AttributeError:  # not all countries worldwide are available
                self.holiday_dates = []
                self.logger.warning(f'Holidays for country {self.country} not available.'
                                    f' No public holidays are considered in this scenario.')

        # prepare for system graph saving later on
        self.path_system_graph_file = os.path.join(
            self.run.path_result_dir,
            f'{self.run.runtimestamp}_{self.run.scenario_file_name}_{self.name}_system_graph.pdf')

        # prepare for dispatch plot saving later on
        self.plot_file_path = os.path.join(self.run.path_result_dir, f'{run.runtimestamp}_'
                                                                     f'{run.scenario_file_name}_'
                                                                     f'{self.name}.html')

        # prepare for cumulative result saving later on
        self.result_summary = pd.DataFrame(columns=['Block', 'Key', self.name])
        self.result_summary = self.result_summary.set_index(['Block', 'Key'])
        self.path_result_summary_tempfile = os.path.join(self.run.path_result_dir,
                                                         f'{self.name}_summary_temp.csv')

        self.result_timeseries = pd.DataFrame(index=self.dti_sim_extd)
        self.path_result_file = os.path.join(
            self.run.path_result_dir,
            f'{self.run.runtimestamp}_{self.run.scenario_file_name}_{self.name}_results.csv')

        self.exception = None  # placeholder for possible infeasibility

        # Energy System Blocks --------------------------------

        # create all block objects defined in the scenario DataFrame under "scenario/blocks" as a dict
        self.blocks = self.create_block_objects()
        self.commodity_systems = {block.name: block for block in self.blocks.values()
                                  if isinstance(block, blocks.CommoditySystem)}

        # initialize variable to store initial investment costs given in scenario definition
        self.capex_init_existing = 0
        # get all initial capex costs for existing components
        for block in self.blocks.values():
            block.calc_capex_init_existing()
        if self.invest_max is not None and self.invest_max < self.capex_init_existing:
            self.logger.error(f'Initial investment costs of {self.capex_init_existing:.2f} {self.currency}'
                              f' exceed maximum investment limit of {self.invest_max} {self.currency}')
            raise ValueError(f'Initial investment costs of {self.capex_init_existing:.2f} {self.currency}'
                              f' exceed maximum investment limit of {self.invest_max} {self.currency}')

        # Execute commodity system discrete event simulation
        # can only be started after all blocks have been initialized, as the different systems depend on each other.
        if any([cs.data_source in ['demand', 'usecases'] for cs in self.commodity_systems.values()]):
            dispatch.execute_des(self, self.run)

        for cs in [cs for cs in self.commodity_systems.values() if cs.data_source in ['usecases', 'demand']]:
            for commodity in cs.commodities.values():
                commodity.data = cs.data.loc[:, (commodity.name, slice(None))].droplevel(0, axis=1)

        # ToDo: move to checker.py
        # check example parameter configuration for model dump
        if self.run.dump_model and self.strategy == 'rh':
            self.logger.warning('Model file dump not implemented for RH operating strategy - ' +
                                'File dump deactivated for current scenario')

        # check example parameter configuration of rulebased charging for validity
        if cs_unlim := [cs for cs in self.commodity_systems.values() if
                        (cs.mode_scheduling in self.run.apriori_lvls)
                        and cs.mode_scheduling != 'uc'
                        and not cs.power_lim_static]:
            if [block for block in self.blocks.values() if getattr(block, 'invest', False)]:
                raise ValueError(f'Rulebased charging except for uncoordinated charging (uc) '
                                 f'without static load management (lm_static) is not compatible'
                                 f' with size optimization')
            if [block for block in self.blocks.values() if isinstance(block, blocks.StationaryEnergyStorage)]:
                raise ValueError(f'Rulebased charging except for uncoordinated charging (uc) '
                                 f'without static load management (lm_static) is not implemented for systems with '
                                 f'stationary energy storage')
            if len(set([cs.mode_scheduling for cs in cs_unlim])) > 1:
                raise ValueError(f'All rulebased CommoditySystems with dynamic load management '
                                 f'have to follow the same strategy. Different strategies are not possible')
            if cs_unlim[0].mode_scheduling == 'equal' and len(set([cs.bus_connected for cs in cs_unlim])) > 1:
                raise ValueError(f'If strategy "equal" is chosen for CommoditySystems with'
                                 f' dynamic load management, all CommoditySystems with dynamic load management have to'
                                 f' be connected to the same bus')

        self.scheduler = None
        if any([cs for cs in self.commodity_systems.values() if cs.mode_scheduling in self.run.apriori_lvls]):
            self.scheduler = scheduler.AprioriPowerScheduler(scenario=self)

        # Result variables --------------------------------
        self.figure = None  # placeholder for plotting

        self.cashflows = pd.DataFrame()

        self.objective_opt = None  # placeholder for objective optimised by the optimizer. Not used for Rolling Horizon

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
                         f' LCOE {f"{self.lcoe_wocs * 1e5:,.2f}" if pd.notna(self.lcoe_wocs) else "-"} {self.currency}-ct/kWh -'
                         f' mIRR {f"{self.mirr * 100:,.2f}" if pd.notna(self.mirr) else "-"} % -'
                         f' Renewable Share:'
                         f' {f"{self.renewable_share * 100:.1f}" if pd.notna(self.renewable_share) else "-"} % -'
                         f' Renewable Curtailment:'
                         f' {f"{self.renewable_curtailment * 100:.1f}" if pd.notna(self.renewable_curtailment) else "-"} %')

    def create_block_objects(self):
        class_dict = self.blocks
        objects = {}
        for name, class_name in class_dict.items():
            class_obj = getattr(blocks, class_name, None)
            if class_obj is not None and isinstance(class_obj, type):
                objects[name] = class_obj(name, self)
            else:
                raise ValueError(f'Class "{class_name}" not found in blocks.py file - '
                                 f'Check for typos or add class.')
        return objects

    def end_timing(self):
        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 2)
        self.logger.info(f'Scenario finished - runtime {self.runtime_len} s')

    def generate_plots(self):

        self.figure = plotly.subplots.make_subplots(specs=[[{'secondary_y': True}]])

        for block in self.blocks.values():
            block.add_power_trace()
            if hasattr(block, 'add_soc_trace'):  # should affect CommoditySystems and StationaryEnergyStorage
                block.add_soc_trace()
            if hasattr(block, 'add_curtailment_trace'):  # should affect PVSource and WindSource
                block.add_curtailment_trace()

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
            block.calc_energy()
            block.calc_expenses()
            block.calc_revenue()
            block.calc_cashflows()

    def print_results(self):
        for block in [block for block in self.blocks.values() if isinstance(block, blocks.InvestBlock)]:
            unit = 'kWh' if isinstance(block, (blocks.CommoditySystem, blocks.StationaryEnergyStorage)) else 'kW'
            if isinstance(block, blocks.SystemCore) and block.invest:
                self.logger.info(f'Optimized size of AC/DC power in component "{block.name}":'
                                 f' {block.size_acdc / 1e3:.1f} {unit}'
                                 f' (existing: {block.size_acdc_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_acdc_additional / 1e3:.1f} {unit})')
                self.logger.info(f'Optimized size of DC/AC power in component "{block.name}":'
                                 f' {block.size_dcac / 1e3:.1f} {unit}'
                                 f' (existing: {block.size_dcac_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_dcac_additional / 1e3:.1f} {unit})')
            elif isinstance(block, blocks.GridConnection):
                if block.invest:
                    self.logger.info(f'Optimized size of g2s power in component "{block.name}":'
                                     f' {block.size_g2s / 1e3:.1f} {unit}' + \
                                     f' (existing: {block.size_g2s_existing / 1e3:.1f} {unit}'
                                     f' - additional: {block.size_g2s_additional / 1e3:.1f} {unit})')
                    self.logger.info(f'Optimized size of s2g power in component "{block.name}":'
                                     f' {block.size_s2g / 1e3:.1f} {unit}'
                                     f' (existing: {block.size_s2g_existing / 1e3:.1f} {unit}'
                                     f' - additional: {block.size_s2g_additional / 1e3:.1f} {unit})')
                if block.peakshaving:
                    for interval in block.peakshaving_ints.index:
                        if block.peakshaving_ints.loc[interval, 'start'] <= self.dti_sim[-1]:
                            self.logger.info(f'Optimized peak power in component "{block.name}" for interval'
                                             f' {interval}: {block.peakshaving_ints.loc[interval, "power"] / 1e3:.1f} {unit}'
                                             f' - OPEX: {block.opex_spec_peak * block.peakshaving_ints.loc[interval, ["period_fraction", "power"]].prod():.2f} {self.currency}')

            elif isinstance(block, blocks.CommoditySystem) and block.invest:
                for commodity in block.commodities.values():
                    self.logger.info(f'Optimized size of commodity "{commodity.name}" in component "{block.name}":'
                                     f' {commodity.size / 1e3:.1f} {unit}'
                                     f' (existing: {commodity.size_existing / 1e3:.1f} {unit}'
                                     f' - additional: {commodity.size_additional / 1e3:.1f} {unit})')
            elif hasattr(block, 'invest') and block.invest:
                self.logger.info(f'Optimized size of component "{block.name}": {block.size / 1e3:.1f} {unit}'
                                 f' (existing: {block.size_existing / 1e3:.1f} {unit}'
                                 f' - additional: {block.size_additional / 1e3:.1f} {unit})')

    def save_plots(self):
        self.figure.write_html(self.plot_file_path)

    def save_result_summary(self):
        """
        Saves all int, float and str attributes of run, scenario (incl. technoeconomic KPIs) and all blocks to the
        results dataframe
        :return: none
        """

        def write_values(name, block):
            keys = [key for key in block.__dict__.keys()
                    if (isinstance(block.__dict__[key], result_types)) or (name, key) == ('scenario', 'blocks')]

            for key in keys:
                value = block.__dict__[key]
                if isinstance(value, int):
                    self.result_summary.loc[(name, key), self.name] = float(value)
                elif (name, key) == ('scenario', 'blocks'):
                    # blocks dict contains objects, but summary shall contain class names of the blocks
                    self.result_summary.loc[(name, key), self.name] = str({block: value[block].__class__.__name__
                                                                           for block in value.keys()})
                else:
                    self.result_summary.loc[(name, key), self.name] = value

        result_types = (int, float, str, bool, type(None))
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
                        self.result_summary.loc[(block_name, f'power_opex_spec_{interval}'), self.name] = (
                            float(block_obj.peakshaving_ints.loc[interval, 'opex_spec']))
                        self.result_summary.loc[(block_name, f'power_opex_{interval}'), self.name] = (
                            block_obj.peakshaving_ints.loc[interval, ['period_fraction', 'power', 'opex_spec']].prod())

        self.result_summary.to_csv(self.path_result_summary_tempfile, index=True)

    def save_result_timeseries(self):
        for block in self.blocks.values():
            block.get_timeseries_results()
        self.result_timeseries.to_csv(self.path_result_file)

    def show_plots(self):
        self.figure.show(renderer='browser')


class SimulationRun:

    def __init__(self, path_scenarios, path_settings, rerun=False, rerun_infeasible=True, execute=False):

        self.name = 'run'
        self.path_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.process = self.process_num = None

        self.scenarios_file_path = path_scenarios
        self.settings_file_path = path_settings
        self.rerun = rerun
        self.rerun_infeasible = rerun_infeasible

        self.runtime_start = time.perf_counter()
        if not self.rerun:
            self.runtimestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
        else:
            self.runtimestamp = '_'.join(os.path.basename(os.path.normpath(self.rerun)).split('_')[0:2])
        self.runtime_end = self.runtime_len = None

        self.version_solph = solph.__version__
        self.version_revoletion = importlib.metadata.version('revoletion')

        try:
            self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()[0:6]
        except subprocess.CalledProcessError:
            self.commit_hash = 'unknown'

        self.settings = pd.read_csv(self.settings_file_path, index_col=[0])
        self.settings = self.settings.map(utils.infer_dtype)
        for key, value in self.settings['value'].items():
            setattr(self, key, value)  # this sets all the parameters defined in the settings file

        self.scenario_file_name = pathlib.Path(self.scenarios_file_path).stem  # file name without extension
        self.define_paths()
        self.scenario_data = pd.read_csv((self.path_result_scenario_file if self.rerun else self.scenarios_file_path),
                                         index_col=[0, 1],
                                         keep_default_na=False)
        self.scenario_data = self.scenario_data.sort_index(sort_remaining=True).map(utils.infer_dtype)
        self.scenario_names = [name for name in self.scenario_data.columns if not name.startswith('#')]  # Get list of column names, each column is one scenario


        if self.rerun:
            # only run scenarios which have not been optimized successfully (or were infeasible)
            self.scenario_status = pd.read_csv(os.path.join(self.path_result_dir,
                                                            f'{self.runtimestamp}_'
                                                            f'{self.scenario_file_name}_'
                                                            f'scenarios_status.csv'),
                                               index_col=0)

            dont_rerun = ['successful', 'infeasible'] if self.rerun_infeasible else ['successful']
            scenarios_failed = self.scenario_status[~self.scenario_status['status'].isin(dont_rerun)].index.to_list()
            self.scenario_names = [name for name in self.scenario_names if name in scenarios_failed]

            # delete all temporary results of files which are rerun (happens if SimulationRun terminates unexpected)
            for scenario in self.scenario_names:
                for file in [f'{scenario}_summary_temp.csv',
                             f'{scenario}_results.csv',
                             f'{scenario}_graph.pdf',
                             f'{scenario}_graph.html']:
                    if os.path.isfile(os.path.join(self.path_result_dir, file)):
                        os.remove(os.path.join(self.path_result_dir, file))

            # reset status of scenarios to be run to 'queued'
            self.scenario_status.loc[self.scenario_names, ['status', 'exception', 'traceback']] = (
                    [['queued', pd.NA, pd.NA]] * len(self.scenario_names))
            self.scenario_status.to_csv(self.path_result_status_file, index=True)

        else:
            self.scenario_status = pd.DataFrame(index=self.scenario_names,
                                                data={'status': 'queued',
                                                      'exception': None,
                                                      'traceback': None}).rename_axis('scenario')
        self.scenario_num = len(self.scenario_names)

        self.input_checker = checker.InputChecker(self)
        self.input_checker.check_settings()

        self.get_process_num()
        if not self.rerun:
            self.copy_scenario_file()

        self.logger = None
        self.define_logger()

        # integration levels at which power consumption is determined a priori
        self.result_df = pd.DataFrame  # blank DataFrame for technoeconomic result saving
        self.apriori_lvls = ['uc', 'fcfs', 'equal', 'soc']

        if execute:
            self.execute_simulation()

    def copy_scenario_file(self):
        try:
            shutil.copy2(self.scenarios_file_path, self.path_result_scenario_file)  # writes metadata
        except PermissionError:  # can happen if metadata is not writable, e.g. on network drives
            shutil.copyfile(self.scenarios_file_path, self.path_result_scenario_file)  # does not write metadata

    def define_logger(self):

        self.logger = logging.getLogger()
        log_formatter = logging.Formatter(f'%(levelname)-{len("WARNING")}s'
                                          f'  %(name)-{max([len(el) for el in list(self.scenario_names) + ["root"]])}s'
                                          f'  %(message)s')
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setFormatter(log_formatter)
        log_file_handler = logging.FileHandler(os.environ.get('LOGFILE', self.path_log_file))
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
                self.logger.setLevel(os.environ.get('LOGLEVEL', 'DEBUG'))
                log_stream_handler.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))
                log_stream_handler.setLevel(logging.INFO)

        # plural extensions
        pe1 = 's' if self.scenario_num > 1 else ''
        pe2 = 'es' if self.process_num > 1 else ''

        mode = f'parallel mode with {self.process_num} process{pe2}' if self.parallel else 'sequential mode'
        self.logger.info(f'Global settings read - running {self.scenario_num} scenario{pe1} in {mode}')

        # make sure that uncaught errors (i.e. errors occurring outside simulate_scenario method) are logged to logfile
        sys.excepthook = self.handle_exception

    def define_paths(self):

        if self.path_input_data == 'example':
            self.path_input_data = os.path.join(self.path_pkg, 'example')
        elif self.path_input_data == 'scenario_file_dir':
            self.path_input_data = os.path.dirname(self.scenarios_file_path)
        elif os.path.isdir(self.path_input_data):
            pass  # no modification of path necessary
        else:
            raise NotADirectoryError(f'Input directory not found: {self.path_input_data}')

        if self.path_output_data == 'package':
            self.path_output_data = os.path.join(self.path_pkg, 'results')
        elif os.path.isdir(self.path_output_data):
            pass  # no modification of path necessary
        else:
            raise NotADirectoryError(f'Output directory not found: {self.path_output_data}')

        self.path_data_immut = os.path.join(self.path_pkg, 'data')

        self.path_result_dir = os.path.join(self.path_output_data,
                                            f'{self.runtimestamp}_{self.scenario_file_name}')

        if not os.path.isdir(self.path_result_dir):
            os.mkdir(self.path_result_dir)

        self.path_result_scenario_file = os.path.join(self.path_result_dir,
                                                      f'{self.runtimestamp}_{self.scenario_file_name}_scenarios.csv')
        self.path_result_summary_file = os.path.join(self.path_result_dir,
                                                     f'{self.runtimestamp}_{self.scenario_file_name}_summary.csv')
        self.path_result_status_file = os.path.join(self.path_result_dir,
                                                    f'{self.runtimestamp}_{self.scenario_file_name}_scenarios_status.csv')
        self.path_dump_file = os.path.join(self.path_result_dir, f'{self.runtimestamp}_{self.scenario_file_name}.lp')
        self.path_log_file = os.path.join(self.path_result_dir, f'{self.runtimestamp}_{self.scenario_file_name}.log')

    def end_timing(self):

        self.runtime_end = time.perf_counter()
        self.runtime_len = self.runtime_end - self.runtime_start
        self.logger.info(f'Total runtime for all scenarios: {self.runtime_len:.1f} s')

    def execute_simulation(self):
        # parallelization activated in settings file
        if self.parallel:
            with mp.Manager() as manager:
                lock = manager.Lock()

                status_queue = manager.Queue()
                status_thread = threading.Thread(target=self.read_status_queue, args=(status_queue,))
                status_thread.start()

                log_queue = manager.Queue()
                log_thread = threading.Thread(target=logger_fcs.read_mplogger_queue, args=(log_queue,))
                log_thread.start()

                with mp.Pool(processes=self.process_num) as pool:
                    pool.starmap(self.simulate_scenario,
                                 zip(self.scenario_names,
                                     itertools.repeat(log_queue),
                                     itertools.repeat(status_queue),
                                     itertools.repeat(lock)))
                status_queue.put(None)
                status_thread.join()
                log_queue.put(None)
                log_thread.join()
        else:
            for scenario_name in self.scenario_names:
                self.simulate_scenario(scenario_name)

        self.end_timing()

        if self.save_results:
            self.join_results()

    def get_process_num(self):
        if self.max_process_num == 'max':
            self.max_process_num = os.cpu_count()
        elif self.max_process_num == 'physical':
            self.max_process_num = psutil.cpu_count(logical=False)
        else:
            self.max_process_num = int(self.max_process_num)
        self.process_num = min(self.scenario_num, os.cpu_count(), self.max_process_num)

        if (len(self.scenario_names) <= 1 or self.process_num == 1) and self.parallel:
            # logger not defined yet, use print as logger definition needs to be done after process_num is defined
            print('Single scenario or process: Parallel mode not possible - switching to sequential mode')
            self.parallel = False

        if len(self.scenario_names) <= 0:
            raise ValueError('No executable scenarios found in scenario file')

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error(f'Exception type: {exc_type.__name__}')
        self.logger.error(f'Exception message: {str(exc_value)}')
        self.logger.error('Traceback:')
        self.logger.error(''.join(traceback.format_tb(exc_traceback)))

        self.logger.error('Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback))

    def join_results(self):

        files = [filename for filename in os.listdir(self.path_result_dir) if filename.endswith('_summary_temp.csv')]

        scenario_frames = []

        for file in files:
            file_path = os.path.join(self.path_result_dir, file)
            file_results = pd.read_csv(file_path, index_col=[0, 1], header=[0], low_memory=False)
            scenario_frames.append(file_results)

        if len(scenario_frames) > 0:  # empty scenario_frames, if all scenarios fail during initialization
            joined_results = pd.concat(scenario_frames, axis=1)
            joined_results.loc[('run', 'runtime_end'), :] = self.runtime_end
            joined_results.loc[('run', 'runtime_len'), :] = self.runtime_len
            if self.rerun and os.path.isfile(os.path.join(self.path_result_summary_file)):
                results_summary_prev = pd.read_csv(os.path.join(self.path_result_summary_file),
                                                   index_col=[0, 1])
                joined_results = pd.concat([results_summary_prev, joined_results], axis=1)
            # apply same order of scenarios as in scenario input file
            joined_results = joined_results[[col for col in self.scenario_data.columns if col in joined_results.columns]]
            joined_results.to_csv(self.path_result_summary_file, index=True)
            self.logger.info('Technoeconomic output file created')

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in files:
            file_path = os.path.join(self.path_result_dir, file)
            os.remove(file_path)

    def read_status_queue(self, queue):
        while True:
            status_msg = queue.get()
            if status_msg is None:  # Exit signal
                break
            self.update_scenario_status(status_msg)

    def simulate_scenario(self, name: str, log_queue=None, status_queue=None, lock=None):
        logger = logger_fcs.setup_logger(name, log_queue, self)

        self.process = mp.current_process() if self.parallel else None

        self.trigger_scenario_status_update(queue=status_queue,
                                            status_msg={'scenario': name,
                                                        'status': 'started'})

        scenario = None

        try:
            scenario = Scenario(name, self, logger, lock)  # Create scenario instance

            self.trigger_scenario_status_update(queue=status_queue,
                                                status_msg={'scenario': name,
                                                            'status': 'fully initialized'})

            # ToDo: move to checker.py
            if scenario.strategy not in ['go', 'rh']:
                scenario.exception = f'Optimization strategy "{scenario.strategy}" unknown'
                raise ValueError(scenario.exception)

            for horizon_index in range(scenario.nhorizons):  # Inner optimization loop over all prediction horizons
                horizon = PredictionHorizon(horizon_index, scenario)
                horizon.run_optimization()
                horizon.get_results()
                self.trigger_scenario_status_update(queue=status_queue,
                                                    status_msg={'scenario': name,
                                                                'status': f'completed horizon {horizon_index + 1} out of'
                                                                          f' {scenario.nhorizons}'})

            self.trigger_scenario_status_update(queue=status_queue,
                                                status_msg={'scenario': name,
                                                            'status': 'successful'})

            scenario.end_timing()

        except Exception as e:
            # Scenario has failed -> store scenario name to dataframe containing failed scenarios
            status = 'infeasible' if isinstance(e, OptimizationError) else 'failed'
            self.trigger_scenario_status_update(queue=status_queue,
                                                status_msg={'scenario': name,
                                                            'status': status,
                                                            'exception': str(e),
                                                            'traceback': traceback.format_exc()})

            # show error message and traceback in console; suppress traceback if problem was infeasible or unbounded
            logger.error(msg=f'{str(e)} - continue on next scenario', exc_info=(not isinstance(e, OptimizationError)))

            if scenario is not None:  # scenario initialization can fail
                scenario.exception = str(e)
                scenario.end_timing()

        finally:
            try:
                # Process all results
                if self.save_results or self.print_results:
                    scenario.get_results()
                    scenario.calc_meta_results()
                    if self.save_results:
                        scenario.save_result_summary()
                        scenario.save_result_timeseries()
                    if self.print_results:
                        scenario.print_results()
                if self.save_plots or self.show_plots:
                    scenario.generate_plots()
                    if self.save_plots:
                        scenario.save_plots()
                    if self.show_plots:
                        scenario.show_plots()
            except Exception as e:
                logger.error(e, exc_info=True)

            logging.shutdown()

    def trigger_scenario_status_update(self, queue, status_msg):
        if queue is not None:
            queue.put(status_msg)
        else:
            self.update_scenario_status(status_msg)

    def update_scenario_status(self, status_msg):
        for col in [key for key, value in status_msg.items() if key != 'scenario' and value is not None]:
            self.scenario_status.loc[status_msg['scenario'], col] = status_msg[col]
        self.scenario_status.to_csv(self.path_result_status_file,
                                    index=True)
