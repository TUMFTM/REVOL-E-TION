#!/usr/bin/env python3
"""
simulation.py

--- Description ---
This script provides the main simulation procedure classes for the oemof mg_ev toolset.
For further information, see readme.md

--- Created by ---
Philipp Rosner

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Imports
###############################################################################

import json
import logging
import logging.handlers
import math
import os
import pprint
import sys
import time

import multiprocessing as mp
import oemof.solph as solph
import pandas as pd
import plotly.graph_objects as go
import PySimpleGUI as psg

from datetime import datetime, timedelta
from pathlib import Path
from plotly.subplots import make_subplots

import blocks
import commodity_des as des
import tum_colors as col
from aprioripowerscheduler import AprioriPowerScheduler

###############################################################################
# Class definitions
###############################################################################


class PredictionHorizon:

    def __init__(self, index, scenario, run):

        self.index = index

        run.logger.info(f'Scenario \"{scenario.name}\" - Horizon {self.index + 1} of {scenario.nhorizons}:'
                        f' Building linear optimization model')

        # Time and data slicing --------------------------------
        self.starttime = scenario.starttime + (index * scenario.ch_len)  # calc both start times
        self.ch_endtime = self.starttime + scenario.ch_len
        self.ph_endtime = self.starttime + scenario.ph_len
        self.timestep = scenario.timestep

        if self.ph_endtime > scenario.sim_endtime:
            self.ph_endtime = scenario.sim_endtime

        # last steps need to be deleted as time index relates to beginning of timestep
        self.ph_dti = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=scenario.timestep).delete(-1)
        self.ch_dti = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=scenario.timestep).delete(-1)

        for block in [block for block in scenario.blocks.values() if hasattr(block, 'data')]:
            block.ph_data = block.data[self.starttime:self.ph_endtime]
            if isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities.values():
                    commodity.ph_data = commodity.data[self.starttime:self.ph_endtime]

        # if apriori power scheduling is necessary, calculate power schedules:
        if scenario.scheduler:
            scenario.scheduler.calc_schedule(self.ph_dti)

        for block in scenario.blocks.values():
            block.update_input_components(scenario)  # (re)define solph components that need input slices

        self.results = None
        self.meta_results = None

        # Build energy system model --------------------------------

        logging.debug(f'Horizon {self.index + 1} of {scenario.nhorizons}: building energy system instance')

        self.es = solph.EnergySystem(timeindex=self.ph_dti,
                                     infer_last_interval=True)  # initialize energy system model instance

        for component in scenario.components:
            self.es.add(component)  # add components to this horizon's energy system

        run.logger.debug(f'Horizon {self.index + 1} of {scenario.nhorizons}: creating optimization model')

        self.model = solph.Model(self.es, debug=run.debugmode)  # Build the mathematical linear optimization model with pyomo

        if run.dump_model:
            if scenario.strategy == 'go':
                self.model.write(run.dump_file_path, io_options={'symbolic_solver_labels': True})
            elif scenario.strategy == 'rh':
                run.logger.warning('Model file dump not implemented for RH operating strategy - no file created')

        run.logger.debug(f'Horizon {self.index + 1} of {scenario.nhorizons}: model build completed')

    def get_results(self, scenario, run):
        """
        Get result data slice for current CH from results and save in result dataframes for later analysis
        Get (possibly optimized) component sizes from results to handle outputs more easily
        """

        run.logger.debug(f'Horizon {self.index} of {scenario.nhorizons}: getting results')

        self.results = solph.processing.results(self.model)  # Get the results of the solved horizon from the solver

        if run.print_results:
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
                block.calc_aging(scenario, self)

    def run_optimization(self, scenario, run):
        run.logger.info(f'Scenario \"{scenario.name}\" - Horizon {self.index + 1} of {scenario.nhorizons}:'
                        f' Model built, starting optimization')
        try:
            self.model.solve(solver=run.solver, solve_kwargs={'tee': run.debugmode})
        except UserWarning as exc:
            run.logger.warning(f'Scenario \"{scenario.name}\" failed or infeasible - continue on next scenario')
            scenario.exception = str(exc)
            # TODO does not jump to next scenario properly (at least in parallel mode)

    def run_lfs(self, scenario, run):
        # TODO implement load following rule based dispatch strategy
        pass

    def run_ccs(self, scenario, run):
        # TODO implement cycle charging rule based dispatch strategy
        pass


class Scenario:

    def __init__(self, scenario_name, run):

        self.name = scenario_name
        self.run = run

        # General Information --------------------------------

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        self.worker = mp.current_process()

        run.logger.info(f'Scenario \"{self.name}\" initialized on {self.worker}')

        self.parameters = run.scenario_data[self.name]
        for key, value in self.parameters.loc['scenario', :].items():
            setattr(self, key, value)  # this sets all the parameters defined in the json file

        curr_dict = {'eur': 'EUR',
                     'usd': 'USD',}
        if self.currency in curr_dict.keys():
            self.currency = curr_dict[self.currency]

        # convert to datetime and calculate time(delta) values
        self.starttime = datetime.strptime(self.starttime,
                                           '%d/%m/%Y')  # simulation and project timeframe start simultaneously
        self.sim_duration = timedelta(days=self.sim_duration)
        self.sim_endtime = self.starttime + self.sim_duration
        self.prj_duration = timedelta(days=self.prj_duration * 365)  # no leap years accounted for
        self.prj_duration_yrs = self.prj_duration.days / 365
        self.prj_endtime = self.starttime + self.prj_duration

        # generate a datetimeindex for the energy system model to run on
        self.sim_dti = pd.date_range(start=self.starttime, end=self.sim_endtime, freq=self.timestep).delete(-1)

        # generate variables for calculations
        self.timestep_hours = self.sim_dti.freq.nanos / 1e9 / 3600
        self.timestep_td = pd.Timedelta(hours=self.timestep_hours)
        self.sim_yr_rat = self.sim_duration.days / 365  # no leap years
        self.sim_prj_rat = self.sim_duration.days / self.prj_duration.days

        # prepare for dispatch plot saving later on
        self.plot_file_path = os.path.join(run.result_folder_path, f'{run.runtimestamp}_'
                                                                   f'{run.scenario_file_name}_'
                                                                   f'{self.name}.html')

        # prepare for cumulative result saving later on
        self.results = pd.DataFrame(columns=['Block', 'Key', self.name])
        self.results = self.results.set_index(['Block', 'Key'])
        self.result_file_path = os.path.join(run.result_folder_path, f'{self.name}.json')

        self.exception = None  # placeholder for possible infeasibility

        if self.strategy == 'rh':
            self.ph_len = timedelta(hours=self.ph_len)
            self.ch_len = timedelta(hours=self.ch_len)
            # number of timesteps for PH
            self.ph_nsteps = math.ceil(self.ph_len.total_seconds() / 3600 / self.timestep_hours)
            # number of timesteps for CH
            self.ch_nsteps = math.ceil(self.ch_len.total_seconds() / 3600 / self.timestep_hours)
            self.nhorizons = int(self.sim_duration // self.ch_len)  # number of timeslices to run
        elif self.strategy in ['go', 'lfs']:
            self.ph_len = self.sim_duration
            self.ch_len = self.sim_duration
            self.nhorizons = 1

        # Energy System Blocks --------------------------------

        self.components = []  # placeholder

        # create all block objects defined in the scenario DataFrame under "scenario/blocks" as a dict
        self.blocks = self.create_block_objects(self.blocks, run)
        self.commodity_systems = {block.name: block for block in self.blocks.values() if isinstance(block, blocks.CommoditySystem)}

        # Execute commodity system discrete event simulation
        # can only be started after all blocks have been initialized, as the different systems depend on each other.
        if any([cs.filename == 'run_des' for cs in self.commodity_systems.values()]):
            des.execute_des(self, run.save_des_results, run.result_folder_path)

        for cs in [cs for cs in self.commodity_systems.values() if cs.filename == 'run_des']:
            for commodity in cs.commodities.values():
                commodity.data = cs.data.loc[:, (commodity.name, slice(None))].droplevel(0, axis=1)

        # ToDo: remove section; old code!
        # for cs in [cs for cs in self.commodity_systems.values() if cs.int_lvl in cs.apriori_lvls]:
        #     for commodity in cs.commodities.values():
        #         commodity.calc_uc_power(self)

        # ToDo:
        self.scheduler = None
        if any([cs for cs in self.commodity_systems.values() if cs.int_lvl in cs.apriori_lvls]):
            self.scheduler = AprioriPowerScheduler(scenario=self)

        # Result variables --------------------------------
        self.figure = None  # placeholder for plotting

        # Result variables - Energy
        self.e_sim_del = self.e_yrl_del = self.e_prj_del = self.e_dis_del = 0
        self.e_sim_pro = self.e_yrl_pro = self.e_prj_pro = self.e_dis_pro = 0
        self.e_sim_ext = self.e_yrl_ext = self.e_prj_ext = self.e_dis_ext = 0  # external charging
        self.e_eta = None

        self.renewable_curtailment = None

        # Result variables - Cost
        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = 0
        self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = 0
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = 0
        self.opex_sim_ext = self.opex_yrl_ext = self.opex_prj_ext = self.opex_dis_ext = self.opex_ann_ext = 0
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = 0
        self.lcoe = self.lcoe_dis = None

        run.logger.debug(f'Scenario {self.name} initialization completed')

    def accumulate_results(self, run):

        for block in self.blocks.values():
            block.calc_results(self)

        # optional metrics
        # TODO implement renewable energy share evaluation
        # TODO implement commodity v2mg usage share
        # TODO implement energy storage usage share
        # TODO implement SAIDI (Average interruption time)
        # TODO implement SAIFI (Interruption frequency per customer per year)

        try:
            self.e_eta = self.e_sim_del / self.e_sim_pro
        except (ZeroDivisionError, RuntimeWarning):
            run.logger.warning(f'Scenario {self.name} - total efficiency calculation: division by zero')

        try:
            self.lcoe = self.totex_dis / self.e_prj_del
            self.lcoe_dis = self.totex_dis / self.e_dis_del
        except (ZeroDivisionError, RuntimeWarning):
            self.lcoe = self.lcoe_dis = -1e-5  # prevent errors in further calculations and force end result to -1
            run.logger.warning(f'Scenario {self.name} - LCOE calculation: division by zero')

        re_blx = [block for block in self.blocks.values() if isinstance(block, (blocks.PVSource, blocks.WindSource))]
        e_pot = 0
        e_curt = 0
        for block in re_blx:
            block.curtailment = sum(block.e_curt) / sum(block.e_pot) if sum(block.e_pot) > 0 else 0
            e_pot += sum(block.e_pot)
            e_curt += sum(block.e_curt)
        self.renewable_curtailment = e_curt / e_pot if e_pot > 0 else 0

        lcoe_display = round(self.lcoe_dis * 1e5, 1)
        npc_display = round(self.totex_dis)
        npc_display_ext = round(self.opex_dis_ext)
        e_display_ext = round(self.e_dis_ext * 1e-3, 1)
        run.logger.info(f'Scenario \"{self.name}\" - NPC {npc_display} {self.currency} - LCOE {lcoe_display} {self.currency}-ct/kWh')
        run.logger.info(f'Scenario \"{self.name}\" - NPC external charging {npc_display_ext} {self.currency} - '
                        f'External charged energy: {e_display_ext} kWh')
        run.logger.info(f'Scenario \"{self.name}\" - external charging {self.opex_sim_ext} {self.currency} - '
                        f'External charged energy: {self.e_sim_ext * 1e-3} kWh')

    def create_block_objects(self, class_dict, run):
        # todo implement anti-infeasibility controllable source? (unlimited size, very high soe, no sce, no sme)
        objects = {}
        for name, class_name in class_dict.items():
            class_obj = getattr(blocks, class_name, None)
            if class_obj is not None and isinstance(class_obj, type):
                objects[name] = class_obj(name, self, run)
            else:
                raise ValueError(f"Class '{class_name}' not found in blocks.py file - Check for typos or add class.")
        return objects

    def end_timing(self, run):
        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 2)
        run.logger.info(f'Scenario \"{self.name}\" finished - runtime {self.runtime_len} s')

    def generate_plots(self, run):

        self.figure = make_subplots(specs=[[{'secondary_y': True}]])

        # types for which positive flow values are power taken out of the core
        invert_types = (blocks.FixedDemand, blocks.StationaryEnergyStorage, blocks.CommoditySystem, blocks.GridConnection)

        for block in [block for block in self.blocks.values() if not isinstance(block, blocks.SystemCore)]:

            if hasattr(block, 'size'):
                if isinstance(block, blocks.CommoditySystem):
                    legentry_p = f"{block.name} total power"
                    display_size = round(block.size / 1e3, 1)
                if isinstance(block, blocks.StationaryEnergyStorage):
                    display_size = round(block.size / 1e3, 1)
                    display_dpwr = round(block.dis_crate * block.size / 1e3, 1)
                    legentry_p = f"{block.name} power ({display_dpwr} kW)"
                else:
                    display_size = round(block.size / 1e3, 1)
                    legentry_p = f"{block.name} power ({display_size} kW)"
            else:
                legentry_p = f"{block.name} power"

            invert_power = isinstance(block, invert_types)  # TODO show as stacked plot
            self.figure.add_trace(go.Scatter(x=block.flow.index,
                                             y=block.flow * {True: -1,
                                                             False: 1}[invert_power],
                                             mode='lines',
                                             name=legentry_p,
                                             line=dict(width=2, dash=None)),  # TODO introduce TUM colors
                                  secondary_y=False)

            if isinstance(block, blocks.StationaryEnergyStorage):
                legentry_soc = f"{block.name} SOC ({display_size} kWh)"
                self.figure.add_trace(go.Scatter(x=block.soc.index,
                                                 y=block.soc,
                                                 mode='lines',
                                                 name=legentry_soc,
                                                 line=dict(width=2, dash=None),
                                                 visible='legendonly'),  # TODO introduce TUM colors
                                      secondary_y=True)
            elif isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities.values():
                    self.figure.add_trace(go.Scatter(x=commodity.flow.index,  # .to_pydatetime(),
                                                     y=commodity.flow * -1,
                                                     mode='lines',
                                                     name=f"{commodity.name} power",
                                                     line=dict(width=2, dash=None),
                                                     visible='legendonly'),  # TODO introduce TUM colors
                                          secondary_y=False)
                    commodity_display_size = round(commodity.size / 1e3, 1)
                    self.figure.add_trace(go.Scatter(x=commodity.soc.index.to_pydatetime(),
                                                     y=commodity.soc,
                                                     mode='lines',
                                                     name=f"{commodity.name} SOC ({commodity_display_size} kWh)",
                                                     line=dict(width=2, dash=None),
                                                     visible='legendonly'),  # TODO introduce TUM colors
                                          secondary_y=True)

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
            self.figure.update_layout(
                title=f'Global Optimum Results ({run.scenario_file_name} - Scenario: {self.name})')
        if self.strategy == 'rh':
            self.figure.update_layout(title=f'Rolling Horizon Results ({run.scenario_file_name} - Scenario: {self.name}'
                                            f'- PH:{self.ph_len}h/CH:{self.ch_len}h)')

    def print_results(self, run):
        print('#################')
        run.logger.info(f'Results for Scenario {self.name}:')
        for block in [block for block in self.blocks.values() if hasattr(block, 'opt') and block.opt]:
            run.logger.info(f'Optimized size of component {block.name}: {round(block.size / 1e3)} kW(h)')
        # ToDo: state that these results are internal costs of minigrid only neglecting costs for external charging
        run.logger.info(f'Total simulated cost: {str(round(self.totex_sim / 1e6, 2))} million {self.currency}')
        run.logger.info(f'Levelized cost of electricity: {str(round(1e5 * self.lcoe_dis, 2))} {self.currency}-ct/kWh')
        print('#################')

    def save_plots(self):
        self.figure.write_html(self.plot_file_path)

    def save_results(self, run):
        """
        Saves all int, float and str attributes of run, scenario (incl. technoeconomic KPIs) and all blocks to the
        results dataframe
        :param run: SimulationRun
        :return: none
        """

        result_types = (int, float, str, bool)
        result_blocks = {'run': run, 'scenario': self}
        result_blocks.update(self.blocks)

        for name, block in result_blocks.items():
            for key in [key for key in block.__dict__.keys() if isinstance(block.__dict__[key], result_types)]:
                value = block.__dict__[key]
                if isinstance(value, int):
                    self.results.loc[(name, key), self.name] = float(value)
                else:
                    self.results.loc[(name, key), self.name] = value

        self.results.reset_index(inplace=True, names=['block', 'key'])
        self.results.to_json(self.result_file_path, orient='records', lines=True)

    def show_plots(self):
        self.figure.show(renderer='browser')


class SimulationRun:

    def __init__(self):

        self.name = 'run'
        self.cwd = os.getcwd()

        # make sure that errors are logged to logfile
        sys.excepthook = self.handle_exception

        if len(sys.argv) == 1:  # if no arguments have been passed
            self.scenarios_file_path, self.settings_file_path, self.result_path = input_gui(self.cwd)
        elif len(sys.argv) == 2:  # only one argument, default result storage
            self.scenarios_file_path = os.path.join(self.cwd, 'input', 'scenarios', sys.argv[1])
            self.settings_file_path = os.path.join(self.cwd, 'input', 'settings', 'default.json')
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
        self.scenario_data = pd.read_json(self.scenarios_file_path, orient='records', lines=True)
        self.scenario_data = self.scenario_data.map(lambda x: x.lower() if isinstance(x, str) else x)
        self.scenario_data.set_index(['block', 'key'], inplace=True)
        self.scenario_data = self.scenario_data.apply(json_parse_bool, axis=1)
        self.scenario_names = self.scenario_data.columns  # Get list of column names, each column is one scenario
        self.scenario_num = len(self.scenario_names)

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder
        self.runtimestamp = datetime.now().strftime('%y%m%d_%H%M%S')  # create str of runtime_start

        with open(self.settings_file_path) as file:
            settings = json.load(file, object_hook=json_parse_bool)

        # check if the settings dict contains all necessary items
        if not all(item in settings.keys() for item in ["solver",
                                                        "parallel",
                                                        "max_process_num",
                                                        "save_results",
                                                        "save_des_results",
                                                        "print_results",
                                                        "save_plots",
                                                        "show_plots",
                                                        "dump_model",
                                                        "debugmode",
                                                        "eps_cost"]):
            raise Exception('incomplete settings file')

        for key, value in settings.items():  # TODO convert True/False strings to bool
            setattr(self, key, value)  # this sets all the parameters defined in the json file

        self.max_process_num = int(self.max_process_num)
        self.process_num = min(self.scenario_num, os.cpu_count(), self.max_process_num)

        self.input_data_path = os.path.join(self.cwd, 'input')
        self.result_folder_path = os.path.join(self.result_path, f'{self.runtimestamp}_{self.scenario_file_name}')
        self.result_file_path = os.path.join(self.result_folder_path,
                                             f'{self.runtimestamp}_{self.scenario_file_name}.json')
        self.dump_file_path = os.path.join(self.result_folder_path, f'{self.runtimestamp}_{self.scenario_file_name}.lp')
        self.log_file_path = os.path.join(self.result_folder_path, f'{self.runtimestamp}_{self.scenario_file_name}.log')
        self.result_df = pd.DataFrame  # blank DataFrame for technoeconomic result saving

        if self.save_results or self.save_des_results:
            os.mkdir(self.result_folder_path)

        log_formatter = logging.Formatter(logging.BASIC_FORMAT)
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setFormatter(log_formatter)
        log_file_handler = logging.FileHandler(os.environ.get("LOGFILE", self.log_file_path))
        log_file_handler.setFormatter(log_formatter)
        self.logger = logging.getLogger()
        self.logger.addHandler(log_stream_handler)
        self.logger.addHandler(log_file_handler)  # TODO global messages not getting through to logs in parallel mode

        if self.parallel:
            log_stream_handler.setLevel(logging.INFO)
            self.logger.info(f'Global settings read - '
                             f'simulating {self.scenario_num} scenario(s)'
                             f' in parallel mode with {self.process_num} process(es)')
        else:
            if self.debugmode:
                self.logger.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
                log_stream_handler.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
                log_stream_handler.setLevel(logging.INFO)
            self.logger.info(f'Global settings read - simulating {self.scenario_num} scenario(s) in sequential mode')

    def end_timing(self):

        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 1)
        self.logger.info(f'Total runtime for all scenarios: {str(self.runtime_len)} s')

    def handle_exception(self,exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        exit()

    def join_results(self):

        files = [filename for filename in os.listdir(self.result_folder_path) if filename.endswith('.json')]

        scenario_frames = []

        for file in files:
            file_path = os.path.join(self.result_folder_path, file)
            file_results = pd.read_json(file_path, orient='records', lines=True)
            file_results.set_index(['block', 'key'], drop=True, inplace=True)
            scenario_frames.append(file_results)

        joined_results = pd.concat(scenario_frames, axis=1)
        joined_results.reset_index(inplace=True, names=['block', 'key'])  # necessary for saving in json
        joined_results.to_json(self.result_file_path, orient='records', lines=True)
        self.logger.info("Technoeconomic output file created")

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in files:
            file_path = os.path.join(self.result_folder_path, file)
            os.remove(file_path)

###############################################################################
# global functions
###############################################################################


def json_parse_bool(dct: dict) -> dict:
    for key, value in dct.items():
        if isinstance(value, str):
            if value.lower() == 'true':
                dct[key] = True
            elif value.lower() == 'false':
                dct[key] = False
    return dct


def input_gui(directory):
    """
    GUI to choose input pickle file containing scenario definition DataFrame
    :return:
    """

    input_default = os.path.join(directory, 'input', 'scenarios')
    input_default_file = os.path.join(input_default, 'example.json')
    input_default_file_show = os.path.relpath(input_default_file, directory)
    settings_default = os.path.join(directory, 'input', 'settings')
    settings_default_file = os.path.join(settings_default, 'default.json')
    settings_default_file_show = os.path.relpath(settings_default_file, directory)
    results_default = os.path.join(directory, 'results')
    results_default_show = os.path.relpath(results_default, directory)

    input_file = [[psg.Text('Choose scenario definition file')],
                  [psg.Input(key='file',
                             default_text=input_default_file_show),
                   psg.FileBrowse(initial_folder=input_default,
                                  file_types=(('.json files',
                                               '.json'),))],
                  ]

    settings_file = [[psg.Text('Choose settings file')],
                  [psg.Input(key='file2',
                             default_text=settings_default_file_show),
                   psg.FileBrowse(initial_folder=settings_default,
                                  file_types=(('.json files',
                                               '.json'),))],
                  ]

    result_folder = [[psg.Text('Choose result storage folder')],
                     [psg.Input(key='folder',
                                default_text=results_default_show),
                      psg.FolderBrowse(initial_folder=results_default), ],
                     ]

    layout = [
        [psg.Column(input_file)],
        [psg.HSeparator()],
        [psg.Column(settings_file)],
        [psg.HSeparator()],
        [psg.Column(result_folder)],
        [psg.HSeparator()],
        [psg.OK(bind_return_key=True), psg.Cancel()],
    ]

    event, values = psg.Window('EV_ESM toolset - select scenario file, settings file and result path', layout).read(close=True)

    try:
        scenarios_filename = os.path.abspath(values['file'])
        settings_filename =  os.path.abspath(values['file2'])
        results_foldername = os.path.abspath(values['folder'])
        if scenarios_filename == '.' or settings_filename == '.' or results_foldername == '.':
            print('not all required paths entered - exiting')
            exit()
        return scenarios_filename, settings_filename, results_foldername
    except TypeError:
        print('GUI window closed manually - exiting')
        exit()
