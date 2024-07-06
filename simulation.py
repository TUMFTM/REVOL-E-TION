#!/usr/bin/env python3
"""
simulation.py

--- Description ---
This script provides the main simulation procedure classes for the REVOL-E-TION toolset.
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

import ast
import graphviz
import logging
import logging.handlers
import math
import numpy as np
import os
import pickle
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
import plotly.graph_objects as go
import tkinter as tk
import tkinter.filedialog

from pathlib import Path
from plotly.subplots import make_subplots

import blocks
import commodity_des as des
from additional_constraints import apply_additional_constraints
import tum_colors as col
from ensys_interface import call_ensys_interface


###############################################################################
# Functions
###############################################################################


def infer_dtype(value):
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() in ['none', 'null', 'nan']:
        return None

    try:
        evaluated = ast.literal_eval(value)
        if isinstance(evaluated, dict):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    return value.lower()


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
    if not scenarios_filename: scenarios_filename = scenarios_default_filename

    # get settings file
    settings_default_dir = os.path.join(directory, 'input', 'settings')
    settings_default_filename = os.path.join(settings_default_dir, 'default.csv')
    settings_filename = tk.filedialog.askopenfilename(initialdir=settings_default_dir, title="Select settings file",
                                                      filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not settings_filename: settings_filename = settings_default_filename

    # get result folder
    results_default_dir = os.path.join(directory, 'results')
    results_foldername = tk.filedialog.askdirectory(initialdir=results_default_dir,
                                                    title="Select result storage folder")
    if not results_foldername: results_foldername = results_default_dir

    return scenarios_filename, settings_filename, results_foldername


###############################################################################
# Class definitions
###############################################################################


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

        if self.ph_endtime > scenario.sim_endtime:
            self.ph_endtime = scenario.sim_endtime

        # Create datetimeindex for ph and ch; neglect last timestep as this is the first timestep of the next ph / ch
        self.dti_ph = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=scenario.timestep, inclusive='left')
        self.dti_ch = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=scenario.timestep, inclusive='left')

        scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - ' +
                             f'Start: {self.starttime} - ' +
                             (f'CH end: {self.ch_endtime} - ' if self.ch_endtime != self.ph_endtime else '') +
                             (f'PH end: {self.ph_endtime}' if self.ch_endtime != self.ph_endtime else f'End: {self.ph_endtime}'))

        scenario.logger.info(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                             f'Initializing model build')

        for block in [block for block in scenario.blocks.values() if hasattr(block, 'data')]:
            block.data_ph = block.data[self.starttime:self.ph_endtime]
            if isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities.values():
                    commodity.data_ph = commodity.data[self.starttime:self.ph_endtime]

        if scenario.strategy == 'rl':
            call_ensys_interface(scenario, run, 8, "DQN")
        # if apriori power scheduling is necessary, calculate power schedules:
        elif scenario.scheduler:
            scenario.scheduler.calc_ph_schedule(self)

        for block in scenario.blocks.values():
            block.update_input_components()  # (re)define solph components that need input slices

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
            self.model = solph.Model(self.es,
                                     debug=run.debugmode)  # Build the mathematical linear optimization model with pyomo
        except IndexError:
            msg = (f'Horizon {self.index + 1} of {scenario.nhorizons} -'
                   f'Input data not matching time index - check input data and time index consistency')
            scenario.logger.error(msg)
            raise IndexError(msg)

        apply_additional_constraints(model=self.model, prediction_horizon=self, scenario=scenario, run=run)

        if run.dump_model:
            if scenario.strategy in ['go', 'rl']:
                self.model.write(run.path_dump_file, io_options={'symbolic_solver_labels': True})
            elif scenario.strategy == 'rh':
                scenario.logger.warning('Model file dump not implemented for RH operating strategy - no file created')

        scenario.logger.debug(f'Horizon {self.index + 1} of {scenario.nhorizons} - '
                              f'Model build completed')

    def draw_energy_system(self, scenario, run):

        # Creates the Directed-Graph
        dot = graphviz.Digraph(filename=run.path_system_graph_file, format='pdf')

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
                scenario.logger.debug(f'Scenario: \"{scenario.name}"\ - System Node {nd.label} - Type {type(nd)} not recognized')

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

        scenario.logger.debug(f'Horizon {self.index} of {scenario.nhorizons}: getting results')

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
                                         na_values=['NaN', 'nan'],  # this inhibits None/Null being read as float NaN
                                         keep_default_na=False)
        self.scenario_data = self.scenario_data.sort_index(sort_remaining=True).map(infer_dtype)
        self.scenario_names = self.scenario_data.columns  # Get list of column names, each column is one scenario
        self.scenario_num = len(self.scenario_names)

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder
        self.runtimestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')  # create str of runtime_start

        self.commit_hash = self.get_git_commit_hash()

        settings = pd.read_csv(self.settings_file_path, index_col=[0])
        settings = settings.map(infer_dtype)

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
        self.path_system_graph_file = os.path.join(self.path_result_folder,
                                                   f'{self.runtimestamp}_{self.scenario_file_name}_system')

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

        joined_results = pd.concat(scenario_frames, axis=1)
        joined_results.reset_index(inplace=True, names=['block', 'key'])  # necessary for saving in csv
        joined_results.to_csv(self.path_result_summary_file)
        self.logger.info("Technoeconomic output file created")

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in files:
            file_path = os.path.join(self.path_result_folder, file)
            os.remove(file_path)
