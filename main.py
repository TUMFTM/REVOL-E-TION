#!/usr/bin/env python3
"""
main.py

--- Description ---
This script is the main executable for the oemof mg_ev toolset.
For further information, see readme.md

--- Created by ---
Philipp Rosner

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Module imports
###############################################################################

import logging
import logging.handlers
import math
import os
import pickle
import pprint
import sys
import time
import threading
import warnings

import multiprocessing as mp
import oemof.solph as solph
import pandas as pd
import plotly.graph_objects as go
import pylightxl as xl
import PySimpleGUI as psg

from datetime import datetime, timedelta
from itertools import repeat
from pathlib import Path
from plotly.subplots import make_subplots

import blocks
import tum_colors as col

warnings.filterwarnings("error")  # needed for catching UserWarning during infeasibility of scenario

###############################################################################
# Class definitions
###############################################################################


class PredictionHorizon:

    def __init__(self, index, scenario, run):

        self.index = index

        run.logger.info(f'Horizon {index+1} of {scenario.horizon_num} in scenario \"{scenario.name}\" initialized')

        # Time and data slicing --------------------------------
        self.starttime = scenario.sim_starttime + (index * scenario.ch_len)  # calc both start times
        self.ch_endtime = self.starttime + scenario.ch_len
        self.ph_endtime = self.starttime + scenario.ph_len
        self.timestep = scenario.sim_timestep

        if self.ph_endtime > scenario.sim_endtime:
            self.ph_endtime = scenario.sim_endtime

        self.ph_dti = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=scenario.sim_timestep).delete(-1)
        self.ch_dti = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=scenario.sim_timestep).delete(-1)

        for block in [block for block in scenario.blocks if hasattr(block, 'data')]:
            block.ph_data = block.data[self.starttime:self.ph_endtime]
            if isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities:
                    commodity.ph_data = commodity.data[self.starttime:self.ph_endtime]

        for block in scenario.blocks:
            block.update_input_components(scenario)  # (re)define solph components that need input slices

        self.results = None
        self.meta_results = None

        # Build energy system model --------------------------------

        self.es = solph.EnergySystem(timeindex=self.ph_dti,
                                     infer_last_interval=True)  # initialize energy system model instance

        for component in scenario.components:
            self.es.add(component)  # add components to this horizon's energy system

        if run.solver_debugmode:  # Build the mathematical linear optimization model with pyomo
            self.model = solph.Model(self.es, debug=True)
        else:
            self.model = solph.Model(self.es, debug=False)

        if run.dump_model:
            if scenario.strategy == 'go':
                self.model.write(run.dump_file_path, io_options={'symbolic_solver_labels': True})
            elif scenario.strategy == 'rh':
                run.logger.warning('Model file dump not implemented for RH operating strategy - no file created')

    def get_results(self, scenario, run):
        """
        Get result data slice for current CH from results and save in result dataframes for later analysis
        Get (possibly optimized) component sizes from results to handle outputs more easily
        """

        self.results = solph.processing.results(self.model)  # Get the results of the solved horizon from the solver

        if run.print_results:
            self.meta_results = solph.processing.meta_results(self.model)
            pprint.pprint(self.meta_results)

        source_types = (blocks.PVSource, blocks.WindSource, blocks.ControllableSource)

        for block in scenario.blocks:
            if isinstance(block, blocks.StationaryEnergyStorage) and block.opt:
                block.size = self.results[(block.ess, None)]["scalars"]["invest"]
            elif isinstance(block, source_types) and block.opt:
                block.size = self.results[(block.src, block.bus)]['scalars']['invest']
            elif isinstance(block, blocks.CommoditySystem) and block.opt:
                pass  # TODO get sizes for optimized commodities
            elif isinstance(block, blocks.SystemCore) and block.opt:
                block.acdc_size = self.results[(block.ac_bus, block.ac_dc)]['scalars']['invest']
                block.dcac_size = self.results[(block.dc_bus, block.dc_ac)]['scalars']['invest']

            block.get_ch_results(self, scenario)

    def run_optimization(self, scenario, run):
        run.logger.info(f'Optimization for horizon {self.index+1} in scenario \"{scenario.name}\" initialized')
        try:
            self.model.solve(solver=run.solver, solve_kwargs={'tee': run.solver_debugmode})
        except UserWarning as exc:
            run.logger.warning(f'Scenario {scenario.name} failed or infeasible - continue on next scenario')
            scenario.exception = str(exc)
            # TODO does not jump to next scenario properly (at least in parallel mode)

    def run_lfs(self, scenario, run):
        pass

    def run_ccs(self, scenario, run):
        pass

    def run_pss(self, scenario, run):
        pass


class Scenario:

    def __init__(self, scenario_name, run):

        self.name = scenario_name

        # General Information --------------------------------

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        run.logger.info(f'Scenario \"{self.name}\" initialized')  # TODO state process number

        self.prj_starttime = datetime.strptime(xread('prj_start', self.name, run), '%Y/%m/%d')
        self.prj_duration_yrs = xread('prj_duration', self.name, run)
        self.prj_duration = timedelta(days=self.prj_duration_yrs * 365)  # no leap years
        self.prj_endtime = self.prj_starttime + self.prj_duration
        self.prj_duration_days = (self.prj_endtime.date() - self.prj_starttime.date()).days

        self.sim_starttime = self.prj_starttime  # simulation timeframe is at beginning of project timeframe
        self.sim_timestep = xread('sim_timestep', self.name, run)
        self.sim_duration = timedelta(days=xread('sim_duration', self.name, run))
        self.sim_endtime = self.sim_starttime + self.sim_duration
        self.sim_dti = pd.date_range(start=self.sim_starttime, end=self.sim_endtime, freq=self.sim_timestep).delete(-1)
        self.sim_timestep_hours = self.sim_dti.freq.nanos / 1e9 / 3600

        self.sim_yr_rat = self.sim_duration.days / 365  # no leap years
        self.sim_prj_rat = self.sim_duration.days / self.prj_duration_days
        self.wacc = xread('wacc', self.name, run)

        self.plot_file_path = os.path.join(run.result_folder_path, f'{run.runtimestamp}_'
                                                                   f'{run.scenarios_file_name}_'
                                                                   f'{self.name}.html')

        self.results = dict()  # for cumulative result saving as pickle later on
        self.results['scenario_name'] = self.name  # saving scenario name for pickle
        self.result_file_path = os.path.join(run.result_folder_path, f'{self.name}.pkl')

        # Operational strategy --------------------------------

        self.strategy = xread('sim_os', self.name, run)
        self.exception = None  # placeholder for possible infeasibility

        if self.strategy == 'rh':
            self.ph_len_hrs = xread('rh_ph', self.name, run)
            self.ch_len_hrs = xread('rh_ch', self.name, run)
            self.ph_len = timedelta(hours=self.ph_len_hrs)
            self.ch_len = timedelta(hours=self.ch_len_hrs)
            # number of timesteps for PH
            self.ph_steps = math.ceil(self.ph_len.total_seconds() / 3600 / self.sim_timestep_hours)
            # number of timesteps for CH
            self.ch_steps = math.ceil(self.ch_len.total_seconds() / 3600 / self.sim_timestep_hours)
            self.horizon_num = int(self.sim_duration // self.ch_len)  # number of timeslices to run
        elif self.strategy in ['go', 'lfs']:
            self.ph_len = self.sim_duration
            self.ch_len = self.sim_duration
            self.horizon_num = 1

        # Energy System Blocks --------------------------------

        self.blocks = []
        self.components = []
        self.blocks_enable = dict(dem=(xread('dem_enable', self.name, run) == 'True'),
                                  wind=(xread('wind_enable', self.name, run) == 'True'),
                                  pv=(xread('pv_enable', self.name, run) == 'True'),
                                  gen=(xread('gen_enable', self.name, run) == 'True'),
                                  grid=(xread('grid_enable', self.name, run) == 'True'),
                                  ess=(xread('ess_enable', self.name, run) == 'True'),
                                  bev=(xread('bev_enable', self.name, run) == 'True'),
                                  brs=(xread('brs_enable', self.name, run) == 'True'))

        self.core = blocks.SystemCore('core', self, run)  # always enabled

        for name in [name for name, enable in self.blocks_enable.items() if enable]:
            if name == 'dem':
                self.dem = blocks.FixedDemand('dem', self, run)
            elif name == 'wind':
                self.wind = blocks.WindSource('wind', self, run)
            elif name == 'pv':
                self.pv = blocks.PVSource('pv', self, run)
            elif name == 'gen':
                self.gen = blocks.ControllableSource('gen', self, run)
            elif name == 'grid':
                self.grid = blocks.ControllableSource('grid', self, run)
            elif name == 'ess':
                self.ess = blocks.StationaryEnergyStorage('ess', self, run)
            elif name == 'bev':
                self.bev = blocks.CommoditySystem('bev', self, run)
            elif name == 'brs':
                self.brs = blocks.CommoditySystem('brs', self, run)

        # Result variables --------------------------------

        self.figure = None  # placeholder for plotting

        self.e_sim_del = self.e_yrl_del = self.e_prj_del = self.e_dis_del = 0
        self.e_sim_pro = self.e_yrl_pro = self.e_prj_pro = self.e_dis_pro = 0
        self.e_eta = None

        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = 0
        self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = 0
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = 0
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = 0
        self.lcoe = self.lcoe_dis = None

    def accumulate_results(self, run):

        for block in self.blocks:
            block.accumulate_results(self)

        #  TODO find a metric for curtailed energy and calculate

        try:
            self.e_eta = self.e_sim_del / self.e_sim_pro
        except ZeroDivisionError or RuntimeWarning:
            run.logger.warning("Efficiency calculation: division by zero")

        try:
            self.lcoe = self.totex_dis / self.e_prj_del
            self.lcoe_dis = self.totex_dis / self.e_dis_del
        except ZeroDivisionError or RuntimeWarning:
            run.logger.warning("LCOE calculation: division by zero")


        lcoe_display = round(self.lcoe_dis * 1e5, 1)
        npc_display = round(self.totex_dis)
        run.logger.info(f'Scenario \"{self.name}\" - NPC {npc_display} USD - LCOE {lcoe_display} USct/kWh')

    def end_timing(self, run):
        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 2)
        run.logger.info(f'Scenario \"{self.name}\" finished - runtime {self.runtime_len} s')

    def generate_plots(self, run):

        self.figure = make_subplots(specs=[[{'secondary_y': True}]])

        # types for which positive flow values are power taken out of the core
        invert_types = (blocks.FixedDemand, blocks.StationaryEnergyStorage, blocks.CommoditySystem)

        for block in [block for block in self.blocks if not isinstance(block, blocks.SystemCore)]:

            if hasattr(block, 'size'):
                if isinstance(block, blocks.CommoditySystem):
                    legentry_p = f"{block.name} total power"
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
                for commodity in block.commodities:
                    self.figure.add_trace(go.Scatter(x=commodity.flow.index,  # .to_pydatetime(),
                                                     y=commodity.flow * -1,
                                                     mode='lines',
                                                     name=f"{commodity.name} power",
                                                     line=dict(width=2, dash=None),
                                                     visible='legendonly'),  # TODO introduce TUM colors
                                          secondary_y=False)
                    self.figure.add_trace(go.Scatter(x=commodity.soc.index.to_pydatetime(),
                                                     y=commodity.soc,
                                                     mode='lines',
                                                     name=f"{commodity.name} SOC ({display_size} kWh)",
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
            self.figure.update_layout(title=f'Global Optimum Results ({run.scenarios_file_name} - Scenario: {self.name})')
        if self.strategy == 'rh':
            self.figure.update_layout(title=f'Rolling Horizon Results ({run.scenarios_file_name} - Scenario: {self.name}'
                                            f'- PH:{self.ph_len}h/CH:{self.ch_len}h)')

    def print_results(self):
        print('#################')
        run.logger.info(f'Results for Scenario {self.name}:')
        for block in [block for block in self.blocks if hasattr(block, 'opt') and block.opt]:
            run.logger.info(f'Optimized size of component {block.name}: {round(block.size / 1e3)} kW(h)')
        run.logger.info(f'Total simulated cost: {str(round(self.totex_sim / 1e6, 2))} million USD')
        run.logger.info(f'Levelized cost of electricity: {str(round(1e5 * self.lcoe_dis, 2))} USct/kWh')
        print('#################')

    def save_exception(self, run):
        """
        Dump error message in result excel file if optimization did not succeed
        """
        # scenario_name is already added in __init__
        self.results['title'] = f'Global Optimum Results ({run.result_path} ' \
                                f'- Sheet: {self.name})'
        self.results['runtimestamp'] = run.runtimestamp
        self.results['runtime'] = self.runtime_len
        self.results['exception'] = self.exception

        with open(self.result_file_path, 'wb') as file:
            pickle.dump(self.results, file)

    def save_plots(self):
        self.figure.write_html(self.plot_file_path)
                
    def save_results(self, run):

        # scenario_name is already added in __init__
        if self.strategy == 'go':
            self.results['title'] = f'Global Optimum Results ({run.result_path} ' \
                                    f'- Sheet: {self.name})'
        elif self.strategy == 'rh':
            self.results['title'] = f'Rolling Horizon Results ({run.result_path} ' \
                                    f'- Sheet: {self.name} ' \
                                    f'- PH: {self.ph_len_hrs} ' \
                                    f'- CH: {self.ch_len_hrs})'

        self.results['runtimestamp'] = run.runtimestamp
        self.results['runtime'] = self.runtime_len

        result_types = (int, float, str)
        result_blocks = [run, self] + self.blocks

        for index, obj in enumerate(result_blocks):

            self.results[obj.name] = dict()

            for item in [item for item in obj.__dict__.items() if isinstance(item[1], result_types)]:
                if isinstance(item[1], int):
                    self.results[obj.name][item[0]] = float(item[1])
                else:
                    self.results[obj.name][item[0]] = item[1]

        with open(self.result_file_path, 'wb') as file:
            pickle.dump(self.results, file)

    def show_plots(self):
        self.figure.show(renderer='browser')


class SimulationRun:

    def __init__(self):

        self.name = 'run'
        self.cwd = os.getcwd()
        if len(sys.argv) == 1:  # if no arguments have been passed
            self.scenarios_file_path, self.result_path = input_gui(self.cwd)
        elif len(sys.argv) == 2:  # only one argument, default result storage
            self.scenarios_file_path = os.path.join(self.cwd, 'input', '_excel', sys.argv[1])
            self.result_path = os.path.join(self.cwd, 'results')
        else:
            self.scenarios_file_path = os.path.join(self.cwd, 'input', '_excel', sys.argv[1])
            self.result_path = sys.argv[2]

        self.scenarios_file_name = Path(self.scenarios_file_path).stem  # Gives file name without extension
        self.input_xdb = xl.readxl(fn=self.scenarios_file_path)  # Excel database of selected file
        self.scenario_names = self.input_xdb.ws_names  # Get list of sheet names, 1 sheet is 1 scenario

        try:
            self.scenario_names.remove('global_settings')
        except ValueError:
            print('Excel File does not include global settings - exiting')
            exit()

        self.scenario_num = len(self.scenario_names)
        self.process_num = min(self.scenario_num, os.cpu_count())

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder
        self.runtimestamp = datetime.now().strftime('%y%m%d_%H%M%S')  # create str of runtime_start

        self.global_sheet = 'global_settings'
        self.solver = xread('solver', self.global_sheet, self)
        self.parallel = (xread('parallel', self.global_sheet, self) == 'True')
        self.save_results = (xread('save_results', self.global_sheet, self) == 'True')
        self.print_results = (xread('print_results', self.global_sheet, self) == 'True')
        self.save_plots = (xread('save_plots', self.global_sheet, self) == 'True')
        self.show_plots = (xread('show_plots', self.global_sheet, self) == 'True')
        self.dump_model = (xread('dump_model', self.global_sheet, self) == 'True')
        self.solver_debugmode = (xread('solver_debugmode', self.global_sheet, self) == 'True')
        self.eps_cost = float(xread('eps_cost', self.global_sheet, self))

        self.cwd = os.getcwd()
        self.input_data_path = os.path.join(self.cwd, 'input')
        self.result_folder_path = os.path.join(self.result_path, f'{self.runtimestamp}_{self.scenarios_file_name}')
        self.result_file_path = os.path.join(self.result_folder_path, f'{self.runtimestamp}_{self.scenarios_file_name}.xlsx')
        self.dump_file_path = os.path.join(self.result_folder_path, f'{self.runtimestamp}_{self.scenarios_file_name}.lp')
        self.log_file_path = os.path.join(self.result_folder_path, f'{self.runtimestamp}_{self.scenarios_file_name}.log')
        self.result_xdb = xl.Database()  # blank excel database for result saving

        if self.save_results:
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
            self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
            log_stream_handler.setLevel(logging.INFO)
            self.logger.info(f'Global settings read - simulating {self.scenario_num} scenario(s) in sequential mode')

    def end_timing(self):

        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 1)
        self.logger.info(f'Total runtime for all scenarios: {str(self.runtime_len)} s')

    def join_results(self):

        path = self.result_folder_path
        files = [filename for filename in os.listdir(path) if filename.endswith('.pkl')]

        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, 'rb') as pickle_file:
                results = pickle.load(pickle_file)
            if 'exception' in results.keys():  # scenario infeasible
                self.save_pickle_exception(results)
            else:  # scenario feasible
                self.save_pickle_results(results)
            os.remove(file_path)

        if self.save_results:
            xl.writexl(db=self.result_xdb, fn=self.result_file_path)
            self.logger.info("Excel output file created")

    @staticmethod
    def save_pickle_exception(res: dict):
        """
        Dump error message in result excel file if optimization did not succeed
        """
        ws = res['scenario_name']
        run.result_xdb.add_ws(ws=ws)

        run.result_xdb.ws(ws=ws).update_index(row=1, col=1, val=res['title'])
        run.result_xdb.ws(ws=ws).update_index(row=2, col=1, val='Timestamp')
        run.result_xdb.ws(ws=ws).update_index(row=2, col=2, val=res['runtimestamp'])
        run.result_xdb.ws(ws=ws).update_index(row=3, col=1, val='Runtime')
        run.result_xdb.ws(ws=ws).update_index(row=3, col=2, val=res['runtime'])
        run.result_xdb.ws(ws=ws).update_index(row=4, col=1, val='Optimization unsuccessful!')
        run.result_xdb.ws(ws=ws).update_index(row=5, col=1, val='Message')
        run.result_xdb.ws(ws=ws).update_index(row=5, col=2, val=res['exception'])

    @staticmethod
    def save_pickle_results(res: dict):

        ws = res['scenario_name']
        run.result_xdb.add_ws(ws=ws)

        run.result_xdb.ws(ws=ws).update_index(row=1, col=1, val=res['title'])
        run.result_xdb.ws(ws=ws).update_index(row=2, col=1, val='Timestamp')
        run.result_xdb.ws(ws=ws).update_index(row=2, col=2, val=res['runtimestamp'])
        run.result_xdb.ws(ws=ws).update_index(row=3, col=1, val='Runtime')
        run.result_xdb.ws(ws=ws).update_index(row=3, col=2, val=res['runtime'])

        header_row = 5

        for index, obj in enumerate([value for key, value in res.items() if isinstance(value, dict)]):
            col_id = 1 + index * 4
            row_id = header_row + 1

            obj_name = obj['name']

            if obj_name == res['scenario_name']:  # if object is scenario
                run.result_xdb.ws(ws=ws).update_index(row=header_row, col=col_id, val='scenario data')
            else:

                run.result_xdb.ws(ws=ws).update_index(row=header_row, col=col_id, val=f'{obj_name} data')

            for key, value in obj.items():
                run.result_xdb.ws(ws=ws).update_index(row=row_id, col=col_id, val=key)
                if isinstance(value, int):
                    run.result_xdb.ws(ws=ws).update_index(row=row_id, col=col_id + 1, val=float(value))
                else:
                    run.result_xdb.ws(ws=ws).update_index(row=row_id, col=col_id + 1, val=value)
                row_id += 1


###############################################################################
# Function definitions
###############################################################################


def input_gui(directory):
    """
    GUI to choose input excel file
    :return:
    """

    input_default = os.path.join(directory, 'input', '_excel')
    input_default_file = os.path.join(input_default, 'example.xlsx')
    input_default_file_show = os.path.relpath(input_default_file, directory)
    results_default = os.path.join(directory, 'results')
    results_default_show = os.path.relpath(results_default, directory)

    input_file = [[psg.Text('Choose input settings file')],
                  [psg.Input(key='file',
                             default_text=input_default_file_show),
                   psg.FileBrowse(initial_folder=input_default,
                                  file_types=(('Excel worksheets',
                                               '.xlsx'),))],
                  ]

    result_folder = [[psg.Text('Choose result storage folder')],
                     [psg.Input(key='folder',
                                default_text=results_default_show),
                      psg.FolderBrowse(initial_folder=results_default), ],
                     ]

    layout = [
        [psg.Column(input_file)],
        [psg.HSeparator()],
        [psg.Column(result_folder)],
        [psg.HSeparator()],
        [psg.OK(bind_return_key=True), psg.Cancel()],
    ]

    event, values = psg.Window('MGEV toolset - select input file and result path', layout).read(close=True)

    try:
        scenarios_filename = os.path.abspath(values['file'])
        results_foldername = os.path.abspath(values['folder'])
        if scenarios_filename == '.' or results_foldername == '.':
            print('not all required paths entered - exiting')
            exit()
        return scenarios_filename, results_foldername
    except TypeError:
        print('GUI window closed manually - exiting')
        exit()


def read_mplogger_queue(queue):
    while True:
        record = queue.get()
        if record is None:
            break
        # run.logger.handle(record)  # This line causes double logger outputs on Linux


def simulate_scenario(name: str, run: SimulationRun, log_queue):  # needs to be a function for starpool

    if run.parallel:
        run.process = mp.current_process()
        run.queue_handler = logging.handlers.QueueHandler(log_queue)
        run.logger = logging.getLogger()
        if run.solver_debugmode:
            run.logger.setLevel(logging.DEBUG)
        else:
            run.logger.setLevel(logging.INFO)
        run.logger.addHandler(run.queue_handler)

    scenario = Scenario(name, run)  # Create scenario instance & read data from Excel sheet.

    for horizon_index in range(scenario.horizon_num):  # Inner optimization loop over all prediction horizons
        #try:
        horizon = PredictionHorizon(horizon_index, scenario, run)
#        except IndexError:
#            scenario.exception = 'Input data does not cover full sim timespan'
#            logging.warning(f'Input data in scenario \"{scenario.name}\" does not cover full simulation timespan'
#                            f' - continuing on next scenario')
#            scenario.save_exception(run)
#            break

        if scenario.strategy == 'lfs':
            pass  # rule_based.lfs(horizon)
        elif scenario.strategy == 'ccs':
            pass  # rule_based.ccs(horizon)
        elif scenario.strategy in ['go', 'rh']:
            horizon.run_optimization(scenario, run)

        if scenario.exception and run.save_results:
            scenario.save_exception(run)
            break
        else:
            horizon.get_results(scenario, run)

    scenario.end_timing(run)

    if not scenario.exception:
        if run.save_results or run.print_results:
            scenario.accumulate_results(run)
            if run.save_results:
                scenario.save_results(run)
            if run.print_results:
                scenario.print_results()

        if run.save_plots or run.show_plots:
            scenario.generate_plots(run)
            if run.save_plots:
                scenario.save_plots()
            if run.show_plots:
                scenario.show_plots()


def xread(param, sheet, run):
    """
    Reading parameters from external excel file
    """
    value = None
    try:
        value = run.input_xdb.ws(ws=sheet).keyrow(key=param, keyindex=1)[1]
    except IndexError:
        run.logger.warning(f'Key \"{param}\" not found in Excel worksheet - exiting')
        exit()  # TODO enable jump to next scenario
    return value


###############################################################################
# Execution code
###############################################################################


if __name__ == '__main__':

    run = SimulationRun()  # get all global information about the run

    if run.parallel:
        with mp.Manager() as manager:
            log_queue = manager.Queue()
            log_thread = threading.Thread(target=read_mplogger_queue, args=(log_queue,))
            log_thread.start()
            with mp.Pool(processes=run.process_num) as pool:
                pool.starmap(simulate_scenario, zip(run.scenario_names, repeat(run), repeat(log_queue)))
            log_queue.put(None)
            log_thread.join()
    else:
        for scenario_name in run.scenario_names:
            simulate_scenario(scenario_name, run, None)  # no logger queue

    if run.save_results:
        run.join_results()

    run.end_timing()
