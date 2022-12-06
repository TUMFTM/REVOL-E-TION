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
import multiprocessing
import sys

import oemof.solph as solph
import os
import pandas as pd
import plotly.graph_objects as go
import pprint
import pylightxl as xl
import PySimpleGUI as psg
import time
import warnings
warnings.filterwarnings("error")  # needed for catching UserWarning during infeasibility of scenario

from datetime import datetime, timedelta
from itertools import repeat
from pathlib import Path
from plotly.subplots import make_subplots

import blocks
import tum_colors as col


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

        self.ph_dti = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=scenario.sim_timestep).delete(-1)
        self.ch_dti = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=scenario.sim_timestep).delete(-1)

        for block in [block for block in scenario.blocks if hasattr(block, 'data')]:
            # block.ph_data = block.data.loc[block.data.index.isin(self.ph_dti)]#.reset_index(drop=True)
            block.ph_data = block.data[self.starttime:self.ph_endtime][:-1]  # todo fix workaround, prefer above

        for block in scenario.blocks:
            block.update_input_components(scenario)  # (re)define solph components that need input slices

        self.results = None
        self.meta_results = None

        # Build energy system model --------------------------------

        self.es = solph.EnergySystem(timeindex=self.ph_dti)  # initialize energy system model instance

        for component in scenario.components:
            self.es.add(component)  # add components to this horizon's energy system

        self.model = solph.Model(self.es)  # Build the mathematical linear optimization model with pyomo

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

            block.get_ch_results(self, scenario)

    def run_optimization(self, scenario, run):
        run.logger.info(f'Optimization for horizon {self.index+1} initialized')
        try:
            self.model.solve(solver=run.solver, solve_kwargs={'tee': run.solver_debugmode})
        except UserWarning as exc:
            run.logger.warning(f'Scenario {scenario.name} failed or infeasible - continue on next scenario')
            scenario.exception = str(exc)


class Scenario:

    def __init__(self, scenario_name, run):

        self.name = scenario_name

        # General Information --------------------------------

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        run.logger.info(f'Scenario \"{self.name}\" initialized')  # TODO state process number

        self.prj_starttime = datetime.strptime(xread('prj_start', self.name, run.input_xdb), '%Y/%m/%d')
        self.prj_duration_yrs = xread('prj_duration', self.name, run.input_xdb)
        self.prj_duration = timedelta(days=self.prj_duration_yrs * 365)  # no leap years
        self.prj_endtime = self.prj_starttime + self.prj_duration
        self.prj_duration_days = (self.prj_endtime.date() - self.prj_starttime.date()).days

        self.sim_starttime = self.prj_starttime  # simulation timeframe is at beginning of project timeframe
        self.sim_timestep = xread('sim_timestep', self.name, run.input_xdb)
        self.sim_duration = timedelta(days=xread('sim_duration', self.name, run.input_xdb))
        self.sim_endtime = self.sim_starttime + self.sim_duration
        self.sim_dti = pd.date_range(start=self.sim_starttime, end=self.sim_endtime, freq=self.sim_timestep).delete(-1)

        self.sim_yr_rat = self.sim_duration.days / 365.25
        self.sim_prj_rat = self.sim_duration.days / self.prj_duration_days
        self.wacc = xread('wacc', self.name, run.input_xdb)

        self.plot_file_path = os.path.join(run.result_path, f'{run.runtimestamp}_'
                                                            f'{run.scenarios_file_name}_'
                                                            f'{self.name}.html')

        # Operational strategy --------------------------------

        self.strategy = xread('sim_os', self.name, run.input_xdb)
        self.exception = None  # placeholder for possible infeasibility

        if self.strategy == 'rh':
            self.ph_len = timedelta(hours=xread('rh_ph', self.name, run.input_xdb))
            self.ch_len = timedelta(hours=xread('rh_ch', self.name, run.input_xdb))
            self.ph_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ph_len  # number of timesteps for PH
            self.ch_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ch_len  # number of timesteps for CH
            self.horizon_num = int(self.sim_duration // self.ch_len)  # number of timeslices to run
        elif self.strategy == 'go':
            self.ph_len = self.sim_duration
            self.ch_len = self.sim_duration
            self.horizon_num = 1

        # Energy System Blocks --------------------------------

        self.blocks = []
        self.components = []
        self.blocks_enable = dict(dem=(xread('dem_enable', self.name, run.input_xdb) == 'True'),
                                  wind=(xread('wind_enable', self.name, run.input_xdb) == 'True'),
                                  pv=(xread('pv_enable', self.name, run.input_xdb) == 'True'),
                                  gen=(xread('gen_enable', self.name, run.input_xdb) == 'True'),
                                  ess=(xread('ess_enable', self.name, run.input_xdb) == 'True'),
                                  bev=(xread('bev_enable', self.name, run.input_xdb) == 'True'),
                                  brs=(xread('brs_enable', self.name, run.input_xdb) == 'True'))

        self.core = blocks.SystemCore('core', self, run)

        for name in [name for name, enable in self.blocks_enable.items() if enable]:
            if name == 'dem':
                self.dem = blocks.FixedDemand('dem', self, run)
            elif name == 'wind':
                self.wind = blocks.WindSource('wind', self, run)
            elif name == 'pv':
                self.pv = blocks.PVSource('pv', self, run)
            elif name == 'gen':
                self.gen = blocks.ControllableSource('gen', self, run)
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

    def accumulate_results(self):

        for block in self.blocks:
            block.accumulate_results(self)

        #  TODO find a metric for curtailed energy and calculate

        try:
            self.e_eta = self.e_sim_del / self.e_sim_pro
        except ZeroDivisionError:
            run.logger.warning("Efficiency calculation: division by zero")

        try:
            self.lcoe = self.totex_dis / self.e_prj_del
            self.lcoe_dis = self.totex_dis / self.e_dis_del
        except ZeroDivisionError:
            run.logger.warning("LCOE calculation: division by zero")

    def end_timing(self, run):
        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 2)
        run.logger.info(f'Scenario \"{self.name}\" finished - runtime {self.runtime_len} s')

    def generate_plots(self, run):

        self.figure = make_subplots(specs=[[{'secondary_y': True}]])

        demand_types = (blocks.FixedDemand, blocks.StationaryEnergyStorage, blocks.CommoditySystem)

        for block in [block for block in self.blocks if not isinstance(block, blocks.SystemCore)]:

            if isinstance(block, demand_types):
                self.figure.add_trace(go.Scatter(x=block.flow.index,  # .to_pydatetime(),
                                                 y=block.flow * -1,
                                                 mode='lines',
                                                 name=f"{block.name} Power",  # TODO print sizing in plot
                                                 line=dict(width=2, dash=None)),  # TODO introduce TUM colors
                                      secondary_y=False)
            else:
                self.figure.add_trace(go.Scatter(x=block.flow.index,  # .to_pydatetime(),
                                                 y=block.flow,
                                                 mode='lines',
                                                 name=f"{block.name} Power",   # TODO print sizing in plot
                                                 line=dict(width=2, dash=None)),  # TODO introduce TUM colors
                                      secondary_y=False)

            if isinstance(block, blocks.StationaryEnergyStorage):
                self.figure.add_trace(go.Scatter(x=block.soc.index,  # .to_pydatetime(),
                                                 y=block.soc,
                                                 mode='lines',
                                                 name=f"{block.name} SOC",  # TODO print sizing in plot
                                                 line=dict(width=2, dash=None),
                                                 visible=None),  # TODO introduce TUM colors
                                      secondary_y=True)

            if isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities:
                    self.figure.add_trace(go.Scatter(x=commodity.soc.index.to_pydatetime(),
                                                     y=commodity.soc,
                                                     mode='lines',
                                                     name=f"{commodity.name} SOC",    # TODO print sizing in plot, denote whether single or combined value
                                                     line=dict(width=2, dash=None),
                                                     visible=None),  # TODO introduce TUM colors
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
            self.figure.update_layout(title=f'Global Optimum Results ({run.scenarios_file_name} - Sheet: {self.name})')
        if self.strategy == 'rh':
            self.figure.update_layout(title=f'Rolling Horizon Results ({run.scenarios_file_name} - Sheet: {self.name}'
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
        if run.save_results:
            run.result_xdb.add_ws(ws=self.name)
            ws_title = f'Results ({run.result_path} - Sheet: {self.name})'
            run.result_xdb.ws(ws=self.name).update_index(row=1, col=1, val=ws_title)
            run.result_xdb.ws(ws=self.name).update_index(row=2, col=1, val='Timestamp')
            run.result_xdb.ws(ws=self.name).update_index(row=2, col=2, val=run.runtimestamp)
            run.result_xdb.ws(ws=self.name).update_index(row=3, col=1, val='Runtime')
            run.result_xdb.ws(ws=self.name).update_index(row=3, col=2, val=self.runtime_len)

            run.result_xdb.ws(ws=self.name).update_index(row=4, col=1, val='Optimization unsuccessful!')
            run.result_xdb.ws(ws=self.name).update_index(row=5, col=1, val='Message')
            run.result_xdb.ws(ws=self.name).update_index(row=5, col=2, val=self.exception)

    def save_plots(self):
        self.figure.write_html(self.plot_file_path)

    def save_results(self, run):

        run.result_xdb.add_ws(ws=self.name)

        if self.strategy == 'go':
            ws_title = f'Global Optimum Results ({run.result_path} - Sheet: {self.name})'
        elif self.strategy == 'rh':
            ws_title = f'Rolling Horizon Results ({run.result_path} - Sheet: {self.name} - PH: {self.ph_len}' \
                       f' - CH: {self.ch_len})'
        else:
            ws_title = f'Unknown Strategy Results ({run.result_path} - Sheet: {self.name})'

        run.result_xdb.ws(ws=self.name).update_index(row=1, col=1, val=ws_title)
        run.result_xdb.ws(ws=self.name).update_index(row=2, col=1, val='Timestamp')
        run.result_xdb.ws(ws=self.name).update_index(row=2, col=2, val=run.runtimestamp)
        run.result_xdb.ws(ws=self.name).update_index(row=3, col=1, val='Runtime')
        run.result_xdb.ws(ws=self.name).update_index(row=3, col=2, val=self.runtime_len)

        header_row = 5
        excel_types = (int, float, str)
        excel_blocks = [run, self] + self.blocks

        for index, obj in enumerate(excel_blocks):
            col_id = 1 + index * 4
            row_id = header_row + 1

            if isinstance(obj, Scenario):
                run.result_xdb.ws(ws=self.name).update_index(row=header_row, col=col_id, val='scenario data')
            else:
                run.result_xdb.ws(ws=self.name).update_index(row=header_row, col=col_id, val=f'{obj.name} data')

            for item in [item for item in obj.__dict__.items() if isinstance(item[1], excel_types)]:
                run.result_xdb.ws(ws=self.name).update_index(row=row_id, col=col_id, val=item[0])
                if isinstance(item[1], int):
                    run.result_xdb.ws(ws=self.name).update_index(row=row_id, col=col_id + 1, val=float(item[1]))
                else:
                    run.result_xdb.ws(ws=self.name).update_index(row=row_id, col=col_id + 1, val=item[1])
                row_id += 1

    def show_plots(self):
        self.figure.show()


class SimulationRun:

    def __init__(self):

        self.name = 'run'

        self.cwd = os.getcwd()
        self.scenarios_file_path, self.result_path = input_gui(self.cwd)
        self.scenarios_file_name = Path(self.scenarios_file_path).stem  # Gives file name without extension
        self.input_xdb = xl.readxl(fn=self.scenarios_file_path)  # Excel database of selected file
        self.scenario_names = self.input_xdb.ws_names  # Get list of sheet names, 1 sheet is 1 scenario

        try:
            self.scenario_names.remove('global_settings')
        except ValueError:
            print('Excel File does not include global settings - exiting')
            exit()

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder
        self.runtimestamp = datetime.now().strftime('%y%m%d_%H%M%S')  # create str of runtime_start

        self.global_sheet = 'global_settings'
        self.solver = xread('solver', self.global_sheet, self.input_xdb)
        self.parallel = (xread('parallel', self.global_sheet, self.input_xdb) == 'True')
        self.save_results = (xread('save_results', self.global_sheet, self.input_xdb) == 'True')
        self.print_results = (xread('print_results', self.global_sheet, self.input_xdb) == 'True')
        self.save_plots = (xread('save_plots', self.global_sheet, self.input_xdb) == 'True')
        self.show_plots = (xread('show_plots', self.global_sheet, self.input_xdb) == 'True')
        self.dump_model = (xread('dump_model', self.global_sheet, self.input_xdb) == 'True')
        self.solver_debugmode = (xread('solver_debugmode', self.global_sheet, self.input_xdb) == 'True')
        self.eps_cost = float(xread('eps_cost', self.global_sheet, self.input_xdb))

        self.cwd = os.getcwd()
        self.input_data_path = os.path.join(self.cwd, 'input')
        self.dump_file_path = os.path.join(self.result_path, f'{self.runtimestamp}_{self.scenarios_file_name}.lp')
        self.log_file_path = os.path.join(self.result_path, f'{self.runtimestamp}_{self.scenarios_file_name}.log')
        self.result_file_path = os.path.join(self.result_path, f'{self.runtimestamp}_{self.scenarios_file_name}.xlsx')
        self.result_xdb = xl.Database()  # blank excel database for cumulative result saving

        self.log_formatter = logging.Formatter(logging.BASIC_FORMAT)
        self.log_stream_handler = logging.StreamHandler(sys.stdout)
        self.log_stream_handler.setFormatter(formatter)
        handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", self.log_file_path))

        if self.parallel:
            self.logger = multiprocessing.get_logger()
            self.logger.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
            self.log_stream_handler.setLevel(logging.WARNING)
            self.logger.addHandler(self.log_stream_handler)
            self.logger.info(f'Global settings read - simulating {len(self.scenario_names)} scenarios in parallel mode')
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
            self.log_stream_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(self.log_stream_handler)
            self.logger.info(f'Global settings read - simulating {len(self.scenario_names)} scenarios in sequential mode')

        handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", self.log_file_path))
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def end_run(self):

        if self.save_results:
            xl.writexl(db=self.result_xdb, fn=self.result_file_path)
            self.logger.info("Excel output file created")

        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 1)
        self.logger.info(f'Total runtime for all scenarios: {str(self.runtime_len)} s')


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


def simulate_scenario(name: str, run: SimulationRun):  # needs to be a function for starpool - multiprocessing

    scenario = Scenario(name, run)  # Create scenario instance & read data from Excel sheet.

    for horizon_index in range(scenario.horizon_num):  # Inner optimization loop over all prediction horizons
        horizon = PredictionHorizon(horizon_index, scenario, run)
        horizon.run_optimization(scenario, run)

        if scenario.exception:
            scenario.save_exception(run)
            break
        else:
            horizon.get_results(scenario, run)
            if run.save_results or run.print_results:
                scenario.accumulate_results()
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

    scenario.end_timing(run)


def xread(param, sheet, db):
    """
    Reading parameters from external excel file
    """
    value = None
    try:
        value = db.ws(ws=sheet).keyrow(key=param, keyindex=1)[1]
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
        with multiprocessing.Pool() as pool:
            pool.starmap(simulate_scenario, zip(run.scenario_names, repeat(run)))
    else:
        for scenario_name in run.scenario_names:
            simulate_scenario(scenario_name, run)

    run.end_run()
