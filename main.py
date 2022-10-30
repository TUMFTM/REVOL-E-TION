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
import multiprocessing

import oemof.solph as solph
import os
import pandas as pd
import plotly.graph_objects as go
import pprint
import pylightxl as xl
import PySimpleGUI as psg
import time

from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import repeat
from oemof.tools import logger
from pathlib import Path
from plotly.subplots import make_subplots

import system_blocks as blocks
import tum_colors as col


###############################################################################
# Class definitions
###############################################################################


class PredictionHorizon:

    def __init__(self, index, scenario, run):

        self.index = index

        # Time and data slicing --------------------------------
        self.starttime = scenario.sim_starttime + (index * scenario.ch_len)  # calc all start times
        self.ch_endtime = self.starttime + scenario.ch_len
        self.ph_endtime = self.starttime + scenario.ph_len
        self.timestep = scenario.sim_timestep

        self.ph_dti = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=scenario.sim_timestep).delete(-1)
        self.ch_dti = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=scenario.sim_timestep).delete(-1)

        for component in [component for component in scenario.component_sets if hasattr(component, 'data')]:
            component.ph_data = component.data.loc(component.data['time'].isin(self.ph_dti)).reset_index(drop=True)

        self.results = None
        self.meta_results = None

        # Build energy system model --------------------------------

        self.es = solph.EnergySystem(timeindex=self.ph_dti)  # initialize energy system model instance

        for component_set in scenario.component_sets:
            component_set.update_input_components(scenario, self)  # (re)define solph components that need input slices

        for solph_component in scenario.solph_components:
            self.es.add(solph_component)  # add components to this horizon's energy system

        self.model = solph.Model(self.es)  # Build the mathematical linear optimization model with pyomo

        if run.dump_model:
            if scenario.strategy == 'go':
                self.model.write(run.dump_file_path, io_options={'symbolic_solver_labels': True})
            elif scenario.strategy == 'rh':
                logging.warning('Model file dump not implemented for RH operating strategy - no file created')

    def get_results(self, scenario, horizon, run):
        """
        Get result data slice for current CH from results and save in result dataframes for later analysis
        Get (possibly optimized) component sizes from results to handle outputs more easily
        """

        self.results = solph.processing.results(self.model)  # Get the results of the solved horizon from the solver

        if run.print_results:  # TODO does this need an individual trigger?
            self.meta_results = solph.processing.meta_results(self.model)
            pprint.pprint(self.meta_results)

        for component_set in scenario.component_sets:

            if component_set.opt:
                component_set.size = self.results[(component_set.src, component_set.bus)]['scalars']['invest']
                # TODO check whether this definition fits all component sets

            component_set.get_ch_results(self, horizon, scenario)

    def run_optimization(self, scenario):
        try:
            self.model.solve(solver=run.solver, solve_kwargs={'tee': run.solver_debugmode})
        except KeyError:  # TODO raise own or find proper exception
            logging.warning(f'Scenario {scenario.name} failed or infeasible - continue on next scenario')
            scenario.feasible = False
        finally:
            scenario.handle_error()


class Scenario:

    def __init__(self, name, run):

        # General Information --------------------------------

        self.name = name
        self.runtime_start = time.time()  #TODO use perfcounter
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        logging.info(f'Scenario \"{self.name}\" initialized')  # TODO state process number

        self.prj_starttime = datetime.strptime(xread('prj_start', self.name, run.input_xdb), '%Y/%m/%d')
        self.prj_duration = relativedelta(years=xread('prj_duration', self.name, run.input_xdb))
        self.prj_endtime = self.prj_starttime + self.prj_duration

        self.sim_starttime = self.prj_starttime  # simulation timeframe is at beginning of project timeframe
        self.sim_timestep = xread('sim_timestep', self.name, run.input_xdb)
        self.sim_duration = relativedelta(days=xread('sim_duration', self.name, run.input_xdb))
        self.sim_endtime = self.sim_starttime + self.sim_duration
        self.sim_dti = pd.date_range(start=self.sim_starttime, end=self.sim_endtime, freq=self.sim_timestep).delete(-1)

        self.sim_yr_rat = self.sim_duration.days / 365.25
        # self.sim_prj_rat = self.sim_duration / self.prj_duration  # TODO convert to datetime timedeltas to be divisible

        self.wacc = xread('wacc', self.name, run.input_xdb)

        self.plot_file_path = os.path.join(run.result_path, f'{run.runtimestamp}_'
                                                            f'{run.scenarios_file_name}_'
                                                            f'{self.name}.html')

        # Operational strategy --------------------------------

        self.strategy = xread('sim_os', self.name, run.input_xdb)
        self.feasible = None  # trigger for infeasible conditions

        if self.strategy == 'rh':
            self.ph_len = relativedelta(hours=xread('rh_ph', self.name, run.input_xdb))
            self.ch_len = relativedelta(days=xread('rh_ch', self.name, run.input_xdb))
            self.ph_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ph_len  # number of timesteps for PH
            self.ch_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ch_len  # number of timesteps for CH
            self.horizon_num = int(self.sim_duration.hours / self.ch_len.hours)  # number of timeslices to run
        elif self.strategy == 'go':
            self.ph_len = self.sim_duration
            self.ch_len = self.sim_duration
            self.horizon_num = 1

        # Creation of static core energy system components --------------------------------

        """
        dc_bus              ac_bus
          |                   |
          |---dc_ac---------->|
          |                   |
          |<----------ac_dc---|
        """

        self.solph_components = []

        self.ac_bus = solph.Bus(label='ac_bus')
        self.solph_components.append(self.ac_bus)

        self.dc_bus = solph.Bus(label='dc_bus')
        self.solph_components.append(self.dc_bus)

        self.ac_dc = solph.Transformer(label='ac_dc',
                                       inputs={self.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.dc_bus: solph.Flow()},
                                       conversion_factors={self.dc_bus: xread('ac_dc_eff', self.name, run.input_xdb)})
        self.solph_components.append(self.ac_dc)

        self.dc_ac = solph.Transformer(label='dc_ac',
                                       inputs={self.dc_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.ac_bus: solph.Flow()},
                                       conversion_factors={self.ac_bus: xread('dc_ac_eff', self.name, run.input_xdb)})
        self.solph_components.append(self.dc_ac)

        # Other Component Sets --------------------------------

        self.components_enable = dict(dem=(xread('dem_enable', self.name, run.input_xdb) == 'True'),
                                      wind=(xread('wind_enable', self.name, run.input_xdb) == 'True'),
                                      pv=(xread('pv_enable', self.name, run.input_xdb) == 'True'),
                                      gen=(xread('gen_enable', self.name, run.input_xdb) == 'True'),
                                      ess=(xread('ess_enable', self.name, run.input_xdb) == 'True'),
                                      bev=(xread('bev_enable', self.name, run.input_xdb) == 'True'))

        self.component_sets = []  # Todo rename to blocks

        for component_name in [name for name, enable in self.components_enable.items() if enable]:
            if component_name == 'dem':
                dem = blocks.StatSink('dem', self, run)
                self.component_sets.append(dem)
            elif component_name == 'wind':
                wind = blocks.WindSource('wind', self, run)
                self.component_sets.append(wind)
            elif component_name == 'pv':
                pv = blocks.PVSource('pv', self, run)
                self.component_sets.append(pv)
            elif component_name == 'gen':
                gen = blocks.ControllableSource('gen', self, run)
                self.component_sets.append(gen)
            elif component_name == 'bev':
                bev = blocks.CommoditySystem('bev', self, run)
                self.component_sets.append(bev)
            elif component_name == 'mb':
                mb = blocks.CommoditySystem('mb', self, run)
                self.component_sets.append(mb)

        self.figure = None  # figure placeholder for result plotting

        # Result variables --------------------------------

        self.e_sim_del = 0
        self.e_yrl_del = 0
        self.e_prj_del = 0
        self.e_dis_del = 0

        self.e_sim_pro = 0
        self.e_yrl_pro = 0
        self.e_prj_pro = 0
        self.e_dis_pro = 0

        self.e_eta = None

        self.capex_init = 0
        self.capex_prj = 0
        self.capex_dis = 0
        self.capex_ann = 0

        self.mntex_yrl = 0
        self.mntex_prj = 0
        self.mntex_dis = 0
        self.mntex_ann = 0

        self.opex_sim = 0
        self.opex_yrl = 0
        self.opex_prj = 0
        self.opex_dis = 0
        self.opex_ann = 0

        self.totex_sim = 0
        self.totex_prj = 0
        self.totex_dis = 0
        self.totex_ann = 0

        self.lcoe = None
        self.lcoe_dis = None

    def accumulate_results(self):

        for component in self.component_sets:
            component.accumulate_results(self)

        #  TODO find a metric for curtailed energy and calculate

        try:
            self.e_eta = self.e_sim_del / self.e_sim_pro
        except ZeroDivisionError:
            self.e_eta = -1

        try:
            self.lcoe = self.totex_dis / self.e_prj_del
            self.lcoe_dis = self.totex_dis / self.e_dis_del
        except ZeroDivisionError:
            self.lcoe = -1
            self.lcoe_dis = -1

    def end_timing(self):

        self.feasible = True  # model optimization or simulation seems to have been successful

        self.runtime_end = time.time()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 2)
        logging.info(f'Scenario \"{self.name}\" finished - runtime {self.runtime_len}')

    def generate_plots(self, run):

        self.figure = make_subplots(specs=[[{'secondary_y': True}]])

        for component_set in self.component_sets:
            self.figure.add_trace(go.Scatter(x=component_set.flow.index.to_pydatetime(),
                                             y=component_set.flow,  # TODO invert for sink components
                                             mode='lines',
                                             name=component_set.name,   # TODO print sizing in plot
                                             line=dict(width=2, dash=None)),  # TODO introduce TUM colors
                                  secondary_y=False)

            if isinstance(component_set, blocks.StationaryEnergyStorage):
                self.figure.add_trace(go.Scatter(x=component_set.soc.index.to_pydatetime(),
                                                 y=component_set.soc,
                                                 mode='lines',
                                                 name=component_set.name,  # TODO print sizing in plot
                                                 line=dict(width=2, dash=None)),  # TODO introduce TUM colors
                                      secondary_y=True)

            if isinstance(component_set, blocks.CommoditySystem):
                for commodity in component_set.commodities:
                    self.figure.add_trace(go.Scatter(x=commodity.soc.index.to_pydatetime(),
                                                     y=commodity.soc,
                                                     mode='lines',
                                                     name=commodity.name,    # TODO print sizing in plot, denote whether single or combined value
                                                     line=dict(width=2, dash=None)),  # TODO introduce TUM colors
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

    def handle_error(self, run):

        """
        Dump error message in result excel file if optimization did not succeed
        """

        if run.save_results:
            # logging.warning("Error occurred, save scenario data")
            #
            # results_filepath = os.path.join(sim['resultpath'], 'results_' + os.path.basename(sim['settings_file']))
            #
            # if sim['run'] == 0:
            #     db = xl.Database()  # create a blank db
            # else:
            #     db = xl.readxl(fn=results_filepath)
            #
            # # add a blank worksheet to the db
            # db.add_ws(ws=sim['sheet'])
            #
            # # header of ws
            # if sim['op_strat'] == 'go':
            #     ws_title = 'Global Optimum Results (' + sim['settings_file'] + ' - Sheet: ' + sim['sheet'] + ')'
            # if sim['op_strat'] == 'rh':
            #     ws_title = 'Rolling Horizon Results (' + sim['settings_file'] + ' - Sheet: ' + sim[
            #         'sheet'] + ', ' + str(
            #         sim['rh_ph']) + 'h, CH: ' + str(sim['rh_ch']) + 'h)'
            #
            # db.ws(ws=sim['sheet']).update_index(row=1, col=1, val=ws_title)
            #
            # # add sim name
            # db.ws(ws=sim['sheet']).update_index(row=3, col=1, val='Logfile:')
            # db.ws(ws=sim['sheet']).update_index(row=3, col=2, val=sim['name'])
            #
            # # write error message
            # db.ws(ws=sim['sheet']).update_index(row=5, col=1,
            #                                     val='ERROR - Optimization could NOT succeed for these simulation settings')
            #
            # # write out the db
            # xl.writexl(db=db, fn=results_filepath)
            pass  # TODO write to excel and continue with next in scenario for loop

    def print_results(self):
        print('#####Results#####')
        print(f'Total simulated cost: {str(round(self.totex_sim / 1e6, 2))} million USD')
        print(f'Levelized cost of electricity: {str(round(1e5 * self.lcoe_dis, 3))} USct/kWh')
        print('#################')

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
            ws_title = None

        run.result_xdb.ws(ws=self.name).update_index(row=1, col=1, val=ws_title)
        run.result_xdb.ws(ws=self.name).update_index(row=2, col=1, val='Timestamp')
        run.result_xdb.ws(ws=self.name).update_index(row=2, col=2, val=run.runtimestamp)
        run.result_xdb.ws(ws=self.name).update_index(row=3, col=1, val='Runtime')
        run.result_xdb.ws(ws=self.name).update_index(row=3, col=2, val=self.runtime_len)

        header_row = 5
        for index, component_set in enumerate(self.component_sets):  # TODO scenario integration
            col_id = 1 + index * 4
            row_id = header_row + 1
            run.result_xdb.ws(ws=self.name).update_index(row=header_row, col=col_id, val=component_set.name)
            component_set_dict = component_set.__dict__
            component_set_dict.pop('name')
            for key in component_set_dict.keys():
                run.result_xdb.ws(ws=self.name).update_index(row=row_id, col=col_id, val=key)
                run.result_xdb.ws(ws=self.name).update_index(row=row_id, col=col_id + 1, val=component_set_dict[key])
                row_id += 1

        xl.writexl(db=run.result_xdb, fn=run.result_file_path)  # TODO check to not write for every scenario, just once!

    def show_plots(self):
        self.figure.show()


class SimulationRun:

    def __init__(self):

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

        self.runtime_start = time.time()  # TODO better timing method
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

        logger.define_logging(logfile=self.log_file_path)
        logging.info('Global settings read - initializing scenarios')

    def end_run(self):
        self.runtime_end = time.time()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 1)
        logging.info(f'Total runtime {str(self.runtime_len)} s')


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
        [psg.OK(), psg.Cancel()],
    ]

    event, values = psg.Window('MGEV toolset - select input file and result path', layout).read(close=True)

    try:
        scenarios_filename = os.path.abspath(values['file'])
        results_foldername = os.path.abspath(values['folder'])
        if scenarios_filename == '.' or results_foldername == '.':
            logging.warning('not all required paths entered - exiting')
            exit()
        return scenarios_filename, results_foldername
    except TypeError:
        logging.warning('GUI window closed manually - exiting')
        exit()


def simulate_scenario(name: str, run: SimulationRun):

    scenario = Scenario(name, run)  # Create scenario instance & read data from Excel sheet.

    for horizon_index in range(scenario.horizon_num):  # Inner optimization loop over all prediction horizons
        horizon = PredictionHorizon(horizon_index, scenario, run)
        horizon.run_optimization(scenario)
        horizon.get_results(scenario, horizon, run)

    scenario.end_timing()

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


def xread(param, sheet, db):
    """
    Reading parameters from external excel file
    """
    value = db.ws(ws=sheet).keyrow(key=param, keyindex=1)[1]
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
