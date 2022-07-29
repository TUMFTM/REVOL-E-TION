"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Department of Mobility Systems Engineering
School of Engineering and Design
Technical University of Munich
philipp.rosner@tum.de
September 2nd, 2021

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022

--- Detailed Description ---
This script is the main model generator and optimizer for the toolset.
Its results are output to files and key ones printed to the terminal.
Visualization is done via different scripts (to be done)
For further information, see readme

--- Input & Output ---
The model requires an Excel input file for scenario definition.
Additionally, several .csv-files for timeseries data are required.
For further information, see readme.

--- Requirements ---
For package requirements, see requirements.txt
For all other requirements, see readme.

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Imports
###############################################################################

from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
# import multiprocessing as mprcs  # TODO parallelize scenario loop
import oemof.solph as solph
import os
import pandas as pd
import pylightxl as xl
import PySimpleGUI as psg
import time

import economics as eco
import preprocessing as pre
import postprocessing as post


###############################################################################
# Class definitions
###############################################################################


class InvestComponent:

    def __init__(self):

        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation

    def concatenate_results(self):
        pass

    def calc_adj_capex(self, scenario):
        self.adj_capex = eco.adj_ce(self.spec_capex,
                                    self.spec_mntex,
                                    self.lifespan,
                                    scenario.wacc)  # adjusted ce (including maintenance) of the component in $/W

    def calc_eq_pr_cost(self, scenario):
        self.eq_pres_cost = eco.ann_recur(self.adj_capex,
                                          self.lifespan,
                                          scenario.prj_duration_yrs,
                                          scenario.wacc,
                                          self.cost_decr)

    def xread_params(self, scenario, run):
        self.spec_capex = xread(self.name + '_sce', scenario.name, run.xdb)
        self.spec_mntex = xread(self.name + '_sme', scenario.name, run.xdb)
        self.spec_opex = xread(self.name + '_soe', scenario.name, run.xdb)

        self.lifespan = xread(self.name + '_ls', scenario.name, run.xdb)
        self.cost_decr = xread(self.name + '_cdc', scenario.name, run.xdb)

        if not scenario.cs_opt[self.name]: self.size = xread('wind_cs', scenario.name, run.xdb)

        self.calc_adj_capex(scenario)
        self.calc_eq_pr_cost(scenario)


class Run:

    def __init__(self):
        self.scenarios_file, self.result_path = self.input_gui()
        self.xdb = xl.readxl(fn=self.scenarios_file)  # Excel database of selected file
        self.scenario_names = self.xdb.ws_names  # Get list of sheet names, 1 sheet is 1 scenario

        try:
            self.scenario_names.remove('global_settings')
        except ValueError:
            print("Excel File does not include global settings - exiting")
            exit()

        self.runtimestart = time.time()
        self.runtimestamp = datetime.now().strftime("%y%m%d_%H%M%S")  # create str of runtimestart


        self.solver = xread(('solver', 'global_settings', self.xdb))
        self.print_results = xread(('print_results', 'global_settings', self.xdb))
        self.show_plots = xread(('show_plots', 'global_settings', self.xdb))
        self.save_plots = xread(('save_plots', 'global_settings', self.xdb))
        self.dump_model = xread(('dump_model', 'global_settings', self.xdb))
        self.solver_debugmode = (xread('solver_debugmode', 'global_settings', self.xdb) == 'True')
        self.eps_cost = xread(('eps_cost', 'global_settings', self.xdb))

        self.input_data_path = os.path.join(os.getcwd(), "input_data")
        self.dump_model_file = os.path.join(os.getcwd(), "lp_models", self.runtimestamp + "_model.lp")
        self.log_file = os.path.join(os.getcwd(), "logfiles", self.runtimestamp + "_log.log")

        logging.define_logging(logfile=self.log_file)
        logging.info("Global settings read - initializing scenarios")


    def input_gui(self):
        """
        GUI to choose excel scenarios file
        """

        scenarios_default = os.path.join(os.getcwd(), "settings")
        results_default = os.path.join(os.getcwd(), "results")

        input_file = [[psg.Text('Choose input settings file')],
                      [psg.Input(), psg.FileBrowse(initial_folder=scenarios_default)],
                      ]

        result_folder = [[psg.Text("Choose result storage folder")],
                         [psg.Input(), psg.FolderBrowse(initial_folder=results_default), ],
                         ]

        layout = [
            [psg.Column(input_file)],
            [psg.HSeparator()],
            [psg.Column(result_folder)],
            [psg.HSeparator()],
            [psg.OK(), psg.Cancel()],
        ]

        event, values = psg.Window('Get settings file', layout).read(close=True)

        try:
            scenarios_filename = os.path.normpath(values['Browse'])
            results_foldername = os.path.normpath(values['Browse0'])
        except TypeError:
            print("WARNING: GUI window closed manually - exiting")
            exit()

        if scenarios_filename == "." or results_foldername == ".":
            print("WARNING: not all required paths entered - exiting")
            exit()

        return scenarios_filename, results_foldername


class Scenario:

    def __init__(self, run):

        logging.info(f"Scenario {scenario_index} of {len(run.scenario_names)} started - {scenario_name}")

        # General Information --------------------------------
        self.name = scenario_name

        self.prj_starttime = datetime.strptime(xread('proj_start', self.name, run.xdb), '%Y/%m/%d')
        self.prj_duration = relativedelta(years=xread('prj_duration', self.name, run.xdb))
        self.prj_endtime = self.prj_starttime + self.prj_duration

        self.sim_starttime = self.prj_starttime  # simulation timeframe is at beginning of project timeframe
        self.sim_timestep = xread('sim_timestep', self.name, run.xdb)
        self.sim_duration = relativedelta(days=xread('sim_duration', self.name, run.xdb))
        self.sim_endtime = self.sim_starttime + self.sim_duration_days
        self.sim_dti = pd.date_range(start=self.sim_starttime, end=self.sim_endtime, freq=self.sim_timestep).delete(-1)

        self.sim_yrrat = self.sim_duration.days / 365.25
        self.sim_prjrat = self.sim_duration.hours / self.prj_duration.hours

        self.wacc = xread('wacc', self.name, run.xdb)

        # Operational strategy --------------------------------
        self.strategy = xread('sim_os', self.name, run.xdb)

        if self.strategy == 'rh':
            self.ph_len = xread('rh_ph', self.name, run.xdb)
            self.ch_len = xread('rh_ch', self.name, run.xdb)
            self.ph_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ph_len  # number of timesteps for predicted horizon
            self.ch_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ch_len  # number of timesteps for control horizon
            self.horizon_num = int(len(sim['dti']) / sim['ch_len'])  # number of CH timeslices for simulated date range
        elif self.strategy == 'go':
            self.ph_len = self.sim_duration
            self.ch_len = self.sim_duration
            self.horizon_num = 1

        self.horizons = [PredictionHorizon(i, scenario) for i in range(self.horizon_num)]

        # Components --------------------------------
        self.components = dict(dem=(xread('dem_enable', self.name, run.xdb) == 'True'),
                               wind=(xread('wind_enable', sim['sheet'], sim['settings_file']) == 'True'),
                               pv=(xread('pv_enable', sim['sheet'], sim['settings_file']) == 'True'),
                               gen=(xread('gen_enable', sim['sheet'], sim['settings_file']) == 'True'),
                               ess=(xread('ess_enable', sim['sheet'], sim['settings_file']) == 'True'),
                               bev=(xread('bev_enable', sim['sheet'], sim['settings_file']) == 'True'))

        self.cs_opt = dict(wind=(xread('wind_enable_cs', sim['sheet'], sim['settings_file']) == 'True'),
                           pv=(xread('pv_enable_cs', sim['sheet'], sim['settings_file']) == 'True'),
                           gen=(xread('gen_enable_cs', sim['sheet'], sim['settings_file']) == 'True'),
                           ess=(xread('ess_enable_cs', sim['sheet'], sim['settings_file']) == 'True'),
                           bev=(xread('bev_enable_cs', sim['sheet'], sim['settings_file']) == 'True'))

        for component in [name for name, enable in self.components.items() if enable]:  # only enabled components
            if component == 'dem': dem = StatSink('dem', scenario, run)
            elif component == 'wind': wind = WindSource('wind', scenario, run)
            elif component == 'pv': pv = PVSource('pv', scenario, run)
            elif component == 'gen': gen = FossilSource('gen', scenario, run)
            elif component == 'bev': bev = CommoditySystem('bev', scenario, run)
            elif component == 'mb': mb = CommoditySystem('mb', scenario, run)

        if True in self.cs_opt.values() and self.strategy != 'go':
            logging.error('Error: Rolling horizon strategy is not feasible if component sizing is active')
            logging.error('Please disable sim_cs in settings file')
            exit()  # TODO switch to next scenario instead of exiting



    def accumulate_results(self):

        self.e_sim_del = 0
        self.e_yrl_del = 0
        self.e_prj_del = 0
        self.e_dis_del = 0

        self.e_sim_pro = 0
        self.e_yrl_pro = 0
        self.e_prj_pro = 0
        self.e_dis_pro = 0

        self.e_eta = 0

        self.init_capex = 0
        self.prj_capex = 0
        self.dis_capex = 0
        self.ann_capex = 0

        self.sim_mntex = 0
        self.prj_mntex = 0
        self.dis_mntex = 0
        self.ann_mntex = 0

        self.sim_opex = 0
        self.prj_opex = 0
        self.dis_opex = 0
        self.ann_opex = 0

        self.sim_totex = 0
        self.prj_totex = 0
        self.dis_totex = 0
        self.ann_totex = 0


class StatSink:

    def __init__(self, name, scenario, run):

        self.name = name
        self.input_file_name = xread('dem_filename', scenario.name, run.xdb)
        self.input_file_path = os.path.join(run.input_data_path,"load_profile_data",self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=",",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)
        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation
        

class WindSource(InvestComponent):
    
    def __init__(self, name, scenario, run):
        
        self.name = name

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.xdb)
        self.input_file_path = os.path.join(run.input_data_path,self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=",",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.transformer_eff = xread(self.name + '_eff', scenario.name, run.xdb)
        
        self.xread_params(scenario, run)


class PVSource(InvestComponent):

    def __init__(self, name, scenario, run):

        self.name = name

        if run.pv_source == 'api':  # API input selected
             pass # TODO: API input goes here
        else:  # data input from fixed csv file
            self.input_file_name = xread(self.name + '_filename', scenario.name, run.xdb)
            self.input_file_path = os.path.join(run.input_data_path, self.input_file_name + ".csv")
            self.data = pd.read_csv(self.input_file_path,
                                    sep=",",
                                    header=10,
                                    skip_blank_lines=False,
                                    skipfooter=13,
                                    engine='python')

        self.data['time'] = pd.to_datetime(self.data['time'],
                                           format='%Y%m%d:%H%M').dt.round('H')  # for direct PVGIS input

        self.data['P'] = self.data['P'] / 1e3  # data is in W for a 1kWp PV array -> convert to specific power

        self.transformer_eff = xread(self.name + '_eff', scenario.name, run.xdb)

        self.xread_params(scenario, run)


class FossilSource(InvestComponent):

    def __init__(self, name, scenario, run):

        self.name = name
        self.xread_params(scenario, run)


class EnergyStorage(InvestComponent):

    def __init__(self, name, scenario, run):

        self.name = name

        self.xread_params(scenario, run)

        self.chg_eff = xread(self.name + '_chg_eff', scenario.name, run.xdb)
        self.dis_eff = xread(self.name + '_dis_eff', scenario.name, run.xdb)
        self.chg_crate = xread(self.name + '_chg_crate', scenario.name, run.xdb)
        self.dis_crate = xread(self.name + '_dis_crate', scenario.name, run.xdb)

        self.init_soc = xread(self.name + '_init_soc', scenario.name, run.xdb)
        self.ph_init_soc = self.init_soc  # TODO actually necessary?
        self.bal = False  # ESS SOC at end of prediction horizon must not be forced equal to initial SOC

        self.flow_out = self.flow_bal = self.soc = pd.Series(dtype='float64')  # empty dataframes for result concat


class CommoditySystem(InvestComponent):

    def __init__(self, name, scenario, run):

        self.name = name

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.xdb)
        self.input_file_path = os.path.join(run.result_path, self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=";",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.xread_params(scenario, run)

        self.item_num = xread(self.name + '_num', scenario.name, run.xdb)  # TODO: integrate multiple vehicle classes
        self.agr_sim = xread(self.name + '_agr', scenario.name, run.xdb)

        self.chg_pwr = xread(self.name + '_chg_pwr', scenario.name, run.xdb)
        self.dis_pwr = xread(self.name + '_dis_pwr', scenario.name, run.xdb)
        self.chg_eff = xread(self.name + '_charge_eff', scenario.name, run.xdb)
        self.dis_eff = xread(self.name + '_discharge_eff', scenario.name, run.xdb)

        self.int_lvl = xread(self.name + '_int_lvl', scenario.name, run.xdb)  # charging integration level

        self.items = [MobileCommodity(self.name + str(i)) for i in range(self.item_num)]

        self.flow_out = self.flow_bal = self.soc = pd.Series(dtype='float64')  # empty dataframes for result concat


class MobileCommodity:

    def __init__(self, name):

        self.name = name
        self.system_name = self.name[:-1]  # remove index to get CommoditySystem name
        self.init_soc = xread(self.system_name + '_init_soc', scenario.name, run.xdb)  # TODO: Don't we want to define this at random?
        self.ph_init_soc = self.init_soc  # TODO: actually necessary?
        self.flow_out = self.flow_bal = self.soc = pd.Series(dtype='float64')  # empty dataframes for result concat


class PredictionHorizon:

    def __init__(self, index, scenario):

        self.starttime = scenario.sim_starttime + (index * scenario.ch_len)  # calc all start times
        self.ch_endtime = self.starttime + scenario.ch_len
        self.ph_endtime = self.starttime + scenario.ph_len
        self.timestep = scenario.sim_timestep

        self.ph_dti = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=scenario.sim_timestep).delete(-1)
        self.ch_dti = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=scenario.sim_timestep).delete(-1)

        for component in [name for name, enable in scenario.components.items() if enable]:  # only enabled components
            cond = data['time'].isin(dti)  # create boolean series marking indices within current PH
            sliced_data = data.loc[cond]  # select correct data slice
            sliced_data = sliced_data.reset_index(drop=True)  # reset data index to start from 0


    def build_esm(self):

        self.es = solph.EnergySystem(timeindex=sim['ph_dti'])
        pass



###############################################################################
# Class-free function definitions
###############################################################################


def xread(param_name, sheet, db):
    """
    Reading parameters from external excel file
    """
    value = db.ws(ws=sheet).keyrow(key=param_name, keyindex=1)[1]
    return value


##########################################################################
# Actual Simulation
##########################################################################
run = Run()  # get all global information about the run

for scenario_index, scenario_name in enumerate(run.scenario_names):
    try:

        scenario = Scenario(run)  # Create scenario instance & read data from excel sheet

        for horizon in scenario.horizons:  # Optimization loop over all prediction horizons
            horizon.build_esm()  # EnergySystemModel instance needs to be built for every horizon
            horizon.om.solve(solver=run.solver, solve_kwargs={"tee": run.solver_debugmode})
            horizon.get_results()

        dem, wind, pv, gen, ess, bev, cres = post.acc_energy(sim, prj, dem, wind, pv, gen, ess, bev, cres)
        wind, pv, gen, ess, bev, cres = post.acc_eco(sim, prj, wind, pv, gen, ess, bev, cres)

        logging.info("Displaying key results")
        post.print_results(sim, wind, pv, gen, ess, bev, cres)

        sim = post.end_timing(sim)

        post.plot_results(sim, dem, wind, pv, gen, ess, bev)
        post.save_results(sim, dem, wind, pv, gen, ess, bev, cres)

    except:
        logging.warning('Model optimization infeasible - continuing on next worksheet')
        post.save_results_err(sim)
        continue