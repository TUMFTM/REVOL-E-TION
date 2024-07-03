import math
import multiprocessing as mp
import os
import time

import numpy_financial as npf
import pandas as pd
import pytz
import timezonefinder
from plotly.subplots import make_subplots

import blocks
import commodity_des as des
import tum_colors as col
from aprioripowerscheduler import AprioriPowerScheduler


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
            setattr(self, key, value)  # this sets all the parameters defined in the csv file

        # noinspection PyUnresolvedReferences
        self.currency = self.currency.upper()

        self.tzfinder = timezonefinder.TimezoneFinder()
        self.timezone = pytz.timezone(self.tzfinder.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        # convert to datetime and calculate time(delta) values
        # simulation and project timeframe start simultaneously
        self.starttime = pd.to_datetime(self.starttime, format='%d.%m.%Y').tz_localize(self.timezone)
        self.sim_duration = pd.Timedelta(days=self.sim_duration)
        self.sim_endtime = self.starttime + self.sim_duration
        self.prj_duration_yrs = self.prj_duration
        self.prj_duration = pd.Timedelta(days=self.prj_duration * 365)  # todo: no leap years accounted for
        self.prj_endtime = self.starttime + self.prj_duration

        # generate a datetimeindex for the energy system model to run on
        self.dti_sim = pd.date_range(start=self.starttime, end=self.sim_endtime, freq=self.timestep, inclusive='left')

        # generate variables for calculations
        self.timestep_hours = self.dti_sim.freq.nanos / 1e9 / 3600
        self.timestep_td = pd.Timedelta(hours=self.timestep_hours)
        self.sim_yr_rat = self.sim_duration.days / 365  # no leap years
        self.sim_prj_rat = self.sim_duration.days / self.prj_duration.days

        # prepare for dispatch plot saving later on
        self.plot_file_path = os.path.join(run.path_result_folder, f'{run.runtimestamp}_'
                                                                   f'{run.scenario_file_name}_'
                                                                   f'{self.name}.html')

        # prepare for cumulative result saving later on
        self.result_summary = pd.DataFrame(columns=['Block', 'Key', self.name])
        self.result_summary = self.result_summary.set_index(['Block', 'Key'])
        self.path_result_summary_tempfile = os.path.join(run.path_result_folder, f'{self.name}_tempresults.csv')

        self.result_timeseries = pd.DataFrame(index=self.dti_sim)
        self.path_result_file = os.path.join(run.path_result_folder, f'{run.runtimestamp}_{self.name}_results.csv')

        self.exception = None  # placeholder for possible infeasibility

        if self.strategy == 'rh':
            self.len_ph = pd.Timedelta(hours=self.len_ph)
            self.len_ch = pd.Timedelta(hours=self.len_ch)
            # number of timesteps for PH
            self.ph_nsteps = math.ceil(self.len_ph.total_seconds() / 3600 / self.timestep_hours)
            # number of timesteps for CH
            self.ch_nsteps = math.ceil(self.len_ch.total_seconds() / 3600 / self.timestep_hours)
            self.nhorizons = int(self.sim_duration // self.len_ch)  # number of timeslices to run
        elif self.strategy in ['go', 'lfs', 'rl']:
            self.len_ph = self.sim_duration
            self.len_ch = self.sim_duration
            self.nhorizons = 1

        # Energy System Blocks --------------------------------

        self.components = []  # placeholder
        self.equal_variables = []

        # create all block objects defined in the scenario DataFrame under "scenario/blocks" as a dict
        self.blocks = self.create_block_objects(self.blocks, run)
        self.commodity_systems = {block.name: block for block in self.blocks.values() if isinstance(block, blocks.CommoditySystem)}

        # Execute commodity system discrete event simulation
        # can only be started after all blocks have been initialized, as the different systems depend on each other.
        if any([cs.filename == 'run_des' for cs in self.commodity_systems.values()]):
            des.execute_des(self, run.save_des_results, run.path_result_folder)

        for cs in [cs for cs in self.commodity_systems.values() if cs.filename == 'run_des']:
            for commodity in cs.commodities.values():
                commodity.data = cs.data.loc[:, (commodity.name, slice(None))].droplevel(0, axis=1)

        # check input parameter configuration of rulebased charging for validity
        if cs_unlim := [cs for cs in self.commodity_systems.values() if cs.int_lvl in [x for x in cs.apriori_lvls if
                                                                               x != 'uc'] and not cs.lm_static]:
            if [block for block in self.blocks.values() if getattr(block, 'opt', False)]:
                run.logger.error(f'Scenario {self.name} - Rulebased charging except for uncoordinated charging (uc)'
                                 f' without static load management (lm_static) is not compatible with size optimization')
                exit()  # TODO exit scenario instead of run
            if [block for block in self.blocks.values() if isinstance(block, blocks.StationaryEnergyStorage)]:
                run.logger.error(f'Scenario {self.name} - Rulebased charging except for uncoordinated charging (uc)'
                                 f' without static load management (lm_static) is not implemented for systems with'
                                 f' stationary energy storage')
                exit()  # TODO exit scenario instead of run
            if len(set([cs.int_lvl for cs in cs_unlim])) > 1:
                run.logger.error(f'Scenario {self.name} - All rulebased CommoditySystems with dynamic load management'
                                 f' have to follow the same strategy. Different strategies are not possible')
                exit()  # TODO exit scenario instead of run
            if cs_unlim[0].int_lvl == 'equal' and len(set([cs.bus_connected for cs in cs_unlim])) > 1:
                run.logger.error(f'Scenario {self.name} - If strategy "equal" is chosen for CommoditySystems with'
                                 f' dynamic load management, all CommoditySystems with dynamic load management have to'
                                 f' be connected to the same bus')
                exit()  # TODO exit scenario instead of run

        self.scheduler = None
        if any([cs for cs in self.commodity_systems.values() if cs.int_lvl in cs.apriori_lvls]):
            self.scheduler = AprioriPowerScheduler(run=run, scenario=self)

        # Result variables --------------------------------
        self.figure = None  # placeholder for plotting

        self.cashflows = pd.DataFrame()

        # Result variables - Energy
        self.e_sim_del = self.e_yrl_del = self.e_prj_del = self.e_dis_del = 0
        self.e_sim_pro = self.e_yrl_pro = self.e_prj_pro = self.e_dis_pro = 0
        self.e_sim_ext = self.e_yrl_ext = self.e_prj_ext = self.e_dis_ext = 0  # external charging
        self.e_eta = 0
        self.renewable_curtailment = self.e_renewable_act = self.e_renewable_pot = self.e_renewable_curt = 0

        # Result variables - Cost
        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = 0
        self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = 0
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = 0
        self.opex_sim_ext = self.opex_yrl_ext = self.opex_prj_ext = self.opex_dis_ext = self.opex_ann_ext = 0
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = 0
        self.crev_sim = self.crev_yrl = self.crev_prj = self.crev_dis = 0
        self.lcoe = self.lcoe_dis = 0

        run.logger.debug(f'Scenario \"{self.name}\" initialization completed')

    def calc_meta_results(self, run):

        # TODO implement commodity v2mg usage share
        # TODO implement energy storage usage share

        #self.e_eta = None
        if self.e_sim_pro == 0:
            run.logger.warning(f'Scenario {self.name} - core efficiency calculation: division by zero')
        else:
            try:
                self.e_eta = self.e_sim_del / self.e_sim_pro
            except ZeroDivisionError:
                run.logger.warning(f'Scenario \"{self.name}\" - core efficiency calculation: division by zero')

        #self.renewable_curtailment = None
        if self.e_renewable_pot == 0:
            run.logger.warning(f'Scenario \"{self.name}\" - renewable curtailment calculation: division by zero')
        else:
            try:
                self.renewable_curtailment = self.e_renewable_curt / self.e_renewable_pot
            except ZeroDivisionError:
                run.logger.warning(f'Scenario \"{self.name}\" - renewable curtailment calculation: division by zero')

        #self.renewable_share = None
        if self.e_sim_pro == 0:
            run.logger.warning(f'Scenario \"{self.name}\" - renewable share calculation: division by zero')
        else:
            try:
                self.renewable_share = self.e_renewable_act / self.e_sim_pro
            except ZeroDivisionError:
                run.logger.warning(f'Scenario \"{self.name}\" - renewable share calculation: division by zero')

        totex_dis_cs = sum([cs.totex_dis for cs in self.commodity_systems.values()])
        if self.e_dis_del == 0:
            run.logger.warning(f'Scenario \"{self.name}\" - LCOE calculation: division by zero')
        else:
            try:
                self.lcoe = self.totex_dis / self.e_dis_del
                self.lcoe_wocs = (self.totex_dis - totex_dis_cs) / self.e_dis_del
            except ZeroDivisionError:
                self.lcoe = self.lcoe_wocs = None
                run.logger.warning(f'Scenario \"{self.name}\" - LCOE calculation: division by zero')

        self.npv = self.crev_dis - self.totex_dis
        self.irr = npf.irr(self.cashflows.sum(axis=1).to_numpy())
        self.mirr = npf.mirr(self.cashflows.sum(axis=1).to_numpy(), self.wacc, self.wacc)

        # print basic results
        run.logger.info(f'Scenario \"{self.name}\" -'
                        f' NPC {round(self.totex_dis) if self.totex_dis else "-":,} {self.currency} -'
                        f' NPV {round(self.npv) if self.npv else "-":,} {self.currency} -'
                        f' LCOE {round(self.lcoe_wocs * 1e5, 1) if self.lcoe_wocs else "-"} {self.currency}-ct/kWh -'
                        f' mIRR {round(self.mirr * 100, 1) if self.mirr else "-"} % -'
                        f' Renewable Share: {round(self.renewable_share * 100, 1) if self.renewable_share else "-"} % -'
                        f' Renewable Curtailment: {round(self.renewable_curtailment * 100, 1) if self.renewable_curtailment else "-"} %')

    def create_block_objects(self, class_dict, run):
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

        if self.strategy in ['go', 'rl']:
            self.figure.update_layout(title=f'Global Optimum Results - '
                                            f'{run.scenario_file_name} - '
                                            f'Scenario: {self.name}')
        if self.strategy == 'rh':
            self.figure.update_layout(title=f'Rolling Horizon Results - '
                                            f'{run.scenario_file_name} - '
                                            f'Scenario: {self.name} - '
                                            f'PH: {self.len_ph}h - '
                                            f'CH: {self.len_ch}h')

    def get_results(self, run):
        for block in self.blocks.values():
            block.calc_energy(self)
            block.calc_expenses(self)
            block.calc_revenue(self)
            block.calc_cashflows(self)

    def print_results(self, run):
        print('#################')
        run.logger.info(f'Results for Scenario \"{self.name}\":')
        for block in [block for block in self.blocks.values() if hasattr(block, 'opt') and block.opt]:
            unit = 'kWh' if isinstance(block, (blocks.CommoditySystem, blocks.StationaryEnergyStorage)) else 'kW'
            if isinstance(block, blocks.SystemCore):
                if block.opt_acdc:
                    run.logger.info(f'Optimized size of AC/DC power in component \"{block.name}\": {round(block.size_acdc / 1e3)} {unit}')
                if block.opt_dcac:
                    run.logger.info(f'Optimized size of DC/AC power in component \"{block.name}\": {round(block.size_dcac / 1e3)} {unit}')
            elif isinstance(block, blocks.GridConnection):
                if block.opt_g2mg:
                    run.logger.info(f'Optimized size of g2mg power in component \"{block.name}\": {round(block.size_g2mg / 1e3)} {unit}')
                if block.opt_mg2g:
                    run.logger.info(f'Optimized size of mg2g power in component \"{block.name}\": {round(block.size_mg2g / 1e3)} {unit}')
            elif isinstance(block, blocks.CommoditySystem):
                for commodity in block.commodities.values():
                    run.logger.info(f'Optimized size of commodity \"{commodity.name}\" in component \"{block.name}\": {round(commodity.size / 1e3, 1)} {unit}')
            else:
                run.logger.info(f'Optimized size of component \"{block.name}\": {round(block.size / 1e3)} {unit}')
        # ToDo: state that these results are internal costs of minigrid only neglecting costs for external charging
        run.logger.info(f'Total simulated cost: {str(round(self.totex_sim / 1e6, 2))} million {self.currency}')
        run.logger.info(f'Levelized cost of electricity: {str(round(1e5 * self.lcoe_dis, 2)) if self.lcoe_dis else "-"} {self.currency}-ct/kWh')
        print('#################')

    def save_plots(self):
        self.figure.write_html(self.plot_file_path)

    def save_result_summary(self, run):
        """
        Saves all int, float and str attributes of run, scenario (incl. technoeconomic KPIs) and all blocks to the
        results dataframe
        :param run: SimulationRun
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
        result_blocks = {'run': run, 'scenario': self}
        result_blocks.update(self.blocks)

        for name, block in result_blocks.items():
            write_values(name, block)
            if isinstance(block, blocks.CommoditySystem):
                for name, commodity in block.commodities.items():
                    write_values(name, commodity)

        self.result_summary.reset_index(inplace=True, names=['block', 'key'])
        self.result_summary.to_csv(self.path_result_summary_tempfile, index=False)

    def save_result_timeseries(self):
        for block in self.blocks.values():
            block.get_timeseries_results(self)
        #self.result_timeseries.to_pickle(self.path_result_file.replace('.csv', '.pkl'))
        self.result_timeseries.to_csv(self.path_result_file)

    def show_plots(self):
        self.figure.show(renderer='browser')
