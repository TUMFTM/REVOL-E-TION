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
from custom_constraints import CustomConstraints


class Scenario:

    def __init__(self, scenario_name, run, logger):

        self.name = scenario_name
        self.run = run
        self.logger = logger
        self.logger.propagate = False
        pass

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

        self.currency = self.currency.upper()

        self.tzfinder = timezonefinder.TimezoneFinder()
        self.timezone = pytz.timezone(self.tzfinder.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        # convert to datetime and calculate time(delta) values
        # simulation and project timeframe start simultaneously
        self.starttime = pd.to_datetime(self.starttime, format='%d.%m.%Y').tz_localize(self.timezone)
        self.sim_duration = pd.Timedelta(days=self.sim_duration)
        self.sim_endtime = self.starttime + self.sim_duration
        self.prj_duration_yrs = self.prj_duration
        self.prj_endtime = self.starttime + pd.DateOffset(years=self.prj_duration)
        self.prj_duration = self.prj_endtime - self.starttime

        # generate a datetimeindex for the energy system model to run on
        self.dti_sim = pd.date_range(start=self.starttime, end=self.sim_endtime, freq=self.timestep, inclusive='left')

        # generate variables for calculations
        self.timestep_td = pd.Timedelta(self.dti_sim.freq)
        self.timestep_hours = self.timestep_td.total_seconds() / 3600
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
        self.constraints = CustomConstraints(scenario=self)

        # create all block objects defined in the scenario DataFrame under "scenario/blocks" as a dict
        self.blocks = self.create_block_objects(self.blocks, run)
        self.commodity_systems = {block.name: block for block in self.blocks.values()
                                  if isinstance(block, blocks.CommoditySystem)}

        # Execute commodity system discrete event simulation
        # can only be started after all blocks have been initialized, as the different systems depend on each other.
        if any([cs.filename == 'run_des' for cs in self.commodity_systems.values()]):
            des.execute_des(self, run.save_des_results, run.path_result_folder)

        for cs in [cs for cs in self.commodity_systems.values() if cs.filename == 'run_des']:
            for commodity in cs.commodities.values():
                commodity.data = cs.data.loc[:, (commodity.name, slice(None))].droplevel(0, axis=1)

        # ToDo: put into extra function
        # check input parameter configuration of rulebased charging for validity
        if self.strategy == 'rl':
            pass
        elif cs_unlim := [cs for cs in self.commodity_systems.values() if cs.int_lvl in [x for x in self.run.apriori_lvls if
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
        if self.strategy == 'rl':
            pass
        elif any([cs for cs in self.commodity_systems.values() if cs.int_lvl in self.run.apriori_lvls]):
            self.scheduler = AprioriPowerScheduler(scenario=self)

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

        if self.strategy in ['go', 'rl']:
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
                        self.logger.info(f'Optimized peak power in component \"{block.name}\" for interval'
                                         f' {interval}: {block.peakshaving_ints.loc[interval, "power"] / 1e3:.2f} {unit}')
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
                    self.result_summary.loc[(block_name, f'peakshaving_peak_power_{interval}'), self.name] = float(block_obj.peakshaving_ints.loc[interval, 'power'])
                    self.result_summary.loc[(block_name, f'peakshaving_opex_spec_{interval}'), self.name] = float(block_obj.peakshaving_ints.loc[interval, 'opex_spec'])

        self.result_summary.reset_index(inplace=True, names=['block', 'key'])
        self.result_summary.to_csv(self.path_result_summary_tempfile, index=False)

    def save_result_timeseries(self):
        for block in self.blocks.values():
            block.get_timeseries_results(self)
        self.result_timeseries.to_csv(self.path_result_file)

    def show_plots(self):
        self.figure.show(renderer='browser')


