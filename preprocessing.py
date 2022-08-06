"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
February 2nd, 2022

--- Contributors ---
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022

--- Detailed Description ---
This script defines various functions used by main.py for preprocessing of input data and during the optimization loop

--- Input & Output ---
see individual functions

--- Requirements ---
none

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
from oemof.tools import logger
import oemof.solph as solph
import os
import pandas as pd
import pprint
import pylightxl as xl
import PySimpleGUI as psg
import time

import economics as eco

###############################################################################
# Class definitions
###############################################################################

class InvestComponent:

    def __init__(self, name, scenario, run):

        self.name = name  # to be set in child class

        self.opt = xread(f'{self.name}_opt', scenario.name, run.xdb)

        if self.opt and scenario.strategy != 'go':
            logging.error('Error: Rolling horizon strategy is not feasible if component sizing is active')
            logging.error('Please disable sim_cs in settings file')
            exit()  # TODO switch to next scenario instead of exiting

        if self.opt:
            self.size = None
        else:
            self.size = xread(self.name + '_cs', scenario.name, run.xdb)

        self.spec_capex = xread(self.name + '_sce', scenario.name, run.xdb)
        self.spec_mntex = xread(self.name + '_sme', scenario.name, run.xdb)
        self.spec_opex = xread(self.name + '_soe', scenario.name, run.xdb)

        self.lifespan = xread(self.name + '_ls', scenario.name, run.xdb)
        self.cost_decr = xread(self.name + '_cdc', scenario.name, run.xdb)
        self.transformer_eff = xread(self.name + '_eff', scenario.name, run.xdb)

        self.adj_capex = eco.adj_ce(self.spec_capex,
                                    self.spec_mntex,
                                    self.lifespan,
                                    scenario.wacc)  # adjusted ce (including maintenance) of the component in $/W

        self.eq_pres_cost = eco.ann_recur(self.adj_capex,
                                          self.lifespan,
                                          scenario.prj_duration,
                                          scenario.wacc,
                                          self.cost_decr)

        self.flow_ch = pd.Series(dtype='float64')  # empty dataframe for result concatenation
        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation

    def concatenate_results(self):
        pass


class AggregateCommoditySystem(InvestComponent):
    """
    Option 1: aggregated vehicles (x denotes the flow measurement point)
    ac_bus             bev_bus
      |<-x-------bev_ac---|<--bev_src
      |                   |
      |-x-ac_bev--------->|<->bev_ess
      |                   |
                          |-->bev_snk
    """
    pass


class CommoditySystem(InvestComponent):

    def __init__(self, name, scenario, run):

        super().__init__(name)

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.xdb)
        self.input_file_path = os.path.join(run.result_path, self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=";",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.commodity_num = xread(self.name + '_num', scenario.name, run.xdb)
        self.commodity_agr = xread(self.name + '_agr', scenario.name, run.xdb)  # TODO enable aggregated simulation

        self.chg_pwr = xread(self.name + '_chg_pwr', scenario.name, run.xdb)
        self.dis_pwr = xread(self.name + '_dis_pwr', scenario.name, run.xdb)
        self.chg_eff = xread(self.name + '_charge_eff', scenario.name, run.xdb)
        self.dis_eff = xread(self.name + '_discharge_eff', scenario.name, run.xdb)

        self.int_lvl = xread(self.name + '_int_lvl', scenario.name, run.xdb)  # charging integration level

        self.flow_out = self.flow_bal = self.soc = pd.Series(dtype='float64')  # empty dataframe for result concatenation

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results
        
        ac_bus             mc_bus
          |<-x--------mc_ac---|---(CommoditySystem Instance)
          |                   |
          |-x-ac_mc---------->|---(CommoditySystem Instance)
                              |
                              |---(CommoditySystem Instance)
        """

        self.bus = solph.Bus(label=f"{self.name}_bus")
        scenario.solph_components.append(self.bus)

        self.inflow = solph.Transformer(label=f"ac_{self.name}",
                                       inputs={scenario.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.bus: solph.Flow()},
                                       conversion_factors={self.bus: 1})
        scenario.solph_components.append(self.inflow)

        self.outflow = solph.Transformer(label=f"ac_{self.name}",
                                         inputs={self.bus: solph.Flow(
                                             nominal_value={'uc': 0,
                                                            'cc': 0,
                                                            'tc': 0,
                                                            'v2v': 0,
                                                            'v2g': None}[self.int_lvl],
                                             variable_costs=run.eps_cost)},
                                         outputs={scenario.ac_bus: solph.Flow()},
                                         conversion_factors={scenario.ac_bus: 1})
        scenario.solph_components.append(self.outflow)

        self.commodities = [MobileCommodity(self.name + str(i), self, scenario, run) for i in range(self.commodity_num)]

    def define_input_components(self):

        for commodity in self.commodities: commodity.update_input_components()

    def get_results(self, horizon, scenario):

        for commodity in self.commodities: commodity.get_results(horizon, scenario)


class StationaryEnergyStorage(InvestComponent):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.chg_eff = xread(self.name + '_chg_eff', scenario.name, run.xdb)
        self.dis_eff = xread(self.name + '_dis_eff', scenario.name, run.xdb)
        self.chg_crate = xread(self.name + '_chg_crate', scenario.name, run.xdb)
        self.dis_crate = xread(self.name + '_dis_crate', scenario.name, run.xdb)

        self.init_soc = xread(self.name + '_init_soc', scenario.name, run.xdb)
        self.ph_init_soc = self.init_soc  # TODO actually necessary?

        self.flow_in_ch = self.flow_out_ch = pd.Series(dtype='float64')  # result data
        self.sc_ch = self.soc_ch = self.soc = pd.Series(dtype='float64')  # result data

        """
        x denotes the flow measurement point in results
        
        dc_bus
          |
          |<-x->ess
          |
        """

        if self.opt:
            self.ess = solph.components.GenericStorage(label="ess",
                                                       inputs={scenario.dc_bus: solph.Flow()},
                                                       outputs={scenario.dc_bus: solph.Flow(variable_cost=self.spec_opex)},
                                                       loss_rate=0,  # TODO proper self discharge
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.ph_init_soc,
                                                       invest_relation_input_capacity=self.chg_crate,
                                                       invest_relation_output_capacity=self.dis_crate,
                                                       inflow_conversion_factor=self.chg_eff,
                                                       outflow_conversion_factor=self.dis_eff,
                                                       investment=solph.Investment(ep_costs=self.eq_pres_cost))
        else:
            self.ess = solph.components.GenericStorage(label="ess",
                                                       inputs={scenario.dc_bus: solph.Flow()},
                                                       outputs={
                                                           scenario.dc_bus: solph.Flow(variable_cost=self.spec_opex)},
                                                       loss_rate=0,  # TODO proper self discharge
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.ph_init_soc,
                                                       invest_relation_input_capacity=self.chg_crate,
                                                       invest_relation_output_capacity=self.dis_crate,
                                                       inflow_conversion_factor=self.chg_eff,
                                                       outflow_conversion_factor=self.dis_eff,
                                                       nominal_storage_capacity=self.size)
        scenario.solph_components.append(self.ess)

    def update_input_components(self):

        self.ess.initial_storage_level = self.ph_init_soc

    def get_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.ess, scenario.dc_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[(scenario.dc_bus, self.ess)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ch = self.flow_in_ch - self.flow_out_ch  # inflow is positive

        self.flow = pd.concat([self.flow, self.flow_ch])

        self.sc_ch = solph.views.node(horizon.results, self.name)['sequences'][
            ((self.name, 'None'), 'storage_content')][horizon.ch_dti].shift(periods=1, freq=scenario.sim_timestep)
        # shift is needed as sc/soc is stored for end of timestep
        self.soc_ch = self.sc_ch / self.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge
        self.ph_init_soc = self.soc.iloc[-1]  # reset initial SOC for next prediction horizon


class ControllableSource(InvestComponent):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results
        
        ac_bus
          |
          |<-x-gen
          |
        """

        if self.opt:
            self.src = solph.Source(label='gen_src',
                                    outputs={scenario.ac_bus: solph.Flow(investment=solph.Investment(ep_costs=self.eq_pres_cost),
                                                                         variable_costs=self.spec_opex)})
        else:
            self.src = solph.Source(label='gen_src',
                                    outputs={scenario.ac_bus: solph.Flow(nominal_value=self.size,
                                                                         variable_costs=self.spec_opex)})
        scenario.solph_components.append(self.src)

    def update_input_components(self):
        pass  # no sliced input data needed for controllable source, but function needs to be callable

    def get_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.src, scenario.ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat(self.flow, self.flow_ch)


class MobileCommodity:

    def __init__(self, name, parent, scenario, run):

        self.name = name
        self.parent = parent

        self.init_soc = xread(self.parent.name + '_init_soc', scenario.name, run.xdb)  # TODO: add random init soc?
        self.ph_init_soc = self.init_soc  # set first PH's initial state variables (only SOC)

        self.flow_in_ch = self.flow_out_ch = self.flow_ch = self.flow = pd.Series(dtype='float64')  # result data
        self.sc_ch = self.soc_ch = self.soc = pd.Series(dtype='float64')  # result data

        # Creation of permanent energy system components --------------------------------

        """
        mc_bus              mc1_bus
          |<---------mc1_mc-x-|<->mc1_ess
          |                   |
          |---mc_mc1-------x->|-->mc1_snk
          |
          |                 mc2_bus
          |<---------mc2_mc---|<->mc2_ess
          |                   |
          |---mc_mc2--------->|-->mc2_snk
        """

        self.bus = solph.Bus(label=f"{self.name}_bus")
        scenario.solph_components.append(self.bus)

        self.inflow = solph.Transformer(label=f"mc_{self.name}",
                                        inputs={self.parent.bus: solph.Flow(nominal_value=self.parent.chg_pwr,
                                                                               max=self.ph_data[
                                                                                   f"at_charger_{self.name}"],
                                                                               variable_costs=run.eps_cost)},
                                        outputs={self.bus: solph.Flow()},
                                        conversion_factors={self.bus: self.parent.chg_eff})
        scenario.solph_components.append(self.inflow)

        self.outflow = solph.Transformer(label=f"{self.name}_mc",
                                        inputs={self.bus: solph.Flow(nominal_value={'uc': 0,
                                                                                        'cc': 0,
                                                                                        'tc': 0,
                                                                                        'v2v': 1,
                                                                                        'v2g': 1}[
                                                                                           self.parent.int_lvl] * self.parent.dis_pwr,
                                                                         max=self.ph_data[f"at_charger_{self.name}"],
                                                                         variable_costs=run.eps_cost)},
                                        outputs={self.parent.mc_bus: solph.Flow()},
                                        conversion_factors={self.parent.mc_bus: self.parent.dis_eff})
        scenario.solph_components.append(self.outflow)

        if self.parent.opt:
            self.ess = solph.components.GenericStorage(label=f"{self.name}_ess",
                                                       inputs={self.bus: solph.Flow()},
                                                       outputs={self.bus: solph.Flow(
                                                           variable_cost=self.parent.spec_opex)},
                                                       loss_rate=0,  # TODO integrate self discharge
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.ph_init_soc,
                                                       inflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       outflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       max_storage_level=1,
                                                       min_storage_level=self.ph_data[f"min_soc_{self.name}"],
                                                       # TODO is commodity ph_data actually created?
                                                       investment=solph.Investment(
                                                           ep_costs=self.parent.eq_pres_cost))
        else:
            self.ess = solph.components.GenericStorage(label=f"{self.name}_ess",
                                                       inputs={self.bus: solph.Flow()},
                                                       outputs={self.bus: solph.Flow(
                                                           variable_cost=self.parent.spec_opex)},
                                                       loss_rate=0,  # TODO integrate self discharge
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.ph_init_soc,
                                                       inflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       outflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       max_storage_level=1,
                                                       min_storage_level=self.ph_data[f"min_soc_{self.name}"],
                                                       # TODO is commodity ph_data actually created?
                                                       nominal_storage_capacity=self.parent.size, )  # TODO does size exist?
        scenario.solph_components.append(self.ess)

        self.snk = solph.Sink(label=f"{self.name}_snk",
                              inputs={self.bus: solph.Flow(fix=self.ph_data[f"sink_data_{self.name}"],
                                                           nominal_value=1)})
        scenario.solph_components.append(self.snk)

    def update_input_components(self):

        # enable/disable transformers to mcx_bus depending on whether the commodity is at base
        self.inflow.inputs[self.parent.mc_bus].max = self.ph_data[f"{self.name}_at_base"]
        self.outflow.inputs[self.bus].max = self.ph_data[f"{self.name}_at_base"]

        # define consumption data for sink (only enabled when detached from base
        self.snk.inputs[self.bus].fix = self.ph_data[f"{self.name}_consumption"]

        # set initial storage level for coming prediction horizon
        self.ess.initial_storage_level = self.ph_init_soc

    def get_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.bus, self.outflow)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[(self.inflow, self.bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ch = self.flow_in_ch - self.flow_out_ch  # inflow is positive

        self.flow = pd.concat(self.flow, self.flow_ch)

        self.sc_ch = solph.views.node(
            horizon.results, f"{self.name}_ess")['sequences'][((f"{self.name}_ess", 'None'), 'storage_content')][
            horizon.ch_dti].shift(periods=1, freq=scenario.sim_timestep)
        # shift is needed as sc/soc is stored for end of timestep
        self.soc_ch = self.sc_ch / self.parent.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge
        self.ph_init_soc = self.soc.iloc[-1]  # reset initial SOC for next prediction horizon


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

        logging.info('Building energy system model')

        self.es = solph.EnergySystem(timeindex=self.ph_dti)  # initialize energy system model instance

        for component_set in scenario.component_sets:
            component_set.update_input_components(scenario, self)  # (re)define solph components that need input slices

        for solph_component in scenario.solph_components:
            self.es.add(solph_component)  # add components to this horizon's energy system

        self.model = solph.Model(self.es)  # Build the mathematical linear optimization model with pyomo

        if run.dump_model:
            if scenario.strategy == 'go':
                self.model.write(run.dump_model_file, io_options={'symbolic_solver_labels': True})
            elif scenario.strategy == 'rh':
                logging.warning('Model file dump not implemented for RH operating strategy - no file created')

    def get_results(self, scenario, run):
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
                component_set.size = self.results[(component_set.src, component_set.bus)]["scalars"]["invest"]
                # TODO check whether this definition fits all component sets

            component_set.get_results(self, scenario)


class PVSource(InvestComponent):  # TODO combine to RenewableSource?

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        if run.pv_source == 'api':  # API input selected
            pass  # TODO: API input goes here
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

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results

        dc_bus              pv_bus
          |                   |
          |<--x-------pv_dc---|<--pv_src
          |                   |
                              |-->pv_exc
        """

        self.bus = solph.Bus(label=f"{self.name}_bus")
        scenario.solph_components.append(self.bus)

        self.outflow = solph.Transformer(label=f"{self.name}_dc",
                                         inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                         outputs={scenario.dc_bus: solph.Flow()},
                                         conversion_factors={self.bus: 1})  # TODO proper efficiency
        scenario.solph_components.append(self.outflow)

        if self.opt:
            self.src = solph.Source(label=f"{self.name}_src",
                                    outputs={self.bus: solph.Flow(fix=self.ph_data['P'],
                                                                     investment=solph.Investment(ep_costs=self.eq_pres_cost),
                                                                     variable_cost=self.spec_opex)})
        else:
            self.src = solph.Source(label=f"{self.name}_src",
                                    outputs={self.bus: solph.Flow(fix=self.ph_data['P'],
                                                                     nominal_value=self.size,
                                                                     variable_cost=self.spec_opex)})
        scenario.solph_components.append(self.src)

        self.exc = solph.Sink(label=f"{self.name}_exc",
                              inputs={self.bus: solph.Flow()})
        scenario.solph_components.append(self.exc)

    def update_input_components(self):

        self.src.outputs[self.bus].fix = self.ph_data['P']

    def get_flow_ch(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.dc_bus)]['sequences']['flow'][horizon.ch_dti]


class Scenario:

    def __init__(self, run, name):

        # General Information --------------------------------

        self.name = name

        self.prj_starttime = datetime.strptime(xread('prj_start', self.name, run.xdb), '%Y/%m/%d')
        self.prj_duration = relativedelta(years=xread('prj_duration', self.name, run.xdb))
        self.prj_endtime = self.prj_starttime + self.prj_duration

        self.sim_starttime = self.prj_starttime  # simulation timeframe is at beginning of project timeframe
        self.sim_timestep = xread('sim_timestep', self.name, run.xdb)
        self.sim_duration = relativedelta(days=xread('sim_duration', self.name, run.xdb))
        self.sim_endtime = self.sim_starttime + self.sim_duration
        self.sim_dti = pd.date_range(start=self.sim_starttime, end=self.sim_endtime, freq=self.sim_timestep).delete(-1)

        self.sim_yrrat = self.sim_duration.days / 365.25
        # self.sim_prjrat = self.sim_duration / self.prj_duration  # TODO convert to datetime timedeltas to be divisible

        self.wacc = xread('wacc', self.name, run.xdb)

        # Operational strategy --------------------------------

        self.strategy = xread('sim_os', self.name, run.xdb)

        if self.strategy == 'rh':
            self.ph_len = relativedelta(hours=xread('rh_ph', self.name, run.xdb))
            self.ch_len = relativedelta(days=xread('rh_ch', self.name, run.xdb))
            self.ph_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ph_len  # number of timesteps for PH
            self.ch_steps = {'H': 1, 'T': 60}[self.sim_timestep] * self.ch_len  # number of timesteps for CH
            self.horizon_num = int(self.sim_duration.hours / self.ch_len.hours)  # number of timeslices to run
        elif self.strategy == 'go':
            self.ph_len = self.sim_duration
            self.ch_len = self.sim_duration
            self.horizon_num = 1

        # Components --------------------------------

        self.components_enable = dict(dem=(xread('dem_enable', self.name, run.xdb) == 'True'),
                                      wind=(xread('wind_enable', self.name, run.xdb) == 'True'),
                                      pv=(xread('pv_enable', self.name, run.xdb) == 'True'),
                                      gen=(xread('gen_enable', self.name, run.xdb) == 'True'),
                                      ess=(xread('ess_enable', self.name, run.xdb) == 'True'),
                                      bev=(xread('bev_enable', self.name, run.xdb) == 'True'))

        self.component_sets = []
        self.solph_components = []

        for component in [[name for name, enable in self.components_enable.items() if enable]]:
            if component == 'dem':
                dem = StatSink('dem', self, run)
                self.component_sets.append(dem)
            elif component == 'wind':
                wind = WindSource('wind', self, run)
                self.component_sets.append(wind)
            elif component == 'pv':
                pv = PVSource('pv', self, run)
                self.component_sets.append(pv)
            elif component == 'gen':
                gen = ControllableSource('gen', self, run)
                self.component_sets.append(gen)
            elif component == 'bev':
                bev = CommoditySystem('bev', self, run)
                self.component_sets.append(bev)
            elif component == 'mb':
                mb = CommoditySystem('mb', self, run)
                self.component_sets.append(mb)

        # Creation of static core energy system components --------------------------------

        """
        dc_bus              ac_bus
          |                   |
          |---dc_ac---------->|
          |                   |
          |<----------ac_dc---|
        """

        self.ac_bus = solph.Bus(label="ac_bus")
        self.solph_components.append(self.ac_bus)

        self.dc_bus = solph.Bus(label="dc_bus")
        self.solph_components.append(self.dc_bus)

        self.ac_dc = solph.Transformer(label="ac_dc",
                                       inputs={self.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.dc_bus: solph.Flow()},
                                       conversion_factors={self.dc_bus: xread('ac_dc_eff', self.name, run.xdb)})
        self.solph_components.append(self.ac_dc)

        self.dc_ac = solph.Transformer(label="dc_ac",
                                       inputs={self.dc_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.ac_bus: solph.Flow()},
                                       conversion_factors={self.ac_bus: xread('dc_ac_eff', self.name, run.xdb)})
        self.solph_components.append(self.dc_ac)

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


class SimulationRun:

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

        self.global_sheet = 'global_settings'
        self.solver = xread('solver', self.global_sheet, self.xdb)
        self.print_results = (xread('print_results', self.global_sheet, self.xdb) == 'True')
        self.show_plots = (xread('show_plots', self.global_sheet, self.xdb) == 'True')
        self.save_plots = (xread('save_plots', self.global_sheet, self.xdb) == 'True')
        self.dump_model = (xread('dump_model', self.global_sheet, self.xdb) == 'True')
        self.solver_debugmode = (xread('solver_debugmode', self.global_sheet, self.xdb) == 'True')
        self.eps_cost = float(xread('eps_cost', self.global_sheet, self.xdb))

        self.input_data_path = os.path.join(os.getcwd(), "input_data")
        self.dump_model_file = os.path.join(os.getcwd(), "lp_models", self.runtimestamp + "_model.lp")
        self.log_file = os.path.join(os.getcwd(), "logfiles", self.runtimestamp + ".log")

        logger.define_logging(logfile=self.log_file)
        logging.info("Global settings read - initializing scenarios")

    def input_gui(self):
        '''
        GUI to choosr input excel file
        :return:
        '''

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
            if scenarios_filename == "." or results_foldername == ".":
                print("WARNING: not all required paths entered - exiting")
                exit()
            return scenarios_filename, results_foldername
        except TypeError:
            print("WARNING: GUI window closed manually - exiting")
            exit()


class StatSink:

    def __init__(self, name, scenario, run):

        self.name = name
        self.input_file_name = xread('dem_filename', scenario.name, run.xdb)
        self.input_file_path = os.path.join(run.input_data_path, "load_profile_data", self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=",",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.flow_ch = pd.Series(dtype='float64')  # empty dataframe for result concatenation
        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation

        """
        x denotes the flow measurement point in results

        ac_bus
          |
          |-x->dem_snk
          |
        """

        self.snk = solph.Sink(label='dem_snk',
                              inputs={scenario.ac_bus:
                                          solph.Flow(fix=self.ph_data["P"],
                                                     nominal_value=1)})


    def update_input_components(self, scenario):

        self.snk.inputs[scenario.ac_bus].fix = self.ph_data["P"]

    def get_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(scenario.ac_bus, self.snk)]['sequences']['flow'][horizon.ch_dti]


class WindSource(InvestComponent):  # TODO combine to RenewableSource?

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.xdb)
        self.input_file_path = os.path.join(run.input_data_path, self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=",",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.transformer_eff = xread(self.name + '_eff', scenario.name, run.xdb)

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results
        
        ac_bus             wind_bus
          |                   |
          |<--x-----wind_ac---|<--wind_src
          |                   |
                              |-->wind_exc
        """

        self.bus = solph.Bus(label=f"{self.name}_bus")
        scenario.solph_components.append(self.bus)

        self.outflow = solph.Transformer(label=f"{self.name}_ac",
                                         inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                         outputs={scenario.ac_bus: solph.Flow()},
                                         conversion_factors={scenario.ac_bus: 1})  # TODO proper efficiency
        scenario.solph_components.append(self.outflow)

        self.exc = solph.Sink(label=f"{self.name}_exc",
                              inputs={self.bus: solph.Flow()})
        scenario.solph_components.append(self.exc)

        if self.opt:
            self.src = solph.Source(label=f"{self.name}_src",
                                    outputs={self.bus: solph.Flow(fix=self.ph_data["P"],
                                                                  investment=solph.Investment(ep_costs=run.eps_cost,
                                                                                              variable_cost=self.spec_opex))})
        else:
            self.src = solph.Source(label=f"{self.name}_src",
                                    outputs={scenario.solph_components.wind_bus: solph.Flow(fix=self.ph_data["P"],
                                                                                            nominal_value=self.size,
                                                                                            variable_cost=self.spec_opex)})
        scenario.solph_components.append(self.src)

    def update_input_components(self):

        self.src.outputs[self.bus].fix = self.ph_data["P"]

    def get_flow_ch(self, horizon, scenario):
        self.flow_ch = horizon.results[(self.outflow, scenario.ac_bus)]['sequences']['flow'][horizon.ch_dti]


###############################################################################
# Function definitions
###############################################################################


def add_bev(sim, es, bev):
    pass
    # if bev['agr'] and bev['chg_lvl'] != 'uc':  # When vehicles are aggregated into three basic components
    #     sim['components']['bev_snk'] = solph.Sink(  # Aggregated sink component modelling leaving vehicles
    #         label="bev_snk",
    #         inputs={sim['components']['bev_bus']: solph.Flow(actual_value=bev['ph_data']['sink_data'],
    #                                                          fixed=True,
    #                                                          nominal_value=1)})
    #     sim['components']['bev_src'] = solph.Source(  # Aggregated source component modelling arriving vehicles
    #         label='bev_src',
    #         outputs={sim['components']['bev_bus']: solph.Flow(actual_value=bev['ph_data']['source_data'],
    #                                                           fixed=True,
    #                                                           nominal_value=1)})
    #     sim['components']['bev_ess'] = solph.components.GenericStorage(  # Aggregated storage models connected vehicles
    #         label="bev_ess",
    #         inputs={sim['components']['bev_bus']: solph.Flow()},
    #         outputs={sim['components']['bev_bus']: solph.Flow(variable_cost=bev['spec_opex'])},
    #         nominal_storage_capacity=bev['num'] * bev['cs'],  # Storage capacity is set to the maximum available,
    #         # adaptation to different numbers of vehicles happens with the min/max storage levels
    #         loss_rate=0,
    #         balanced=False,
    #         initial_storage_level=None,  # TODO: Check for validity!
    #         inflow_conversion_factor=1,
    #         outflow_conversion_factor=1,
    #         min_storage_level=bev['ph_data']['min_charge'],
    #         # This models the varying storage capacity with (dis)connects
    #         max_storage_level=bev['ph_data'][
    #             'max_charge'])  # This models the varying storage capacity with (dis)connects

    # if bev['chg_lvl'] == 'uc':  # When charging level "uncoordinated charging"
    #     sim['components']['bev_snk'] = solph.Sink(  # Aggregated sink component for charging vehicles
    #         label="bev_snk",
    #         inputs={sim['components']['bev_bus']: solph.Flow(fix=bev['ph_data']['uc_power'] / bev['charge_eff'],
    #                                                          nominal_value=1)})


def xread(param_name, sheet, db):
    """
    Reading parameters from external excel file
    """
    value = db.ws(ws=sheet).keyrow(key=param_name, keyindex=1)[1]
    return value
