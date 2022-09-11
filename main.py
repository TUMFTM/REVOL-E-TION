#!/usr/bin/env python3

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
Created September 2nd, 2021

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022
Hannes Henglein, B.Sc. - Semester Thesis in progress
Marc Alsina Planelles, B.Sc. - Master Thesís in progress

--- Detailed Description ---
This script is the main model generator and optimizer for the toolset.
Its results are output to files and key ones printed to the terminal.

For further information, see readme

--- Input & Output ---
The model requires a specific folder structure containing
    - a Microsoft Excel input file controlling general parameters
    - several .csv files containing timeseries data as referenced by the input file

For further information, see readme.

--- Requirements ---

For package requirements, see requirements.txt

For further information, see readme.

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

import logging
# import multiprocessing as multi  # TODO parallelize scenario loop
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

        self.opt = xread(f'{self.name}_opt', scenario.name, run.input_xdb)

        if self.opt and scenario.strategy != 'go':
            logging.error('Error: Rolling horizon strategy is not feasible if component sizing is active')
            logging.error('Please disable sim_cs in settings file')
            exit()  # TODO switch to next scenario instead of exiting

        if self.opt:
            self.size = None
        else:
            self.size = xread(self.name + '_cs', scenario.name, run.input_xdb)

        self.spec_capex = xread(self.name + '_sce', scenario.name, run.input_xdb)  # TODO rename capex_spec
        self.spec_mntex = xread(self.name + '_sme', scenario.name, run.input_xdb)  # TODO rename mntex_spec
        self.spec_opex = xread(self.name + '_soe', scenario.name, run.input_xdb)  # TODO rename opex_spec

        self.lifespan = xread(self.name + '_ls', scenario.name, run.input_xdb)
        self.cost_decr = xread(self.name + '_cdc', scenario.name, run.input_xdb)
        self.transformer_eff = xread(self.name + '_eff', scenario.name, run.input_xdb)

        self.adj_capex = eco.adj_ce(self.spec_capex,  # TODO rename capex_adj
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

        self.e_sim = self.e_yrl = self.e_prj = self.e_dis = None  # empty placeholders for cumulative results
        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = None
        self.mntex_sim = self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = None
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = None

    def accumulate_invest_results(self, scenario):

        self.e_sim = self.flow.sum()
        self.e_yrl = self.e_sim / scenario.sim_prj_rat
        self.e_prj = self.e_yrl * scenario.prj_duration
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration, scenario.wacc)

        self.capex_init = self.size * self.spec_capex
        self.capex_prj = eco.tce(self.capex_init,
                                 self.capex_init,  # TODO integrate cost decrease
                                 self.lifespan,
                                 scenario.prj_duration)
        self.capex_dis = eco.pce(self.capex_init,
                                 self.capex_init,  # TODO integrate cost decrease
                                 self.lifespan,
                                 scenario.prj_duration,
                                 scenario.wacc)
        self.capex_ann = eco.ann_recur(self.capex_init,
                                       self.lifespan,
                                       scenario.prj_duration,
                                       scenario.wacc,
                                       self.cost_decr)
        scenario.capex_init += self.capex_init
        scenario.capex_prj += self.capex_prj
        scenario.capex_dis += self.capex_dis
        scenario.capex_ann += self.capex_ann

        self.mntex_yrl = self.size * self.spec_mntex  # time-based maintenance
        self.mntex_sim = self.mntex_yrl * scenario.sim_yr_rat
        self.mntex_prj = self.mntex_yrl * scenario.prj_duration
        self.mntex_dis = eco.acc_discount(self.mntex_yrl,
                                          scenario.prj_duration,
                                          scenario.wacc)
        self.mntex_ann = eco.ann_recur(self.mntex_yrl,
                                       1,  # lifespan of 1 yr -> mntex happening yearly
                                       scenario.prj_duration,
                                       scenario.wacc,
                                       1)  # no cost decrease in mntex
        scenario.mntex_yrl += self.mntex_yrl
        scenario.mntex_prj += self.mntex_prj
        scenario.mntex_dis += self.mntex_dis
        scenario.mntex_ann += self.mntex_ann

        self.opex_sim = self.e_sim * self.spec_opex
        self.opex_yrl = self.opex_sim / scenario.sim_yr_rat  # linear scaling i.c.o. longer or shorter than 1 year
        self.opex_prj = self.opex_yrl * scenario.prj_duration
        self.opex_dis = eco.acc_discount(self.opex_yrl,
                                         scenario.prj_duration,
                                         scenario.wacc)
        self.opex_ann = eco.ann_recur(self.opex_yrl,
                                      1,  # lifespan of 1 yr -> opex happening yearly
                                      scenario.prj_duration,
                                      scenario.wacc,
                                      1)  # no cost decrease in opex
        scenario.opex_sim += self.opex_sim
        scenario.opex_yrl += self.opex_yrl
        scenario.opex_prj += self.opex_prj
        scenario.opex_dis += self.opex_dis
        scenario.opex_ann += self.opex_ann

        self.totex_sim = self.capex_init + self.mntex_sim + self.opex_sim
        self.totex_prj = self.capex_prj + self.mntex_prj + self.opex_prj
        self.totex_dis = self.capex_dis + self.mntex_dis + self.opex_dis
        self.totex_ann = self.capex_ann + self.mntex_ann + self.opex_ann

        scenario.totex_sim += self.totex_sim
        scenario.totex_prj += self.totex_prj
        scenario.totex_dis += self.totex_dis
        scenario.totex_ann += self.totex_ann

    def accumulate_energy_results_source(self, scenario):

        scenario.e_sim_pro += self.e_sim
        scenario.e_yrl_pro += self.e_yrl
        scenario.e_prj_pro += self.e_prj
        scenario.e_dis_pro += self.e_dis

    def accumulate_energy_results_sink(self, scenario):

        scenario.e_sim_del += self.e_sim
        scenario.e_yrl_del += self.e_yrl
        scenario.e_prj_del += self.e_prj
        scenario.e_dis_del += self.e_dis


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

        super().__init__(name, scenario, run)

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.result_path, self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=";",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.commodity_num = xread(self.name + '_num', scenario.name, run.input_xdb)
        self.commodity_agr = xread(self.name + '_agr', scenario.name, run.input_xdb)  # TODO enable aggregated simulation

        self.chg_pwr = xread(self.name + '_chg_pwr', scenario.name, run.input_xdb)
        self.dis_pwr = xread(self.name + '_dis_pwr', scenario.name, run.input_xdb)
        self.chg_eff = xread(self.name + '_charge_eff', scenario.name, run.input_xdb)
        self.dis_eff = xread(self.name + '_discharge_eff', scenario.name, run.input_xdb)

        self.int_lvl = xread(self.name + '_int_lvl', scenario.name, run.input_xdb)  # charging integration level

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

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)  # TODO check for validity - multiple resources included!
        self.accumulate_energy_results_sink(scenario)  # CommoditySystem is a sink as positive power/energy exits the core

        for commodity in self.commodities:
            commodity.accumulate_results()

    def get_ch_results(self, scenario):

        for commodity in self.commodities:
            commodity.get_ch_results(scenario)
        # TODO get system level flows, not only individual ones

    def update_input_components(self):

        for commodity in self.commodities:
            commodity.update_input_components()


class StationaryEnergyStorage(InvestComponent):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.chg_eff = xread(self.name + '_chg_eff', scenario.name, run.input_xdb)
        self.dis_eff = xread(self.name + '_dis_eff', scenario.name, run.input_xdb)
        self.chg_crate = xread(self.name + '_chg_crate', scenario.name, run.input_xdb)
        self.dis_crate = xread(self.name + '_dis_crate', scenario.name, run.input_xdb)

        self.init_soc = xread(self.name + '_init_soc', scenario.name, run.input_xdb)
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
                                                       outputs={
                                                           scenario.dc_bus: solph.Flow(variable_cost=self.spec_opex)},
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

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_sink(scenario)  # StationaryEnergyStorage is a sink as positive power/energy exits the core

    def get_ch_results(self, horizon, scenario):

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

    def update_input_components(self):

        self.ess.initial_storage_level = self.ph_init_soc


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
                                    outputs={scenario.ac_bus: solph.Flow(
                                        investment=solph.Investment(ep_costs=self.eq_pres_cost),
                                        variable_costs=self.spec_opex)})
        else:
            self.src = solph.Source(label='gen_src',
                                    outputs={scenario.ac_bus: solph.Flow(nominal_value=self.size,
                                                                         variable_costs=self.spec_opex)})
        scenario.solph_components.append(self.src)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.src, scenario.ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat(self.flow, self.flow_ch)

    def update_input_components(self):
        pass  # no sliced input data needed for controllable source, but function needs to be callable


class MobileCommodity:

    def __init__(self, name, parent, scenario, run):

        self.name = name
        self.parent = parent

        self.init_soc = xread(self.parent.name + '_init_soc', scenario.name, run.input_xdb)  # TODO: add random init soc?
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

    def accumulate_results(self):
        pass  # TODO individual commodity result accumulation

    def get_ch_results(self, horizon, scenario):

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

    def update_input_components(self):

        # enable/disable transformers to mcx_bus depending on whether the commodity is at base
        self.inflow.inputs[self.parent.mc_bus].max = self.ph_data[f"{self.name}_at_base"]
        self.outflow.inputs[self.bus].max = self.ph_data[f"{self.name}_at_base"]

        # define consumption data for sink (only enabled when detached from base
        self.snk.inputs[self.bus].fix = self.ph_data[f"{self.name}_consumption"]

        # set initial storage level for coming prediction horizon
        self.ess.initial_storage_level = self.ph_init_soc


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
                component_set.size = self.results[(component_set.src, component_set.bus)]["scalars"]["invest"]
                # TODO check whether this definition fits all component sets

            component_set.get_ch_results(self, horizon, scenario)


class PVSource(InvestComponent):  # TODO combine to RenewableSource?

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        if run.pv_source == 'api':  # API input selected
            pass  # TODO: API input goes here
        else:  # data input from fixed csv file
            self.input_file_name = xread(self.name + '_filename', scenario.name, run.input_xdb)
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
                                                                  investment=solph.Investment(
                                                                      ep_costs=self.eq_pres_cost),
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

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.dc_bus)]['sequences']['flow'][horizon.ch_dti]

    def update_input_components(self):

        self.src.outputs[self.bus].fix = self.ph_data['P']


class Scenario:

    def __init__(self, run, index, name):

        # General Information --------------------------------

        self.index = index
        self.name = name
        self.runtime_start = time.time()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

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

        # Components --------------------------------

        self.components_enable = dict(dem=(xread('dem_enable', self.name, run.input_xdb) == 'True'),
                                      wind=(xread('wind_enable', self.name, run.input_xdb) == 'True'),
                                      pv=(xread('pv_enable', self.name, run.input_xdb) == 'True'),
                                      gen=(xread('gen_enable', self.name, run.input_xdb) == 'True'),
                                      ess=(xread('ess_enable', self.name, run.input_xdb) == 'True'),
                                      bev=(xread('bev_enable', self.name, run.input_xdb) == 'True'))

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

        # Result variables --------------------------------

        self.e_sim_del = 0
        self.e_yrl_del = 0
        self.e_prj_del = 0
        self.e_dis_del = 0

        self.e_sim_pro = 0
        self.e_yrl_pro = 0
        self.e_prj_pro = 0
        self.e_dis_pro = 0

        self.e_eta = 0

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
                                       conversion_factors={self.dc_bus: xread('ac_dc_eff', self.name, run.input_xdb)})
        self.solph_components.append(self.ac_dc)

        self.dc_ac = solph.Transformer(label="dc_ac",
                                       inputs={self.dc_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.ac_bus: solph.Flow()},
                                       conversion_factors={self.ac_bus: xread('dc_ac_eff', self.name, run.input_xdb)})
        self.solph_components.append(self.dc_ac)

    def end_timing(self):

        self.feasible = True  # model optimization or simulation seems to have been successful

        self.runtime_end = time.time()
        self.runtime_len = self.runtime_end - self.runtime_start
        logging.info(f'Scenario {self.index} ({self.name}) finished - runtime {self.runtime_len}')

    def generate_plots(self):
        pass

    def accumulate_results(self):

        for component in self.component_sets:
            component.accumulate_results(self)

        self.e_eta = self.e_sim_del / self.e_sim_pro

    def print_results(self):
        pass

    def save_plots(self):
        pass

    def save_results(self, run):

        logging.info(f'Saving results for scenario "{self.name}" with index {self.index}')

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

        column_names = ['Cumulative results',  # TODO make fit for arbitrary component set combinations
                        'Demand results',
                        'Wind power results',
                        'PV power results',
                        'Fossil power results',
                        'Energy storage results',
                        'CommoditySystem results']
        for i, header in enumerate(column_names):
            run.result_xdbdb.ws(ws=self.name).update_index(row=5, col=1 + i * 4, val=header)

        # # function to add component data to the worksheet  # TODO working point marker
        # def add_ws(comp, col):
        #     keys = list(comp.keys())
        #     vals = list(comp.values())
        #     row_id = 6
        #     for i in range(len(vals)):
        #         if type(vals[i]) == np.float64 or type(vals[i]) == np.float or type(vals[i]) == np.int:
        #             db.ws(ws=sim['sheet']).update_index(row=row_id, col=col, val=keys[i])
        #             db.ws(ws=sim['sheet']).update_index(row=row_id, col=col + 1, val=vals[i])
        #             row_id += 1
        #     return None
        #
        # add_ws(cres, 1)
        # if sim['enable']['dem']:
        #     add_ws(dem, 5)
        # if sim['enable']['wind']:
        #     add_ws(wind, 9)
        # if sim['enable']['pv']:
        #     add_ws(pv, 13)
        # if sim['enable']['gen']:
        #     add_ws(gen, 17)
        # if sim['enable']['ess']:
        #     add_ws(ess, 21)
        # if sim['enable']['bev']:
        #     add_ws(bev, 25)
        #
        # # write out the db
        # xl.writexl(db=db, fn=results_filepath)  # TODO check to now write for every scenario, just once!





    def show_plots(self):
        pass


class SimulationRun:

    def __init__(self):
        self.scenarios_file_path, self.result_path = self.input_gui()
        self.scenarios_file_name = Path(self.scenarios_file_path).stem  # Gives file name without extension
        self.input_xdb = xl.readxl(fn=self.scenarios_file_path)  # Excel database of selected file
        self.scenario_names = self.input_xdb.ws_names  # Get list of sheet names, 1 sheet is 1 scenario

        try:
            self.scenario_names.remove('global_settings')
        except ValueError:
            print("Excel File does not include global settings - exiting")
            exit()

        self.runtimestart = time.time()  # TODO better timing method
        self.runtimestamp = datetime.now().strftime("%y%m%d_%H%M%S")  # create str of runtimestart

        self.global_sheet = 'global_settings'
        self.solver = xread('solver', self.global_sheet, self.input_xdb)
        self.save_results = (xread('save_results', self.global_sheet, self.input_xdb) == 'True')
        self.print_results = (xread('print_results', self.global_sheet, self.input_xdb) == 'True')
        self.save_plots = (xread('save_plots', self.global_sheet, self.input_xdb) == 'True')
        self.show_plots = (xread('show_plots', self.global_sheet, self.input_xdb) == 'True')
        self.dump_model = (xread('dump_model', self.global_sheet, self.input_xdb) == 'True')
        self.solver_debugmode = (xread('solver_debugmode', self.global_sheet, self.input_xdb) == 'True')
        self.eps_cost = float(xread('eps_cost', self.global_sheet, self.input_xdb))

        self.cwd = os.getcwd()
        self.input_data_path = os.path.join(self.cwd, "input_data")
        self.dump_file_path = os.path.join(self.cwd, "lp_models", self.runtimestamp + "_model.lp")
        self.log_file_path = os.path.join(self.cwd, "logfiles", self.runtimestamp + ".log")
        self.result_file_path = os.path.join(self.result_path, self.scenarios_file_name, ".xlsx")
        self.result_xdb = xl.Database()  # blank excel database for cumulative result saving

        logger.define_logging(logfile=self.log_file_path)
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
        self.input_file_name = xread('dem_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, "load_profile_data", self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=",",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.flow_ch = pd.Series(dtype='float64')  # empty dataframe for result concatenation
        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation

        self.e_sim = 0  # empty placeholder for cumulative results
        self.e_yrl = 0  # empty placeholder for cumulative results
        self.e_prj = 0  # empty placeholder for cumulative results
        self.e_dis = 0  # empty placeholder for cumulative results

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results

        ac_bus
          |
          |-x->dem_snk
          |
        """

        self.snk = solph.Sink(label='dem_snk',
                              inputs={scenario.ac_bus:solph.Flow(fix=self.ph_data["P"],
                                                                 nominal_value=1)})
        scenario.solph_components.append(self.snk)

    def accumulate_results(self, scenario):

        # No super function as StatSink is not an InvestComponent child (where accumulate_invest_results lives)

        self.e_sim = self.flow.sum()
        self.e_yrl = self.e_sim / scenario.sim_prj_rat
        self.e_prj = self.e_yrl * scenario.prj_duration
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration, scenario.wacc)

        scenario.e_sim_del += self.e_sim
        scenario.e_yrl_del += self.e_yrl
        scenario.e_prj_del += self.e_prj
        scenario.e_dis_del += self.e_dis

    def get_ch_results(self, horizon, scenario):
        self.flow_ch = horizon.results[(scenario.ac_bus, self.snk)]['sequences']['flow'][horizon.ch_dti]

    def update_input_components(self, scenario):
        self.snk.inputs[scenario.ac_bus].fix = self.ph_data["P"]


class WindSource(InvestComponent):  # TODO combine to RenewableSource?

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, self.input_file_name + ".csv")
        self.data = pd.read_csv(self.input_file_path,
                                sep=",",
                                skip_blank_lines=False)
        self.data['time'] = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.transformer_eff = xread(self.name + '_eff', scenario.name, run.input_xdb)

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

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.ac_bus)]['sequences']['flow'][horizon.ch_dti]

    def update_input_components(self):

        self.src.outputs[self.bus].fix = self.ph_data["P"]


###############################################################################
# Function definitions
###############################################################################


def handle_scenario_error():
    pass


def simulate_scenario():
    '''
    Main function optimizing and simulating a single scenario as defined by a scenarios excel sheet
    '''

    try:
        scenario = Scenario(run, scenario_index, scenario_name)  # Create scenario instance & read data from excel sheet.

        for horizon_index in range(scenario.horizon_num):  # Inner optimization loop over all prediction horizons
            horizon = PredictionHorizon(horizon_index, scenario, run)
            horizon.model.solve(solver=run.solver, solve_kwargs={"tee": run.solver_debugmode})
            horizon.get_results(scenario, horizon, run)

        scenario.end_timing()

        if run.save_results or run.print_results:
            scenario.accumulate_results()
            if run.save_results:
                scenario.save_results(run)
            if run.print_results:
                scenario.print_results()

        if run.save_plots or run.show_plots:
            scenario.generate_plots()
            if run.save_plots:
                scenario.save_plots()
            if run.show_plots:
                scenario.show_plots()

    except KeyError:  # TODO proper error handle to only catch infeasible conditions - current one is only a dummy
        logging.warning(f"Scenario {scenario_name} failed or infeasible - continue on next scenario")
        try:
            scenario.feasible = False
        finally:
            handle_scenario_error()


def xread(param_name, sheet, db):
    """
    Reading parameters from external excel file
    """
    value = db.ws(ws=sheet).keyrow(key=param_name, keyindex=1)[1]
    return value

###############################################################################
# Execution code
###############################################################################


if __name__ == '__main__':

    run = SimulationRun()  # get all global information about the run

    for scenario_index, scenario_name in enumerate(run.scenario_names):
        simulate_scenario(run)  # TODO integrate multiprocessing


        #post.print_results(sim, wind, pv, gen, ess, bev, cres)  # TODO OO

        #sim = post.end_timing(sim)  # TODO OO

        #post.plot_results(sim, dem, wind, pv, gen, ess, bev)  # TODO OO
        #post.save_results(sim, dem, wind, pv, gen, ess, bev, cres)  # TODO OO





