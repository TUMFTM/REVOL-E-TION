"""
system_blocks.py

--- Description ---
This script defines the energy system blocks for the oemof mg_ev toolset.

For further information, see readme

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
import oemof.solph as solph
import os
import pandas as pd

import economics as eco

from main import xread

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
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = None

    def accumulate_invest_results(self, scenario):  # TODO check whether CommoditySystems accumulate all commodity costs correctly

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
        self.input_file_path = os.path.join(run.result_path, self.input_file_name + '.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=';',
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

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.solph_components.append(self.bus)

        self.inflow = solph.Transformer(label=f'ac_{self.name}',
                                        inputs={scenario.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                        outputs={self.bus: solph.Flow()},
                                        conversion_factors={self.bus: 1})
        scenario.solph_components.append(self.inflow)

        self.outflow = solph.Transformer(label=f'ac_{self.name}',
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

    def get_ch_results(self, horizon, scenario):

        for commodity in self.commodities:
            commodity.get_ch_results(horizon, scenario)
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
            self.ess = solph.components.GenericStorage(label='ess',
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
            self.ess = solph.components.GenericStorage(label='ess',
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

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.solph_components.append(self.bus)

        self.inflow = solph.Transformer(label=f'mc_{self.name}',
                                        inputs={self.parent.bus: solph.Flow(nominal_value=self.parent.chg_pwr,
                                                                            max=self.ph_data[
                                                                                f'at_charger_{self.name}'],
                                                                            variable_costs=run.eps_cost)},
                                        outputs={self.bus: solph.Flow()},
                                        conversion_factors={self.bus: self.parent.chg_eff})
        scenario.solph_components.append(self.inflow)

        self.outflow = solph.Transformer(label=f'{self.name}_mc',
                                         inputs={self.bus: solph.Flow(nominal_value={'uc': 0,
                                                                                     'cc': 0,
                                                                                     'tc': 0,
                                                                                     'v2v': 1,
                                                                                     'v2g': 1}[
                                                                                        self.parent.int_lvl] * self.parent.dis_pwr,
                                                                      max=self.ph_data[f'at_charger_{self.name}'],
                                                                      variable_costs=run.eps_cost)},
                                         outputs={self.parent.mc_bus: solph.Flow()},
                                         conversion_factors={self.parent.mc_bus: self.parent.dis_eff})
        scenario.solph_components.append(self.outflow)

        if self.parent.opt:
            self.ess = solph.components.GenericStorage(label=f'{self.name}_ess',
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
                                                       min_storage_level=self.ph_data[f'min_soc_{self.name}'],
                                                       # TODO is commodity ph_data actually created?
                                                       investment=solph.Investment(
                                                           ep_costs=self.parent.eq_pres_cost))
        else:
            self.ess = solph.components.GenericStorage(label=f'{self.name}_ess',
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
                                                       min_storage_level=self.ph_data[f'min_soc_{self.name}'],
                                                       # TODO is commodity ph_data actually created?
                                                       nominal_storage_capacity=self.parent.size, )  # TODO does size exist?
        scenario.solph_components.append(self.ess)

        self.snk = solph.Sink(label=f'{self.name}_snk',
                              inputs={self.bus: solph.Flow(fix=self.ph_data[f'sink_data_{self.name}'],
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
            horizon.results, f'{self.name}_ess')['sequences'][((f'{self.name}_ess', 'None'), 'storage_content')][
            horizon.ch_dti].shift(periods=1, freq=scenario.sim_timestep)
        # shift is needed as sc/soc is stored for end of timestep
        self.soc_ch = self.sc_ch / self.parent.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge
        self.ph_init_soc = self.soc.iloc[-1]  # reset initial SOC for next prediction horizon

    def update_input_components(self):

        # enable/disable transformers to mcx_bus depending on whether the commodity is at base
        self.inflow.inputs[self.parent.mc_bus].max = self.ph_data[f'{self.name}_at_base']
        self.outflow.inputs[self.bus].max = self.ph_data[f'{self.name}_at_base']

        # define consumption data for sink (only enabled when detached from base
        self.snk.inputs[self.bus].fix = self.ph_data[f'{self.name}_consumption']

        # set initial storage level for coming prediction horizon
        self.ess.initial_storage_level = self.ph_init_soc


class PVSource(InvestComponent):  # TODO combine to RenewableSource?

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        if run.pv_source == 'api':  # API input selected
            pass  # TODO: API input goes here
        else:  # data input from fixed csv file
            self.input_file_name = xread(self.name + '_filename', scenario.name, run.input_xdb)
            self.input_file_path = os.path.join(run.input_data_path, self.input_file_name + '.csv')
            self.data = pd.read_csv(self.input_file_path,
                                    sep=',',
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

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.solph_components.append(self.bus)

        self.outflow = solph.Transformer(label=f'{self.name}_dc',
                                         inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                         outputs={scenario.dc_bus: solph.Flow()},
                                         conversion_factors={self.bus: 1})  # TODO proper efficiency
        scenario.solph_components.append(self.outflow)

        if self.opt:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={self.bus: solph.Flow(fix=self.ph_data['P'],
                                                                  investment=solph.Investment(
                                                                      ep_costs=self.eq_pres_cost),
                                                                  variable_cost=self.spec_opex)})
        else:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={self.bus: solph.Flow(fix=self.ph_data['P'],
                                                                  nominal_value=self.size,
                                                                  variable_cost=self.spec_opex)})
        scenario.solph_components.append(self.src)

        self.exc = solph.Sink(label=f'{self.name}_exc',
                              inputs={self.bus: solph.Flow()})
        scenario.solph_components.append(self.exc)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.dc_bus)]['sequences']['flow'][horizon.ch_dti]

    def update_input_components(self):

        self.src.outputs[self.bus].fix = self.ph_data['P']


class StatSink:

    def __init__(self, name, scenario, run):

        self.name = name
        self.input_file_name = xread('dem_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, 'load_profile_data', self.input_file_name)
        self.data = pd.read_csv(self.input_file_path,
                                sep=',',
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
                              inputs={scenario.ac_bus: solph.Flow(fix=self.ph_data['P'],  # TODO definition without fix possible? - fix is added in update_input components...
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

        # TODO ph data needs to be created here
        self.snk.inputs[scenario.ac_bus].fix = self.ph_data['P']


class WindSource(InvestComponent):  # TODO combine to RenewableSource?

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, self.input_file_name + '.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=',',
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

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.solph_components.append(self.bus)

        self.outflow = solph.Transformer(label=f'{self.name}_ac',
                                         inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                         outputs={scenario.ac_bus: solph.Flow()},
                                         conversion_factors={scenario.ac_bus: 1})  # TODO proper efficiency
        scenario.solph_components.append(self.outflow)

        self.exc = solph.Sink(label=f'{self.name}_exc',
                              inputs={self.bus: solph.Flow()})
        scenario.solph_components.append(self.exc)

        if self.opt:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={self.bus: solph.Flow(fix=self.ph_data['P'],
                                                                  investment=solph.Investment(ep_costs=run.eps_cost,
                                                                                              variable_cost=self.spec_opex))})
        else:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={scenario.solph_components.wind_bus: solph.Flow(fix=self.ph_data['P'],
                                                                                            nominal_value=self.size,
                                                                                            variable_cost=self.spec_opex)})
        scenario.solph_components.append(self.src)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.ac_bus)]['sequences']['flow'][horizon.ch_dti]

    def update_input_components(self):

        self.src.outputs[self.bus].fix = self.ph_data['P']