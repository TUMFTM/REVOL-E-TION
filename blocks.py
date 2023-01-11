"""
blocks.py

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

import oemof.solph as solph
import os
import pandas as pd
import pvlib
import pytz
import timezonefinder

import economics as eco

from main import xread

###############################################################################
# Class definitions
###############################################################################


class SystemCore:

    def __init__(self, name, scenario, run):

        """
        dc_bus              ac_bus
          |                   |
          |---dc_ac---------->|
          |                   |
          |<----------ac_dc---|
        """

        self.name = name

        self.ac_bus = solph.Bus(label='ac_bus')
        scenario.components.append(self.ac_bus)

        self.dc_bus = solph.Bus(label='dc_bus')
        scenario.components.append(self.dc_bus)

        self.ac_dc = solph.Transformer(label='ac_dc',
                                       inputs={self.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.dc_bus: solph.Flow()},
                                       conversion_factors={self.dc_bus: xread('ac_dc_eff', scenario.name, run.input_xdb)})
        scenario.components.append(self.ac_dc)

        self.dc_ac = solph.Transformer(label='dc_ac',
                                       inputs={self.dc_bus: solph.Flow(variable_costs=run.eps_cost)},
                                       outputs={self.ac_bus: solph.Flow()},
                                       conversion_factors={self.ac_bus: xread('dc_ac_eff', scenario.name, run.input_xdb)})
        scenario.components.append(self.dc_ac)

        scenario.blocks.append(self)

    def accumulate_results(self, *_):
        pass  # function needs to be callable

    def get_ch_results(self, *_):
        pass  # function needs to be callable

    def update_input_components(self, *_):
        pass  # function needs to be callable


class InvestBlock:

    def __init__(self, name, scenario, run):

        self.name = name

        self.opt = (xread(f'{self.name}_opt', scenario.name, run.input_xdb) == 'True')

        if self.opt and scenario.strategy != 'go':
            run.logger.warning(f'{self.name} component size optimization not implemented for any'
                               f' other strategy than \"GO\" - disabling size optimization')
            self.opt = False

        if self.opt:
            self.size = None
        else:
            self.size = xread(self.name + '_cs', scenario.name, run.input_xdb)

        self.spec_capex = xread(self.name + '_sce', scenario.name, run.input_xdb)  # TODO rename capex_spec
        self.spec_mntex = xread(self.name + '_sme', scenario.name, run.input_xdb)  # TODO rename mntex_spec
        self.spec_opex = xread(self.name + '_soe', scenario.name, run.input_xdb)  # TODO rename opex_spec

        self.lifespan = xread(self.name + '_ls', scenario.name, run.input_xdb)
        self.cost_decr = xread(self.name + '_cdc', scenario.name, run.input_xdb)
        self.transformer_eff = xread(self.name + '_eff', scenario.name, run.input_xdb)  # TODO make sure eff is used

        self.adj_capex = eco.adj_ce(self.spec_capex,  # TODO rename capex_adj
                                    self.spec_mntex,
                                    self.lifespan,
                                    scenario.wacc)  # adjusted ce (including maintenance) of the component in $/W

        self.eq_pres_cost = eco.ann_recur(self.adj_capex,
                                          self.lifespan,
                                          scenario.prj_duration_yrs,
                                          scenario.wacc,
                                          self.cost_decr)

        self.flow_ch = pd.Series(dtype='float64')  # empty dataframe for result concatenation
        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation

        self.e_sim = self.e_yrl = self.e_prj = self.e_dis = None  # empty placeholders for cumulative results
        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = None
        self.mntex_sim = self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = None
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = None
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = None

        scenario.blocks.append(self)

    def accumulate_invest_results(self, scenario):  # TODO check whether CommoditySystems accumulate all commodity costs correctly

        self.e_sim = self.flow.sum()
        self.e_yrl = self.e_sim / scenario.sim_yr_rat
        self.e_prj = self.e_yrl * scenario.prj_duration_yrs
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration_yrs, scenario.wacc)

        self.capex_init = self.size * self.spec_capex
        self.capex_prj = eco.tce(self.capex_init,
                                 self.capex_init,  # TODO integrate cost decrease
                                 self.lifespan,
                                 scenario.prj_duration_yrs)
        self.capex_dis = eco.pce(self.capex_init,
                                 self.capex_init,  # TODO integrate cost decrease
                                 self.lifespan,
                                 scenario.prj_duration_yrs,
                                 scenario.wacc)
        self.capex_ann = eco.ann_recur(self.capex_init,
                                       self.lifespan,
                                       scenario.prj_duration_yrs,
                                       scenario.wacc,
                                       self.cost_decr)
        scenario.capex_init += self.capex_init
        scenario.capex_prj += self.capex_prj
        scenario.capex_dis += self.capex_dis
        scenario.capex_ann += self.capex_ann

        self.mntex_yrl = self.size * self.spec_mntex  # time-based maintenance
        self.mntex_sim = self.mntex_yrl * scenario.sim_yr_rat
        self.mntex_prj = self.mntex_yrl * scenario.prj_duration_yrs
        self.mntex_dis = eco.acc_discount(self.mntex_yrl,
                                          scenario.prj_duration_yrs,
                                          scenario.wacc)
        self.mntex_ann = eco.ann_recur(self.mntex_yrl,
                                       1,  # lifespan of 1 yr -> mntex happening yearly
                                       scenario.prj_duration_yrs,
                                       scenario.wacc,
                                       1)  # no cost decrease in mntex
        scenario.mntex_yrl += self.mntex_yrl
        scenario.mntex_prj += self.mntex_prj
        scenario.mntex_dis += self.mntex_dis
        scenario.mntex_ann += self.mntex_ann

        self.opex_sim = self.e_sim * self.spec_opex
        self.opex_yrl = self.opex_sim / scenario.sim_yr_rat  # linear scaling i.c.o. longer or shorter than 1 year
        self.opex_prj = self.opex_yrl * scenario.prj_duration_yrs
        self.opex_dis = eco.acc_discount(self.opex_yrl,
                                         scenario.prj_duration_yrs,
                                         scenario.wacc)
        self.opex_ann = eco.ann_recur(self.opex_yrl,
                                      1,  # lifespan of 1 yr -> opex happening yearly
                                      scenario.prj_duration_yrs,
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


class AggregateCommoditySystem(InvestBlock):
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


class CommoditySystem(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        # todo integrate Monte Carlo sim here

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, self.name, self.input_file_name + '.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=';',
                                skip_blank_lines=False)
        self.data.index = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)
        self.ph_data = None  # placeholder, is filled in "update_input_components"

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

        ac_bus               bus
          |<-x--------mc_ac---|---(CommoditySystem Instance)
          |                   |
          |-x-ac_mc---------->|---(CommoditySystem Instance)
                              |
                              |---(CommoditySystem Instance)
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.components.append(self.bus)

        self.inflow = solph.Transformer(label=f'ac_{self.name}',
                                        inputs={scenario.core.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                        outputs={self.bus: solph.Flow()},
                                        conversion_factors={self.bus: 1})
        scenario.components.append(self.inflow)

        self.outflow = solph.Transformer(label=f'{self.name}_ac',
                                         inputs={self.bus: solph.Flow(
                                             nominal_value={'uc': 0,
                                                            'cc': 0,
                                                            'tc': 0,
                                                            'v2v': 0,
                                                            'v2g': None}[self.int_lvl],
                                             variable_costs=run.eps_cost)},
                                         outputs={scenario.core.ac_bus: solph.Flow()},
                                         conversion_factors={scenario.core.ac_bus: 1})
        scenario.components.append(self.outflow)

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

    def update_input_components(self, *_):
        for commodity in self.commodities:
            commodity.update_input_components()


class StationaryEnergyStorage(InvestBlock):

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
                                                       inputs={scenario.core.dc_bus: solph.Flow()},
                                                       outputs={
                                                           scenario.core.dc_bus: solph.Flow(variable_cost=self.spec_opex)},
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
                                                       inputs={scenario.core.dc_bus: solph.Flow()},
                                                       outputs={
                                                           scenario.core.dc_bus: solph.Flow(variable_cost=self.spec_opex)},
                                                       loss_rate=0,  # TODO proper self discharge
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.ph_init_soc,
                                                       invest_relation_input_capacity=self.chg_crate,
                                                       invest_relation_output_capacity=self.dis_crate,
                                                       inflow_conversion_factor=self.chg_eff,
                                                       outflow_conversion_factor=self.dis_eff,
                                                       nominal_storage_capacity=self.size)
        scenario.components.append(self.ess)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_sink(scenario)  # StationaryEnergyStorage is a sink as positive power/energy exits the core

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.ess, scenario.core.dc_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[(scenario.core.dc_bus, self.ess)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ch = self.flow_in_ch - self.flow_out_ch  # inflow is positive

        self.flow = pd.concat([self.flow, self.flow_ch])

        self.sc_ch = solph.views.node(horizon.results, self.name)['sequences'][
            ((self.name, 'None'), 'storage_content')][horizon.ch_dti].shift(periods=1, freq=scenario.sim_timestep)
        # shift is needed as sc/soc is stored for end of timestep
        self.soc_ch = self.sc_ch / self.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge
        self.ph_init_soc = self.soc.iloc[-1]  # reset initial SOC for next prediction horizon

    def update_input_components(self, *_):

        self.ess.initial_storage_level = self.ph_init_soc


class ControllableSource(InvestBlock):

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

        self.bus = scenario.core.ac_bus

        if self.opt:
            self.src = solph.Source(label='gen_src',
                                    outputs={scenario.core.ac_bus: solph.Flow(
                                        investment=solph.Investment(ep_costs=self.eq_pres_cost),
                                        variable_costs=self.spec_opex)})
        else:
            self.src = solph.Source(label='gen_src',
                                    outputs={scenario.core.ac_bus: solph.Flow(nominal_value=self.size,
                                                                         variable_costs=self.spec_opex)})
        scenario.components.append(self.src)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.src, scenario.core.ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, *_):
        pass  # no sliced input data needed for controllable source, but function needs to be callable


class MobileCommodity:

    def __init__(self, name, parent, scenario, run):

        self.name = name
        self.parent = parent
        self.data = self.parent.data
        self.ph_data = None  # placeholder, is filled in update_input_components

        self.init_soc = xread(self.parent.name + '_init_soc', scenario.name, run.input_xdb)
        self.ph_init_soc = self.init_soc  # set first PH's initial state variables (only SOC)

        self.flow_in_ch = self.flow_out_ch = self.flow_ch = self.flow = pd.Series(dtype='float64')  # result data
        self.sc_ch = self.soc_ch = self.soc = pd.Series(dtype='float64')  # result data

        # Creation of permanent energy system components --------------------------------

        """
         bus               mc1_bus
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
        scenario.components.append(self.bus)

        self.inflow = solph.Transformer(label=f'mc_{self.name}',
                                        inputs={self.parent.bus: solph.Flow(nominal_value=self.parent.chg_pwr,
                                                                            variable_costs=run.eps_cost)},
                                        outputs={self.bus: solph.Flow()},
                                        conversion_factors={self.bus: self.parent.chg_eff})
        scenario.components.append(self.inflow)

        self.outflow = solph.Transformer(label=f'{self.name}_mc',
                                         inputs={self.bus: solph.Flow(nominal_value={'uc': 0,
                                                                                     'cc': 0,
                                                                                     'tc': 0,
                                                                                     'v2v': 1,
                                                                                     'v2g': 1}[
                                                                                        self.parent.int_lvl] * self.parent.dis_pwr,
                                                                      variable_costs=run.eps_cost)},
                                         outputs={self.parent.bus: solph.Flow()},
                                         conversion_factors={self.parent.bus: self.parent.dis_eff})
        scenario.components.append(self.outflow)

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
                                                       nominal_storage_capacity=self.parent.size)
        scenario.components.append(self.ess)

        self.snk = solph.Sink(label=f'{self.name}_snk',
                              inputs={self.bus: solph.Flow(nominal_value=1)})
        scenario.components.append(self.snk)

    def accumulate_results(self):
        pass  # TODO individual commodity result accumulation

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.bus, self.outflow)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[(self.inflow, self.bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ch = self.flow_in_ch - self.flow_out_ch  # inflow is positive

        self.flow = pd.concat([self.flow, self.flow_ch])

        self.sc_ch = solph.views.node(
            horizon.results, f'{self.name}_ess')['sequences'][((f'{self.name}_ess', 'None'), 'storage_content')][
            horizon.ch_dti].shift(periods=1, freq=scenario.sim_timestep)
        # shift is needed as sc/soc is stored for end of timestep
        self.soc_ch = self.sc_ch / self.parent.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge
        self.ph_init_soc = self.soc.iloc[-1]  # reset initial SOC for next prediction horizon

    def update_input_components(self, *_):

        self.ph_data = self.parent.ph_data

        # enable/disable transformers to mcx_bus depending on whether the commodity is at base
        self.inflow.inputs[self.parent.bus].max = self.ph_data[f'{self.name}_atbase']
        self.outflow.inputs[self.bus].max = self.ph_data[f'{self.name}_atbase']

        # define consumption data for sink (only enabled when detached from base)
        self.snk.inputs[self.bus].fix = self.ph_data[f'{self.name}_consumption']

        # set initial and minimum storage levels for coming prediction horizon
        self.ess.initial_storage_level = self.ph_init_soc
        self.ess.min_storage_level = self.ph_data[f'{self.name}_minsoc']


class PVSource(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.use_api = (xread(self.name + '_use_api', scenario.name, run.input_xdb) == 'True')

        self.input_file_name = xread(self.name + '_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, 'pv', f'{self.input_file_name}.csv')

        self.ph_data = None  # placeholder, is filled in "update_input_components"

        tf = timezonefinder.TimezoneFinder()
        self.utc = pytz.timezone('UTC')

        if self.use_api:  # API input selected
            self.latitude = xread(self.name + '_latitude', scenario.name, run.input_xdb)
            self.longitude = xread(self.name + '_longitude', scenario.name, run.input_xdb)
            self.timezone = pytz.timezone(tf.certain_timezone_at(lat=self.latitude, lng=self.longitude))
            self.api_startyear = self.timezone.localize(scenario.sim_starttime).astimezone(self.utc).year
            self.api_endyear = self.timezone.localize(scenario.sim_endtime).astimezone(self.utc).year
            self.data, self.meta, self.inputs = pvlib.iotools.get_pvgis_hourly(self.latitude,
                                                                               self.longitude,
                                                                               start=self.api_startyear,
                                                                               end=self.api_endyear,
                                                                               #url='https://re.jrc.ec.europa.eu/api/v5_2/',
                                                                               #raddatabase='PVGIS-SARAH2',  # TODO reinstate version and db for data post 2016
                                                                               components=False,
                                                                               outputformat='json',
                                                                               pvcalculation=True,
                                                                               peakpower=1,
                                                                               pvtechchoice='crystSi',
                                                                               mountingplace='free',
                                                                               loss=0,
                                                                               optimalangles=True,
                                                                               map_variables=True)
            # self.api_file_path = os.path.join(run.input_data_path, 'pv', f'pv_{scenario.name}_{run.runtimestamp}.csv')
            # self.data.to_csv(sep=',', path_or_buf=self.api_file_path)

        else:  # data input from fixed csv file
            self.data, self.meta, self.inputs = pvlib.iotools.read_pvgis_hourly(self.input_file_path,
                                                                                map_variables=True)
            self.latitude = self.meta['latitude']
            self.longitude = self.meta['longitude']
            self.timezone = pytz.timezone(tf.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        # convert to local time and remove timezone-awareness (model is only in one timezone)
        self.data.index = self.data.index.tz_convert(tz=self.timezone).tz_localize(tz=None)
        # PVGIS gives time slots as XX:06 - round to full hour
        self.data.index = self.data.index.round('H')
        # data is in W for a 1kWp PV array -> convert to specific power
        self.data['p_spec'] = self.data['P'] / 1e3

        self.data = self.data[['p_spec', 'wind_speed']]

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
        scenario.components.append(self.bus)

        self.outflow = solph.Transformer(label=f'{self.name}_dc',
                                         inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                         outputs={scenario.core.dc_bus: solph.Flow()},
                                         conversion_factors={self.bus: self.transformer_eff})
        scenario.components.append(self.outflow)

        # input data from PVGIS is added in function "update_input_components"
        if self.opt:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={self.bus: solph.Flow(investment=solph.Investment(
                                        ep_costs=self.eq_pres_cost),
                                        variable_cost=self.spec_opex)})
        else:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={self.bus: solph.Flow(nominal_value=self.size,
                                                                  variable_cost=self.spec_opex)})
        scenario.components.append(self.src)

        self.exc = solph.Sink(label=f'{self.name}_exc',
                              inputs={self.bus: solph.Flow()})
        scenario.components.append(self.exc)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.core.dc_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, *_):

        self.src.outputs[self.bus].fix = self.ph_data['p_spec']


class FixedDemand:

    def __init__(self, name, scenario, run):

        self.name = name

        self.input_file_name = xread('dem_filename', scenario.name, run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, 'dem', f'{self.input_file_name}.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=',',
                                skip_blank_lines=False)
        self.data.index = pd.date_range(start=scenario.sim_starttime,
                                        periods=len(self.data),
                                        freq=scenario.sim_timestep)
        self.ph_data = None  # placeholder

        self.flow_ch = pd.Series(dtype='float64')  # empty dataframe for result concatenation
        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation

        self.e_sim = 0  # empty placeholder for cumulative results
        self.e_yrl = 0  # empty placeholder for cumulative results
        self.e_prj = 0  # empty placeholder for cumulative results
        self.e_dis = 0  # empty placeholder for cumulative results

        scenario.blocks.append(self)

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results

        ac_bus
          |
          |-x->dem_snk
          |
        """

        self.snk = solph.Sink(label='dem_snk',
                              inputs={scenario.core.ac_bus: solph.Flow(nominal_value=1)})
        scenario.components.append(self.snk)

    def accumulate_results(self, scenario):

        # No super function as FixedDemand is not an InvestBlock child (where accumulate_invest_results lives)

        self.e_sim = self.flow.sum()
        self.e_yrl = self.e_sim / scenario.sim_yr_rat
        self.e_prj = self.e_yrl * scenario.prj_duration_yrs
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration_yrs, scenario.wacc)

        scenario.e_sim_del += self.e_sim
        scenario.e_yrl_del += self.e_yrl
        scenario.e_prj_del += self.e_prj
        scenario.e_dis_del += self.e_dis

    def get_ch_results(self, horizon, scenario):
        self.flow_ch = horizon.results[(scenario.core.ac_bus, self.snk)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, scenario):
        # new ph data slice is created during initialization of the PredictionHorizon
        self.snk.inputs[scenario.core.ac_bus].fix = self.ph_data['Power']


class WindSource(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.input_file_name = xread(self.name + '_filename', scenario. run.input_xdb)
        self.input_file_path = os.path.join(run.input_data_path, 'wind', self.input_file_name + '.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=',',
                                skip_blank_lines=False)
        self.data.index = pd.date_range(start=scenario.sim_starttime,
                                          periods=len(self.data),
                                          freq=scenario.sim_timestep)

        self.ph_data = None  # placeholder, is filled in "update_input_components"

        # self.transformer_eff = xread(self.name + '_eff', scenario.name, run.input_xdb)  # TODO check if this can be omitted

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
        scenario.components.append(self.bus)

        self.outflow = solph.Transformer(label=f'{self.name}_ac',
                                         inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                         outputs={scenario.core.ac_bus: solph.Flow()},
                                         conversion_factors={scenario.core.ac_bus: 1})  # TODO proper efficiency
        scenario.components.append(self.outflow)

        self.exc = solph.Sink(label=f'{self.name}_exc',
                              inputs={self.bus: solph.Flow()})
        scenario.components.append(self.exc)

        if self.opt:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={self.bus: solph.Flow(fix=self.ph_data['P'],
                                                                  investment=solph.Investment(ep_costs=run.eps_cost,
                                                                                              variable_cost=self.spec_opex))})
        else:
            self.src = solph.Source(label=f'{self.name}_src',
                                    outputs={scenario.components.wind_bus: solph.Flow(fix=self.ph_data['P'],
                                                                                            nominal_value=self.size,
                                                                                            variable_cost=self.spec_opex)})
        scenario.components.append(self.src)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.core.ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, *_):

        self.src.outputs[self.bus].fix = self.ph_data['P']