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


class InvestBlock:

    def __init__(self, name, scenario, run):

        self.name = name

        self.opt = (xread(f'{self.name}_opt', scenario.name, run) == 'True')

        # TODO add "existing" block for grid connection

        if self.opt and scenario.strategy != 'go':
            run.logger.warning(f'{self.name} component size optimization not implemented for any'
                               f' other strategy than \"GO\" - disabling size optimization')
            self.opt = False

        if self.opt or isinstance(self, SystemCore):  # SystemCore has two sizes and is initialized in its own __init__
            self.size = None
        else:
            self.size = xread(self.name + '_cs', scenario.name, run)

        self.capex_spec = xread(self.name + '_sce', scenario.name, run)
        self.mntex_spec = xread(self.name + '_sme', scenario.name, run)
        self.opex_spec = xread(self.name + '_soe', scenario.name, run)

        self.lifespan = xread(self.name + '_ls', scenario.name, run)
        self.cost_decr = xread(self.name + '_cdc', scenario.name, run)
        self.transformer_eff = xread(self.name + '_eff', scenario.name, run)

        self.capex_adj = eco.adj_ce(self.capex_spec,
                                    self.mntex_spec,
                                    self.lifespan,
                                    scenario.wacc)  # adjusted ce (including maintenance) of the component in $/W

        self.eq_pres_cost = eco.ann_recur(self.capex_adj,
                                          self.lifespan,
                                          scenario.prj_duration_yrs,
                                          scenario.wacc,
                                          self.cost_decr)

        self.flow_ch = pd.Series(dtype='float64')  # empty dataframe for result concatenation
        self.flow = pd.Series(dtype='float64')  # empty dataframe for result concatenation

        '''For bidirectional blocks (CommoditySystem, SystemCore, and StationaryEnergyStorage instances, these
        denote the difference between the directions to give a total flow. For economic calculations however,
        we need the sum of both. This is initialized in every single class.'''

        self.e_sim = self.e_yrl = self.e_prj = self.e_dis = None  # empty placeholders for cumulative results
        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = None
        self.mntex_sim = self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = None
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = None
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = None

        scenario.blocks.append(self)

    def accumulate_invest_results(self, scenario):

        if hasattr(self, 'flow_sum'):
            self.e_sim = self.flow_sum.sum()  # for all bidirectional components
        else:
            self.e_sim = self.flow.sum()
        self.e_yrl = self.e_sim / scenario.sim_yr_rat
        self.e_prj = self.e_yrl * scenario.prj_duration_yrs
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration_yrs, scenario.wacc)

        if isinstance(self, CommoditySystem):
            self.capex_init = self.size * self.capex_spec * self.commodity_num
        elif isinstance(self, SystemCore):
            self.capex_init = (self.acdc_size + self.dcac_size) * self.capex_spec
        else:
            self.capex_init = self.size * self.capex_spec

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

        if isinstance(self, CommoditySystem):
            self.mntex_yrl = self.size * self.mntex_spec * self.commodity_num  # time-based maintenance
        elif isinstance(self, SystemCore):
            self.mntex_yrl = (self.acdc_size + self.dcac_size) * self.mntex_spec
        else:
            self.mntex_yrl = self.size * self.mntex_spec  # time-based maintenance

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

        self.opex_sim = self.e_sim * self.opex_spec
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
    pass  # Feature not included anymore, but theoretically possible


class CommoditySystem(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        # todo integrate DES trigger here

        self.input_file_name = xread(self.name + '_filename', scenario.name, run)
        self.input_file_path = os.path.join(run.input_data_path, self.name, self.input_file_name + '.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=';',
                                skip_blank_lines=False)
        self.data.index = pd.date_range(start=scenario.sim_starttime,
                                        periods=len(self.data),
                                        freq=scenario.sim_timestep)
        self.ph_data = None  # placeholder, is filled in "update_input_components"

        self.apriori_lvls = ['uc']  # integration levels at which power consumption is determined a priori

        self.commodity_num = xread(self.name + '_num', scenario.name, run)
        self.commodity_agr = xread(self.name + '_agr', scenario.name, run)  # TODO enable aggregated simulation

        self.init_soc = xread(self.name + '_init_soc', scenario.name, run)

        self.chg_pwr = xread(self.name + '_chg_pwr', scenario.name, run)
        self.dis_pwr = xread(self.name + '_dis_pwr', scenario.name, run)
        self.chg_eff = xread(self.name + '_charge_eff', scenario.name, run)
        self.dis_eff = xread(self.name + '_discharge_eff', scenario.name, run)

        self.int_lvl = xread(self.name + '_int_lvl', scenario.name, run)  # charging integration level

        self.flow_in_ch = self.flow_out_ch = pd.Series(dtype='float64')  # result data
        self.flow_sum_ch = self.flow_sum = pd.Series(dtype='float64')  # result data for cost calculation

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results

        ac_bus               bus
          |<-x--------mc_ac---|---(MobileCommodity Instance)
          |                   |
          |-x-ac_mc---------->|---(MobileCommodity Instance)
                              |
                              |---(MobileCommodity Instance)
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.components.append(self.bus)

        self.inflow = solph.components.Transformer(label=f'ac_{self.name}',
                                                   inputs={scenario.core.ac_bus: solph.Flow(
                                                       variable_costs=run.eps_cost)},
                                                   outputs={self.bus: solph.Flow()},
                                                   conversion_factors={self.bus: 1})
        scenario.components.append(self.inflow)

        self.outflow = solph.components.Transformer(label=f'{self.name}_ac',
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

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_sink(scenario)
        # CommoditySystem is a sink as positive power/energy exits the core

        for commodity in self.commodities:
            commodity.accumulate_results(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.outflow, scenario.core.ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[(scenario.core.ac_bus, self.inflow)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ch = self.flow_in_ch - self.flow_out_ch  # for energy considerations, inflow is positive
        self.flow_sum_ch = self.flow_in_ch + self.flow_out_ch  # for cost considerations

        self.flow = pd.concat([self.flow, self.flow_ch])
        self.flow_sum = pd.concat([self.flow_sum, self.flow_sum_ch])

        for commodity in self.commodities:
            commodity.get_ch_results(horizon, scenario)

    def update_input_components(self, *_):
        for commodity in self.commodities:
            commodity.update_input_components()


class ControllableSource(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        """
        x denotes the flow measurement point in results

        ac_bus
          |
          |<-x-gen
          |
        """

        self.bus = scenario.core.ac_bus

        if self.opt:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={scenario.core.ac_bus: solph.Flow(
                                                   investment=solph.Investment(ep_costs=self.eq_pres_cost),
                                                   variable_costs=self.opex_spec)}
                                               )
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={scenario.core.ac_bus: solph.Flow(nominal_value=self.size,
                                                                                         variable_costs=self.opex_spec)}
                                               )
        scenario.components.append(self.src)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.src, scenario.core.ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, *_):
        pass  # no sliced input data needed for controllable source, but function needs to be callable


class FixedDemand:

    def __init__(self, name, scenario, run):
        self.name = name

        self.input_file_name = xread('dem_filename', scenario.name, run)
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

        self.snk = solph.components.Sink(label='dem_snk',
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


class MobileCommodity:

    def __init__(self, name, parent, scenario, run):

        self.name = name
        self.parent = parent
        self.size = self.parent.size
        self.chg_pwr = self.parent.chg_pwr

        colnames = [coln for coln in self.parent.data.columns if self.name in coln]

        self.data = self.parent.data.loc[:, colnames]

        # remove CommoditySystem name and Commodity numbers in column headers
        self.data.columns = self.data.columns.str.split('_').str[1]  # remove commodity's name from column names
        self.ph_data = None  # placeholder, is filled in update_input_components

        self.init_soc = self.parent.init_soc
        self.ph_init_soc = self.init_soc  # set first PH's initial state variables (only SOC)

        self.e_sim = self.e_yrl = self.e_prj = self.e_dis = None  # empty placeholders for cumulative results
        self.flow_in_ch = self.flow_out_ch = self.flow_ch = self.flow = pd.Series(dtype='float64')  # result data

        self.sc_ch = self.soc_ch = pd.Series(dtype='float64')  # result data
        self.soc = pd.Series(data=self.init_soc,
                             index=scenario.sim_dti[0:1],
                             dtype='float64')
        # add initial sc (and later soc) to the timeseries of the first horizon (otherwise not recorded)

        # Creation of permanent energy system components --------------------------------

        """
        in case of integration level "uc" or other rule-based scheduling methods
        bus               mc1_bus
          |<---------mc1_mc-x-|
          |                   |
          |---mc_mc1-------x->|-->mc1_snk
          |
          |                 mc2_bus
          |<---------mc2_mc---|
          |                   |
          |---mc_mc2--------->|-->mc2_snk
        
        in case of higher integration levels
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

        self.inflow = solph.components.Transformer(label=f'mc_{self.name}',
                                                   inputs={
                                                       self.parent.bus: solph.Flow(nominal_value=self.parent.chg_pwr,
                                                                                   variable_costs=run.eps_cost)},
                                                   outputs={self.bus: solph.Flow()},
                                                   conversion_factors={self.bus: self.parent.chg_eff})
        scenario.components.append(self.inflow)

        self.outflow_enable = True if self.parent.int_lvl in ['v2v', 'v2g'] else False
        self.outflow = solph.components.Transformer(label=f'{self.name}_mc',
                                                    inputs={self.bus: solph.Flow(nominal_value=self.outflow_enable
                                                                                               * self.parent.dis_pwr,
                                                                                 variable_costs=run.eps_cost)},
                                                    outputs={self.parent.bus: solph.Flow()},
                                                    conversion_factors={self.parent.bus: self.parent.dis_eff})
        scenario.components.append(self.outflow)

        if self.parent.int_lvl in self.parent.apriori_lvls:  # dispatch is known a priori --> simple sink is sufficient
            self.calc_uc_power(scenario)
        else:  # Storage is only added if MCs have flexibility potential
            if self.parent.opt:  # dispatch is optimized later --> commodity is modeled as storage and sink
                self.ess = solph.components.GenericStorage(label=f'{self.name}_ess',
                                                           inputs={self.bus: solph.Flow()},
                                                           outputs={self.bus: solph.Flow(
                                                               variable_costs=self.parent.opex_spec)},
                                                           loss_rate=0,  # TODO integrate self discharge
                                                           balanced=False,
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
                                                               variable_costs=self.parent.opex_spec)},
                                                           loss_rate=0,  # TODO integrate self discharge
                                                           balanced=False,
                                                           initial_storage_level=self.ph_init_soc,
                                                           inflow_conversion_factor=1,
                                                           # efficiency already modeled in transformers
                                                           outflow_conversion_factor=1,
                                                           # efficiency already modeled in transformers
                                                           max_storage_level=1,
                                                           nominal_storage_capacity=self.parent.size)
            scenario.components.append(self.ess)

        self.snk = solph.components.Sink(label=f'{self.name}_snk',
                                         inputs={self.bus: solph.Flow()})
        # actual values are set later in update_input_components for each prediction horizon
        scenario.components.append(self.snk)

    def accumulate_results(self, scenario):
        self.e_sim = self.flow.sum()
        self.e_yrl = self.e_sim / scenario.sim_yr_rat
        self.e_prj = self.e_yrl * scenario.prj_duration_yrs
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration_yrs, scenario.wacc)

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

    def calc_uc_power(self, scenario):
        """Converting availability and consumption data of commodities into a power timeseries for uncoordinated
         (i.e. unoptimized and starting at full power after return) charging of commodity"""

        uc_power = []
        soc = [self.init_soc]

        minsoc_inz, = self.data['minsoc'].to_numpy().nonzero()
        # get the integer indices of all leaving (nonzero) minsoc rows in the data

        for dtindex, row in self.data.iterrows():
            intindex = self.data.index.get_loc(dtindex)  # get row number
            if row['atbase'] == 1:  # commodity is chargeable
                try:
                    dep_inxt = min(minsoc_inz[minsoc_inz >= intindex])  # find next nonzero min SOC (= next departure)
                    dep_time = self.data.index[dep_inxt]
                    dep_soc = self.data.loc[dep_time, 'minsoc']  # get the SOC to recharge to
                except ValueError:  # when there is no further departure
                    dep_soc = 1
                e_tominsoc = (dep_soc - soc[-1]) * self.size  # energy to be recharged to departure SOC in Wh
                p_tominsoc = e_tominsoc / scenario.sim_timestep_hours  # power to recharge to departure SOC in one step
                p_maxchg = self.chg_pwr * self.parent.chg_eff  # max rechargeable energy in one timestep in Wh
                p_act = min(p_maxchg, p_tominsoc)  # reduce chg power in final step to just reach departure SOC
                p_uc = p_act / self.parent.chg_eff # convert back to grid side power
            else:  # commodity is not chargeable, but might be discharged
                e_dis = -1 * row['consumption']
                p_act = e_dis / scenario.sim_timestep_hours
                p_uc = 0  # discharged energy does not directly affect UC power

            uc_power.append(p_uc)
            soc_delta = p_act * scenario.sim_timestep_hours / self.size
            soc.append(soc[-1] + soc_delta)

        self.data['uc_power'] = uc_power
        self.data['soc'] = soc[:-1]  # TODO check whether SOC indexing fits optimization output
        pass

    def update_input_components(self, *_):

        if self.parent.int_lvl in self.parent.apriori_lvls:
            # define consumption data for sink (as per uc power calculation)
            self.snk.inputs[self.bus].fix = self.ph_data['uc_power']
        else:
            # enable/disable transformers to mcx_bus depending on whether the commodity is at base
            self.inflow.inputs[self.parent.bus].max = self.ph_data['atbase']
            self.outflow.inputs[self.bus].max = self.ph_data['atbase']

            # define consumption data for sink (only enabled when detached from base)
            self.snk.inputs[self.bus].fix = self.ph_data['consumption']

            # set initial and minimum storage levels for coming prediction horizon
            self.ess.initial_storage_level = self.ph_init_soc
            self.ess.min_storage_level = self.ph_data['minsoc']

        # TODO something is still not right with these settings - EVs stay at high SOC and never consume...


class PVSource(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.use_api = (xread(self.name + '_use_api', scenario.name, run) == 'True')

        self.input_file_name = xread(self.name + '_filename', scenario.name, run)
        self.input_file_path = os.path.join(run.input_data_path, 'pv', f'{self.input_file_name}.csv')

        self.ph_data = None  # placeholder, is filled in "update_input_components"

        tf = timezonefinder.TimezoneFinder()
        self.utc = pytz.timezone('UTC')

        if self.use_api:  # API input selected
            self.latitude = xread(self.name + '_latitude', scenario.name, run)
            self.longitude = xread(self.name + '_longitude', scenario.name, run)
            self.timezone = pytz.timezone(tf.certain_timezone_at(lat=self.latitude, lng=self.longitude))
            self.api_startyear = self.timezone.localize(scenario.sim_starttime).astimezone(self.utc).year
            self.api_endyear = self.timezone.localize(scenario.sim_endtime).astimezone(self.utc).year
            self.data, self.meta, self.inputs = pvlib.iotools.get_pvgis_hourly(self.latitude,
                                                                               self.longitude,
                                                                               start=self.api_startyear,
                                                                               end=self.api_endyear,
                                                                               url='https://re.jrc.ec.europa.eu/api/v5_2/',
                                                                               raddatabase='PVGIS-SARAH2',
                                                                               components=False,
                                                                               outputformat='json',
                                                                               pvcalculation=True,
                                                                               peakpower=1,
                                                                               pvtechchoice='crystSi',
                                                                               mountingplace='free',
                                                                               loss=0,
                                                                               optimalangles=True,
                                                                               map_variables=True)

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

        self.outflow = solph.components.Transformer(label=f'{self.name}_dc',
                                                    inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                                    outputs={scenario.core.dc_bus: solph.Flow()},
                                                    conversion_factors={self.bus: self.transformer_eff})
        scenario.components.append(self.outflow)

        # input data from PVGIS is added in function "update_input_components"
        if self.opt:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(investment=solph.Investment(
                                                   ep_costs=self.eq_pres_cost),
                                                   variable_costs=self.opex_spec)})
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(nominal_value=self.size,
                                                                             variable_costs=self.opex_spec)})
        scenario.components.append(self.src)

        self.exc = solph.components.Sink(label=f'{self.name}_exc',
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


class StationaryEnergyStorage(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.chg_eff = xread(self.name + '_chg_eff', scenario.name, run)
        self.dis_eff = xread(self.name + '_dis_eff', scenario.name, run)
        self.chg_crate = xread(self.name + '_chg_crate', scenario.name, run)
        self.dis_crate = xread(self.name + '_dis_crate', scenario.name, run)

        self.init_soc = xread(self.name + '_init_soc', scenario.name, run)
        self.ph_init_soc = self.init_soc  # TODO actually necessary?

        self.flow_in_ch = self.flow_out_ch = pd.Series(dtype='float64')  # result data
        self.flow_sum_ch = self.flow_sum = pd.Series(dtype='float64')  # result data for cost calculation
        self.sc_ch = self.soc_ch = pd.Series(dtype='float64')  # result data
        self.soc = pd.Series(data=self.init_soc,
                             index=scenario.sim_dti[0:1],
                             dtype='float64')
        # add initial sc (and later soc) to the timeseries of the first horizon (otherwise not recorded)

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
                                                           scenario.core.dc_bus: solph.Flow(
                                                               variable_costs=self.opex_spec)},
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
                                                           scenario.core.dc_bus: solph.Flow(
                                                               variable_costs=self.opex_spec)},
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
        self.accumulate_energy_results_sink(
            scenario)  # StationaryEnergyStorage is a sink as positive power/energy exits the core
        # TODO is storage really a sink component and should be added to delivered energy?

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.ess, scenario.core.dc_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[(scenario.core.dc_bus, self.ess)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ch = self.flow_in_ch - self.flow_out_ch  # for energy summary, inflow is positive
        self.flow_sum_ch = self.flow_in_ch + self.flow_out_ch  # for cost calculation

        self.flow = pd.concat([self.flow, self.flow_ch])
        self.flow_sum = pd.concat([self.flow_sum, self.flow_sum_ch])

        self.sc_ch = solph.views.node(horizon.results, self.name)['sequences'][
            ((self.name, 'None'), 'storage_content')][horizon.ch_dti].shift(periods=1, freq=scenario.sim_timestep)
        # shift is needed as sc/soc is stored for end of timestep

        self.soc_ch = self.sc_ch / self.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge
        self.ph_init_soc = self.soc.iloc[-1]  # reset initial SOC for next prediction horizon

    def update_input_components(self, *_):

        self.ess.initial_storage_level = self.ph_init_soc


class SystemCore(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        if self.opt:
            self.acdc_size = None
            self.dcac_size = None
        else:
            self.acdc_size = xread(self.name + '_acdc_cs', scenario.name, run)
            self.dcac_size = xread(self.name + '_dcac_cs', scenario.name, run)

        self.flow_acdc_ch = self.flow_dcac_ch = pd.Series(dtype='float64')  # result data
        self.flow_sum_ch = self.flow_sum = pd.Series(dtype='float64')  # result data

        """
        x denotes the flow measurement point in results
        
        dc_bus              ac_bus
          |                   |
          |-x-dc_ac---------->|
          |                   |
          |<----------ac_dc-x-|
        """

        self.ac_bus = solph.Bus(label='ac_bus')
        scenario.components.append(self.ac_bus)

        self.dc_bus = solph.Bus(label='dc_bus')
        scenario.components.append(self.dc_bus)

        if self.opt:
            self.ac_dc = solph.components.Transformer(label='ac_dc',
                                                      inputs={self.ac_bus: solph.Flow(investment=solph.Investment(
                                                          ep_costs=self.eq_pres_cost),
                                                          variable_costs=self.opex_spec)},
                                                      outputs={self.dc_bus: solph.Flow()},
                                                      conversion_factors={self.dc_bus: self.transformer_eff})

            self.dc_ac = solph.components.Transformer(label='dc_ac',
                                                      inputs={self.dc_bus: solph.Flow(investment=solph.Investment(
                                                          ep_costs=self.eq_pres_cost),
                                                          variable_costs=self.opex_spec)},
                                                      outputs={self.ac_bus: solph.Flow()},
                                                      conversion_factors={self.dc_bus: self.transformer_eff})
        else:
            self.ac_dc = solph.components.Transformer(label='ac_dc',
                                                      inputs={self.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                                      outputs={self.dc_bus: solph.Flow()},
                                                      conversion_factors={self.dc_bus: self.transformer_eff})

            self.dc_ac = solph.components.Transformer(label='dc_ac',
                                                      inputs={self.dc_bus: solph.Flow(variable_costs=run.eps_cost)},
                                                      outputs={self.ac_bus: solph.Flow()},
                                                      conversion_factors={self.dc_bus: self.transformer_eff})

        scenario.components.append(self.ac_dc)
        scenario.components.append(self.dc_ac)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_acdc_ch = horizon.results[(scenario.core.ac_bus, self.ac_dc)]['sequences']['flow'][horizon.ch_dti]
        self.flow_dcac_ch = horizon.results[(scenario.core.dc_bus, self.dc_ac)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ch = self.flow_dcac_ch - self.flow_acdc_ch  # for energy summary, dc->ac is positive
        self.flow_sum_ch = self.flow_dcac_ch + self.flow_acdc_ch  # for cost calculation, flows need to be summed up

        self.flow = pd.concat([self.flow, self.flow_ch])
        self.flow_sum = pd.concat([self.flow_sum, self.flow_sum_ch])

    def update_input_components(self, *_):
        pass  # function needs to be callable


class WindSource(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.input_file_name = xread(self.name + '_filename', scenario.name, run)
        self.input_file_path = os.path.join(run.input_data_path, 'wind', self.input_file_name + '.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=',',
                                skip_blank_lines=False)
        self.data.index = pd.date_range(start=scenario.sim_starttime,
                                        periods=len(self.data),
                                        freq=scenario.sim_timestep)

        self.ph_data = None  # placeholder, is filled in "update_input_components"

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

        self.outflow = solph.components.Transformer(label=f'{self.name}_ac',
                                                    inputs={self.bus: solph.Flow(variable_costs=run.eps_cost)},
                                                    outputs={scenario.core.ac_bus: solph.Flow()},
                                                    conversion_factors={scenario.core.ac_bus: self.transformer_eff})
        scenario.components.append(self.outflow)

        self.exc = solph.components.Sink(label=f'{self.name}_exc',
                                         inputs={self.bus: solph.Flow()})
        scenario.components.append(self.exc)

        # TODO make wind speed from PVGIS usable - then it's also time based and not just stepwise...

        if self.opt:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(investment=solph.Investment(
                                                   ep_costs=self.eq_pres_cost),
                                                   variable_costs=self.opex_spec)})
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(nominal_value=self.size,
                                                                             variable_costs=self.opex_spec)})
        scenario.components.append(self.src)

    def accumulate_results(self, scenario):

        self.accumulate_invest_results(scenario)
        self.accumulate_energy_results_source(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.core.ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, *_):

        self.src.outputs[self.bus].fix = self.ph_data['P']
