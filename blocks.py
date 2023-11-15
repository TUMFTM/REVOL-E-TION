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

import numpy as np
import oemof.solph as solph
import os
import pandas as pd
import pvlib
import pytz
import timezonefinder

import economics as eco


###############################################################################
# Class definitions
###############################################################################


class InvestBlock:

    def __init__(self, name, scenario, run):

        self.name = name

        self.parameters = scenario.parameters.loc[self.name]
        for key, value in self.parameters.items():
            setattr(self, key, value)  # this sets all the parameters defined in the json file

        # TODO add "existing" block for grid connection

        if isinstance(self, SystemCore):
            self.size = None  # SystemCore has two sizes and is initialized in its own __init__
            if self.acdc_size == 'opt' or self.dcac_size == 'opt':
                self.opt = True
                if self.acdc_size == 'opt':
                    self.acdc_size = None
                if self.dcac_size == 'opt':
                    self.dcac_size = None
            else:
                self.opt = False
        elif self.size == 'opt':  # all non-SystemCore blocks that are to be optimzed
            self.opt = True
            # size will now be set when getting results
            self.size = None
        elif isinstance(self.size, float) or self.size is None:  # all non-SystemCore blocks that are not to be optimzed
            self.opt = False
            # size is given per commodity in scenario data
            if isinstance(self, CommoditySystem):
                self.size_pc = self.size  # pc = per commodity
                self.size = self.size * self.num

        else:
            run.logger.warning(f'Scenario {scenario.name}: \"{self.name}_size\" variable in scenario definition'
                               f' needs to be either a number or \"opt\" - exiting')
            exit()  # TODO exit scenario instead of entire execution

        if self.opt and scenario.strategy != 'go':
            run.logger.warning(f'Scenario {scenario.name}: {self.name} component size optimization not implemented'
                               f' for any other strategy than \"GO\" - exiting')
            exit()  # TODO exit scenario instead of entire execution

        # Calculate adjusted ce (including maintenance) of the component in $/W
        self.ace = eco.adj_ce(self.capex_spec, self.mntex_spec, self.ls, scenario.wacc)
        # Calculate equivalent present cost
        self.epc = eco.ann_recur(self.ace, self.ls, scenario.prj_duration_yrs, scenario.wacc, self.cdc)

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

        scenario.blocks[self.name] = self

    def calc_eco_results(self, scenario):
        """
        Calculating cost values from parameters. Objective function results are not directly employed as these count
        e.g. capital expenses for each horizon and disregard economic projection to more than the sim timeframe.
        """

        ###########
        # Initial & recurring capital expenses
        ###########

        # for CommoditySystems, size is the sum of all commodity sizes
        self.capex_init = self.size * self.capex_spec
        self.capex_prj = eco.tce(self.capex_init,
                                 self.capex_init,  # TODO integrate cost decrease
                                 self.ls,
                                 scenario.prj_duration_yrs)
        self.capex_dis = eco.pce(self.capex_init,
                                 self.capex_init,  # TODO integrate cost decrease
                                 self.ls,
                                 scenario.prj_duration_yrs,
                                 scenario.wacc)
        self.capex_ann = eco.ann_recur(self.capex_init,
                                       self.ls,
                                       scenario.prj_duration_yrs,
                                       scenario.wacc,
                                       self.cdc)

        scenario.capex_init += self.capex_init
        scenario.capex_prj += self.capex_prj
        scenario.capex_dis += self.capex_dis
        scenario.capex_ann += self.capex_ann

        ###########
        # Time-based Maintenance Expenses
        ###########

        self.mntex_yrl = self.size * self.mntex_spec
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

        ###########
        # Operational & thoughput-based maintenance expenses
        ###########

        if isinstance(self, SystemCore):
            self.opex_sim = (self.e_sim_dcac + self.e_sim_acdc) * self.opex_spec
        elif isinstance(self, StationaryEnergyStorage):
            self.opex_sim = self.e_sim_in * self.opex_spec
        elif isinstance(self, CommoditySystem):
            self.opex_sys = self.e_sim_in * self.sys_chg_soe + self.e_sim_out * self.sys_dis_soe
            self.opex_commodities = 0
            self.opex_commodities_ext = 0
            for commodity in self.commodities.values():
                commodity.opex_sim = commodity.e_sim_in * self.opex_spec
                commodity.opex_sim_ext = commodity.flow_ext_ac @ self.ext_ac_costs + \
                                         commodity.flow_ext_dc @ self.ext_dc_costs
                self.opex_commodities += commodity.opex_sim
                self.opex_commodities_ext += commodity.opex_sim_ext
            self.opex_sim = self.opex_sys + self.opex_commodities
        elif isinstance(self, ControllableSource):
            self.opex_sim = self.flow @ self.opex_spec * scenario.timestep_hours  # @ is dot product (Skalarprodukt)
        else:  # all unidirectional source & sink blocks
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

        ###########
        # Total expenses
        ###########

        self.totex_sim = self.capex_init + self.mntex_sim + self.opex_sim
        self.totex_prj = self.capex_prj + self.mntex_prj + self.opex_prj
        self.totex_dis = self.capex_dis + self.mntex_dis + self.opex_dis
        self.totex_ann = self.capex_ann + self.mntex_ann + self.opex_ann

        scenario.totex_sim += self.totex_sim
        scenario.totex_prj += self.totex_prj
        scenario.totex_dis += self.totex_dis
        scenario.totex_ann += self.totex_ann

    def calc_energy_results_bidi(self, scenario):
        """
        Calculate the energy results for bidirectional blocks (CommoditySystems and StationaryEnergyStorages).
        SystemCore is handled differently as there is no in/out, rather acdc/dcac.
        """

        self.e_sim_in = self.flow_in.sum() * scenario.timestep_hours  # flow values are powers --> conversion to Wh
        self.e_sim_out = self.flow_out.sum() * scenario.timestep_hours
        self.e_yrl_in = self.e_sim_in / scenario.sim_yr_rat
        self.e_yrl_out = self.e_sim_out / scenario.sim_yr_rat
        self.e_prj_in = self.e_yrl_in * scenario.prj_duration_yrs
        self.e_prj_out = self.e_yrl_out * scenario.prj_duration_yrs
        self.e_dis_in = eco.acc_discount(self.e_yrl_in, scenario.prj_duration_yrs, scenario.wacc)
        self.e_dis_out = eco.acc_discount(self.e_yrl_out, scenario.prj_duration_yrs, scenario.wacc)

        self.flow = self.flow_in - self.flow_out  # for plotting

        if self.e_sim_in > self.e_sim_out:
            self.e_sim_del = self.e_sim_in - self.e_sim_out
            self.e_yrl_del = self.e_sim_del / scenario.sim_yr_rat
            self.e_prj_del = self.e_yrl_del * scenario.prj_duration_yrs
            self.e_dis_del = eco.acc_discount(self.e_yrl_del, scenario.prj_duration_yrs, scenario.wacc)

            scenario.e_sim_del += self.e_sim_del
            scenario.e_yrl_del += self.e_yrl_del
            scenario.e_prj_del += self.e_prj_del
            scenario.e_dis_del += self.e_dis_del

        else:  # storage was emptied
            self.e_sim_pro = self.e_sim_out - self.e_sim_in
            self.e_yrl_pro = self.e_sim_pro / scenario.sim_yr_rat
            self.e_prj_pro = self.e_yrl_pro * scenario.prj_duration_yrs
            self.e_dis_pro = eco.acc_discount(self.e_yrl_pro, scenario.prj_duration_yrs, scenario.wacc)

            scenario.e_sim_pro += self.e_sim_pro
            scenario.e_yrl_pro += self.e_yrl_pro
            scenario.e_prj_pro += self.e_prj_pro
            scenario.e_dis_pro += self.e_dis_pro
        pass

    def calc_energy_results_source(self, scenario):

        self.e_sim = self.flow.sum() * scenario.timestep_hours  # flow values are powers --> conversion to Wh
        self.e_yrl = self.e_sim / scenario.sim_yr_rat
        self.e_prj = self.e_yrl * scenario.prj_duration_yrs
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration_yrs, scenario.wacc)

        scenario.e_sim_pro += self.e_sim
        scenario.e_yrl_pro += self.e_yrl
        scenario.e_prj_pro += self.e_prj
        scenario.e_dis_pro += self.e_dis

    def calc_energy_results_sink(self, scenario):

        self.e_sim = self.flow.sum() * scenario.timestep_hours  # flow values are powers --> conversion to Wh
        self.e_yrl = self.e_sim / scenario.sim_yr_rat
        self.e_prj = self.e_yrl * scenario.prj_duration_yrs
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration_yrs, scenario.wacc)

        scenario.e_sim_del += self.e_sim
        scenario.e_yrl_del += self.e_yrl
        scenario.e_prj_del += self.e_prj
        scenario.e_dis_del += self.e_dis

    def get_opt_size(self, horizon):

        """
        Get back the optimal size from solver results. Can only be reached in GO strategy, as only there,
        size opt is feasible. Before, size variables to be optimized are none.
        :param horizon: recently optimized PredictionHorizon
        :return: none, saves self.size value
        """

        source_types = (PVSource, WindSource, ControllableSource)

        if isinstance(self, StationaryEnergyStorage):
            self.size = horizon.results[(self.ess, None)]["scalars"]["invest"]
        elif isinstance(self, source_types):
            self.size = horizon.results[(self.src, self.bus)]['scalars']['invest']
        elif isinstance(self, CommoditySystem):
            for commodity in self.commodities.values():
                commodity.size = horizon.results[(commodity.ess, None)]["scalars"]["invest"]
            self.size = sum([commodity.size for commodity in self.commodities.values()])
        elif isinstance(self, SystemCore):
            self.acdc_size = horizon.results[(self.ac_bus, self.ac_dc)]['scalars']['invest']
            self.dcac_size = horizon.results[(self.dc_bus, self.dc_ac)]['scalars']['invest']
            self.size = self.dcac_size + self.acdc_size

    def load_opex(self, var_name, input_data_path, scenario, name):
        # get opex variable
        opex = getattr(self, var_name)
        # In case of filename for operations cost read csv file
        if isinstance(opex, str):
            # Open csv file and use first column as index; also directly convert dates to DateTime objects
            opex = pd.read_csv(os.path.join(input_data_path, name, f'{opex}.csv'),
                               index_col=0,
                               parse_dates=True)
            # Resample input data and extract relevant timesteps using start and end of simulation
            opex = opex.resample(scenario.timestep, axis=0).mean().ffill().bfill()
            opex = opex[(opex.index >= scenario.starttime) &
                        (opex.index < scenario.sim_endtime)]
            # Convert data column of cost DataFrame into Series
            setattr(self, var_name, opex[opex.columns[0]])
        else:  # opex_spec is given as a scalar directly in scenario file
            # Use sequence of values for variable costs to unify computation of results
            setattr(self, var_name, pd.Series(opex, index=scenario.sim_dti))


class CommoditySystem(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        if self.filename == 'run_des':  # if commodity system shall use a predefined behavior file
            self.data = None
        else:  # use pregenerated file
            self.input_file_path = os.path.join(run.input_data_path, self.name, self.filename + '.csv')
            self.data = pd.read_csv(self.input_file_path,
                                    header=[0, 1],
                                    index_col=0,
                                    parse_dates=True)
            self.data = self.data.resample(scenario.timestep, axis=0).ffill().bfill()
            self.data = self.data.resample(scenario.timestep, axis=0).ffill().bfill()

        self.ph_data = None  # placeholder, is filled in "update_input_components"

        self.load_opex('ext_ac_costs', 'bev', scenario, name)
        self.load_opex('ext_dc_costs', 'bev', scenario, name)

        self.apriori_lvls = ['uc']  # integration levels at which power consumption is determined a priori

        # Setting the transformer cost of the main feed(back) transformers of the system to either eps or the set values
        self.sys_chg_soe = run.eps_cost if self.sys_chg_soe == 0 else self.sys_chg_soe
        self.sys_dis_soe = run.eps_cost if self.sys_dis_soe == 0 else self.sys_dis_soe

        self.flow_in_ch = self.flow_out_ch = pd.Series(dtype='float64')  # result data
        self.flow_in = self.flow_out = pd.Series(dtype='float64')

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
                                                   inputs={scenario.blocks['core'].ac_bus: solph.Flow(
                                                       variable_costs=self.sys_chg_soe)},
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
                                                        variable_costs=self.sys_dis_soe)},
                                                    outputs={scenario.blocks['core'].ac_bus: solph.Flow()},
                                                    conversion_factors={scenario.blocks['core'].ac_bus: 1})
        scenario.components.append(self.outflow)

        self.commodities = {f'{self.name}{str(i)}':
                                MobileCommodity(self.name + str(i), self, scenario, run) for i in range(self.num)}

    def calc_results(self, scenario):

        for commodity in self.commodities.values():
            commodity.calc_results(scenario)

        self.calc_energy_results_bidi(scenario)  # bidirectional block
        self.calc_eco_results(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[
            (self.outflow, scenario.blocks['core'].ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[
            (scenario.blocks['core'].ac_bus, self.inflow)]['sequences']['flow'][horizon.ch_dti]

        self.flow_in = pd.concat([self.flow_in, self.flow_in_ch])
        self.flow_out = pd.concat([self.flow_out, self.flow_out_ch])

        for commodity in self.commodities.values():
            commodity.get_ch_results(horizon, scenario)

    def update_input_components(self, *_):
        for commodity in self.commodities.values():
            commodity.update_input_components()


class BatteryCommoditySystem(CommoditySystem):

    def __init__(self, name, scenario, run):
        super().__init__(name, scenario, run)


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

        self.bus = scenario.blocks['core'].ac_bus

        # Load sequence of opex_spec from csv file or create a constant sequence from a value
        self.load_opex(input_data_path=run.input_data_path,
                       var_name='opex_spec',
                       scenario=scenario,
                       name=name)

        if self.opt:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={scenario.blocks['core'].ac_bus: solph.Flow(
                                                   investment=solph.Investment(ep_costs=self.epc),
                                                   variable_costs=self.opex_spec)}
                                               )
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={
                                                   scenario.blocks['core'].ac_bus: solph.Flow(nominal_value=self.size,
                                                                                              variable_costs=self.opex_spec)}
                                               )
        scenario.components.append(self.src)

    def calc_results(self, scenario):

        self.calc_energy_results_source(scenario)  # unidirectional block
        self.calc_eco_results(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.src, scenario.blocks['core'].ac_bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, *_):
        pass  # no sliced input data needed for controllable source, but function needs to be callable


class FixedDemand:

    def __init__(self, name, scenario, run):
        self.name = name

        self.parameters = scenario.parameters.loc[self.name]
        for key, value in self.parameters.items():
            setattr(self, key, value)  # this sets all the parameters defined in the json file

        self.input_file_path = os.path.join(run.input_data_path, 'dem', f'{self.filename}.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=',',
                                skip_blank_lines=False)

        if 'Timestamp' in self.data:
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data.set_index('Timestamp', drop=True, inplace=True)
            self.data = self.data.tz_localize(None)  # Remove timezone-awareness of index while not converting values
            # resample to timestep, fill upsampling NaN values with previous ones (or next ones, if not available)
            self.data = self.data.resample(scenario.timestep, axis=0).mean().ffill().bfill()
        else:  # In this case, there is no check whether the frequency of inputs actually matches the simulation!
            self.data['Timestamp'] = pd.date_range(start=scenario.starttime,
                                                   periods=len(self.data),
                                                   freq=scenario.timestep)
            self.data.set_index('Timestamp', drop=True, inplace=True)

        self.ph_data = None  # placeholder

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

        self.snk = solph.components.Sink(label='dem_snk',
                                         inputs={scenario.blocks['core'].ac_bus: solph.Flow(nominal_value=1)})
        scenario.components.append(self.snk)

        scenario.blocks[self.name] = self

    def calc_results(self, scenario):

        # No super function as FixedDemand is not an InvestBlock child

        self.e_sim = self.flow.sum() * scenario.timestep_hours  # flow values are powers --> Wh
        self.e_yrl = self.e_sim / scenario.sim_yr_rat
        self.e_prj = self.e_yrl * scenario.prj_duration_yrs
        self.e_dis = eco.acc_discount(self.e_yrl, scenario.prj_duration_yrs, scenario.wacc)

        scenario.e_sim_del += self.e_sim
        scenario.e_yrl_del += self.e_yrl
        scenario.e_prj_del += self.e_prj
        scenario.e_dis_del += self.e_dis

    def get_ch_results(self, horizon, scenario):
        self.flow_ch = horizon.results[(scenario.blocks['core'].ac_bus, self.snk)]['sequences']['flow'][horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, scenario):
        # new ph data slice is created during initialization of the PredictionHorizon
        self.snk.inputs[scenario.blocks['core'].ac_bus].fix = self.ph_data['power']


class MobileCommodity:

    def __init__(self, name, parent, scenario, run):

        self.name = name
        self.parent = parent
        self.size = None if self.parent.opt else self.parent.size / self.parent.num
        self.chg_pwr = self.parent.chg_pwr

        self.ext_ac = None  # prepare for external chargers
        self.ext_dc = None  # prepare for external chargers

        if self.parent.filename == 'run_des':
            self.data = None  # parent data does not exist yet, filtering is done later
        else:  # predetermined files
            self.data = self.parent.data.loc[:, (self.name, slice(None))].droplevel(0, axis=1)

        # add columns for external AC and DC charging to data if not existing
        # ToDo: find better position for using with FCFS, etc. already add columns when reading data?
        for col in ['atac', 'atdc']:
            if col not in self.data.columns:
                self.data[col] = False

        self.ph_data = None  # placeholder, is filled in update_input_components

        self.init_soc = self.parent.init_soc
        self.ph_init_soc = self.init_soc  # set first PH's initial state variables (only SOC)

        self.e_sim_in = self.e_yrl_in = self.e_prj_in = self.e_dis_in = None
        self.e_sim_out = self.e_yrl_out = self.e_prj_out = self.e_dis_out = None

        self.flow_in_ch = self.flow_out_ch = pd.Series(dtype='float64')  # result data
        self.flow_in = self.flow_out = pd.Series(dtype='float64')

        # prepare for energy results of external chargers
        self.e_ext_ac_sim = self.e_ext_ac_yrl = self.e_ext_ac_prj = self.e_ext_ac_dis = None
        self.e_ext_dc_sim = self.e_ext_dc_yrl = self.e_ext_dc_prj = self.e_ext_dc_dis = None

        self.flow_ext_ac_ch = self.flow_ext_dc_ch = pd.Series(dtype='float64')  # result data
        self.flow_ext_ac = self.flow_ext_dc = pd.Series(dtype='float64')

        self.sc_ch = self.soc_ch = pd.Series(dtype='float64')  # result data
        self.soc = pd.Series(data=self.init_soc,
                             index=scenario.sim_dti[0:1],
                             dtype='float64')
        # add initial sc (and later soc) to the timeseries of the first horizon (otherwise not recorded)

        # Creation of permanent energy system components --------------------------------

        """
         bus               mc1_bus
          |<---------mc1_mc-x-|<->mc1_ess
          |                   |
          |                   |<--x--mc1_ext_ac
          |                   |
          |                   |<--x--mc1_ext_dc
          |                   |
          |---mc_mc1-------x->|-->mc1_snk
          |
          |                 mc2_bus
          |<---------mc2_mc---|<->mc2_ess
          |                   |
          |                   |<--x--mc2_ext_ac
          |                   |
          |                   |<--x--mc2_ext_dc
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

        self.snk = solph.components.Sink(label=f'{self.name}_snk',
                                         inputs={self.bus: solph.Flow(nominal_value=1)})
        # actual values are set later in update_input_components for each prediction horizon
        scenario.components.append(self.snk)

        if self.parent.opt:  # dispatch is optimized later --> commodity is modeled as storage and sink
            self.ess = solph.components.GenericStorage(label=f'{self.name}_ess',
                                                       inputs={self.bus: solph.Flow(
                                                           variable_costs=self.parent.opex_spec)},
                                                       outputs={self.bus: solph.Flow()},
                                                       loss_rate=0,  # TODO integrate self discharge
                                                       balanced=False,
                                                       initial_storage_level=self.ph_init_soc,
                                                       inflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       outflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       max_storage_level=1,
                                                       investment=solph.Investment(
                                                           ep_costs=self.parent.epc))
        else:
            self.ess = solph.components.GenericStorage(label=f'{self.name}_ess',
                                                       inputs={self.bus: solph.Flow(
                                                           variable_costs=self.parent.opex_spec)},
                                                       outputs={self.bus: solph.Flow()},
                                                       loss_rate=0,  # TODO integrate self discharge
                                                       balanced=False,
                                                       initial_storage_level=self.ph_init_soc,
                                                       inflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       outflow_conversion_factor=1,
                                                       # efficiency already modeled in transformers
                                                       max_storage_level=1,
                                                       nominal_storage_capacity=self.size)
        scenario.components.append(self.ess)

        # always add charger -> reduce different paths of result calculations; no chargers -> power is set to 0 kW
        # add external AC charger as new energy source
        self.ext_ac = solph.components.Source(label=f'{self.name}_ext_ac',
                                              outputs={self.bus: solph.Flow(nominal_value=1,
                                                                            variable_costs=self.parent.ext_ac_costs)}
                                              )
        scenario.components.append(self.ext_ac)

        # add external DC charger as new energy source
        self.ext_dc = solph.components.Source(label=f'{self.name}_ext_dc',
                                              outputs={self.bus: solph.Flow(nominal_value=1,
                                                                            variable_costs=self.parent.ext_dc_costs)}
                                              )
        scenario.components.append(self.ext_dc)

    # noinspection DuplicatedCode
    def calc_results(self, scenario):

        # energy result calculation does not count towards delivered/produced energy (already done at the system level)
        self.e_sim_in = self.flow_in.sum() * scenario.timestep_hours  # flow values are powers --> conversion to Wh
        self.e_sim_out = self.flow_out.sum() * scenario.timestep_hours
        self.e_yrl_in = self.e_sim_in / scenario.sim_yr_rat
        self.e_yrl_out = self.e_sim_out / scenario.sim_yr_rat
        self.e_prj_in = self.e_yrl_in * scenario.prj_duration_yrs
        self.e_prj_out = self.e_yrl_out * scenario.prj_duration_yrs
        self.e_dis_in = eco.acc_discount(self.e_yrl_in, scenario.prj_duration_yrs, scenario.wacc)
        self.e_dis_out = eco.acc_discount(self.e_yrl_out, scenario.prj_duration_yrs, scenario.wacc)

        # energy results for external chargers
        self.e_ext_ac_sim = self.flow_ext_ac.sum() * scenario.timestep_hours
        self.e_ext_dc_sim = self.flow_ext_dc.sum() * scenario.timestep_hours
        self.e_ext_ac_yrl = self.e_ext_ac_sim / scenario.sim_yr_rat
        self.e_ext_dc_yrl = self.e_ext_dc_sim / scenario.sim_yr_rat
        self.e_ext_ac_prj = self.e_ext_ac_yrl * scenario.prj_duration_yrs
        self.e_ext_dc_prj = self.e_ext_dc_yrl * scenario.prj_duration_yrs
        self.e_ext_ac_dis = eco.acc_discount(self.e_ext_ac_yrl, scenario.prj_duration_yrs, scenario.wacc)
        self.e_ext_dc_dis = eco.acc_discount(self.e_ext_dc_yrl, scenario.prj_duration_yrs, scenario.wacc)

        self.flow = self.flow_in - self.flow_out  # for plotting

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.bus, self.outflow)]['sequences']['flow'][horizon.ch_dti]
        self.flow_in_ch = horizon.results[(self.inflow, self.bus)]['sequences']['flow'][horizon.ch_dti]

        self.flow_in = pd.concat([self.flow_in, self.flow_in_ch])
        self.flow_out = pd.concat([self.flow_out, self.flow_out_ch])

        # Get results of external chargers
        self.flow_ext_ac_ch = horizon.results[(self.ext_ac, self.bus)]['sequences']['flow'][horizon.ch_dti]
        self.flow_ext_dc_ch = horizon.results[(self.ext_dc, self.bus)]['sequences']['flow'][horizon.ch_dti]

        self.flow_ext_ac = pd.concat([self.flow_ext_ac, self.flow_ext_ac_ch])
        self.flow_ext_dc = pd.concat([self.flow_ext_dc, self.flow_ext_dc_ch])

        # ToDo: Distinguishing shouldn't be necessary anymore, due to activation of storage for all int_lvls
        # if self.parent.int_lvl in self.parent.apriori_lvls:
        #     self.soc_ch = self.ph_data.loc[horizon.ch_dti, 'soc']
        # else:
        self.sc_ch = solph.views.node(
            horizon.results, f'{self.name}_ess')['sequences'][((f'{self.name}_ess', 'None'), 'storage_content')][
            horizon.ch_dti].shift(periods=1, freq=scenario.timestep)
        # shift is needed as sc/soc is stored for end of timestep by oemof
        self.soc_ch = self.sc_ch / self.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge
        self.ph_init_soc = self.soc.iloc[-1]  # reset initial SOC for next prediction horizon

    def calc_uc_power(self, scenario):
        """Converting availability and consumption data of commodities into a power timeseries for uncoordinated
         (i.e. unoptimized and starting at full power after return) charging of commodity"""

        self.uc_flows = pd.DataFrame({'p_int_ac': 0,
                                      'p_ext_ac': 0,
                                      'p_ext_dc': 0}, index=self.data.index)

        soc = [self.init_soc]  # initialize list of socs

        # get the indices of all nonzero minsoc rows in the data
        minsoc_inz = self.data.index[self.data['minsoc'] != 0]

        # get first timesteps, where vehicle has left the base
        dep_inz = self.data.index[self.data['atbase'] & ~self.data['atbase'].shift(-1, fill_value=False)]

        # get first timesteps, where vehicle is at base again
        arr_inz = self.data.index[self.data['atbase'] & ~self.data['atbase'].shift(fill_value=False)][1:]

        # get first timesteps, where vehicle is parking at destination
        arr_parking_inz = self.data.index[self.data['atac'] & ~self.data['atac'].shift(fill_value=False)]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        chg_inz = self.data.index[self.data[['atbase', 'atac', 'atdc']].any(axis=1)]

        parking_charging = False

        for dtindex, row in self.data.iterrows():
            # Heuristic:
            # - Vehicle is charged immediately, if atbase is True
            # - Vehicle gets charged during driving, if soc in next timestep is going to fall below threshold value
            # - Vehicle is charged during parking at destination, if current SOC is not enough for trip back to base
            # ToDo: influence of min soc for atbase charging: min soc should not have any influence:
            #  vehicle starts charging, when returning and charges at full speed, until battery is full
            #  min soc doesn't accelerate things
            #  -> remove minsoc in its current function from atbase
            if row['atbase'] == 1:  # commodity is at base and chargeable
                try:
                    minsoc_inxt = minsoc_inz[minsoc_inz >= dtindex][0]  # find next nonzero min SOC
                    dep_inxt = dep_inz[dep_inz >= dtindex][0]  # find next departure
                    if minsoc_inxt > dep_inxt:  # Next min soc is not defined before next departure
                        dep_soc = 1  # ToDo: implement input dependent target soc (e.g. 80%)
                    else:  # next min_soc is defined before departure and therefore valid for this charging session
                        dep_soc = self.data.loc[minsoc_inxt, 'minsoc']  # get the SOC to recharge to
                except:  # when there is no further departure
                    dep_soc = 1  # ToDo: implement input dependent target soc (e.g. 80%)

                # Execute AC charging at base
                self.uc_flows.loc[dtindex, 'p_int_ac'] = self.uc_charge(soc_target=dep_soc,
                                                                        soc_current=soc[-1],
                                                                        p_maxchg=self.chg_pwr,
                                                                        chg_eff=self.parent.chg_eff,
                                                                        scenario=scenario)

                parking_charging = False  # ToDo: find better position to set variable

            elif row['atac'] == 1:  # parking at destination
                if dtindex in arr_parking_inz:  # plugging in only happens when parking starts
                    # use current int-index and next arrival index to calculate consumption and convert to SOC
                    try:  # Fails, if current trip is last trip and doesn't end within prediction horizon
                        arr_inxt = arr_inz[arr_inz >= dtindex][0]
                    except:
                        arr_inxt = self.data.index[-1]
                    consumption_remaining = self.data.loc[dtindex:arr_inxt, 'consumption'].sum()
                    # set charging to True, if charging is necessary
                    if consumption_remaining > ((soc[-1] + self.data.loc[arr_inxt, 'minsoc']) * self.size):
                        parking_charging = True

                if parking_charging is True:
                    dep_soc = 1  # ToDo: implement input dependent target soc (e.g. 80%)

                    # Execute AC charging at destination parking
                    self.uc_flows.loc[dtindex, 'p_ext_ac'] = self.uc_charge(soc_target=dep_soc,
                                                                            soc_current=soc[-1],
                                                                            p_maxchg=self.parent.ext_ac_power,
                                                                            chg_eff=1,
                                                                            scenario=scenario)

            elif row['atdc'] == 1:  # vehicle is driving with possibility to charge on-route
                # activate charging, if SOC will fall below threshold, before next possibility to charge
                chg_inxt = chg_inz[chg_inz > dtindex][0]
                chg_soc = soc[-1] - self.data.loc[dtindex:chg_inxt, 'consumption'].sum() * scenario.timestep_hours / self.size
                if chg_soc < 0.05:
                    dep_soc = 0.8  # fast-charging only up to SOC of 80 %

                    # Execute DC charging on-route
                    self.uc_flows.loc[dtindex, 'p_ext_dc'] = self.uc_charge(soc_target=dep_soc,
                                                                            soc_current=soc[-1],
                                                                            p_maxchg=self.parent.ext_dc_power,
                                                                            chg_eff=1,
                                                                            scenario=scenario)

            # update SOC
            print(dtindex)
            soc_delta = (self.uc_flows.loc[dtindex, 'p_int_ac'] + \
                         self.uc_flows.loc[dtindex, 'p_ext_ac'] + \
                         self.uc_flows.loc[dtindex, 'p_ext_dc'] - \
                         row['consumption']) * scenario.timestep_hours / self.size
            soc.append(soc[-1] + soc_delta)  # TODO check whether SOC indexing fits optimization output
            if (soc[-1] < 0) or (soc[-1] > 1):
                # ToDo: Raise exception
                print("Error! Calculation of UC charging profile failed. SOC out of bounds")
        pass

    def uc_charge(self, soc_target, soc_current, p_maxchg, chg_eff, scenario):
        soc_target = min(soc_target, 1)  # soc must not get bigger than 1
        e_tominsoc = max(0, (soc_target - soc_current) * self.size)  # energy to be recharged to departure SOC in Wh
        p_tominsoc = e_tominsoc / scenario.timestep_hours  # power to recharge to departure SOC in one step
        p_act = min(p_maxchg * chg_eff, p_tominsoc)  # reduce chg power in final step to just reach departure SOC
        return p_act

    def update_input_components(self):

        # set vehicle consumption data for sink
        self.snk.inputs[self.bus].fix = self.ph_data['consumption']

        # set initial storage levels for coming prediction horizon
        self.ess.initial_storage_level = self.ph_init_soc

        if self.parent.int_lvl in self.parent.apriori_lvls:
            # define charging powers (as per uc power calculation)
            self.inflow.outputs[self.bus].fix = self.uc_flows['p_int_ac']
            self.ext_ac.outputs[self.bus].fix = self.uc_flows['p_ext_ac']
            self.ext_dc.outputs[self.bus].fix = self.uc_flows['p_ext_dc']
        else:
            # enable/disable transformers to mcx_bus depending on whether the commodity is at base
            self.inflow.inputs[self.parent.bus].max = self.ph_data['atbase'].astype(int)
            self.outflow.inputs[self.bus].max = self.ph_data['atbase'].astype(int)

            # set minimum storage levels for coming prediction horizon
            self.ess.min_storage_level = self.ph_data['minsoc']

            # enable/disable ac and dc charging station dependent on input data
            self.ext_ac.outputs.data[self.bus].max = self.ph_data['atac'].astype(int) * self.parent.ext_ac_power
            self.ext_dc.outputs.data[self.bus].max = self.ph_data['atdc'].astype(int) * self.parent.ext_dc_power


class PVSource(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.ph_data = self.input_file_name = self.input_file_path = None  # placeholders, are filled later
        self.api_startyear = self.api_endyear = None
        self.timezone = self.data = self.meta = None

        self.tf = timezonefinder.TimezoneFinder()
        self.utc = pytz.timezone('UTC')

        self.get_timeseries_data(scenario, run)

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
                                                    outputs={scenario.blocks['core'].dc_bus: solph.Flow()},
                                                    conversion_factors={scenario.blocks['core'].dc_bus: self.eff})
        scenario.components.append(self.outflow)

        # input data from PVGIS is added in function "update_input_components"
        if self.opt:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(investment=solph.Investment(
                                                   ep_costs=self.epc),
                                                   variable_costs=self.opex_spec)})
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(nominal_value=self.size,
                                                                             variable_costs=self.opex_spec)})
        scenario.components.append(self.src)

        self.exc = solph.components.Sink(label=f'{self.name}_exc',
                                         inputs={self.bus: solph.Flow()})
        scenario.components.append(self.exc)

    def calc_results(self, scenario):

        self.calc_energy_results_source(scenario)
        self.calc_eco_results(scenario)

    def calc_power_solcast(self):

        u0 = 26.9  # W/(˚C.m2) - cSi Free standing
        u1 = 6.2  # W.s/(˚C.m3) - cSi Free standing
        mod_temp = self.data['AirTemp'] + (self.data['GtiFixedTilt'] / (u0 + (u1 * self.data['WindSpeed10m'])))

        # PVGIS temperature and irradiance coefficients for cSi panels as per Huld T., Friesen G., Skoczek A.,
        # Kenny R.P., Sample T., Field M., Dunlop E.D. A power-rating model for crystalline silicon PV modules
        # Solar Energy Materials & Solar Cells. 2011 95, 3359-3369.
        k1 = -0.017237
        k2 = -0.040465
        k3 = -0.004702
        k4 = 0.000149
        k5 = 0.000170
        k6 = 0.000005
        g = self.data['GtiFixedTilt'] / 1000
        t = mod_temp - 25
        lng = np.zeros_like(g)
        lng[g != 0] = np.log(g[g != 0])  # ln(g) ignoring zeros

        # Faiman, D. Assessing the outdoor operating temperature of photovoltaic modules.
        # Prog. Photovolt. Res. Appl.2008, 16, 307–315
        eff_rel = 1 + \
                  (k1 * lng) + \
                  (k2 * (lng ** 2)) + \
                  (k3 * t) + \
                  (k4 * t * lng) + \
                  (k5 * t * (lng ** 2)) + \
                  (k6 * (t ** 2))
        eff_rel = eff_rel.fillna(0)

        # calculate power of a 1kWp array, limited to 0 (negative values fail calculation)
        self.data['P'] = np.maximum(0, eff_rel * self.data['GtiFixedTilt'])

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.blocks['core'].dc_bus)]['sequences']['flow'][
            horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def get_timeseries_data(self, scenario, run):

        if self.data_source == 'PVGIS API':  # API input selected
            self.timezone = pytz.timezone(self.tf.certain_timezone_at(lat=self.latitude, lng=self.longitude))
            self.api_startyear = self.timezone.localize(scenario.starttime).astimezone(self.utc).year
            self.api_endyear = self.timezone.localize(scenario.sim_endtime).astimezone(self.utc).year
            self.data, self.meta, _ = pvlib.iotools.get_pvgis_hourly(self.latitude,
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

            # PVGIS gives time slots as XX:06 - round to full hour
            self.data.index = self.data.index.round('H')
        else:
            self.input_file_path = os.path.join(run.input_data_path, 'pv', f'{self.filename}.csv')

            if self.data_source == 'PVGIS file':  # data input from fixed PVGIS csv file
                self.data, self.meta, _ = pvlib.iotools.read_pvgis_hourly(self.input_file_path,
                                                                          map_variables=True)
                self.latitude = self.meta['latitude']
                self.longitude = self.meta['longitude']
                # PVGIS gives time slots as XX:06 - round to full hour
                self.data.index = self.data.index.round('H')
            elif self.data_source.lower() == 'solcast file':  # data input from fixed Solcast csv file, no lat/lon contained
                self.data = pd.read_csv(self.input_file_path)
                self.data['PeriodStart'] = pd.to_datetime(self.data['PeriodStart'])
                self.data['PeriodEnd'] = pd.to_datetime(self.data['PeriodEnd'])
                self.data.set_index(pd.DatetimeIndex(self.data['PeriodStart']), inplace=True)
                self.data['wind_speed'] = self.data['WindSpeed10m']
                self.calc_power_solcast()
            else:
                run.logger.warning('No usable PV input type specified - exiting')
                exit()  # TODO exit scenario instead of entire execution

            self.timezone = pytz.timezone(self.tf.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        # resample to timestep, fill NaN values with previous ones (or next ones, if not available)
        self.data = self.data.resample(scenario.timestep, axis=0).mean().ffill().bfill()
        # convert to local time and remove timezone-awareness (model is only in one timezone)
        self.data.index = self.data.index.tz_convert(tz=self.timezone).tz_localize(tz=None)
        # data is in W for a 1kWp PV array -> convert to specific power
        self.data['p_spec'] = self.data['P'] / 1e3

        self.data = self.data[['p_spec', 'wind_speed']]

    def update_input_components(self, *_):

        self.src.outputs[self.bus].fix = self.ph_data['p_spec']


class StationaryEnergyStorage(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.ph_init_soc = self.init_soc

        self.flow_in_ch = self.flow_out_ch = pd.Series(dtype='float64')  # result data
        self.flow_in = self.flow_out = pd.Series(dtype='float64')

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
                                                       inputs={scenario.blocks['core'].dc_bus: solph.Flow(
                                                           variable_costs=self.opex_spec)},
                                                       outputs={scenario.blocks['core'].dc_bus: solph.Flow()},
                                                       loss_rate=0,  # TODO proper self discharge
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.ph_init_soc,
                                                       invest_relation_input_capacity=self.chg_crate,
                                                       invest_relation_output_capacity=self.dis_crate,
                                                       inflow_conversion_factor=self.chg_eff,
                                                       outflow_conversion_factor=self.dis_eff,
                                                       investment=solph.Investment(ep_costs=self.epc))
        else:
            self.ess = solph.components.GenericStorage(label='ess',
                                                       inputs={scenario.blocks['core'].dc_bus: solph.Flow(
                                                           variable_costs=self.opex_spec)},
                                                       outputs={scenario.blocks['core'].dc_bus: solph.Flow()},
                                                       loss_rate=0,  # TODO proper self discharge
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.ph_init_soc,
                                                       invest_relation_input_capacity=self.chg_crate,
                                                       invest_relation_output_capacity=self.dis_crate,
                                                       inflow_conversion_factor=self.chg_eff,
                                                       outflow_conversion_factor=self.dis_eff,
                                                       nominal_storage_capacity=self.size)
        scenario.components.append(self.ess)

    def calc_results(self, scenario):

        self.calc_energy_results_bidi(scenario)
        self.calc_eco_results(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.ess, scenario.blocks['core'].dc_bus)]['sequences']['flow'][
            horizon.ch_dti]
        self.flow_in_ch = horizon.results[(scenario.blocks['core'].dc_bus, self.ess)]['sequences']['flow'][
            horizon.ch_dti]

        self.flow_in = pd.concat([self.flow_in, self.flow_in_ch])
        self.flow_out = pd.concat([self.flow_out, self.flow_out_ch])

        self.sc_ch = solph.views.node(horizon.results, self.name)['sequences'][
            ((self.name, 'None'), 'storage_content')][horizon.ch_dti].shift(periods=1, freq=scenario.timestep)
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
            self.size = self.acdc_size + self.dcac_size

        self.flow_acdc_ch = self.flow_dcac_ch = pd.Series(dtype='float64')  # result data
        self.flow_acdc = self.flow_dcac = pd.Series(dtype='float64')

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
                                                          ep_costs=self.epc),
                                                          variable_costs=self.opex_spec)},
                                                      outputs={self.dc_bus: solph.Flow()},
                                                      conversion_factors={self.dc_bus: self.acdc_eff})

            self.dc_ac = solph.components.Transformer(label='dc_ac',
                                                      inputs={self.dc_bus: solph.Flow(investment=solph.Investment(
                                                          ep_costs=self.epc),
                                                          variable_costs=self.opex_spec)},
                                                      outputs={self.ac_bus: solph.Flow()},
                                                      conversion_factors={self.ac_bus: self.dcac_eff})
        else:
            self.ac_dc = solph.components.Transformer(label='ac_dc',
                                                      inputs={self.ac_bus: solph.Flow(variable_costs=run.eps_cost)},
                                                      outputs={self.dc_bus: solph.Flow()},
                                                      conversion_factors={self.dc_bus: self.acdc_eff})

            self.dc_ac = solph.components.Transformer(label='dc_ac',
                                                      inputs={self.dc_bus: solph.Flow(variable_costs=run.eps_cost)},
                                                      outputs={self.ac_bus: solph.Flow()},
                                                      conversion_factors={self.ac_bus: self.dcac_eff})

        scenario.components.append(self.ac_dc)
        scenario.components.append(self.dc_ac)

    def calc_results(self, scenario):

        # energy result calculation is different from any other block as there is no in/out definition of flow
        self.e_sim_dcac = self.flow_dcac.sum() * scenario.timestep_hours  # flow values are powers --> conversion to Wh
        self.e_sim_acdc = self.flow_acdc.sum() * scenario.timestep_hours
        self.e_yrl_dcac = self.e_sim_dcac / scenario.sim_yr_rat
        self.e_yrl_acdc = self.e_sim_acdc / scenario.sim_yr_rat
        self.e_prj_dcac = self.e_yrl_dcac * scenario.prj_duration_yrs
        self.e_prj_acdc = self.e_yrl_acdc * scenario.prj_duration_yrs
        self.e_dis_dcac = eco.acc_discount(self.e_yrl_dcac, scenario.prj_duration_yrs, scenario.wacc)
        self.e_dis_acdc = eco.acc_discount(self.e_yrl_acdc, scenario.prj_duration_yrs, scenario.wacc)

        self.calc_eco_results(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_acdc_ch = horizon.results[(scenario.blocks['core'].ac_bus, self.ac_dc)]['sequences']['flow'][
            horizon.ch_dti]
        self.flow_dcac_ch = horizon.results[(scenario.blocks['core'].dc_bus, self.dc_ac)]['sequences']['flow'][
            horizon.ch_dti]

        self.flow_acdc = pd.concat([self.flow_acdc, self.flow_acdc_ch])
        self.flow_dcac = pd.concat([self.flow_dcac, self.flow_dcac_ch])

    def update_input_components(self, *_):
        pass  # function needs to be callable


class VehicleCommoditySystem(CommoditySystem):

    def __init__(self, name, scenario, run):
        super().__init__(name, scenario, run)


class WindSource(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.input_file_path = os.path.join(run.input_data_path, 'wind', self.filename + '.csv')
        self.data = pd.read_csv(self.input_file_path,
                                sep=',',
                                skip_blank_lines=False)
        self.data.index = pd.date_range(start=scenario.starttime,
                                        periods=len(self.data),
                                        freq=scenario.timestep)

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
                                                    outputs={scenario.blocks['core'].ac_bus: solph.Flow()},
                                                    conversion_factors={scenario.blocks['core'].ac_bus: self.eff})
        scenario.components.append(self.outflow)

        self.exc = solph.components.Sink(label=f'{self.name}_exc',
                                         inputs={self.bus: solph.Flow()})
        scenario.components.append(self.exc)

        # TODO make wind speed from PVGIS usable - then it's also time based and not just stepwise...

        if self.opt:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(investment=solph.Investment(
                                                   ep_costs=self.epc),
                                                   variable_costs=self.opex_spec)})
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus: solph.Flow(nominal_value=self.size,
                                                                             variable_costs=self.opex_spec)})
        scenario.components.append(self.src)

    def calc_results(self, scenario):

        self.calc_energy_results_source(scenario)  # unidirectional block
        self.calc_eco_results(scenario)

    def get_ch_results(self, horizon, scenario):

        self.flow_ch = horizon.results[(self.outflow, scenario.blocks['core'].ac_bus)]['sequences']['flow'][
            horizon.ch_dti]
        self.flow = pd.concat([self.flow, self.flow_ch])

    def update_input_components(self, *_):

        self.src.outputs[self.bus].fix = self.ph_data['P']
