"""
blocks.py

--- Description ---
This script defines the energy system blocks for the REVOL-E-TION toolset.

For further information, see readme

--- Created by ---
Philipp Rosner

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

import ast
import numpy as np
import oemof.solph as solph
import os
import pandas as pd
import pvlib
import pytz
import statistics
import windpowerlib

import battery as bat
import economics as eco

import plotly.graph_objects as go


class Block:

    def __init__(self, name, scenario, run):
        self.name = name
        scenario.blocks[self.name] = self

        self.parameters = scenario.parameters.loc[self.name]
        for key, value in self.parameters.items():
            setattr(self, key, value)  # this sets all the parameters defined in the scenario file

        time_var_params = [var for var in vars(self) if ('opex_spec' in var) or ('crev_spec' in var)]
        for var in time_var_params:
            self.transform_scalar_var(var, scenario, run)

        # Empty result series
        self.flow = self.flow_ch = pd.Series(dtype='float64')
        self.flow_in_ch = self.flow_out_ch = self.flow_in = self.flow_out = pd.Series(dtype='float64')
        # flow direction is specified with respect to the block -> flow_in is from energy system into block

        # Empty result scalar variables
        self.e_sim = self.e_yrl = self.e_prj = self.e_dis = 0
        self.e_sim_in = self.e_yrl_in = self.e_prj_in = self.e_dis_in = 0
        self.e_sim_out = self.e_yrl_out = self.e_prj_out = self.e_dis_out = 0
        self.e_sim_del = self.e_yrl_del = self.e_prj_del = self.e_dis_del = 0
        self.e_sim_pro = self.e_yrl_pro = self.e_prj_pro = self.e_dis_pro = 0
        self.capex_init = self.capex_prj = self.capex_dis = self.capex_ann = 0
        self.mntex_sim = self.mntex_yrl = self.mntex_prj = self.mntex_dis = self.mntex_ann = 0
        self.opex_sim = self.opex_yrl = self.opex_prj = self.opex_dis = self.opex_ann = 0
        self.opex_sim_ext = self.opex_yrl_ext = self.opex_prj_ext = self.opex_dis_ext = self.opex_ann_ext = 0
        self.totex_sim = self.totex_prj = self.totex_dis = self.totex_ann = 0
        self.crev_sim = self.crev_yrl = self.crev_prj = self.crev_dis = 0

        self.cashflows = pd.DataFrame()

        self.apriori_data = None

    def accumulate_crev(self, scenario):
        """
        crev_sim is calculated beforehand for the individual blocks
        """

        self.crev_yrl = eco.scale_sim2year(self.crev_sim, scenario)
        self.crev_prj = eco.scale_year2prj(self.crev_yrl, scenario)
        self.crev_dis = eco.acc_discount(self.crev_yrl, scenario.prj_duration_yrs, scenario.wacc)

        scenario.crev_sim += self.crev_sim
        scenario.crev_yrl += self.crev_yrl
        scenario.crev_prj += self.crev_prj
        scenario.crev_dis += self.crev_dis

    def add_power_trace(self, scenario):
        legentry = self.get_legend_entry()
        scenario.figure.add_trace(go.Scatter(x=self.flow.index,
                                             y=self.flow,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None)),
                                  secondary_y=False)

    def calc_cashflows(self, scenario):

        capex = pd.Series(dtype='float64', index=range(scenario.prj_duration_yrs), data=0)
        capex[0] = self.capex_init
        if hasattr(self, 'ls'):
            for year in eco.repllist(self.ls, scenario.prj_duration_yrs):
                capex[year] = self.capex_init * (self.cdc ** year)
        self.cashflows[f'capex_{self.name}'] = -1 * capex  # expenses are negative cashflows (outgoing)

        self.cashflows[f'mntex_{self.name}'] = -1 * self.mntex_yrl
        self.cashflows[f'opex_{self.name}'] = -1 * self.opex_yrl
        self.cashflows[f'crev_{self.name}'] = self.crev_yrl

        scenario.cashflows = pd.concat([scenario.cashflows, self.cashflows], axis=1)

    def calc_energy_bidi(self, scenario):
        """
        Calculate the energy results for bidirectional blocks (CommoditySystems and StationaryEnergyStorages).
        Bidirectional blocks can be either counted towards energy production or delivery, depending on their balance.
        """
        self.calc_energy_common(scenario)

        if self.e_sim_in > self.e_sim_out:
            self.e_sim_del = self.e_sim_in - self.e_sim_out
            self.e_yrl_del = eco.scale_sim2year(self.e_sim_del, scenario)
            self.e_prj_del = eco.scale_year2prj(self.e_yrl_del, scenario)
            self.e_dis_del = eco.acc_discount(self.e_yrl_del, scenario.prj_duration_yrs, scenario.wacc)

            scenario.e_sim_del += self.e_sim_del
            scenario.e_yrl_del += self.e_yrl_del
            scenario.e_prj_del += self.e_prj_del
            scenario.e_dis_del += self.e_dis_del

        else:  # storage was emptied
            self.e_sim_pro = self.e_sim_out - self.e_sim_in
            self.e_yrl_pro = eco.scale_sim2year(self.e_sim_pro, scenario)
            self.e_prj_pro = eco.scale_year2prj(self.e_yrl_pro, scenario)
            self.e_dis_pro = eco.acc_discount(self.e_yrl_pro, scenario.prj_duration_yrs, scenario.wacc)

            scenario.e_sim_pro += self.e_sim_pro
            scenario.e_yrl_pro += self.e_yrl_pro
            scenario.e_prj_pro += self.e_prj_pro
            scenario.e_dis_pro += self.e_dis_pro

    def calc_energy_source_sink(self, scenario):
        """
        Accumulating results for sources and sinks
        """
        self.calc_energy_common(scenario)

        scenario.e_sim_pro += self.e_sim_out
        scenario.e_sim_del += self.e_sim_in
        scenario.e_yrl_pro += self.e_yrl_out
        scenario.e_yrl_del += self.e_yrl_in
        scenario.e_prj_pro += self.e_prj_out
        scenario.e_prj_del += self.e_prj_in
        scenario.e_dis_pro += self.e_dis_out
        scenario.e_dis_del += self.e_dis_in

    def calc_energy_common(self, scenario):

        if any(~(self.flow_in == 0) & ~(self.flow_out == 0)):
            scenario.logger.warning(f'Block {self.name} - '
                                    f'simultaneous in- and outflow detected!')

        self.e_sim_in = self.flow_in.sum() * scenario.timestep_hours  # flow values are powers in W --> conversion to Wh
        self.e_sim_out = self.flow_out.sum() * scenario.timestep_hours
        self.e_yrl_in = eco.scale_sim2year(self.e_sim_in, scenario)
        self.e_yrl_out = eco.scale_sim2year(self.e_sim_out, scenario)
        self.e_prj_in = eco.scale_year2prj(self.e_yrl_in, scenario)
        self.e_prj_out = eco.scale_year2prj(self.e_yrl_out, scenario)
        self.e_dis_in = eco.acc_discount(self.e_yrl_in, scenario.prj_duration_yrs, scenario.wacc)
        self.e_dis_out = eco.acc_discount(self.e_yrl_out, scenario.prj_duration_yrs, scenario.wacc)

        if self.flow_in.empty:
            self.flow = self.flow_out
        elif self.flow_out.empty:
            self.flow = -1 * self.flow_in
        else:
            self.flow = self.flow_out - self.flow_in

    def calc_expenses(self, scenario):
        """
        dummy function for code structure simplification.
        Only InvestBlocks have expenses.
        """
        pass

    def calc_revenue(self, scenario):
        """
        dummy function for code structure simplification
        Actually only relevant for CommoditySystems and FixedDemands, where this is implemented separately
        """
        pass

    def get_legend_entry(self):
        """
        Standard legend entry for simple blocks using power as their size
        """
        return f'{self.name} power (max. {round(self.size / 1e3)} kW)'

    def get_timeseries_results(self, scenario):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        block_ts_results = pd.DataFrame({f'{self.name}_flow_in': self.flow_in,
                                         f'{self.name}_flow_out': self.flow_out})
        scenario.result_timeseries = pd.concat([scenario.result_timeseries, block_ts_results], axis=1)

    def read_input_csv(self, path_input_file, scenario, multiheader=False):
        """
        Properly read in timezone-aware input csv files and form correct datetimeindex
        """
        if multiheader:
            df = pd.read_csv(path_input_file, header=[0, 1])
            df.sort_index(axis=1, sort_remaining=True, inplace=True)
            df = df.set_index(pd.to_datetime(df.loc[:, ('time', 'time')], utc=True)).drop(columns='time')
        else:
            df = pd.read_csv(path_input_file)
            df = df.set_index(pd.to_datetime(df['time'], utc=True)).drop(columns='time')

        # parser in to_csv does not create datetimeindex
        df = df.tz_convert(scenario.timezone)
        df = self.resample_to_timestep(df, scenario)
        return df

    @staticmethod
    def resample_to_timestep(data: pd.DataFrame, scenario):
        """
        Resample the data to the timestep of the scenario, conserving the proper index end even in upsampling
        :param data: The input dataframe with DatetimeIndex
        :param scenario: The current scenario object
        :return: resampled dataframe
        """

        dti = data.index
        # Add one element to the dataframe to include the last timesteps
        dti_ext = dti.union(dti.shift(periods=1, freq=pd.infer_freq(dti))[-1:])
        data_ext = data.reindex(dti_ext).ffill()

        def resample_column(column):
            if data_ext[column].dtype == bool:
                return data_ext[column].resample(scenario.timestep).ffill().bfill()
            else:
                return data_ext[column].resample(scenario.timestep).mean().ffill().bfill()

        resampled_data = pd.DataFrame({col: resample_column(col) for col in data_ext.columns})[:-1]
        return resampled_data

    def transform_scalar_var(self, var_name, scenario, run):
        scenario_entry = getattr(self, var_name)
        # In case of filename for operations cost read csv file
        if isinstance(scenario_entry, str):
            # Open csv file and use first column as index; also directly convert dates to DateTime objects
            opex = self.read_input_csv(os.path.join(run.path_input_data, self.name, f'{scenario_entry}.csv'), scenario)
            opex = opex[scenario.starttime:(scenario.sim_endtime - scenario.timestep_td)]
            # Convert data column of cost DataFrame into Series
            setattr(self, var_name, opex[opex.columns[0]])
        else:  # opex_spec is given as a scalar directly in scenario file
            # Use sequence of values for variable costs to unify computation of results
            setattr(self, var_name, pd.Series(scenario_entry, index=scenario.dti_sim))


class InvestBlock(Block):
    """
    An InvestBlock is a block that can be optimized in size. It has therefore incurs expenses.
    """

    def __init__(self, name, scenario, run):

        self.opt = self.size = None

        super().__init__(name, scenario, run)

        self.set_init_size(scenario, run)

        if self.opt and scenario.strategy != 'go':
            scenario.logger.warning(f'\"{self.name}\" component size optimization not implemented'
                                    f' for any other strategy than \"GO\" - exiting')
            exit()  # TODO exit scenario instead of run

        # ace = adjusted capital expenses (including maintenance)
        self.ace = eco.adj_ce(self.capex_spec, self.mntex_spec, self.ls, scenario.wacc)
        # epc = equivalent present cost
        self.epc = eco.ann_recur(self.ace, self.ls, scenario.prj_duration_yrs, scenario.wacc, self.cdc)

    def calc_capex(self, scenario):
        """
        Calculate capital expenses over simulation timeframe and convert to other timeframes.
        """

        self.calc_capex_init(scenario)  # initial investment references to different parameters depending on block type

        self.capex_prj = eco.tce(self.capex_init,
                                 self.cdc,
                                 self.ls,
                                 scenario.prj_duration_yrs)
        self.capex_dis = eco.pce(self.capex_init,
                                 self.cdc,
                                 scenario.wacc,
                                 self.ls,
                                 scenario.prj_duration_yrs)
        self.capex_ann = eco.ann_recur(self.capex_init,
                                       self.ls,
                                       scenario.prj_duration_yrs,
                                       scenario.wacc,
                                       self.cdc)

        scenario.capex_init += self.capex_init
        scenario.capex_prj += self.capex_prj
        scenario.capex_dis += self.capex_dis
        scenario.capex_ann += self.capex_ann

    def calc_capex_init(self, scenario):
        """
        Default function for blocks with a single size value.
        GridConnections, SystemCore and CommoditySystems are more complex and have their own functions
        """
        self.capex_init = self.size * self.capex_spec

    def calc_expenses(self, scenario):

        self.calc_capex(scenario)
        self.calc_mntex(scenario)
        self.calc_opex(scenario)
        if isinstance(self, CommoditySystem):
            self.calc_opex_ext(scenario)

        self.totex_sim = self.capex_init + self.mntex_sim + self.opex_sim
        self.totex_prj = self.capex_prj + self.mntex_prj + self.opex_prj
        self.totex_dis = self.capex_dis + self.mntex_dis + self.opex_dis
        self.totex_ann = self.capex_ann + self.mntex_ann + self.opex_ann

        scenario.totex_sim += self.totex_sim
        scenario.totex_prj += self.totex_prj
        scenario.totex_dis += self.totex_dis
        scenario.totex_ann += self.totex_ann

    def calc_mntex(self, scenario):
        """
        Calculate maintenance expenses over simulation timeframe and convert to other timeframes.
        Maintenance expenses are solely time-based. Throughput-based maintenance should be included in opex.
        """

        self.calc_mntex_yrl()  # maintenance expenses are defined differently depending on the block type

        self.mntex_sim = self.mntex_yrl * scenario.sim_yr_rat
        self.mntex_prj = eco.scale_year2prj(self.mntex_yrl, scenario)
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

    def calc_mntex_yrl(self):
        """
        Default function for simple blocks with a single size value. GridConnection, SystemCore and CommoditySystem
        are more complex.
        """
        self.mntex_yrl = self.size * self.mntex_spec

    def calc_opex(self, scenario):
        """
        Calculate operational expenses over simulation timeframe and convert to other timeframes.
        """

        self.calc_opex_sim(scenario)  # opex is defined differently depending on the block type

        self.opex_yrl = eco.scale_sim2year(self.opex_sim, scenario)
        self.opex_prj = eco.scale_year2prj(self.opex_yrl, scenario)
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

    def get_timeseries_results(self, scenario):
        """
        Dummy method to make Block method available to InvestBlock children classes
        """
        super().get_timeseries_results(scenario)

    def set_init_size(self, scenario, run):
        """
        Default function for components with a single size (i.e. not GridConnection and SystemCore)
        """

        if self.size == 'opt':  # size will be set when getting results
            self.opt = True
            self.size = None

        elif isinstance(self.size, (float, int)):  # fixed size
            self.opt = False


class RenewableInvestBlock(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.data = self.data_ph = self.input_file_name = self.path_input_file = None  # placeholders, are filled later

        self.flow_pot = self.flow_pot_ch = self.flow_curt = self.flow_curt_ch = pd.Series(dtype='float64')
        self.e_pot = self.e_curt = 0

        self.get_timeseries_data(scenario, run)

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results

        bus_connected      self.bus
          |                   |
          |<--x----self_out---|<--self_src
          |                   |
                              |-->self_exc
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.components.append(self.bus)

        self.outflow = solph.components.Converter(label=f'{self.name}_out',
                                                  inputs={self.bus: solph.Flow(variable_costs=scenario.cost_eps)},
                                                  outputs={self.bus_connected: solph.Flow()},
                                                  conversion_factors={self.bus_connected: self.eff})
        scenario.components.append(self.outflow)

        self.exc = solph.components.Sink(label=f'{self.name}_exc',
                                         inputs={self.bus: solph.Flow()})
        scenario.components.append(self.exc)

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

    def add_curtailment_trace(self, scenario):
        legentry = f'{self.name} curtailed power'
        scenario.figure.add_trace(go.Scatter(x=self.flow_curt.index,
                                             y=-1 * self.flow_curt,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=False)

        legentry = f'{self.name} potential power'
        scenario.figure.add_trace(go.Scatter(x=self.flow_pot.index,
                                             y=self.flow_pot,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=False)

    def calc_energy(self, scenario):
        self.calc_energy_source_sink(scenario)

        self.e_pot = self.flow_pot.sum() * scenario.timestep_hours  # flow values are powers in W --> conversion to Wh
        self.e_curt = self.flow_curt.sum() * scenario.timestep_hours

        scenario.e_renewable_act += self.e_sim_out
        scenario.e_renewable_pot += self.e_pot
        scenario.e_renewable_curt += self.e_curt

    def calc_opex_sim(self, scenario):
        self.opex_sim = self.flow_out @ self.opex_spec * scenario.timestep_hours

    def get_ch_results(self, horizon, scenario):

        # flow values are powers
        self.flow_out_ch = horizon.results[(self.outflow, self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flow_pot_ch = horizon.results[(self.src, self.bus)]['sequences']['flow'][horizon.dti_ch]
        self.flow_curt_ch = horizon.results[(self.bus, self.exc)]['sequences']['flow'][horizon.dti_ch]

        self.flow_out = pd.concat([self.flow_out, self.flow_out_ch])
        self.flow_pot = pd.concat([self.flow_pot, self.flow_pot_ch])
        self.flow_curt = pd.concat([self.flow_curt, self.flow_curt_ch])

    def get_legend_entry(self):
        return f'{self.name} power (nom. {round(self.size / 1e3)} kW)'

    def get_opt_size(self, horizon):
        self.size = horizon.results[(self.src, self.bus)]['scalars']['invest']

    def get_timeseries_results(self, scenario):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        super().get_timeseries_results(scenario)  # this goes up to the Block class

        block_ts_results = pd.DataFrame({f'{self.name}_flow_pot': self.flow_pot,
                                         f'{self.name}_flow_curt': self.flow_curt})
        scenario.result_timeseries = pd.concat([scenario.result_timeseries, block_ts_results], axis=1)

    def update_input_components(self):

        self.src.outputs[self.bus].fix = self.data_ph['power_spec']

        if self.apriori_data is not None:
            # Use power calculated in apriori_data for fixed output of block
            self.outflow.outputs[self.bus_connected].fix = self.apriori_data['p']


class CommoditySystem(InvestBlock):

    def __init__(self, name, scenario, run):

        self.size_pc = 0  # placeholder for storage capacity. Might be set in super().__init__

        super().__init__(name, scenario, run)

        if self.filename == 'run_des':  # if commodity system shall use a predefined behavior file
            self.data = None  # data will be generated in a joint DES run after model setup
        else:  # use pregenerated file
            self.path_input_file = os.path.join(run.path_input_data, self.name, self.filename + '.csv')
            self.data = self.read_input_csv(self.path_input_file, scenario, multiheader=True)

            if pd.infer_freq(self.data.index).lower() != scenario.timestep:
                scenario.logger.warning(f'\"{self.name}\" input data does not match timestep'
                                        f' - resampling is experimental')
                consumption_columns = list(filter(lambda x: 'consumption' in x[1], self.data.columns))
                bool_columns = self.data.columns.difference(consumption_columns)
                # mean ensures equal energy consumption after downsampling, ffill and bfill fill upsampled NaN values
                df = self.data[consumption_columns].resample(scenario.timestep).mean().ffill().bfill()
                df[bool_columns] = self.data[bool_columns].resample(scenario.timestep).ffill().bfill()
                self.data = df

        self.data_ph = None  # placeholder, is filled in "update_input_components"

        self.loss_rate = eco.convert_sdr_to_timestep(self.sdr)

        # integration levels at which power consumption is determined a priori
        self.apriori_lvls = ['uc', 'fcfs', 'equal', 'soc']

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results (?c denotes ac or dc, depending on the parameter 'system')

        ac/dc_bus            bus
          |<-x--------mc_xc---|---(MobileCommodity Instance)
          |                   |
          |-x-xc_mc---------->|---(MobileCommodity Instance)
                              |
                              |---(MobileCommodity Instance)
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.components.append(self.bus)

        self.bus_connected = scenario.blocks['core'].ac_bus if self.system == 'ac' else scenario.blocks['core'].dc_bus

        self.inflow = solph.components.Converter(label=f'xc_{self.name}',
                                                 inputs={self.bus_connected: solph.Flow(
                                                     variable_costs=self.opex_spec_sys_chg)},
                                                 outputs={self.bus: solph.Flow(
                                                        variable_costs=scenario.cost_eps)},
                                                 conversion_factors={self.bus: 1})

        self.outflow = solph.components.Converter(label=f'{self.name}_xc',
                                                  inputs={self.bus: solph.Flow(
                                                      nominal_value={'uc': 0,
                                                                     'fcfs': 0,
                                                                     'equal': 0,
                                                                     'soc': 0,
                                                                     'cc': 0,
                                                                     'tc': 0,
                                                                     'v2v': 0,
                                                                     'v2mg': None}[self.int_lvl],
                                                      variable_costs=self.opex_spec_sys_dis)},
                                                  outputs={self.bus_connected: solph.Flow(
                                                      variable_costs=scenario.cost_eps)},
                                                  conversion_factors={self.bus_connected: 1})

        scenario.components.append(self.inflow)
        scenario.components.append(self.outflow)

        # Generate individual commodity instances
        self.commodities = {f'{self.name}{str(i)}': MobileCommodity(self.name + str(i), self, scenario)
                            for i in range(self.num)}

        self.e_sim_ext = self.e_yrl_ext = self.e_prj_ext = self.e_dis_ext = 0  # results of external charging

    def add_power_trace(self, scenario):
        super().add_power_trace(scenario)
        for commodity in self.commodities.values():
            commodity.add_power_trace(scenario)

    def add_soc_trace(self, scenario):
        for commodity in self.commodities.values():
            commodity.add_soc_trace(scenario)

    def calc_aging(self, run, scenario, horizon):
        for commodity in self.commodities.values():
            commodity.calc_aging(run, scenario, horizon)

    def calc_capex_init(self, scenario):
        """
        Default function to calculate initial capex of simple blocks with a single size value.
        GridConnection, SystemCore and CommoditySystem are more complex.
        """
        self.capex_init = np.array([com.size for com in self.commodities.values()]).sum() * self.capex_spec

    def calc_energy(self, scenario):

        # Aggregate energy results for external charging for all MobileCommodities within the CommoditySystem
        for commodity in self.commodities.values():
            commodity.calc_results(scenario)
            scenario.e_sim_ext += (commodity.e_ext_ac_sim + commodity.e_ext_dc_sim)
            scenario.e_yrl_ext += (commodity.e_ext_ac_yrl + commodity.e_ext_dc_yrl)
            scenario.e_prj_ext += (commodity.e_ext_ac_prj + commodity.e_ext_dc_prj)
            scenario.e_dis_ext += (commodity.e_ext_ac_dis + commodity.e_ext_dc_dis)

        self.calc_energy_bidi(scenario)  # bidirectional block

    def calc_mntex_yrl(self):
        self.mntex_yrl = np.array([com.size for com in self.commodities.values()]).sum() * self.mntex_yrl

    def calc_opex_ext(self, scenario):
        """
        Cost calculation for external charging
        """
        self.opex_yrl_ext = eco.scale_sim2year(self.opex_sim_ext, scenario)
        self.opex_prj_ext = eco.scale_year2prj(self.opex_yrl_ext, scenario)
        self.opex_dis_ext = eco.acc_discount(self.opex_yrl_ext,
                                             scenario.prj_duration_yrs,
                                             scenario.wacc)
        self.opex_ann_ext = eco.ann_recur(self.opex_yrl_ext,
                                          1,  # lifespan of 1 yr -> opex happening yearly
                                          scenario.prj_duration_yrs,
                                          scenario.wacc,
                                          1)  # no cost decrease in opex

        scenario.opex_sim_ext += self.opex_sim_ext
        scenario.opex_yrl_ext += self.opex_yrl_ext
        scenario.opex_prj_ext += self.opex_prj_ext
        scenario.opex_dis_ext += self.opex_dis_ext
        scenario.opex_ann_ext += self.opex_ann_ext

    def calc_opex_sim(self, scenario):

        self.opex_sys = self.flow_in @ self.opex_spec_sys_chg + self.flow_out @ self.opex_spec_sys_dis
        self.opex_commodities = 0
        self.opex_commodities_ext = 0

        for commodity in self.commodities.values():
            commodity.opex_sim = commodity.flow_in @ self.opex_spec * scenario.timestep_hours
            commodity.opex_sim_ext = ((commodity.flow_ext_ac @ self.opex_spec_ext_ac) +
                                      (commodity.flow_ext_dc @ self.opex_spec_ext_dc)) * scenario.timestep_hours
            self.opex_commodities += commodity.opex_sim
            self.opex_commodities_ext += commodity.opex_sim_ext

        self.opex_sim = self.opex_sys + self.opex_commodities
        self.opex_sim_ext = self.opex_commodities_ext

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[
            (self.outflow, self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flow_in_ch = horizon.results[
            (self.bus_connected, self.inflow)]['sequences']['flow'][horizon.dti_ch]

        self.flow_in = pd.concat([self.flow_in, self.flow_in_ch])
        self.flow_out = pd.concat([self.flow_out, self.flow_out_ch])

        for commodity in self.commodities.values():
            commodity.get_ch_results(horizon, scenario)

    def get_legend_entry(self):
        return f'{self.name} total power'

    def get_opt_size(self, horizon):
        """
        Size for the commodity system is the sum of all commodity sizes in results
        """

        for commodity in self.commodities.values():
            commodity.size = horizon.results[(commodity.ess, None)]['scalars']['invest']
            if self.aging:
                commodity.aging_model.size = commodity.size
                # Calculate number of cells as a float to correctly represent power split with nonreal cells
                commodity.aging_model.n_cells = commodity.size / commodity.aging_model.e_cell

    def get_timeseries_results(self, scenario):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        super().get_timeseries_results(scenario)  # this goes up to the Block class
        for commodity in self.commodities.values():
            commodity.get_timeseries_results(scenario)

    def set_init_size(self, scenario, run):
        super().set_init_size(scenario, run)
        self.size_pc = self.size  # pc = per commodity
        self.size = self.size_pc * self.num if not self.opt else None

    def update_input_components(self):
        for commodity in self.commodities.values():
            commodity.update_input_components()


class BatteryCommoditySystem(CommoditySystem):
    """
    Dummy class to keep track of the different commodity system types in the energy system
    """

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

        self.bus_connected = scenario.blocks['core'].ac_bus

        if self.opt:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus_connected: solph.Flow(
                                                   investment=solph.Investment(ep_costs=self.epc),
                                                   variable_costs=self.opex_spec)}
                                               )
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus_connected: solph.Flow(nominal_value=1,
                                                                                       max=self.size,
                                                                                       variable_costs=self.opex_spec)}
                                               )
        scenario.components.append(self.src)

    def calc_energy(self, scenario):
        self.calc_energy_source_sink(scenario)

    def calc_opex_sim(self, scenario):
        self.opex_sim = self.flow_out @ self.opex_spec * scenario.timestep_hours

    def get_ch_results(self, horizon, scenario):
        self.flow_out_ch = horizon.results[(self.src, self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flow_out = pd.concat([self.flow_out, self.flow_out_ch])

    def get_opt_size(self, horizon):
        self.size = horizon.results[(self.src, self.bus_connected)]['scalars']['invest']

    def update_input_components(self):
        if self.apriori_data is not None:
            # Use power calculated in apriori_data for fixed output of block
            self.src.outputs[self.bus_connected].fix = self.apriori_data['p']


class GridConnection(InvestBlock):
    def __init__(self, name, scenario, run):

        self.size_g2mg = self.opt_g2mg = self.size_mg2g = self.opt_mg2g = self.equal = None

        super().__init__(name, scenario, run)

        """
        x denotes the flow measurement point in results

        ac_bus
          |
          |<-x-grid source
          |
          |-x->grid sink
          |
        """

        self.bus_connected = scenario.blocks['core'].ac_bus

        if self.opt_g2mg:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus_connected: solph.Flow(
                                                   investment=solph.Investment(ep_costs=self.epc),
                                                   variable_costs=self.opex_spec_g2mg)}
                                               )
        else:
            self.src = solph.components.Source(label=f'{self.name}_src',
                                               outputs={self.bus_connected: solph.Flow(
                                                   nominal_value=1,
                                                   max=self.size_g2mg,
                                                   variable_costs=self.opex_spec_g2mg)}
                                               )
        if self.opt_mg2g:
            self.snk = solph.components.Sink(label=f'{self.name}_snk',
                                             inputs={self.bus_connected: solph.Flow(
                                                 investment=solph.Investment(ep_costs=self.epc),
                                                 variable_costs=self.opex_spec_mg2g)}
                                             )
        else:
            self.snk = solph.components.Sink(label=f'{self.name}_snk',
                                             inputs={self.bus_connected: solph.Flow(
                                                 nominal_value=1,
                                                 max=self.size_mg2g,
                                                 variable_costs=self.opex_spec_mg2g)}
                                             )
        scenario.components.append(self.src)
        scenario.components.append(self.snk)

        if self.opt and self.equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            scenario.equal_variables.append({'var1': {'in': self.src, 'out': self.bus_connected},
                                             'var2': {'in': self.bus_connected, 'out': self.snk},
                                             'factor': 1})

    def calc_capex_init(self, scenario):
        """
        Calculate initial capital expenses
        """
        self.capex_init = np.maximum(self.size_g2mg, self.size_mg2g) * self.capex_spec

    def calc_energy(self, scenario):
        self.calc_energy_source_sink(scenario)

    def calc_mntex_yrl(self):
        self.mntex_yrl = np.maximum(self.size_g2mg, self.size_mg2g) * self.mntex_spec

    def calc_opex_sim(self, scenario):
        # @ is dot product (Skalarprodukt)
        self.opex_sim = self.flow_out @ self.opex_spec_g2mg * scenario.timestep_hours + \
                        self.flow_in @ self.opex_spec_mg2g * scenario.timestep_hours

    def get_ch_results(self, horizon, scenario):

        self.flow_in_ch = horizon.results[(self.bus_connected, self.snk)]['sequences']['flow'][horizon.dti_ch]
        self.flow_out_ch = horizon.results[(self.src, self.bus_connected)]['sequences']['flow'][horizon.dti_ch]

        self.flow_in = pd.concat([self.flow_in if not self.flow_in.empty else None, self.flow_in_ch])
        self.flow_out = pd.concat([self.flow_out if not self.flow_out.empty else None, self.flow_out_ch])

    def get_legend_entry(self):
        return (f'{self.name} power (max. {round(self.size_g2mg / 1e3)} kW from / '
                f'{round(self.size_mg2g / 1e3)} kW to grid)')

    def get_opt_size(self, horizon):

        if self.opt_g2mg:
            self.size_g2mg = horizon.results[(self.src, self.bus_connected)]['scalars']['invest']
        if self.opt_mg2g:
            self.size_mg2g = horizon.results[(self.bus_connected, self.snk)]['scalars']['invest']

    def set_init_size(self, scenario, run):

        if self.size_g2mg == 'equal' and self.size_mg2g == 'equal':
            self.size_g2mg = self.size_mg2g = 'opt'
            scenario.logger.warning(f'\"{self.name}\" component size was defined as "equal" for'
                                    f' the size of g2mg and mg2g. This was changed to optimization of the size of both'
                                    f' components with an additional "equal" constraint')
            self.equal = False
        elif self.size_g2mg == 'equal':
            self.size_g2mg = self.size_mg2g
            self.equal = True
        elif self.size_mg2g == 'equal':
            self.size_mg2g = self.size_g2mg
            self.equal = True
        else:
            self.equal = False

        if (self.size_g2mg != 'opt') and (self.size_mg2g != 'opt'):
            self.opt = self.opt_g2mg = self.opt_mg2g = False

        if self.size_g2mg == 'opt':
            self.opt = self.opt_g2mg = True

        if self.size_mg2g == 'opt':
            self.opt = self.opt_mg2g = True

    def update_input_components(self):
        if self.apriori_data is not None:
            # Use power calculated in apriori_data for fixed output of block
            self.src.outputs[self.bus_connected].fix = self.apriori_data['p'].clip(lower=0)
            self.snk.inputs[self.bus_connected].fix = self.apriori_data['p'].clip(upper=0) * -1


class FixedDemand(Block):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.path_input_file = os.path.join(run.path_input_data, 'dem', f'{self.filename}.csv')
        self.data = self.read_input_csv(self.path_input_file, scenario)

        self.data_ph = None  # placeholder

        # Creation of static energy system components --------------------------------

        """
        x denotes the flow measurement point in results

        ac_bus
          |
          |-x->dem_snk
          |
        """

        self.bus_connected = scenario.blocks['core'].ac_bus

        self.snk = solph.components.Sink(label='dem_snk',
                                         inputs={self.bus_connected: solph.Flow(nominal_value=1)})
        scenario.components.append(self.snk)

    def calc_energy(self, scenario):
        self.calc_energy_source_sink(scenario)

    def calc_revenue(self, scenario):
        self.crev_sim = (self.flow_in @ self.crev_spec) * scenario.timestep_hours  # @ is dot product (Skalarprodukt)
        self.accumulate_crev(scenario)

    def get_ch_results(self, horizon, scenario):
        self.flow_in_ch = horizon.results[(self.bus_connected, self.snk)]['sequences']['flow'][horizon.dti_ch]
        self.flow_in = pd.concat([self.flow_in, self.flow_in_ch])

    def get_legend_entry(self):
        return f'{self.name} power'

    def update_input_components(self):
        # new ph data slice is created during initialization of the PredictionHorizon
        self.snk.inputs[self.bus_connected].fix = self.data_ph['power_w']


class MobileCommodity:

    def __init__(self, name, parent, scenario):

        self.name = name
        self.parent = parent
        self.size = None if self.parent.opt else self.parent.size_pc
        self.pwr_chg = self.parent.pwr_chg
        self.pwr_dis = self.parent.pwr_dis
        self.eff_chg = self.parent.eff_chg
        self.eff_dis = self.parent.eff_dis
        self.temp_battery = self.parent.temp_battery

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

        self.apriori_data = None

        self.data_ph = self.sc_init_ph = None  # placeholder, is filled in update_input_components

        self.soc_init = self.parent.soc_init
        self.soc_init_ph = self.soc_init  # set first PH's initial state variables (only SOC)

        self.soh_init = self.parent.soh_init
        self.soh = pd.Series(index=scenario.dti_sim)
        self.soh.loc[scenario.starttime] = self.soh_init

        self.soc_min = (1 - self.soh_init) / 2
        self.soc_max = 1 - ((1 - self.soh_init) / 2)

        self.e_sim_in = self.e_yrl_in = self.e_prj_in = self.e_dis_in = 0
        self.e_sim_out = self.e_yrl_out = self.e_prj_out = self.e_dis_out = 0
        self.e_ext_ac_sim = self.e_ext_ac_yrl = self.e_ext_ac_prj = self.e_ext_ac_dis = 0
        self.e_ext_dc_sim = self.e_ext_dc_yrl = self.e_ext_dc_prj = self.e_ext_dc_dis = 0
        self.crev_time = self.crev_usage = self.crev_sim = self.crev_yrl = self.crev_prj = self.crev_dis = 0

        # timeseries result initialization
        self.flow_in_ch = self.flow_out_ch = self.flow_in = self.flow_out = self.flow = pd.Series(dtype='float64')
        self.flow_bat_in = self.flow_bat_out = self.flow_bat_in_ch = self.flow_bat_out_ch = pd.Series(dtype='float64')
        self.flow_ext_ac = self.flow_ext_dc = self.flow_ext_ac_ch = self.flow_ext_dc_ch = pd.Series(dtype='float64')

        self.sc_ch = self.soc_ch = self.soc = pd.Series(dtype='float64')

        # Creation of permanent energy system components --------------------------------

        """
         bus               mc1_bus
          |<---------mc1_mc-x-|<->mc1_ess
          |                   |
          |---mc_mc1-------x->|-->mc1_snk
          |                   |
          |                   |<--mc1_ext_ac (external charging AC)
          |                   |
          |                   |<--mc1_ext_dc (external charging DC)
          |
          |                 mc2_bus
          |<---------mc2_mc---|<->mc2_ess
          |                   |
          |---mc_mc2--------->|-->mc2_snk
          |                   |
          |                   |<--mc2_ext_ac (external charging AC)
          |                   |
          |                   |<--mc2_ext_dc (external charging DC)
        """

        self.bus = solph.Bus(label=f'{self.name}_bus')
        scenario.components.append(self.bus)

        self.inflow = solph.components.Converter(label=f'mc_{self.name}',
                                                 inputs={
                                                     self.parent.bus: solph.Flow(nominal_value=self.pwr_chg,
                                                                                 variable_costs=scenario.cost_eps)},
                                                 outputs={self.bus: solph.Flow(nominal_value=1,
                                                                               max=self.pwr_chg * self.eff_chg)},
                                                 conversion_factors={self.bus: self.eff_chg})
        scenario.components.append(self.inflow)

        self.outflow_enable = True if self.parent.int_lvl in ['v2v', 'v2mg'] else False
        self.outflow = solph.components.Converter(label=f'{self.name}_mc',
                                                  inputs={self.bus: solph.Flow(nominal_value=self.outflow_enable *
                                                                                             self.pwr_dis,
                                                                               variable_costs=scenario.cost_eps)},
                                                  outputs={self.parent.bus: solph.Flow()},
                                                  conversion_factors={self.parent.bus: self.eff_dis})
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
                                                       loss_rate=self.parent.loss_rate,
                                                       balanced=False,
                                                       initial_storage_level=self.soc_init_ph,
                                                       inflow_conversion_factor=1,
                                                       # efficiency already modeled in Converters
                                                       outflow_conversion_factor=1,
                                                       # efficiency already modeled in Converters
                                                       investment=solph.Investment(
                                                           ep_costs=self.parent.epc))
        else:
            self.ess = solph.components.GenericStorage(label=f'{self.name}_ess',
                                                       inputs={self.bus: solph.Flow(
                                                           variable_costs=self.parent.opex_spec)},
                                                       outputs={self.bus: solph.Flow()},
                                                       loss_rate=self.parent.loss_rate,
                                                       balanced=False,
                                                       initial_storage_level=self.soc_init_ph,
                                                       inflow_conversion_factor=1,
                                                       # efficiency already modeled in Converters
                                                       outflow_conversion_factor=1,
                                                       # efficiency already modeled in Converters
                                                       nominal_storage_capacity=self.size)
        scenario.components.append(self.ess)

        # always add charger -> reduce different paths of result calculations; no chargers -> power is set to 0 kW
        # add external AC charger as new energy source
        self.ext_ac = solph.components.Source(label=f'{self.name}_ext_ac',
                                              outputs={self.bus: solph.Flow(nominal_value=1,
                                                                            variable_costs=self.parent.opex_spec_ext_ac)}
                                              )
        scenario.components.append(self.ext_ac)

        # add external DC charger as new energy source
        self.ext_dc = solph.components.Source(label=f'{self.name}_ext_dc',
                                              outputs={self.bus: solph.Flow(nominal_value=1,
                                                                            variable_costs=self.parent.opex_spec_ext_dc)}
                                              )
        scenario.components.append(self.ext_dc)

        if self.parent.aging:
            self.aging_model = bat.BatteryPackModel(scenario, self)

    def add_power_trace(self, scenario):
        power_charge_max = round(self.pwr_chg / 1e3, 1)
        power_discharge_max = round(self.pwr_dis * self.parent.eff_chg / 1e3, 1)
        legentry = f'{self.name} power (max. {power_charge_max} kW charge / {power_discharge_max} kW discharge)'
        scenario.figure.add_trace(go.Scatter(x=self.flow.index,
                                             y=self.flow,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=False)

        scenario.figure.add_trace(go.Scatter(x=self.flow_ext_dc.index,
                                             y=self.flow_ext_dc + self.flow_ext_ac,
                                             mode='lines',
                                             name=f'{self.name} external charging power (AC & DC)',
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=False)

    def add_soc_trace(self, scenario):
        legentry = f"{self.name} SOC ({round(self.size/1e3,1)} kWh)"
        scenario.figure.add_trace(go.Scatter(x=self.soc.index,
                                             y=self.soc,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=True)

        legentry = f"{self.name} SOH"
        data = self.soh.dropna()
        scenario.figure.add_trace(go.Scatter(x=data.index,
                                             y=data,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=True)

    def calc_aging(self, run, scenario, horizon):
        self.aging_model.age(self, run, scenario, horizon)

    # noinspection DuplicatedCode
    def calc_results(self, scenario):

        # energy result calculation does not count towards delivered/produced energy (already done at the system level)
        self.e_sim_in = self.flow_in.sum() * scenario.timestep_hours  # flow values are powers --> conversion to Wh
        self.e_sim_out = self.flow_out.sum() * scenario.timestep_hours
        self.e_yrl_in = eco.scale_sim2year(self.e_sim_in, scenario)
        self.e_yrl_out = eco.scale_sim2year(self.e_sim_out, scenario)
        self.e_prj_in = eco.scale_year2prj(self.e_yrl_in, scenario)
        self.e_prj_out = eco.scale_year2prj(self.e_yrl_out, scenario)
        self.e_dis_in = eco.acc_discount(self.e_yrl_in, scenario.prj_duration_yrs, scenario.wacc)
        self.e_dis_out = eco.acc_discount(self.e_yrl_out, scenario.prj_duration_yrs, scenario.wacc)

        # energy results for external chargers
        self.e_ext_ac_sim = self.flow_ext_ac.sum() * scenario.timestep_hours
        self.e_ext_dc_sim = self.flow_ext_dc.sum() * scenario.timestep_hours
        self.e_ext_ac_yrl = eco.scale_sim2year(self.e_ext_ac_sim, scenario)
        self.e_ext_dc_yrl = eco.scale_sim2year(self.e_ext_dc_sim, scenario)
        self.e_ext_ac_prj = eco.scale_year2prj(self.e_ext_ac_yrl, scenario)
        self.e_ext_dc_prj = eco.scale_year2prj(self.e_ext_dc_yrl, scenario)
        self.e_ext_ac_dis = eco.acc_discount(self.e_ext_ac_yrl, scenario.prj_duration_yrs, scenario.wacc)
        self.e_ext_dc_dis = eco.acc_discount(self.e_ext_dc_yrl, scenario.prj_duration_yrs, scenario.wacc)

        self.flow = self.flow_in - self.flow_out  # for plotting

    def calc_revenue(self, scenario):

        # rental time based revenue
        self.crev_time = ((~self.data.loc[scenario.dti_sim, 'atbase'] @ self.parent.crev_spec_time) *
                          scenario.timestep_hours)

        # usage based revenue
        if isinstance(self.parent, VehicleCommoditySystem):
            self.crev_usage = self.data.loc[scenario.dti_sim, 'tour_dist'] @ self.parent.crev_spec_dist
        else:  # BatteryCommoditySystems have no usage based revenue
            self.crev_usage = 0  # Battery rental is a fixed time based price, irrespective of energy consumption

        self.crev_sim = self.crev_time + self.crev_usage

    def get_ch_results(self, horizon, scenario):

        self.flow_bat_out_ch = horizon.results[(self.ess, self.bus)]['sequences']['flow'][horizon.dti_ch]
        self.flow_bat_in_ch = horizon.results[(self.bus, self.ess)]['sequences']['flow'][horizon.dti_ch]

        self.flow_bat_in = pd.concat([self.flow_bat_in if not self.flow_bat_in.empty else None, self.flow_bat_in_ch])
        self.flow_bat_out = pd.concat([self.flow_bat_out if not self.flow_bat_out.empty else None, self.flow_bat_out_ch])

        self.flow_out_ch = horizon.results[(self.bus, self.outflow)]['sequences']['flow'][horizon.dti_ch]
        self.flow_in_ch = horizon.results[(self.inflow, self.bus)]['sequences']['flow'][horizon.dti_ch]

        self.flow_in = pd.concat([self.flow_in if not self.flow_in.empty else None, self.flow_in_ch])
        self.flow_out = pd.concat([self.flow_out if not self.flow_out.empty else None, self.flow_out_ch])

        # Get results of external chargers
        self.flow_ext_ac_ch = horizon.results[(self.ext_ac, self.bus)]['sequences']['flow'][horizon.dti_ch]
        self.flow_ext_dc_ch = horizon.results[(self.ext_dc, self.bus)]['sequences']['flow'][horizon.dti_ch]

        self.flow_ext_ac = pd.concat([self.flow_ext_ac if not self.flow_ext_ac.empty else None, self.flow_ext_ac_ch])
        self.flow_ext_dc = pd.concat([self.flow_ext_dc if not self.flow_ext_dc.empty else None, self.flow_ext_dc_ch])

        # storage content during PH (does not include endtime)
        self.sc_ch = solph.views.node(
            horizon.results, f'{self.name}_ess')['sequences'][((f'{self.name}_ess', 'None'),
                                                               'storage_content')][horizon.dti_ch]
        # storage content at end of ph to initialize next PH
        self.sc_init_ph = solph.views.node(
            horizon.results, f'{self.name}_ess')['sequences'][((f'{self.name}_ess', 'None'),
                                                               'storage_content')][horizon.ch_endtime]

        self.soc_ch = self.sc_ch / self.size
        self.soc_init_ph = self.sc_init_ph / self.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge

    def get_timeseries_results(self, scenario):
        """
        Collect timeseries results of the commodity in a scenario wide dataframe to be saved
        """
        commodity_ts_results = pd.DataFrame({f'{self.name}_flow_in': self.flow_in,
                                             f'{self.name}_flow_out': self.flow_out,
                                             f'{self.name}_flow_bat_in': self.flow_bat_in,
                                             f'{self.name}_flow_bat_out': self.flow_bat_out,
                                             f'{self.name}_flow_ext_dc': self.flow_ext_dc,
                                             f'{self.name}_flow_ext_ac': self.flow_ext_ac,
                                             f'{self.name}_soc': self.soc,
                                             f'{self.name}_soh': self.soh})
        scenario.result_timeseries = pd.concat([scenario.result_timeseries, commodity_ts_results], axis=1)

    def update_input_components(self):

        # set vehicle consumption data for sink
        self.snk.inputs[self.bus].fix = self.data_ph['consumption']

        # set initial storage levels for coming prediction horizon
        # limit and set initial storage level to min and max soc from aging
        self.ess.initial_storage_level = statistics.median([self.soc_min, self.soc_init_ph, self.soc_max])

        if self.apriori_data is not None:
            # define charging powers (as per uc power calculation)
            self.inflow.outputs[self.bus].fix = self.apriori_data['p_int_ac']
            self.ext_ac.outputs[self.bus].fix = self.apriori_data['p_ext_ac']
            self.ext_dc.outputs[self.bus].fix = self.apriori_data['p_ext_dc']
        else:
            # enable/disable Converters to mcx_bus depending on whether the commodity is at base
            self.inflow.inputs[self.parent.bus].max = self.data_ph['atbase'].astype(int)
            self.outflow.inputs[self.bus].max = self.data_ph['atbase'].astype(int)

            # define consumption data for sink (only enabled when detached from base)
            self.snk.inputs[self.bus].fix = self.data_ph['consumption']

            # Adjust min/max storage levels based on state of health for the upcoming prediction horizon
            # nominal_storage_capacity is retained for accurate state of charge tracking and cycle depth
            # relative to nominal capacity
            soc_min_clipped = self.data_ph['minsoc'].clip(lower=self.soc_min, upper=self.soc_max)
            self.ess.min_storage_level = soc_min_clipped
            self.ess.max_storage_level = pd.Series(data=self.soc_max, index=self.data_ph.index)

            # enable/disable ac and dc charging station dependent on input data
            self.ext_ac.outputs.data[self.bus].max = self.data_ph['atac'].astype(int) * self.parent.pwr_ext_ac
            self.ext_dc.outputs.data[self.bus].max = self.data_ph['atdc'].astype(int) * self.parent.pwr_ext_dc


class PVSource(RenewableInvestBlock):

    def __init__(self, name, scenario, run):

        self.api_startyear = self.api_endyear = self.meta = None
        self.bus_connected = scenario.blocks['core'].dc_bus

        super().__init__(name, scenario, run)

    def calc_power_solcast(self):

        u0 = 26.9  # W/(˚C.m2) - cSi Free standing
        u1 = 6.2  # W.s/(˚C.m3) - cSi Free standing
        mod_temp = self.data['temp_air'] + (self.data['GtiFixedTilt'] / (u0 + (u1 * self.data['WindSpeed10m'])))

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
        eff_rel = (1 +
                  (k1 * lng) +
                  (k2 * (lng ** 2)) +
                  (k3 * t) +
                  (k4 * t * lng) +
                  (k5 * t * (lng ** 2)) +
                  (k6 * (t ** 2)))
        eff_rel = eff_rel.fillna(0)

        # calculate power of a 1kWp array, limited to 0 (negative values fail calculation)
        self.data['P'] = np.maximum(0, eff_rel * self.data['GtiFixedTilt'])

    def get_timeseries_data(self, scenario, run):

        if self.data_source == 'pvgis api':  # API input selected
            self.api_startyear = scenario.starttime.tz_convert(pytz.utc).year
            self.api_endyear = scenario.sim_endtime.tz_convert(pytz.utc).year
            self.data, self.meta, _ = pvlib.iotools.get_pvgis_hourly(scenario.latitude,
                                                                     scenario.longitude,
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

            # PVGIS gives time slots as XX:06h - round to full hour
            self.data.index = self.data.index.round('h')

        else:  # input from file instead of API
            self.path_input_file = os.path.join(run.path_input_data, 'pv', f'{self.filename}.csv')

            if self.data_source.lower() == 'pvgis file':  # data input from fixed PVGIS csv file
                self.data, self.meta, _ = pvlib.iotools.read_pvgis_hourly(self.path_input_file, map_variables=True)
                scenario.latitude = self.meta['latitude']
                scenario.longitude = self.meta['longitude']
                # PVGIS gives time slots as XX:06 - round to full hour
                self.data.index = self.data.index.round('h')
            elif self.data_source.lower() == 'solcast file':  # data input from fixed Solcast csv file
                # no lat/lon contained in solcast files
                # self.data = pd.read_csv(self.path_input_file,
                #                         parse_dates=True,
                #                         index_col='PeriodStart')
                self.data = pd.read_csv(self.path_input_file)
                self.data['PeriodStart'] = pd.to_datetime(self.data['PeriodStart'])
                self.data['PeriodEnd'] = pd.to_datetime(self.data['PeriodEnd'])
                self.data.set_index(pd.DatetimeIndex(self.data['PeriodStart']), inplace=True)
                self.data['wind_speed'] = self.data['WindSpeed10m']
                self.data.rename(columns={'AirTemp': 'temp_air'}, inplace=True)  # compatibility with aging model
                self.calc_power_solcast()
            else:
                scenario.logger.warning('No usable PV input type specified - exiting')
                exit()  # TODO exit scenario instead of entire execution

        # resample to timestep, fill NaN values with previous ones (or next ones, if not available)
        self.data = self.data.resample(scenario.timestep).mean().ffill().bfill()
        # convert to local time
        self.data.index = self.data.index.tz_convert(tz=scenario.timezone)
        # data is in W for a 1kWp PV array -> convert to specific power
        self.data['power_spec'] = self.data['P'] / 1e3

        self.data = self.data[['power_spec', 'wind_speed', 'temp_air']]  # only keep relevant columns


class StationaryEnergyStorage(InvestBlock):

    def __init__(self, name, scenario, run):

        super().__init__(name, scenario, run)

        self.soc_init_ph = self.soc_init

        self.apriori_data = self.sc_init_ph =  None

        self.loss_rate = eco.convert_sdr_to_timestep(self.sdr)

        self.flow_in_ch = self.flow_out_ch = pd.Series(dtype='float64')  # result data
        self.flow_in = self.flow_out = pd.Series(dtype='float64')

        self.sc_ch = self.soc_ch = pd.Series(dtype='float64')  # result data
        self.soc = pd.Series()

        self.soh = pd.Series(index=scenario.dti_sim)
        self.soh.loc[scenario.starttime] = self.soh_init

        """
        x denotes the flow measurement point in results

        dc_bus
          |
          |<-x->ess
          |
        """

        self.bus_connected = scenario.blocks['core'].dc_bus

        if self.opt:
            self.ess = solph.components.GenericStorage(label='ess',
                                                       inputs={self.bus_connected: solph.Flow(
                                                           variable_costs=self.opex_spec)},
                                                       outputs={self.bus_connected: solph.Flow()},
                                                       loss_rate=self.loss_rate,
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.soc_init_ph,
                                                       invest_relation_input_capacity=self.crate_chg,
                                                       invest_relation_output_capacity=self.crate_dis,
                                                       inflow_conversion_factor=self.eff_chg,
                                                       outflow_conversion_factor=self.eff_dis,
                                                       investment=solph.Investment(ep_costs=self.epc))
        else:
            self.ess = solph.components.GenericStorage(label='ess',
                                                       inputs={self.bus_connected: solph.Flow(
                                                           nominal_value=1,
                                                           max=self.size * self.crate_chg,
                                                           variable_costs=self.opex_spec)},
                                                       outputs={self.bus_connected: solph.Flow(
                                                           nominal_value=1,
                                                           max=self.size * self.crate_dis,
                                                       )},
                                                       loss_rate=self.loss_rate,
                                                       balanced={'go': True, 'rh': False}[scenario.strategy],
                                                       initial_storage_level=self.soc_init_ph,
                                                       inflow_conversion_factor=self.eff_chg,
                                                       outflow_conversion_factor=self.eff_dis,
                                                       nominal_storage_capacity=self.size)
        scenario.components.append(self.ess)

        if self.aging:
            self.aging_model = bat.BatteryPackModel(scenario, self)

    def add_soc_trace(self, scenario):
        legentry = f"{self.name} SOC ({round(self.size/1e3)} kWh)"
        scenario.figure.add_trace(go.Scatter(x=self.soc.index,
                                             y=self.soc,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None)),
                                  secondary_y=True)

        legentry = f"{self.name} SOH"
        data = self.soh.dropna()
        scenario.figure.add_trace(go.Scatter(x=data.index,
                                             y=data,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=True)

    def calc_aging(self, run, scenario, horizon):
        self.aging_model.age(self, run, scenario, horizon)

    def calc_energy(self, scenario):
        self.calc_energy_bidi(scenario)

    def calc_opex_sim(self, scenario):
        self.opex_sim = self.flow_in @ self.opex_spec * scenario.timestep_hours

    def get_ch_results(self, horizon, scenario):

        self.flow_out_ch = horizon.results[(self.ess, self.bus_connected)]['sequences']['flow'][
            horizon.dti_ch]
        self.flow_in_ch = horizon.results[(self.bus_connected, self.ess)]['sequences']['flow'][
            horizon.dti_ch]

        self.flow_in = pd.concat([self.flow_in if not self.flow_in.empty else None, self.flow_in_ch])
        self.flow_out = pd.concat([self.flow_out if not self.flow_out.empty else None, self.flow_out_ch])

        # storage content during PH (does not include endtime)
        self.sc_ch = solph.views.node(horizon.results, self.name)['sequences'][
            ((self.name, 'None'), 'storage_content')][horizon.dti_ch]
        # storage content at end of ph to initialize next PH
        self.sc_init_ph = solph.views.node(horizon.results, self.name)['sequences'][
            ((self.name, 'None'), 'storage_content')][horizon.ch_endtime]

        self.soc_ch = self.sc_ch / self.size
        self.soc_init_ph = self.sc_init_ph / self.size

        self.soc = pd.concat([self.soc, self.soc_ch])  # tracking state of charge

    def get_opt_size(self, horizon):
        self.size = horizon.results[(self.ess, None)]['scalars']['invest']

    def get_legend_entry(self):
        power_charge = round(self.size * self.crate_chg / 1e3)
        power_discharge = round(self.size * self.crate_dis * self.eff_dis / 1e3)
        return f'{self.name} power (max. {power_charge} kW charge / {power_discharge} kW discharge)'

    def get_timeseries_results(self, scenario):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        super().get_timeseries_results(scenario)  # this goes up to the Block class

        block_ts_results = pd.DataFrame({f'{self.name}_soc': self.soc, f'{self.name}_soh': self.soh})
        scenario.result_timeseries = pd.concat([scenario.result_timeseries, block_ts_results], axis=1)

    def update_input_components(self):

        self.ess.initial_storage_level = self.soc_init_ph

        if self.apriori_data is not None:
            self.ess.inputs[self.bus_connected].fix = self.apriori_data['p'].clip(upper=0) * (-1)
            self.ess.outputs[self.bus_connected].fix = self.apriori_data['p'].clip(lower=0)


class SystemCore(InvestBlock):

    def __init__(self, name, scenario, run):

        self.size_acdc = self.opt_acdc = self.size_dcac = self.opt_dcac = 0  # might be set in super().__init__

        super().__init__(name, scenario, run)

        self.e_sim_acdc = self.e_sim_dcac = self.e_yrl_acdc = self.e_yrl_dcac = 0
        self.e_prj_acdc = self.e_prj_dcac = self.e_dis_acdc = self.e_dis_dcac = 0

        self.flow_acdc = self.flow_dcac = self.flow_acdc_ch = self.flow_dcac_ch = pd.Series(dtype='float64')

        self.equal = None

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

        if self.opt_acdc:
            self.ac_dc = solph.components.Converter(label='ac_dc',
                                                    inputs={self.ac_bus: solph.Flow(investment=solph.Investment(
                                                        ep_costs=self.epc),
                                                        variable_costs=self.opex_spec)},
                                                    outputs={self.dc_bus: solph.Flow(
                                                        variable_costs=scenario.cost_eps)},
                                                    conversion_factors={self.dc_bus: self.eff_acdc})

        else:
            self.ac_dc = solph.components.Converter(label='ac_dc',
                                                    inputs={self.ac_bus: solph.Flow(variable_costs=scenario.cost_eps,
                                                                                    nominal_value=1,
                                                                                    max=self.size_acdc)},
                                                    outputs={self.dc_bus: solph.Flow(
                                                        variable_costs=scenario.cost_eps)},
                                                    conversion_factors={self.dc_bus: self.eff_acdc})

        if self.opt_dcac:
            self.dc_ac = solph.components.Converter(label='dc_ac',
                                                    inputs={self.dc_bus: solph.Flow(investment=solph.Investment(
                                                        ep_costs=self.epc),
                                                        variable_costs=self.opex_spec)},
                                                    outputs={self.ac_bus: solph.Flow(
                                                        variable_costs=scenario.cost_eps)},
                                                    conversion_factors={self.ac_bus: self.eff_dcac})
        else:
            self.dc_ac = solph.components.Converter(label='dc_ac',
                                                    inputs={self.dc_bus: solph.Flow(variable_costs=scenario.cost_eps,
                                                                                    nominal_value=1,
                                                                                    max=self.size_dcac)},
                                                    outputs={self.ac_bus: solph.Flow(
                                                        variable_costs=scenario.cost_eps)},
                                                    conversion_factors={self.ac_bus: self.eff_dcac})

        scenario.components.append(self.ac_dc)
        scenario.components.append(self.dc_ac)

        if self.opt and self.equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            scenario.equal_variables.append({'var1': {'in': self.dc_bus, 'out': self.dc_ac},
                                             'var2': {'in': self.ac_bus, 'out': self.ac_dc},
                                             'factor': 1})

    def add_power_trace(self, scenario):
        legentry = f'{self.name} DC-AC power (max. {round(self.size_dcac/1e3)} kW)'
        scenario.figure.add_trace(go.Scatter(x=self.flow_dcac.index,
                                             y=self.flow_dcac,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=False)

        legentry = f'{self.name} AC-DC power (max. {round(self.size_acdc/1e3)} kW)'
        scenario.figure.add_trace(go.Scatter(x=self.flow_acdc.index,
                                             y=self.flow_acdc,
                                             mode='lines',
                                             name=legentry,
                                             line=dict(width=2, dash=None),
                                             visible='legendonly'),
                                  secondary_y=False)

    def calc_capex_init(self, scenario):
        self.capex_init = (self.size_acdc + self.size_dcac) * self.capex_spec

    def calc_energy(self, scenario):

        # energy result calculation is different from any other block as there is no in/out definition of flow
        self.e_sim_dcac = self.flow_dcac.sum() * scenario.timestep_hours  # flow values are powers --> conversion to Wh
        self.e_sim_acdc = self.flow_acdc.sum() * scenario.timestep_hours
        self.e_yrl_dcac = eco.scale_sim2year(self.e_sim_dcac, scenario)
        self.e_yrl_acdc = eco.scale_sim2year(self.e_sim_acdc, scenario)
        self.e_prj_dcac = eco.scale_year2prj(self.e_yrl_dcac, scenario)
        self.e_prj_acdc = eco.scale_year2prj(self.e_yrl_acdc, scenario)
        self.e_dis_dcac = eco.acc_discount(self.e_yrl_dcac, scenario.prj_duration_yrs, scenario.wacc)
        self.e_dis_acdc = eco.acc_discount(self.e_yrl_acdc, scenario.prj_duration_yrs, scenario.wacc)

    def calc_mntex_yrl(self):
        self.mntex_yrl = (self.size_acdc + self.size_dcac) * self.mntex_spec

    def calc_opex_sim(self, scenario):
        self.opex_sim = (self.flow_acdc + self.flow_dcac) @ self.opex_spec * scenario.timestep_hours

    def get_ch_results(self, horizon, scenario):

        self.flow_acdc_ch = horizon.results[(scenario.blocks['core'].ac_bus, self.ac_dc)]['sequences']['flow'][
            horizon.dti_ch]
        self.flow_dcac_ch = horizon.results[(scenario.blocks['core'].dc_bus, self.dc_ac)]['sequences']['flow'][
            horizon.dti_ch]

        self.flow_acdc = pd.concat([self.flow_acdc, self.flow_acdc_ch])
        self.flow_dcac = pd.concat([self.flow_dcac, self.flow_dcac_ch])

    def get_opt_size(self, horizon):

        if self.opt_acdc:
            self.size_acdc = horizon.results[(self.ac_bus, self.ac_dc)]['scalars']['invest']
        if self.opt_dcac:
            self.size_dcac = horizon.results[(self.dc_bus, self.dc_ac)]['scalars']['invest']

    def get_timeseries_results(self, scenario):
        """
        Collect timeseries results of the block in a scenario wide dataframe to be saved
        """
        block_ts_results = pd.DataFrame({f'{self.name}_flow_dcac': self.flow_dcac,
                                         f'{self.name}_flow_acdc': self.flow_acdc})
        scenario.result_timeseries = pd.concat([scenario.result_timeseries, block_ts_results], axis=1)

    def set_init_size(self, scenario, run):

        if (self.size_acdc == 'equal') and (self.size_dcac == 'equal'):
            self.size_acdc = self.size_dcac = 'opt'
            scenario.logger.warning(f'\"{self.name}\" component size was defined as "equal" for'
                                    f' AC/DC and DC/AC converter. This was changed to optimization of the size of both'
                                    f' components with an additional "equal" constraint')
        elif self.size_acdc == 'equal':
            self.size_acdc = self.size_dcac
            self.equal = True
        elif self.size_dcac == 'equal':
            self.size_dcac = self.size_acdc
            self.equal = True
        else:
            self.equal = False

        if (self.size_acdc != 'opt') and (self.size_dcac != 'opt'):
            self.opt = self.opt_acdc = self.opt_dcac = False

        if self.size_acdc == 'opt':
            self.opt = self.opt_acdc = True

        if self.size_dcac == 'opt':
            self.opt = self.opt_dcac = True

    def update_input_components(self):
        pass  # function needs to be callable


class VehicleCommoditySystem(CommoditySystem):
    """
    VehicleCommoditySystem is a CommoditySystem operating vehicles in a service system generating revenues from rentals
    """

    def __init__(self, name, scenario, run):
        super().__init__(name, scenario, run)

    def calc_revenue(self, scenario):
        for commodity in self.commodities.values():
            commodity.calc_revenue(scenario)
            self.crev_sim += commodity.crev_sim

        self.accumulate_crev(scenario)


class WindSource(RenewableInvestBlock):

    def __init__(self, name, scenario, run):

        self.bus_connected = scenario.blocks['core'].ac_bus

        super().__init__(name, scenario, run)

        self.path_turbine_data_file = self.turbine_data = self.turbine_type = None

    def get_timeseries_data(self, scenario, run):

        if self.data_source in scenario.blocks.keys():  # input from a PV block

            self.data = scenario.blocks[self.data_source].data.copy()
            self.data['wind_speed_adj'] = windpowerlib.wind_speed.hellman(self.data['wind_speed'], 10, self.height)

            self.path_turbine_data_file = os.path.join(run.path_input_data, 'wind', 'turbine_data.pkl')
            self.turbine_type = 'E-53/800'  # smallest fully filled wind turbine in dataseta as per June 2024
            self.turbine_data = pd.read_pickle(self.path_turbine_data_file)
            self.turbine_data = self.turbine_data.loc[self.turbine_data['turbine_type'] == self.turbine_type].reset_index()

            self.data['power_original'] = windpowerlib.power_output.power_curve(
                wind_speed=self.data['wind_speed_adj'],
                power_curve_wind_speeds=ast.literal_eval(self.turbine_data.loc[0, 'power_curve_wind_speeds']),
                power_curve_values=ast.literal_eval(self.turbine_data.loc[0, 'power_curve_values']),
                density_correction=False)
            self.data['power_spec'] = self.data['power_original'] / self.turbine_data.loc[0, 'nominal_power']

        else:  # input from file instead of PV block

            self.path_input_file = os.path.join(run.path_input_data, 'wind', f'{self.filename}.csv')
            self.data = self.read_input_csv(self.path_input_file, scenario)
