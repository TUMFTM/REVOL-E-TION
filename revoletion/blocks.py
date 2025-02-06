#!/usr/bin/env python3

import os

import numpy as np
import oemof.solph as solph
import pandas as pd
import plotly.graph_objects as go

from revoletion import battery as bat
from revoletion import economics as eco
from revoletion import utils


class Block:

    def __init__(self,
                 name: str,
                 scenario,  # todo type hint
                 flow_names: list = None,
                 state_names: list = None,
                 size_names: list = None,
                 poe_names: list = None,
                 params: dict = None,
                 parent=None,  # todo type hint
                 ):
        """
        Initialize (Sub)Block object with its attributes and data structures
        """

        self.name = name
        self.scenario = scenario
        self.parent = parent

        # region set attributes from scenario file or parent
        if self.parent is None:  # is top level block
            scenario.blocks[self.name] = self

            self.parameters = self.scenario.parameters.loc[self.name]
            for key, value in self.parameters.items():
                setattr(self, key, value)  # this sets all the parameters defined in the scenario file

        elif params is not None:  # is subblock with inherited params
            self.parent.subblocks[self.name] = self

            for key, value in params.items():
                setattr(self, key, value)

        else:  # is subblock without params defined
            raise ValueError(f'Subblock {self.name} of {self.parent.name} has no parameters defined')
        # endregion

        # region initialize data structures
        self.subblocks = dict()
        self.components = dict()
        self.bus_connected = None

        self.flows_apriori = pd.DataFrame()  # partially recalculated for every horizon
        self.flows = pd.DataFrame(index=self.scenario.dti_sim,
                                  columns=flow_names if flow_names is not None else [],
                                  data=np.nan,
                                  dtype='float64')
        self.states = pd.DataFrame(index=utils.extend_dti(self.scenario.dti_sim),
                                   columns=state_names if state_names is not None else [],
                                   data=np.nan,
                                   dtype='float64')
        self.energies = pd.DataFrame(index=['total', 'in', 'out', 'del', 'pro'],
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,  # cumulative property
                                     dtype=float)

        self.sizes = pd.DataFrame()
        self.expansion_equal = False
        self.initialize_sizes(sizes=size_names)

        self.poes = {'total': eco.PointOfEvaluation(name='total', block=self, aggregator=True)}  # total covers entire block
        self.poes.update({name: eco.PointOfEvaluation(name=name, block=self, aggregator=False) for name in poe_names} \
            if poe_names is not None else dict())
        self.scenario.capex_init_existing += self.poes['total'].capex['preexisting']

        # todo move ls & ccr to poe
        # self.economic_results = eco.EconomicResults(self) #todo
        # endregion

    def initialize_sizes(self,
                         sizes: list = None):
        """
        Initialize the sizes DataFrame for the block
        """

        sizes = sizes if sizes else ['block']
        self.sizes = pd.DataFrame(index=sizes,
                                  columns=['total', 'preexisting', 'expansion', 'total_max', 'expansion_max'],
                                  data=np.nan,
                                  dtype='float64')
        self.sizes['invest'] = False

        for size in sizes:
            size_str = '' if size == 'block' else f'_{size}'
            self.sizes.loc[size, 'preexisting'] = getattr(self, f'size_preexisting{size_str}', 0)
            self.sizes.loc[size, 'total_max'] = getattr(self, f'size_max{size_str}', 0)
            self.sizes.loc[size, 'invest'] = getattr(self, f'invest{size_str}', False)
            # delete attributes if available
            for attr in ['size_preexisting', 'size_max', 'invest']:
                attr_str = f'{attr}{size_str}'
                if hasattr(self, attr_str):
                    delattr(self, attr_str)

        # expansion_max logic: 0=no investment, NaN=unlimited investment, float=limited investment
        self.sizes['expansion_max'] = self.sizes['total_max'] - self.sizes['preexisting']
        self.sizes.loc[~self.sizes['invest'], 'expansion_max'] = 0

        if self.sizes['invest'].any() and self.scenario.strategy != 'go':
            raise ValueError(f'Block "{self.name}" component size optimization '
                             f'not implemented for any other strategy than "GO"')

    def pre_horizon(self,
                    horizon):
        """
        build energy system components before each horizon
        """
        self.define_oemof_components(horizon=horizon)
        horizon.es.add(*self.components.values())

        for subblock in self.subblocks.values():
            subblock.pre_horizon(horizon=horizon)


class SystemCore(Block):

    def __init__(self,
                 name : str,
                 scenario):
        super().__init__(name=name,
                         scenario=scenario,
                         flow_names=['acdc', 'dcac'],
                         state_names=None,
                         size_names=['acdc', 'dcac'],
                         poe_names=['acdc', 'dcac'],
                         params=None,
                         parent=None)

    def initialize_sizes(self,
                         sizes: list = None):

        self.expansion_equal = True if self.invest_acdc =='equal' or self.invest_dcac == 'equal' else False

        utils.init_equalizable_variables(block=self, name_vars=['invest_acdc', 'invest_dcac'])
        utils.init_equalizable_variables(block=self, name_vars=['size_preexisting_acdc', 'size_preexisting_dcac'])
        utils.init_equalizable_variables(block=self, name_vars=['size_max_acdc', 'size_max_dcac'])

        super().initialize_sizes(sizes=sizes)

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method
        x denotes the flow measurement point in results

          dc          ac
          |-x--dcac-->|
          |           |
          |<---acdc-x-|
        """

        self.components['ac'] = solph.Bus(label='ac')
        self.components['dc'] = solph.Bus(label='dc')

        self.components['acdc'] = solph.components.Converter(
            label='acdc',
            inputs={self.components['ac']: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.poes['acdc'].capex['spec_ep'],
                                               existing=self.sizes.loc['acdc', 'existing'],
                                               maximum=utils.conv_add_max(self.sizes.loc['acdc', 'additional_max'])),
                variable_costs=self.poes['acdc'].opex['spec_ep'][horizon.dti_ph])},
            outputs={self.components['dc']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['dc']: self.eff_acdc})

        self.components['dcac'] = solph.components.Converter(
            label='dcac',
            inputs={self.components['dc']: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.poes['dcac'].capex['spec_ep'],
                                               existing=self.sizes.loc['dcac', 'existing'],
                                               maximum=utils.conv_add_max(self.sizes.loc['dcac', 'additional_max'])),
                variable_costs=self.poes['dcac'].opex['spec_ep'][horizon.dti_ph])},
            outputs={self.components['ac']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['ac']: self.eff_dcac})

        horizon.constraints.add_invest_costs(
            invest=(self.components['ac'], self.components['acdc']),
            capex_spec=self.poes['acdc'].capex['spec'],
            invest_type='flow')

        horizon.constraints.add_invest_costs(
            invest=(self.components['dc'], self.components['dcac']),
            capex_spec=self.poes['dcac'].capex['spec'],
            invest_type='flow')

        if self.equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            horizon.constraints.add_equal_invests([{'in': self.components['dc'], 'out': self.components['dcac']},
                                                   {'in': self.components['ac'], 'out': self.components['acdc']}])


class RenewableSource(Block):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_names=['out', 'total', 'pot', 'curt'],
                         state_names=None,
                         size_names=['block'],
                         poe_names=['block'],
                         params=None,
                         parent=None)


class StorageBlock:

    def __init__(self):

        self.scenario.storage_blocks[self.name] = self

        self.loss_rate = utils.convert_sdr(self.sdr, pd.Timedelta(hours=1))  #todo move function to StorageBlock

        self.states.loc[self.scenario.starttime, 'soc'] = self.soc_init
        delattr(self, 'soc_init')

        self.aging_model = bat.BatteryPackModel(self)
        self.soc_min = (1 - self.states.loc[self.scenario.starttime, 'soh']) / 2  # todo move to states df
        self.soc_max = 1 - ((1 - self.states.loc[self.scenario.starttime, 'soh']) / 2)

    def calc_energies(self):
        """
        post-scenario calculation of energies as integrals of flows
        """
        pass

    def add_state_traces(self):
        """
        post-scenario plotting of SOC and SOH traces in timeseries plot
        """
        legentry = f'{self.name} SOC ({self.size.loc["block", "total"]/1e3:.1f} kWh)'
        self.scenario.figure.add_trace(go.Scatter(x=self.states.index,
                                                  y=self.states['soc'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None)),
                                       secondary_y=True)

        legentry = f'{self.name} SOH'
        data = self.states['soh'].dropna()
        self.scenario.figure.add_trace(go.Scatter(x=data.index,
                                                  y=data,
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=True)







class PVSource(RenewableSource):

    pass


class WindSource(RenewableSource):

    pass


class FixedDemand(Block):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_names=['in', 'total'],
                         state_names=None,
                         size_names=None,
                         poe_names=['block'],
                         params=None,
                         parent=None)

        self.get_flows_apriori()

    def get_flows_apriori(self):
        self.flows_apriori.index = self.scenario.dti_sim_extd
        if self.load_profile in ['h0', 'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'l0', 'l1', 'l2']:
            self.get_demand_from_slp()
        elif self.load_profile in ['const', 'constant']:
            self.flows_apriori['demand'] = self.consumption_yrl / (365 * 24)
        elif isinstance(self.load_profile, str):  # load_profile is a file name
            self.get_demand_from_file()
        else:
            raise ValueError(f'Parameter "load_profile" in block "{self.block.name}" is not valid')

    def get_demand_from_file(self):
        data = utils.read_input_csv(block=self,
                                    path_input_file=os.path.join(self.scenario.run.path_input_data,
                                                                 utils.set_extension(self.load_profile)),
                                    scenario=self.scenario)

        if data.shape[1] != 1:
            self.scenario.logger.warning(f'Input file "{utils.set_extension(self.load_profile)}" for parameter '
                                         f'"load_profile" in block "{self.block.name}" has more than one column. '
                                         f'Sum of all columns is calculated for load profile.')

        data = data.sum(axis=1)[self.flows_apriori.index]  # convert to series and slice to sim timeframe
        self.flows_apriori['demand'] = data

    def get_demand_from_slp(self):
        def get_timeframe(date):
            month = date.month
            day = date.day
            if ((month, day) >= (11, 1)) or ((month, day) <= (3, 20)):
                return 'Winter'
            elif (5, 15) <= (month, day) <= (9, 14):
                return 'Summer'
            else:  # Transition months
                return 'Transition'

        def get_daytype(date, holidays):
            if date.date() in holidays or date.weekday() == 6:
                return 'Sunday'
            # Treat Christmas Eve and New Year's Eve as Saturdays if they are not Sundays
            elif (date.weekday() == 5) or ((date.month, date.day) in [(12, 24), (12, 31)]):
                return 'Saturday'
            else:
                return 'Workday'

        # Read BDEW SLP profiles
        slp = pd.read_csv(os.path.join(self.scenario.run.path_data_immut, 'slp_bdew.csv'),
                          skiprows=[0],
                          header=[0, 1, 2],
                          index_col=0)

        slp.index = pd.to_datetime(slp.index, format='%H:%M').time

        # use a fixed frequency of 15 minutes for the timeseries generation as the SLPs are given with that frequency
        freq_slp = '15min'
        dti_slp = pd.DatetimeIndex(pd.date_range(start=self.scenario.starttime.floor(freq_slp),
                                                 end=max(self.scenario.dti_sim_extd).ceil(freq_slp),
                                                 freq=freq_slp))

        data = pd.Series(index=dti_slp, data=0, dtype='float64')

        data = data.index.to_series().apply(
        lambda x: slp.loc[x.time(), (self.load_profile.upper(), get_timeframe(x), get_daytype(x, self.scenario.holiday_dates))])

        # apply dynamic correction for household profiles
        if self.load_profile == 'h0':
            # for private households use dynamic correction as stated in VDEW manual -> round to 1/10 Watt
            num_day = data.index.dayofyear.astype('int64')
            data = round(data * (-3.92e-10 * num_day ** 4 + 3.2e-7 * num_day ** 3 -
                                 7.02e-5 * num_day ** 2 + 2.1e-3 * num_day ** 1 + 1.24),
                         ndigits=1)

        # scale load profile (given for consumption of 1MWh per year) to specified yearly consumption
        # this calculation leads to small deviations from the specified yearly consumption due to varying holidays and
        # leap years, but is the correct way as stated by the VDEW manual
        data *= (self.consumption_yrl / 1e6)

        # resample to simulation time step
        self.flows_apriori['demand'] = data.resample(self.scenario.timestep).mean().ffill().bfill()


class StationaryBattery(Block, StorageBlock):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_names=['in', 'out', 'total'],
                         state_names=['soc', 'soh', 'q_loss_cal', 'q_loss_cyc'],
                         size_names=['block'],
                         poe_names=['block'],
                         params=None,
                         parent=None)

        StorageBlock.__init__(self)

        self.eff_chg = self.eff_acdc if self.system == 'ac' else 1
        self.eff_dis = self.eff_dcac if self.system == 'ac' else 1

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected   name_bus
             |             |
             |<-x-name_xc--|
             |             |<--->name_ess
             |-x-name_ess->|
             |             |

        """

        self.bus_connected = self.scenario.blocks['core'].components[self.system]

        self.components['bus'] = solph.Bus(
            label=f'{self.name}_bus'
        )

        self.components['inflow'] = solph.components.Converter(
            label=f'xc_{self.name}',
            inputs={self.bus_connected: solph.Flow(
                variable_costs=self.poes['block'].opex['spec_ep'][horizon.dti_ph]
            )},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: self.eff_chg}
        )

        self.components['outflow'] = solph.components.Converter(
            label=f'{self.name}_xc',
            inputs={self.components['bus']: solph.Flow()},
            # cost_eps are needed to prevent storage from being emptied in RH
            outputs={self.bus_connected: solph.Flow(
                variable_costs=self.scenario.cost_eps
            )},
            conversion_factors={self.bus_connected: self.eff_dis}
        )

        self.components['storage'] = solph.components.GenericStorage(
            label=f'{self.name}_storage',
            inputs={self.components['bus']: solph.Flow()},
            outputs={
                self.components['bus']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            loss_rate=self.loss_rate,
            balanced={'go': True, 'rh': False}[self.scenario.strategy],
            initial_storage_level=statistics.median(
                [self.soc_min,
                 self.states.loc[horizon.starttime, 'soc'],
                 self.soc_max]
            ),
            invest_relation_input_capacity=self.crate_chg,  # crate measured "outside" of conversion factor (efficiency)
            invest_relation_output_capacity=self.crate_dis,
            inflow_conversion_factor=np.sqrt(self.eff_roundtrip),
            outflow_conversion_factor=np.sqrt(self.eff_roundtrip),
            nominal_storage_capacity=solph.Investment(
                ep_costs=self.poes['block'].opex['spec_ep'],
                existing=self.sizes.loc['block', 'existing'],
                maximum=utils.conv_add_max(self.size.loc['block', 'additional_max'])),
            max_storage_level=pd.Series(
                data=self.soc_max,
                index=utils.extend_dti(horizon.dti_ph)
            ),
            min_storage_level=pd.Series(
                data=self.soc_min,
                index=utils.extend_dti(horizon.dti_ph)
            )
        )

        horizon.constraints.add_invest_costs(
            invest=(self.components['storage'],),
            capex_spec=self.poes['block'].capex['spec'],
            invest_type='storage')


class ControllableSource(Block):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_names=['out', 'total'],
                         state_names=None,
                         size_names=['block'],
                         poe_names=['block'],
                         params=None,
                         parent=None)


class GridConnection(Block):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_names=['in', 'out', 'total'],
                         state_names=None,
                         size_names=['g2s', 's2g'],
                         poe_names=['g2s', 's2g'],
                         params=None,
                         parent=None)


class GridMarket(Block):

    def __init__(self,
                 name: str,
                 scenario,
                 params,
                 parent):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_names=['in', 'out', 'total'],
                         state_names=None,
                         size_names=['g2s', 's2g'],
                         poe_names=['g2s', 's2g'],
                         params=params,
                         parent=parent)

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method

        parent_bus
            |<---x----name_src
            |
            |----x--->name_snk
            |
        """

        self.components['src'] = solph.components.Source(
            label=f'{self.name}_src',
            outputs={self.parent.components['bus']: solph.Flow(
                nominal_value=(self.pwr_g2s if not pd.isna(self.pwr_g2s) else None),
                variable_costs=self.poes['g2s'].opex['spec_ep'][horizon.dti_ph])
            }
        )

        self.components['snk'] = solph.components.Sink(
            label=f'{self.name}_snk',
            inputs={
                self.parent.components['bus']: solph.Flow(
                    nominal_value=(self.pwr_s2g if not pd.isna(self.pwr_s2g) else None),
                    variable_costs=(self.poes['s2g'].opex['spec_ep'][horizon.dti_ph] +
                                    (self.scenario.cost_eps if self.equal_prices else 0)),
                )
            }
        )


class CommoditySystem:

    def __init__(self):
        self.scenario.commodity_systems[self.name] = self


class NonElectricBlock:

    def define_oemof_components(self, *_):
        """
        Dummy to be callable for all blocks
        """
        pass


class ICEVehicle(Block, NonElectricBlock):

    pass







