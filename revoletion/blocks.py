#!/usr/bin/env python3

import ast
import io
import numpy as np
import oemof.solph as solph
import os
import pandas as pd
import plotly.graph_objects as go
import pvlib
import requests
import statistics
import windpowerlib

from revoletion import battery as bat
from revoletion import economics as eco
from revoletion import mobility
from revoletion import simulation as sim
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

        self.poes = {name: eco.PointOfEvaluation(name=name, block=self) for name in poe_names} \
            if poe_names is not None else dict()

        # todo move ls & ccr to poe

        #self.economic_results = eco.EconomicResults(self) #todo
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
        self.loss_rate = utils.convert_sdr(self.sdr, pd.Timedelta(hours=1))  #todo move function to StorageBlock

        self.states.loc[self.scenario.starttime, 'soc'] = self.soc_init
        delattr(self, 'soc_init')

        self.aging_model = bat.BatteryPackModel(self)
        self.soc_min = (1 - self.states.loc[self.scenario.starttime, 'soh']) / 2  # todo move to states df
        self.soc_max = 1 - ((1 - self.states.loc[self.scenario.starttime, 'soh']) / 2)


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
            self.generate_timeseries_from_slp()
        elif self.load_profile in ['const', 'constant']:
            self.flows_apriori['demand'] = self.consumption_yrl / (365 * 24)
        elif isinstance(self.load_profile, str):  # load_profile is a file name
            utils.transform_scalar_var(self, 'load_profile')  # todo change: utils.read_csv & extract series/slice timeframe
            self.data = self.load_profile

        pass

        # todo append to flows_apriori

    def generate_timeseries_from_slp(self):
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
            num_day = self.data.index.dayofyear.astype('int64')
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






