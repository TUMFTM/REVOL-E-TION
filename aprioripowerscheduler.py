import numpy as np
import pandas as pd
import blocks


class AprioriPowerScheduler:
    def __init__(self, scenario):
        self.scenario = scenario

        '''
        # Get name of system core block
        self.sys_core = [name for name, block in self.scenario.blocks.items() if isinstance(block, blocks.SystemCore)][
            0]

        # remaining power available on the system core converter for the current timestep
        self.p_avail_conv = {}

        # mapping dictionary to execute correct function corresponding to input file specification
        self.func_map = {'uc': self.calc_power_uc,
                         'fcfs': self.calc_power_fcfs,
                         'equal': self.calc_power_equal,
                         'soc': self.calc_power_soc}
        '''
        # get list of commodity systems with apriori integration level
        self.apriori_cs = [cs for cs in self.scenario.commodity_systems.values() if cs.int_lvl in cs.apriori_lvls]

        # define dict of elements controlled by rule based algorithm (all commodity systems)
        # ToDo: necessary or split into different int_lvls?
        self.commodities = {name: EsmCommodity(commodity, self.scenario) for block in
                            self.apriori_cs for name, commodity in block.commodities.items()}

        # get different lists of commodity systems according to the restrictions of the apriori integration level
        self.cs_uc = [cs for cs in self.scenario.commodity_systems.values() if cs.int_lvl in 'uc']
        self.cs_apriori_lm = [cs for cs in self.scenario.commodity_systems.values() if
                              cs.int_lvl in [x for x in cs.apriori_lvls if x != 'uc'] and cs.lm_static]
        self.cs_apriori_unlim = [cs for cs in self.scenario.commodity_systems.values() if
                                 cs.int_lvl in [x for x in cs.apriori_lvls if x != 'uc'] and not cs.lm_static]
        '''
        '''
        # define dict of sources in scenario
        self.sources = {name: EsmSourceBlock(block, self.scenario) for name, block in self.scenario.blocks.items() if
                        isinstance(block, (blocks.PVSource,
                                           blocks.WindSource,
                                           blocks.GridConnection,
                                           blocks.ControllableSource))}

        # define dict of StationaryEnergyStorages in scenario
        self.storages = {name: EsmStationaryEnergyStorage(block, self.scenario) for name, block in self.scenario.blocks.items() if
                         isinstance(block, blocks.StationaryEnergyStorage)}

        # define dict of FixedDemands in scenario
        self.demands = {name: EsmSinkBlock(block, self.scenario) for name, block in self.scenario.blocks.items() if
                        isinstance(block, blocks.FixedDemand)}

        # initialize dataframe for available and fixed power for both the AC and the DC bus
        self.p_available = pd.DataFrame(columns=['ac', 'dc'] + [name for name in {**self.sources,
                                                                                  **self.storages}.keys()]
                                        + [f'{name}_chg' for name in self.storages.keys()], dtype=float)
        self.p_fixed = pd.DataFrame(columns=['ac', 'dc'] + [name for name in self.demands.keys()], dtype=float)
        # Placeholder for timeindex of current prediction horizon
        self.dti_ph = None

    def calc_schedule(self, dti_ph):
        # Set timeindex of current prediction horizon
        self.dti_ph = dti_ph

        # Initialize apriori_data dataframes for all blocks (including index, column names, initial SOC value and
        # additional information about the presence and absence of commodities)
        # for block in {**self.commodities, **self.storages, **self.sources}.values():
        #     block.init_ph(self.dti_ph)
        for block in self.commodities.values():
            block.init_ph(self.dti_ph)

        for dtindex in self.dti_ph:
            # region 1. Calculate power for all CommoditySystems with 'uc' or static load management
            for cs in self.cs_uc + self.cs_apriori_lm:
                # Get the available power of the static load management for the CommoditySystem
                p_avail_lm = np.inf if cs.int_lvl in 'uc' else cs.lm_static
                if cs.int_lvl == 'equal':
                    # get list of commodities in CommoditySystem which are ready for charging
                    coms = [self.commodities[key] for key in cs.commodities.keys() if
                            self.commodities[key].block.data_ph.loc[dtindex, 'atbase']]
                    while p_avail_lm > 0 and len(coms) > 0:
                        # calculate power for each commodity
                        p_share = p_avail_lm / len(coms)
                        for commodity in coms:
                            # ToDo: needs to be done in a prettier way
                            p_chg = min(p_share, commodity.calc_p_chg(dtindex, soc_max=1, mode='int_ac') - commodity.get_data(dtindex, 'p_int_ac') / commodity.block.parent.eff_chg)
                            commodity.set_p(dtindex=dtindex, power=p_chg + commodity.get_data(dtindex, 'p_int_ac') / commodity.block.parent.eff_chg, mode='int_ac')
                            p_avail_lm -= p_chg
                            if p_chg == 0:
                                coms.remove(commodity)
                else:
                    # define sorting functions for the different strategies
                    sort_key_funcs = {
                        'uc': lambda x: x.block.name,  # sorting makes no difference -> dummy function´for uc
                        'fcfs': lambda x: x.get_arr(dtindex),
                        'soc': lambda x: x.get_data(dtindex, 'soc')
                    }
                    # get a list of all available commodities and sort them according to the chosen strategy
                    coms = sorted([self.commodities[key] for key in cs.commodities.keys() if
                                   self.commodities[key].block.data_ph.loc[dtindex, 'atbase']],
                                  key=sort_key_funcs[cs.int_lvl])
                    for commodity in coms:
                        # get maximum charging power based on vehicle and static load management
                        p_chg = min(p_avail_lm, commodity.calc_p_chg(dtindex, soc_max=1, mode='int_ac'))
                        # set charging power
                        commodity.set_p(dtindex=dtindex, power=p_chg, mode='int_ac')
                        # update available power of static load management
                        p_avail_lm -= p_chg
            # update SOCs of all commodities within the CommoditySystem
            for commodity in [com.name for cs in self.cs_uc + self.cs_apriori_lm for com in cs.commodities.values()]:
                self.commodities[commodity].calc_new_soc(dtindex=dtindex)
            # endregion

class EsmBlock:
    def __init__(self, block, scenario):
        self.block = block
        self.scenario = scenario
        # get the system to which the block is connected
        self.system = None
        if isinstance(self.block, (blocks.PVSource, blocks.StationaryEnergyStorage)):
            self.system = 'dc'
        elif isinstance(self.block, (blocks.WindSource, blocks.ControllableSource, blocks.GridConnection, blocks.FixedDemand)):
            self.system = 'ac'
        elif hasattr(self.block, 'parent'):
            if isinstance(self.block.parent, (blocks.VehicleCommoditySystem, blocks.BatteryCommoditySystem)):
                self.system = self.block.parent.system

    def init_ph(self, dti_ph, cols):
        # Initialize apriori_data at start of every new prediction horizon
        self.block.apriori_data = pd.DataFrame(0, index=dti_ph, columns=cols, dtype=float)

    def set_data(self, dtindex, value, col):
        # Set power value of apriori_data of block for given timestamp
        if dtindex == ':':
            self.block.apriori_data.loc[:, col] = value
        else:
            self.block.apriori_data.loc[dtindex, col] = value

    def get_data(self, dtindex, col):
        # Function the get apriori data of a block for a given timestamp
        if dtindex == ':':
            return self.block.apriori_data.loc[:, col]
        else:
            return self.block.apriori_data.loc[dtindex, col]


class EsmSourceBlock(EsmBlock):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)
        # Flag for renewable energy sources (res)
        self.res = False
        if isinstance(self.block, (blocks.PVSource, blocks.WindSource)):
            self.res = True
        # Set the opex_spec of the block (named differently for grid connection)
        if isinstance(self.block, blocks.GridConnection):
            self.opex_spec = getattr(self.block, 'opex_spec_g2mg', 0)
        else:
            self.opex_spec = getattr(self.block, 'opex_spec', 0)
        # ToDo: this should not be the case as all blocks in REVOL-E-TION should have opex as timeseries
        if not isinstance(self.opex_spec, pd.Series):
            self.opex_spec = pd.Series(self.opex_spec, index=self.scenario.dti_sim)

    def init_ph(self, dti_ph):
        super().init_ph(dti_ph, ['p'])

    def set_p(self, dtindex, power):
        super().set_data(dtindex=dtindex, value=power, col=['p'])

    def calc_p(self):
        if isinstance(self.block, blocks.WindSource):
            return self.block.data_ph['power_spec'] * self.block.size * self.block.eff
        elif isinstance(self.block, blocks.PVSource):
            return self.block.data_ph['power_spec'] * self.block.size * self.block.eff
        elif isinstance(self.block, blocks.GridConnection):
            return self.block.size_g2mg * self.block.eff
        else:  # if isinstance(self.block, blocks.ControllableSource): -> ToDo: delete condition, is used as default
            return self.block.size * self.block.eff


class EsmSinkBlock(EsmBlock):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)

    def init_ph(self, dti_ph):
        # No apriori data for sinks
        raise Warning('Function init_ph() is not implemented for EsmSinkBlock as there is no apriori data for sinks')
        return

    def set_data(self, dtindex, value, col):
        # No apriori data for sinks
        raise Warning('Function set_data() is not implemented for EsmSinkBlock as there is no apriori data for sinks')
        return

    def get_data(self, dtindex, col):
        # No apriori data for sinks
        raise Warning('Function get_data() is not implemented for EsmSinkBlock as there is no apriori data for sinks')
        return

    def calc_p(self):
        return self.block.data_ph['power_w'] * -1


class EsmStorage(EsmBlock):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)
        self.loss_rate = None

    def init_ph(self, dti_ph, columns):
        super().init_ph(dti_ph, columns)
        # Set initial SOC value of storage
        self.set_data(dti_ph[0], self.block.soc_init_ph, 'soc')

    def set_p(self, dtindex, power, col):
        super().set_data(dtindex=dtindex, value=power, col=col)

    def calc_p_chg(self, dtindex, p_maxchg, eff=1, soc_max=1, soc_threshold=0):
        # p_maxchg: maximum charging power in W, measured at storage, NOT at bus

        # Only charge if SOC falls below threshold (soc_max - soc_threshold)
        if (soc_current := self.get_data(dtindex, 'soc')) >= soc_max - soc_threshold:
            return 0

        # STORAGE: power to be charged to target SOC in Wh in one timestep using SOC delta (clip soc_target to 1)
        p_tosoc = (soc_max - soc_current * (1 - self.loss_rate)) * self.block.size / self.scenario.timestep_hours
        if p_tosoc < 0:
            print('charging power below 0')

        # BUS: charging power measured at connection to DC bus; reduce power in final step to just reach target SOC
        p_chg = min(p_maxchg, p_tosoc) / eff
        return p_chg

    def calc_p_dis(self, dtindex, p_maxdis, eff=1, soc_min=0):
        # STORAGE: power to be discharged to target SOC in Wh in one timestep using SOC delta (clip soc_target to 0)
        p_tosoc = max(0, (self.get_data(dtindex, 'soc') - max(soc_min, 0)) * self.block.size) / \
                  self.scenario.timestep_hours

        # BUS: discharging power measured at connection to DC bus; reduce power in final step to just reach target SOC
        p_dis = min(p_maxdis, p_tosoc) * eff
        return p_dis

    def calc_new_soc(self, dtindex, power):
        # calculate new soc value
        new_soc = power / self.block.size * self.scenario.timestep_hours + self.get_data(dtindex, 'soc') * (1 - self.loss_rate)
        # Assign new SOC to apriori_data DataFrame of block
        self.set_data(dtindex=dtindex + self.scenario.timestep_td,
                      value=new_soc,
                      col='soc')


class EsmStationaryEnergyStorage(EsmStorage):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)

        # define loss_rate
        self.loss_rate = 1 - (1 - self.block.loss_rate) ** (self.scenario.timestep_td / pd.Timedelta('1h'))

        self.opex_spec = getattr(self.block, 'opex_spec', 0)
        if not isinstance(self.opex_spec, pd.Series):
            self.opex_spec = pd.Series(self.opex_spec, index=self.scenario.dti_sim)

    def init_ph(self, dti_ph, *_):
        columns = ['p', 'soc']
        super().init_ph(dti_ph, columns)

    def set_p(self, dtindex, power, *_):
        super().set_p(dtindex=dtindex,
                      power=power,
                      col='p')

    def calc_p_chg(self, dtindex, soc_max=1, *_):
        return super().calc_p_chg(dtindex=dtindex,
                                  p_maxchg=self.block.size * self.block.crate_chg,
                                  eff=self.block.eff_chg,
                                  soc_max=soc_max)

    def calc_p_dis(self, dtindex, soc_min=0, *_):
        return super().calc_p_dis(dtindex=dtindex,
                                  p_maxdis=self.block.size * self.block.crate_dis,
                                  eff=self.block.eff_dis,
                                  soc_min=soc_min)

    def calc_new_soc(self, dtindex, *_):
        # convert power at connection to DC bus to power at storage:
        power = self.get_data(dtindex, ['p']).sum() * (-1)
        if power >= 0:
            power *= self.block.eff_chg
        else:
            power /= self.block.eff_dis

        super().calc_new_soc(dtindex=dtindex, power=power)


class EsmCommodity(EsmStorage):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)

        # define priorities for charging power from buses of SystemCore
        self.system_map = {'ac': {'prio': 'ac',
                                  'non_prio': 'dc'},
                           'dc': {'prio': 'p_dc',
                                  'non_prio': 'p_ac'}
                           }[self.block.parent.system]

        # Placeholder for variables for power calculations at external charging
        self.minsoc_inz = self.dep_inz = self.arr_inz = self.arr_parking_inz = self.chg_inz = None
        self.parking_charging = None

    def init_ph(self, dti_ph, *_):
        columns = ['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption', 'soc']
        super().init_ph(dti_ph, columns)
        self.set_data(dtindex=':', value=-1 * self.block.data_ph['consumption'], col='p_consumption')

        # define loss_rate
        self.loss_rate = 1 - (1 - self.block.parent.loss_rate) ** (self.scenario.timestep_td / pd.Timedelta('1h'))

        # get the indices of all nonzero minsoc rows in the data
        self.minsoc_inz = self.block.data_ph.index[self.block.data_ph['minsoc'] != 0]

        # get first timesteps, where vehicle has left the base
        self.dep_inz = self.block.data_ph.index[
            self.block.data_ph['atbase'] & ~self.block.data_ph['atbase'].shift(-1, fill_value=False)]

        # get first timesteps, where vehicle is at base again
        # based on data instead of data_ph to include info from previous prediction horizon
        # ToDo: switch to data_ph again and include the data from the previous prediction horizon
        #  [1:] only necessary, if vehicle is atbase at beginning of prediction horizon
        self.arr_inz = self.block.data.index[
                           self.block.data['atbase'] & ~self.block.data['atbase'].shift(fill_value=False)][1:]
        # ToDo: check why this line was added
        # self.arr_inz = self.arr_inz[self.arr_inz <= self.block.data_ph.index[0]]

        # get first timesteps, where vehicle is parking at destination
        self.arr_parking_inz = self.block.data_ph.index[
            self.block.data_ph['atac'] & ~self.block.data_ph['atac'].shift(fill_value=False)]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        self.chg_inz = self.block.data_ph.index[self.block.data_ph[['atbase', 'atac', 'atdc']].any(axis=1)]

        # initialize variable for charging during single parking process
        self.parking_charging = False

    def get_arr(self, dtindex):
        # get latest arrival before current timestep
        try:
            return self.arr_inz[self.arr_inz <= dtindex][-1]
        except:
            # return 1900 if no arrival is found -> at start of new simulation
            return pd.to_datetime('1900').replace(tzinfo=self.scenario.timezone)

    def set_p(self, dtindex, power, mode='int_ac'):
        super().set_p(dtindex=dtindex,
                      power={'int_ac': self.block.parent.eff_chg,
                             'ext_ac': 1,
                             'ext_dc': 1}[mode] * power,
                      col=f'p_{mode}')

    def calc_p_chg(self, dtindex, soc_max=1, mode='int_ac', *_):
        return super().calc_p_chg(dtindex=dtindex,
                                  p_maxchg={'int_ac': self.block.pwr_chg * self.block.parent.eff_chg,
                                            'ext_ac': self.block.parent.pwr_ext_ac,
                                            'ext_dc': self.block.parent.pwr_ext_dc}[mode],
                                  eff={'int_ac': self.block.parent.eff_chg,
                                       'ext_ac': 1,
                                       'ext_dc': 1}[mode],
                                  soc_max=soc_max)

    def calc_p_dis(self, dtindex, soc_min=0, mode='int_ac', *_):
        return 0

    def calc_new_soc(self, dtindex, *_):
        columns = ['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption']
        power = self.get_data(dtindex, columns).sum()

        super().calc_new_soc(dtindex=dtindex, power=power)

