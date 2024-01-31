import numpy as np
import pandas as pd
import blocks


class AprioriPowerScheduler:
    def __init__(self, scenario):
        self.scenario = scenario

        # Get name of system core block
        self.sys_core = [name for name, block in self.scenario.blocks.items() if isinstance(block, blocks.SystemCore)][
            0]

        self.p_avail_conv = {}

        # mapping dictionary to execute correct function corresponding to input file specification
        self.func_map = {'uc': self.calc_power_uc,
                         'fcfs': self.calc_power_fcfs,
                         'equal': self.calc_power_equal,
                         'soc': self.calc_power_soc}

        # get list of commodity systems with apriori integration level
        self.apriori_cs = [cs for cs in self.scenario.commodity_systems.values() if cs.int_lvl in cs.apriori_lvls]

        # define dict of elements controlled by rule based algorithm (all commodity systems)
        self.commodities = {name: Commodity(commodity, self.scenario) for block in
                            self.apriori_cs for name, commodity in block.commodities.items()}

        # define dict of sources in scenario
        self.sources = {name: SourceBlock(block, self.scenario) for name, block in self.scenario.blocks.items() if
                        isinstance(block, (blocks.PVSource,
                                           blocks.WindSource,
                                           blocks.GridConnection,
                                           blocks.ControllableSource))}

        # define dict of StationaryEnergyStorages in scenario
        self.storages = {name: Storage(block, self.scenario) for name, block in self.scenario.blocks.items() if
                         isinstance(block, blocks.StationaryEnergyStorage)}

        # define dict of FixedDemands in scenario
        self.demands = {name: SinkBlock(block, self.scenario) for name, block in self.scenario.blocks.items() if
                        isinstance(block, blocks.FixedDemand)}

        # initialize dataframe for available and fixed power for both the AC and the DC bus
        self.p_available = pd.DataFrame(columns=['ac', 'dc'] + [name for name in {**self.sources,
                                                                                  **self.storages}.keys()]
                                        + [f'{name}_chg' for name in self.storages.keys()])
        self.p_fixed = pd.DataFrame(columns=['ac', 'dc'] + [name for name in self.demands.keys()])

        # Placeholder for timeindex of current prediction horizon
        self.ph_dti = None

    @staticmethod
    def get_bus(bus, which):
        # Takes the bus the block is connected to and returns the specified bus
        return {'ac': {'same': 'ac', 'other': 'dc'},
                'dc': {'same': 'dc', 'other': 'ac'}}[bus][which]

    def get_conv_eff(self, source, target):
        return {'ac': {'ac': 1,
                       'dc': self.scenario.blocks[self.sys_core].acdc_eff},
                'dc': {'ac': self.scenario.blocks[self.sys_core].dcac_eff,
                       'dc': 1}}[source][target]

    def get_conv_cap(self, source, target):
        if source == target:
            return np.inf
        else:
            return self.p_avail_conv[source]

    def p_avail_conv_reset(self):
        # store how much power still can be converted on the system core within current timestep
        # key specifies origin of power
        self.p_avail_conv = {'ac': self.scenario.blocks[self.sys_core].acdc_size,
                             'dc': self.scenario.blocks[self.sys_core].dcac_size}

    def calc_schedule(self, ph_dti):
        # Set timeindex of current prediction horizon
        self.ph_dti = ph_dti

        # Update objects with data of Prediction Horizon
        for block in {**self.commodities, **self.storages, **self.sources}.values():
            block.update_ph(self.ph_dti)

        # Calculate available and fixed power for prediction horizon
        self.p_available = self.calc_power_ph(power=self.p_available, items=self.sources)
        self.p_fixed = self.calc_power_ph(power=self.p_fixed, items=self.demands)

        # Calculate powers in energy system for every timestep in Prediction Horizon
        for dtindex in self.ph_dti:
            # =============================================
            # === 1. Preparing data of current timestep ===
            # =============================================

            # 1.(a) Calculate available power from stationary energy storage and add to available dc power
            for name, storage in self.storages.items():
                self.p_available.loc[dtindex, name] = storage.calc_p_dis(dtindex)
                self.p_available.loc[dtindex, 'dc'] += self.p_available.loc[dtindex, name]

            # 1.(b) Store how much power is available on the buses within the current timestep
            p_avail_sys = {'ac': self.p_available.loc[dtindex, 'ac'],
                           'dc': self.p_available.loc[dtindex, 'dc']}

            # ======================================================
            # === 2. Schedule demand and charging of commodities ===
            # ======================================================

            # 2.(a) reset available power on converter to maximum power
            self.p_avail_conv_reset()

            # 2.(b) Subtract demand from available power on AC bus (demand is specified as a negative power)
            for system in ['ac', 'dc']:
                p_avail_sys = self.draw_power(bus_connected=system,
                                              pwr=(-1) * self.p_fixed.loc[dtindex, system],
                                              p_avail_sys=p_avail_sys)

            # 2.(c) Schedule at base charging of commodities
            for int_lvl in ['uc', 'fcfs', 'equal', 'soc']:
                int_coms = {name: com for name, com in self.commodities.items() if
                            com.block.parent.int_lvl == int_lvl}
                p_avail_sys = self.func_map[int_lvl](dtindex=dtindex,
                                                     p_avail_sys=p_avail_sys,
                                                     int_lvl_commodities=int_coms)

            # ===============================================================
            # === 3. Schedule sources to cover power demand in local grid ===
            # ===============================================================

            # 3.(a) reset available power on converter to maximum power
            self.p_avail_conv_reset()

            # 3.(b) Calculate power of sources and storages to cover demand of both buses
            self.calc_local_grid(dtindex, p_avail_sys)

            # ====================================================
            # === 4. Schedule external charging of commodities ===
            # ====================================================

            for commodity in self.commodities.values():
                commodity.ext_charging(dtindex)

            # =============================================================
            # === 5. Calculate new SOC for all commodities and storages ===
            # =============================================================

            for block in {**self.commodities, **self.storages}.values():
                block.calc_new_soc(dtindex)

        # Apriori data for sources is not set. Reset apriori data of sources to None
        for name, block in self.sources.items():
            block.block.apriori_data = None

    def calc_power_ph(self, power, items):
        # reset power to zero for all timestamps including deleting data from previous prediction horizon
        power = power.reindex(self.ph_dti).fillna(0)

        # compute power resulting from different blocks at the AC and DC bus
        for name, block in items.items():
            power.loc[:, name] += block.calc_p()
            power.loc[:, block.system] += power.loc[:, name]

        return power

    def calc_power_uc(self, dtindex, p_avail_sys, int_lvl_commodities):
        for com in int_lvl_commodities.values():
            if com.block.ph_data.loc[dtindex, 'atbase']:
                # Execute AC charging at base
                pwr_chg = com.calc_p_chg(dtindex=dtindex, soc_max=1, mode='int_ac')
                com.set_p(dtindex=dtindex,
                          power=pwr_chg,
                          mode='int_ac')

                p_avail_sys = self.draw_power(bus_connected=com.system,
                                              pwr=pwr_chg,
                                              p_avail_sys=p_avail_sys)

        return p_avail_sys

    def calc_power_fcfs(self, dtindex, p_avail_sys, int_lvl_commodities):
        # Get list of names and SOCs of all commodities
        prio_list = [(name, com.get_data(dtindex, 'soc')) for name, com in int_lvl_commodities.items()]

        # ToDO: Sort list of commodities based on last arrival timestamp
        prio_list.sort(key=lambda x: x)

        # Calculate charging powers based on calculated priorities
        p_avail_sys = self.prio_charging(dtindex, prio_list, p_avail_sys)

        return p_avail_sys

    def calc_power_equal(self, dtindex, p_avail_sys, int_lvl_commodities):
        return p_avail_sys

    def calc_power_soc(self, dtindex, p_avail_sys, int_lvl_commodities):
        # Get list of names and SOCs of all commodities
        prio_list = [(name, com.get_data(dtindex, 'soc')) for name, com in int_lvl_commodities.items() if
                     com.block.ph_data.loc[dtindex, 'atbase']]

        # Sort list of commodities based on SOC
        prio_list.sort(key=lambda x: x[1])

        # Calculate charging powers based on calculated priorities
        p_avail_sys = self.prio_charging(dtindex, prio_list, p_avail_sys)
        return p_avail_sys

    def prio_charging(self, dtindex, prio_list, p_avail_sys):
        # loop through commodities starting with highest prio and compute charging power
        for com_name, com_soc in prio_list:
            # get preferred bus for charging
            bus_prio = self.get_bus(self.commodities[com_name].system, 'same')
            bus_non_prio = self.get_bus(self.commodities[com_name].system, 'other')

            # ToDo: implement input dependent target soc (e.g. 80%)
            # ToDo: if available power = 0 (then it should be 0 for all following commodities, too)
            # power for commodity considering all limitations (max charging power, available power, converter)
            pwr_chg = min(self.commodities[com_name].calc_p_chg(dtindex=dtindex, soc_max=1, mode='int_ac'),
                          p_avail_sys[bus_prio] + min(p_avail_sys[bus_non_prio],
                                                      self.get_conv_cap(bus_non_prio, bus_prio)) * self.get_conv_eff(
                              bus_non_prio, bus_prio))

            p_avail_sys = self.draw_power(bus_connected=bus_prio,
                                          pwr=pwr_chg,
                                          p_avail_sys=p_avail_sys)

            # assign charging power to commodity
            self.commodities[com_name].set_p(dtindex=dtindex, power=pwr_chg, mode='int_ac')

        return p_avail_sys

    def draw_power(self, bus_connected, pwr, p_avail_sys):
        # deduct power from available power on the corresponding bus(es) and the converter
        pwr_prio = min(p_avail_sys[bus_connected], pwr)
        p_avail_sys[bus_connected] -= pwr_prio

        pwr_non_prio = (pwr - pwr_prio) / self.get_conv_eff(self.get_bus(bus_connected, 'other'), bus_connected)
        p_avail_sys[self.get_bus(bus_connected, 'other')] -= pwr_non_prio
        self.p_avail_conv[self.get_bus(bus_connected, 'other')] -= pwr_non_prio

        # ToDo: check whether is really necessary
        if 0 > p_avail_sys[self.get_bus(bus_connected, 'other')] > -1E-10:
            p_avail_sys[self.get_bus(bus_connected, 'other')] = 0

        if 0 > p_avail_sys[bus_connected] > -1E-10:
            p_avail_sys[bus_connected] = 0

        if p_avail_sys['ac'] < 0 or p_avail_sys['dc'] < 0:
            print('\n\nError! Not enough power available on buses!')

        if self.p_avail_conv['ac'] < 0 or self.p_avail_conv['dc'] < 0:
            print('\n\nError! Not enough power available on SystemCore converter!')

        return p_avail_sys

    def calc_local_grid(self, dtindex, p_avail_sys):
        # Concept:
        # Step 1: Use power with negative costs for both buses (sometimes this happens for grid power)
        # Step 2: Use all power free of charge (mainly renewable sources) to cover demand of AC and DC bus
        # Step 3: Assign excess power of sources free of charge to storages
        # Step 4: Use storage power to cover demand of AC and DC bus
        # Step 5: All other options are scheduled by the optimizer (including negative grid costs)

        # Create DataFrame with available power of all components of the local grid and the used power of both buses
        p_system = self.p_available.loc[dtindex, :]
        p_system[['ac', 'dc']] -= pd.Series(p_avail_sys)

        #######################################
        ### Step 1: Use negative cost power ###
        #######################################

        sources_sorted = sorted(self.sources.items(), key=lambda item: item[1].opex_spec[dtindex])

        # Use power with negative costs (sometimes grid power) to cover demand of AC and DC bus
        for name, block in sources_sorted:
            if getattr(block, "opex_spec", 0)[dtindex] < 0:
                # system here target system
                for demand_system in ['ac', 'dc']:
                    # Power limitations: demand, offer, conversion between buses
                    used = min(p_system[demand_system] / self.get_conv_eff(block.system, demand_system),
                               p_system[name],
                               self.get_conv_cap(block.system, demand_system))
                    # Subtract usage from demand, offer of current source and converter capacity
                    p_system[demand_system] -= used * self.get_conv_eff(block.system, demand_system)
                    p_system[name] -= used
                    if demand_system != block.system:  # only reduce available converter power if converter is used
                        self.p_avail_conv[block.system] -= used

        ########################################
        ### Step 2: Use free of charge power ###
        ########################################

        # Use all power free of charge (mainly renewable sources) to cover demand of AC and DC bus
        # First, try to cover demand using sources at own bus, then try to cover demand using sources at other bus
        for bus_type in ['same', 'other']:
            for system in ['ac', 'dc']:  # system describes system the source is connected to
                for name, block in sources_sorted:
                    if block.system == system and getattr(block, "opex_spec", 0)[dtindex] == 0:
                        # Power limitations: demand, offer, conversion between buses
                        used = min(p_system[self.get_bus(system, bus_type)] / self.get_conv_eff(system, self.get_bus(system, bus_type)),
                                   p_system[name],
                                   self.get_conv_cap(system, self.get_bus(system, bus_type)))
                        # Subtract usage from demand, offer of current source and converter capacity
                        p_system[self.get_bus(system, bus_type)] -= used * self.get_conv_eff(system, self.get_bus(system, bus_type))
                        p_system[name] -= used
                        if bus_type == 'other':  # only reduce available converter power if converter is used
                            self.p_avail_conv[system] -= used

        ##############################################
        ### Step 3: Assign excess power to storage ###
        ##############################################

        # Assign excess power of sources free of charge to storages
        # Start with dc (as this is the same bus as the storages)
        for storage_name, storage_block in self.storages.items():
            p_system[f'{storage_name}_chg'] = storage_block.calc_p_chg(dtindex=dtindex, soc_max=1)

        storages_sorted = sorted(self.storages.items(), key=lambda item: item[1].opex_spec[dtindex])
        for system in ['dc', 'ac']:
            for storage_name, storage_block in storages_sorted:
                for name, block in sources_sorted:
                    if block.system == system and getattr(block, "opex_spec", 0)[dtindex] <= 0:
                        # Power limitations: demand, offer, conversion between buses
                        used = min(p_system[f'{storage_name}_chg'] / self.get_conv_eff(block.system, 'dc'),
                                   p_system[name],
                                   self.get_conv_cap(block.system, 'dc'))
                        p_system[storage_name] += used * self.get_conv_eff(block.system, 'dc')
                        p_system[name] -= used
                        p_system[f'{storage_name}_chg'] -= used * self.get_conv_eff(block.system, 'dc')
                        if system == 'ac':  # only reduce available converter power if converter is used
                            self.p_avail_conv[system] -= used

        ###########################################
        ### Step 4: Cover demand using storages ###
        ###########################################

        for demand_system in ['dc', 'ac']:
            for name, block in storages_sorted:
                # Power limitations: demand, offer, conversion between buses
                used = min(p_system[demand_system] / self.get_conv_eff(block.system, demand_system),
                           p_system[name],
                           self.get_conv_cap(block.system, demand_system))
                # Subtract usage from demand, offer of current source and converter capacity
                p_system[demand_system] -= used * self.get_conv_eff(block.system, demand_system)
                p_system[name] -= used
                if demand_system != block.system:  # only reduce available converter power if converter is used
                    self.p_avail_conv[block.system] -= used

        # apply power to apriori data of sources
        for name, block in {**self.sources, **self.storages}.items():
            # >0: power into grid, <0 power out of grid; measured at connection to corresponding bus
            block.set_p(dtindex=dtindex, power=self.p_available.loc[dtindex, name] - p_system[name])


class EnergySystemModelBlock:
    def __init__(self, block, scenario):
        self.block = block
        self.scenario = scenario
        self.system = None
        if isinstance(self.block, (blocks.PVSource, blocks.StationaryEnergyStorage)):
            self.system = 'dc'
        elif isinstance(self.block, (blocks.WindSource, blocks.ControllableSource, blocks.GridConnection, blocks.FixedDemand)):
            self.system = 'ac'
        elif hasattr(self.block, 'parent'):
            if isinstance(self.block.parent, (blocks.VehicleCommoditySystem, blocks.BatteryCommoditySystem)):
                self.system = self.block.parent.system

    def update_ph(self, ph_dti, cols):
        # Initialize apriori_data at start of every new prediction horizon
        self.block.apriori_data = pd.DataFrame(0, index=ph_dti, columns=cols)

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


class SourceBlock(EnergySystemModelBlock):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)
        # Flag for renewable energy sources (res)
        self.res = False
        if isinstance(self.block, (blocks.PVSource, blocks.WindSource)):
            self.res = True
        if isinstance(self.block, blocks.GridConnection):
            self.opex_spec = getattr(self.block, 'opex_spec_g2mg', 0)
        else:
            self.opex_spec = getattr(self.block, 'opex_spec', 0)
        if not isinstance(self.opex_spec, pd.Series):
            self.opex_spec = pd.Series(self.opex_spec, index=self.scenario.sim_dti)

    def update_ph(self, ph_dti):
        super().update_ph(ph_dti, ['p'])

    def set_p(self, dtindex, power):
        super().set_data(dtindex=dtindex,
                         value=power,
                         col=['p'])

    def calc_p(self):
        if isinstance(self.block, blocks.WindSource):
            return self.block.ph_data['P'] * self.block.size * self.block.eff
        elif isinstance(self.block, blocks.PVSource):
            return self.block.ph_data['p_spec'] * self.block.size * self.block.eff
        elif isinstance(self.block, blocks.ControllableSource):
            return self.block.size * self.block.eff
        elif isinstance(self.block, blocks.GridConnection):
            return self.block.size * self.block.eff


class SinkBlock(EnergySystemModelBlock):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)

    def update_ph(self, ph_dti):
        # No apriori data for sinks
        return

    def set_data(self, dtindex, value, col):
        # No apriori data for sinks
        return

    def get_data(self, dtindex, col):
        # No apriori data for sinks
        return

    def calc_p(self):
        return self.block.ph_data['power'] * -1


class StorageBlock(EnergySystemModelBlock):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)

    def update_ph(self, ph_dti, columns):
        super().update_ph(ph_dti, columns)
        self.set_data(ph_dti[0], self.block.ph_init_soc, 'soc')

    def set_p(self, dtindex, power, col):
        super().set_data(dtindex=dtindex,
                         value=power,
                         col=col)

    def calc_p_chg(self, dtindex, p_maxchg, eff=1, soc_max=1):
        # p_maxchg: maximum charging power in W, measured at storage, NOT at bus

        # STORAGE: power to be charged to target SOC in Wh in one timestep using SOC delta (clip soc_target to 1)
        p_tosoc = max(0, (min(soc_max, 1) - self.get_data(dtindex, 'soc')) * self.block.size) / \
                  self.scenario.timestep_hours

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
        # Assign new SOC to apriori_data DataFrame of block
        self.set_data(dtindex=dtindex + self.scenario.timestep_td,
                      value=power / self.block.size * self.scenario.timestep_hours + self.get_data(dtindex, 'soc'),
                      col='soc')


class Storage(StorageBlock):
    def __init__(self, block, scenario):
        super().__init__(block, scenario)
        self.opex_spec = getattr(self.block, 'opex_spec', 0)
        if not isinstance(self.opex_spec, pd.Series):
            self.opex_spec = pd.Series(self.opex_spec, index=self.scenario.sim_dti)

    def update_ph(self, ph_dti, *_):
        columns = ['p', 'soc']
        super().update_ph(ph_dti, columns)

    def set_p(self, dtindex, power, *_):
        super().set_p(dtindex=dtindex,
                      power=power,
                      col='p')

    def calc_p_chg(self, dtindex, soc_max=1, *_):
        return super().calc_p_chg(dtindex=dtindex,
                                  p_maxchg=self.block.size * self.block.chg_crate,
                                  eff=self.block.chg_eff,
                                  soc_max=soc_max)

    def calc_p_dis(self, dtindex, soc_min=0, *_):
        return super().calc_p_dis(dtindex=dtindex,
                                  p_maxdis=self.block.size * self.block.dis_crate,
                                  eff=self.block.dis_eff,
                                  soc_min=soc_min)

    def calc_new_soc(self, dtindex, *_):
        # convert power at connection to DC bus to power at storage:
        power = self.get_data(dtindex, ['p']).sum() * (-1)
        if power >= 0:
            power *= self.block.chg_eff
        else:
            power /= self.block.dis_eff

        super().calc_new_soc(dtindex=dtindex, power=power)


class Commodity(StorageBlock):
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

    def update_ph(self, ph_dti, *_):
        columns = ['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption', 'soc']
        super().update_ph(ph_dti, columns)
        self.set_data(dtindex=':', value=-1 * self.block.ph_data['consumption'], col='p_consumption')

        # get the indices of all nonzero minsoc rows in the data
        self.minsoc_inz = self.block.ph_data.index[self.block.ph_data['minsoc'] != 0]

        # get first timesteps, where vehicle has left the base
        self.dep_inz = self.block.ph_data.index[
            self.block.ph_data['atbase'] & ~self.block.ph_data['atbase'].shift(-1, fill_value=False)]

        # get first timesteps, where vehicle is at base again
        self.arr_inz = self.block.ph_data.index[
                           self.block.ph_data['atbase'] & ~self.block.ph_data['atbase'].shift(fill_value=False)][1:]

        # get first timesteps, where vehicle is parking at destination
        self.arr_parking_inz = self.block.ph_data.index[
            self.block.ph_data['atac'] & ~self.block.ph_data['atac'].shift(fill_value=False)]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        self.chg_inz = self.block.ph_data.index[self.block.ph_data[['atbase', 'atac', 'atdc']].any(axis=1)]

        # initialize variable for charging during single parking process
        self.parking_charging = False

    def set_p(self, dtindex, power, mode='int_ac'):
        super().set_p(dtindex=dtindex,
                      power={'int_ac': self.block.parent.chg_eff,
                             'ext_ac': 1,
                             'ext_dc': 1}[mode] * power,
                      col=f'p_{mode}')

    def calc_p_chg(self, dtindex, soc_max=1, mode='int_ac', *_):
        return super().calc_p_chg(dtindex=dtindex,
                                  p_maxchg={'int_ac': self.block.chg_pwr * self.block.parent.chg_eff,
                                            'ext_ac': self.block.parent.ext_ac_power,
                                            'ext_dc': self.block.parent.ext_dc_power}[mode],
                                  eff={'int_ac': self.block.parent.chg_eff,
                                       'ext_ac': 1,
                                       'ext_dc': 1}[mode],
                                  soc_max=soc_max)

    def calc_p_dis(self, dtindex, soc_min=0, mode='int_ac', *_):
        return 0

    def calc_new_soc(self, dtindex, *_):
        columns = ['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption']
        power = self.get_data(dtindex, columns).sum()

        super().calc_new_soc(dtindex=dtindex, power=power)

    def ext_charging(self, dtindex):
        if self.block.ph_data.loc[dtindex, 'atac'] == 1:  # parking at destination
            if dtindex in self.arr_parking_inz:  # plugging in only happens when parking starts
                # use current int-index and next arrival index to calculate consumption and convert to SOC
                try:  # Fails, if current trip is last trip and doesn't end within prediction horizon
                    arr_inxt = self.arr_inz[self.arr_inz >= dtindex][0]
                except:
                    arr_inxt = self.block.ph_data.index[-1]
                consumption_remaining = self.block.ph_data.loc[dtindex:arr_inxt,
                                        'consumption'].sum() * self.scenario.timestep_hours
                # set charging to True, if charging is necessary
                if consumption_remaining > (
                        (self.get_data(dtindex, 'soc') + self.block.ph_data.loc[
                            arr_inxt, 'minsoc']) * self.block.size):
                    self.parking_charging = True
                else:
                    self.parking_charging = False

            if self.parking_charging is True:
                # ToDo: implement input dependent target soc (e.g. 80%)
                self.set_p(dtindex=dtindex,
                            power=self.calc_p_chg(dtindex=dtindex, soc_max=1, mode='ext_ac'),
                            mode='ext_ac')

        elif self.block.ph_data.loc[dtindex, 'atdc'] == 1:  # vehicle is driving with possibility to charge on-route
            # activate charging, if SOC will fall below threshold, before next possibility to charge
            chg_inxt = self.chg_inz[self.chg_inz > dtindex][0]
            chg_soc = self.get_data(dtindex, 'soc') - self.block.ph_data.loc[dtindex:chg_inxt,
                                                      'consumption'].sum() * self.scenario.timestep_hours / self.block.size
            if chg_soc < 0.05:
                # fast-charging only up to SOC of 80 %
                self.set_p(dtindex=dtindex,
                            power=self.calc_p_chg(dtindex=dtindex, soc_max=0.8, mode='ext_dc'),
                            mode='ext_dc')
