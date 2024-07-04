import numpy as np
import pandas as pd
import statistics
import blocks


def get_bus(bus, which):
    # Takes the bus the block is connected to and returns the specified bus
    return {'ac': {'same': 'ac', 'other': 'dc'},
            'dc': {'same': 'dc', 'other': 'ac'}}[bus][which]


def get_block_system(block):
    # get the label of the bus a block is connected to
    return block.bus_connected.label[0:2]


class AprioriPowerScheduler:
    def __init__(self, run, scenario):
        self.run = run
        self.scenario = scenario

        # Get name of system core block
        self.sys_core = [block for block in self.scenario.blocks.values() if isinstance(block, blocks.SystemCore)][0]

        # Placeholder for timeindex of current prediction horizon
        self.dti_ph = None

        # remaining power available on the system core converter for the current timestep
        self.p_avail_conv = {}

        # get different lists of commodity systems according to the restrictions of the apriori integration level
        self.cs_uc = [cs for cs in self.scenario.commodity_systems.values() if cs.int_lvl in 'uc']
        self.cs_apriori_lm = [cs for cs in self.scenario.commodity_systems.values() if
                              cs.int_lvl in [x for x in cs.apriori_lvls if x != 'uc'] and cs.lm_static]
        self.cs_apriori_unlim = [cs for cs in self.scenario.commodity_systems.values() if
                                 cs.int_lvl in [x for x in cs.apriori_lvls if x != 'uc'] and not cs.lm_static]

        # get a dict of all commodities within Apriori CommoditySystems
        self.commodities = {name: EsmCommodity(commodity, self.scenario, self.run) for block in
                            self.cs_uc + self.cs_apriori_lm + self.cs_apriori_unlim for name, commodity in
                            block.commodities.items()}

        # initialize dataframe for available and fixed power for both the AC and the DC bus
        self.p_available = pd.DataFrame(columns=['ac', 'dc'], dtype=float)
        self.p_fixed = pd.DataFrame(columns=['ac', 'dc'], dtype=float)

    def calc_schedule(self, dti_ph):
        # Set timeindex of current prediction horizon
        self.dti_ph = dti_ph

        # Initialize apriori_data dataframes for all apriori commodities (including index, column names, initial SOC
        # value and additional information about the presence and absence of commodities)
        for commodity in self.commodities.values():
            commodity.init_ph(self.dti_ph)

        # Calculate available and fixed power for prediction horizon
        # ToDo: create new dataframe instead of reindexing and replacing all values?
        self.p_available = self.p_available.reindex(self.dti_ph).fillna(0)
        self.p_available.loc[:] = 0
        self.p_fixed = self.p_fixed.reindex(self.dti_ph).fillna(0)
        self.p_fixed.loc[:] = 0

        for block in self.scenario.blocks.values():
            if isinstance(block, blocks.GridConnection):
                self.p_available.loc[:, get_block_system(block)] += block.size_g2mg * block.eff
            elif isinstance(block, (blocks.WindSource, blocks.PVSource)):
                self.p_available.loc[:, get_block_system(block)] += block.data_ph['power_spec'] * block.size * block.eff
            elif isinstance(block, blocks.ControllableSource):
                self.p_available.loc[:, get_block_system(block)] += block.size * block.eff
            elif isinstance(block, blocks.FixedDemand):
                self.p_fixed.loc[:, get_block_system(block)] += block.data_ph['power_w'] * -1

        for dtindex in self.dti_ph:
            # Calculate power for all CommoditySystems with 'uc' or static load management and add to consumed power
            p_cs_lim_uc = {'ac': 0, 'dc': 0}
            for cs in self.cs_uc + self.cs_apriori_lm:
                p_cs_lim_uc[cs.system] += self.calc_p_commodities(dtindex=dtindex,
                                                                  commodities=[self.commodities[key]
                                                                               for key in cs.commodities.keys()],
                                                                  int_lvl=cs.int_lvl,
                                                                  p_avail_lm=cs.lm_static)

            # only execute optimization of the local grid if there are rulebased components
            if self.cs_apriori_unlim:
                # reset available power on converter to maximum power
                self.p_avail_conv = {'ac': self.sys_core.size_acdc,
                                     'dc': self.sys_core.size_dcac}

                # Subtract demand from available power on AC bus (demand is specified as a negative power)
                for system in ['ac', 'dc']:
                    self.draw_power(bus_connected=system,
                                    pwr=(-1) * self.p_fixed.loc[dtindex, system] + p_cs_lim_uc[system],
                                    dtindex=dtindex)

                # Schedule at base charging of commodities (int_lvl has to be the same for all CommoditySystems -> [0])
                self.calc_p_commodities(dtindex=dtindex,
                                        commodities=[self.commodities[key] for cs in self.cs_apriori_unlim
                                                     for key in cs.commodities.keys()],
                                        int_lvl=self.cs_apriori_unlim[0].int_lvl,
                                        p_avail_lm=None)

            # Execute external charging of commodities based on the defined criteria
            for commodity in self.commodities.values():
                commodity.ext_charging(dtindex)
                commodity.calc_new_soc(dtindex, self.scenario)

    def calc_p_commodities(self, dtindex, commodities, int_lvl, p_avail_lm):
        # get all commodities which are ready for charging
        commodities = [commodity for commodity in commodities if commodity.block.data_ph.loc[dtindex, 'atbase']]
        p_cs = 0

        if not commodities:
            return p_cs  # no commodities to charge at base within the current timestep

        if int_lvl == 'equal':
            # get maximum available power of dynamic load management if not limited by static load management
            if not p_avail_lm:
                # define bus priority and non-priority
                bus_prio = commodities[0].system
                bus_non_prio = get_bus(bus_prio, 'other')
                # calculate available power for dynamic load management
                p_avail_lm = self.p_available.loc[dtindex, bus_prio] + min(self.p_available.loc[dtindex, bus_non_prio],
                                                                           self.p_avail_conv[bus_non_prio]) * \
                             self.get_conv_eff(bus_non_prio, bus_prio)

            while (p_avail_lm - p_cs) > 0 and len(commodities) > 0:
                # calculate possible power for each commodity
                p_share = (p_avail_lm - p_cs) / len(commodities)
                for commodity in commodities:
                    # get maximum possible charging power for commodity, consider the power already assigned to the
                    # commodity in previous iterations
                    p_chg = min(p_share, commodity.calc_p_chg(dtindex, soc_max=1, mode='int_ac') -
                                commodity.block.apriori_data.loc[dtindex, 'p_int_ac'] / commodity.block.parent.eff_chg)
                    # set charging power -> consider the power already assigned to the commodity in previous iterations
                    commodity.set_p(dtindex=dtindex, power=p_chg + commodity.block.apriori_data.loc[
                        dtindex, 'p_int_ac'] / commodity.block.parent.eff_chg, mode='int_ac')
                    p_cs += p_chg
                    if p_chg == 0:
                        commodities.remove(commodity)
        else:
            # define sorting functions for the different strategies
            sort_key_funcs = {
                'uc': lambda x: x.block.name,  # sorting makes no difference -> dummy function´for uc
                'fcfs': lambda x: x.get_latest_arr(dtindex),
                'soc': lambda x: x.block.apriori_data.loc[dtindex, 'soc']
            }
            # get a list of all available commodities and sort them according to the chosen strategy
            commodities = sorted(commodities, key=sort_key_funcs[int_lvl])
            for commodity in commodities:
                # define bus priority and non-priority for the commodity
                bus_prio = get_bus(commodity.system, 'same')
                bus_non_prio = get_bus(commodity.system, 'other')

                # get limitations of the system (available power on buses and converter or static load managment)
                if int_lvl == 'uc':
                    # no limitation for charging power on CommoditySystem level for uncoordinated charging
                    p_sys_lim = np.inf
                elif p_avail_lm:
                    # limitation for static load management is based on max power and already assigned power
                    p_sys_lim = p_avail_lm - p_cs
                else:
                    # limitation for dynamic load management is based on available power on buses and converter
                    p_sys_lim = self.p_available.loc[dtindex, bus_prio] + \
                                min(self.p_available.loc[dtindex, bus_non_prio],
                                    self.p_avail_conv[bus_non_prio]) * self.get_conv_eff(bus_non_prio, bus_prio)

                # power for commodity considering limitations of the commodity and the system
                pwr_chg = min(commodity.calc_p_chg(dtindex=dtindex, soc_max=1, mode='int_ac'), p_sys_lim)

                # update available power of the system for dynamic load management
                if not p_avail_lm and int_lvl != 'uc':
                    self.draw_power(bus_connected=bus_prio, pwr=pwr_chg, dtindex=dtindex)

                # update power already assigned to commodities
                p_cs += pwr_chg

                # assign charging power to commodity
                commodity.set_p(dtindex=dtindex, power=pwr_chg, mode='int_ac')

        return p_cs

    def draw_power(self, bus_connected, pwr, dtindex):
        # deduct power from available power on the corresponding bus(es) and the converter
        pwr_prio = min(self.p_available.loc[dtindex, bus_connected], pwr)
        self.p_available.loc[dtindex, bus_connected] -= pwr_prio

        pwr_non_prio = (pwr - pwr_prio) / self.get_conv_eff(get_bus(bus_connected, 'other'), bus_connected)
        self.p_available.loc[dtindex, get_bus(bus_connected, 'other')] -= pwr_non_prio
        self.p_avail_conv[get_bus(bus_connected, 'other')] -= pwr_non_prio

        for location, value in {'AC bus': self.p_available.loc[dtindex, 'ac'],
                                'DC bus': self.p_available.loc[dtindex, 'dc'],
                                'AC/DC converter': self.p_avail_conv['ac'],
                                'DC/AC converter': self.p_avail_conv['dc']}.items():
            if value < 0:
                # ToDo: change to self.scenario.logger if new logger structure is merged into branch
                self.run.logger.warning(f'Scenario \"{self.scenario.name}\": Power shortage of {-1 * value:.2E} W on'
                                        f' {location} occurred in AprioriScheduler at {dtindex}!'
                                        f' This shortage may lead to infeasiblity during optimization.')

    def get_conv_eff(self, source, target):
        return {'ac': {'ac': 1,
                       'dc': self.sys_core.eff_acdc},
                'dc': {'ac': self.sys_core.eff_dcac,
                       'dc': 1}}[source][target]


class EsmCommodity:
    def __init__(self, block, scenario, run):
        self.block = block
        self.run = run
        self.scenario = scenario
        # get the system to which the block is connected
        self.system = self.block.parent.system

        self.loss_rate = None

        # Placeholder for variables for power calculations at external charging
        self.minsoc_inz = self.dep_inz = self.arr_inz = self.arr_parking_inz = self.chg_inz = None
        self.parking_charging = None

    def init_ph(self, dti_ph, *_):
        columns = ['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption', 'soc']
        self.block.apriori_data = pd.DataFrame(0,
                                               index=dti_ph,
                                               columns=['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption', 'soc'],
                                               dtype=float)

        # set inital soc for PH -> take aging results (soc_min, soc_max) into account
        self.block.apriori_data.loc[dti_ph[0], 'soc'] = statistics.median([self.block.soc_min,
                                                                           self.block.soc_init_ph,
                                                                           self.block.soc_max])

        self.block.apriori_data.loc[:, 'p_consumption'] = -1 * self.block.data_ph['consumption']

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

        # get first timesteps, where vehicle is parking at destination
        self.arr_parking_inz = self.block.data_ph.index[
            self.block.data_ph['atac'] & ~self.block.data_ph['atac'].shift(fill_value=False)]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        self.chg_inz = self.block.data_ph.index[self.block.data_ph[['atbase', 'atac', 'atdc']].any(axis=1)]

        # initialize variable for charging during single parking process
        self.parking_charging = False

    def get_latest_arr(self, dtindex):
        # get latest arrival before current timestep
        try:
            return self.arr_inz[self.arr_inz <= dtindex][-1]
        except:
            # return 1900 if no arrival is found -> at start of new simulation
            return pd.to_datetime('1900').replace(tzinfo=self.scenario.timezone)

    def set_p(self, dtindex, power, mode='int_ac'):
        p = {'int_ac': self.block.parent.eff_chg,
             'ext_ac': 1,
             'ext_dc': 1}[mode] * power
        col = f'p_{mode}'
        self.block.apriori_data.loc[dtindex, col] = p

    def calc_p_chg(self, dtindex, soc_max=1, mode='int_ac', *_):
        # p_maxchg: maximum charging power in W, measured at storage, NOT at bus
        soc_max = float(soc_max)
        soc_max = min(self.block.soc_max, soc_max)
        soc_threshold = 0
        p_maxchg = {'int_ac': self.block.pwr_chg * self.block.parent.eff_chg,
                    'ext_ac': self.block.parent.pwr_ext_ac,
                    'ext_dc': self.block.parent.pwr_ext_dc}[mode]
        eff = {'int_ac': self.block.parent.eff_chg,
               'ext_ac': 1,
               'ext_dc': 1}[mode]

        # Only charge if SOC falls below threshold (soc_max - soc_threshold)
        if (soc_current := self.block.apriori_data.loc[dtindex, 'soc']) >= soc_max - soc_threshold:
            return 0

        # STORAGE: power to be charged to target SOC in Wh in one timestep using SOC delta
        p_tosoc = (soc_max - soc_current * (1 - self.loss_rate)) * self.block.size / self.scenario.timestep_hours

        # BUS: charging power measured at connection to DC bus; reduce power in final step to just reach target SOC
        p_chg = min(p_maxchg, p_tosoc) / eff

        if p_chg < 0:
            # ToDo: change to self.scenario.logger if new logger structure is merged into branch
            self.run.warning('Charging power below 0 W for commodity {self.block.name} at {dtindex}!')
        return p_chg

    def calc_new_soc(self, dtindex, scenario):
        columns = ['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption']
        power = self.block.apriori_data.loc[dtindex, columns].sum()
        # calculate new soc value
        new_soc = power / self.block.size * self.scenario.timestep_hours + \
                  self.block.apriori_data.loc[dtindex, 'soc'] * (1 - self.loss_rate)

        if new_soc < self.block.soc_min:
            # ToDo: change to self.scenario.logger if new logger structure is merged into branch
            self.run.logger.warning(f'SOC of commodity {self.block.name} falls below minimum SOC of'
                                             f' {self.block.soc_min * 100:.2f} % at {dtindex}!')

        # Assign new SOC to apriori_data DataFrame of block
        self.block.apriori_data.loc[dtindex + self.scenario.timestep_td, 'soc'] = new_soc

    def ext_charging(self, dtindex):
        if self.block.data_ph.loc[dtindex, 'atac'] == 1:  # parking at destination
            if dtindex in self.arr_parking_inz:  # plugging in only happens when parking starts
                # use current int-index and next arrival index to calculate consumption and convert to SOC
                try:  # Fails, if current trip is last trip and doesn't end within prediction horizon
                    arr_inxt = self.arr_inz[self.arr_inz >= dtindex][0]
                except:
                    arr_inxt = self.block.data_ph.index[-1]
                consumption_remaining = self.block.data_ph.loc[dtindex:arr_inxt,
                                        'consumption'].sum() * self.scenario.timestep_hours
                # set charging to True, if charging is necessary
                if consumption_remaining > (
                        (self.get_data(dtindex, 'soc') + self.block.data_ph.loc[
                            arr_inxt, 'minsoc']) * self.block.size):
                    self.parking_charging = True
                else:
                    self.parking_charging = False

            if self.parking_charging is True:
                # ToDo: implement input dependent target soc (e.g. 80%)
                self.set_p(dtindex=dtindex,
                           power=self.calc_p_chg(dtindex=dtindex, soc_max=1, mode='ext_ac'),
                           mode='ext_ac')

        elif self.block.data_ph.loc[dtindex, 'atdc'] == 1:  # vehicle is driving with possibility to charge on-route
            # activate charging, if SOC will fall below threshold, before next possibility to charge
            chg_inxt = self.chg_inz[self.chg_inz > dtindex][0]
            chg_soc = self.get_data(dtindex, 'soc') - self.block.data_ph.loc[dtindex:chg_inxt,
                                                      'consumption'].sum() * self.scenario.timestep_hours / self.block.size
            if chg_soc < 0.05:
                # fast-charging only up to SOC of 80 %
                self.set_p(dtindex=dtindex,
                           power=self.calc_p_chg(dtindex=dtindex, soc_max=0.8, mode='ext_dc'),
                           mode='ext_dc')
