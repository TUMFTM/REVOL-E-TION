import numpy as np
import pandas as pd
import statistics
import blocks


def get_bus(bus, target):
    # Takes the bus the block is connected to and returns the specified bus
    return {'ac': {'same': 'ac', 'other': 'dc'},
            'dc': {'same': 'dc', 'other': 'ac'}}[bus][target]


def get_block_system(block):
    # get the label of the bus a block is connected to
    return block.bus_connected.label[0:2]


class AprioriPowerScheduler:
    def __init__(self, scenario):
        self.scenario = scenario
        self.horizon = None

        # Get name of system core block
        self.sys_core = [block for block in self.scenario.blocks.values() if isinstance(block, blocks.SystemCore)][0]

        # Placeholder for timeindex of current prediction horizon
        self.dti_ph = None

        # remaining power available on the system core converter for the current timestep
        self.p_syscore_conv_avail = {}

        # get different lists of commodity systems according to the restrictions of the apriori integration level
        self.cs_unlim = [cs for cs in self.scenario.commodity_systems.values() if cs.int_lvl in 'uc']
        self.cs_lm_static = [cs for cs in self.scenario.commodity_systems.values() if
                              cs.int_lvl in [x for x in cs.apriori_lvls if x != 'uc'] and cs.lm_static]
        self.cs_lm_dynamic = [cs for cs in self.scenario.commodity_systems.values() if
                                 cs.int_lvl in [x for x in cs.apriori_lvls if x != 'uc'] and not cs.lm_static]

        # get a dict of all commodities within Apriori CommoditySystems
        self.apriori_commodities = {name: AprioriCommodity(commodity, self.scenario) for block in
                                    self.cs_unlim + self.cs_lm_static + self.cs_lm_dynamic for name, commodity in
                                    block.commodities.items()}

        # initialize dataframe for available and fixed power for both the AC and the DC bus
        self.p_esm_avail = pd.DataFrame(columns=['ac', 'dc'], dtype=float)
        self.p_esm_fixed = pd.DataFrame(columns=['ac', 'dc'], dtype=float)

    def calc_ph_schedule(self, horizon):
        # Set timeindex of current prediction horizon
        self.horizon = horizon

        # Initialize apriori_data dataframes for all apriori_commodities (including index, column names, initial SOC
        # value and additional information about the presence and absence of apriori_commodities)
        for commodity in self.apriori_commodities.values():
            commodity.init_ph(self.horizon.dti_ph)

        # Only necessary, if there are CommoditySystems with dynamic load management
        if self.cs_lm_dynamic:
            # Calculate available and fixed power for prediction horizon
            # reset available and fixed power for the current prediction horizon to zero
            self.p_esm_avail = self.p_esm_avail.reindex(self.horizon.dti_ph)
            self.p_esm_avail.loc[:] = 0
            self.p_esm_fixed = self.p_esm_fixed.reindex(self.horizon.dti_ph)
            self.p_esm_fixed.loc[:] = 0

            for block in self.scenario.blocks.values():
                if isinstance(block, blocks.GridConnection):
                    self.p_esm_avail.loc[:, get_block_system(block)] += block.size_g2mg * block.eff
                elif isinstance(block, (blocks.WindSource, blocks.PVSource)):
                    self.p_esm_avail.loc[:, get_block_system(block)] += block.data_ph['power_spec'] * block.size * block.eff
                elif isinstance(block, blocks.ControllableSource):
                    self.p_esm_avail.loc[:, get_block_system(block)] += block.size * block.eff
                elif isinstance(block, blocks.FixedDemand):
                    self.p_esm_fixed.loc[:, get_block_system(block)] += block.data_ph['power_w']

        for dtindex in self.horizon.dti_ph:
            # Calculate power for all CommoditySystems with 'uc' or static load management and add to consumed power
            p_csc_unlim_static = {'ac': 0, 'dc': 0}
            for cs in self.cs_unlim + self.cs_lm_static:
                p_csc_unlim_static[cs.system] += self.calc_p_commodities(dtindex=dtindex,
                                                                         commodities=[self.apriori_commodities[key]
                                                                                      for key in cs.commodities.keys()],
                                                                         int_lvl=cs.int_lvl,
                                                                         p_csc_avail_total=cs.lm_static)

            # only execute optimization of the local grid if there are rulebased components
            if self.cs_lm_dynamic:
                # reset available power on converter to maximum power
                self.p_syscore_conv_avail = {'ac': self.sys_core.size_acdc,
                                             'dc': self.sys_core.size_dcac}

                # Draw demand power (FixedDemand, cs_unlim, cs_lm_static) from available power on AC bus
                for system in ['ac', 'dc']:
                    self.draw_power(bus_connected=system,
                                    pwr=self.p_esm_fixed.loc[dtindex, system] + p_csc_unlim_static[system],
                                    dtindex=dtindex)

                # Schedule at base charging of commodities (int_lvl has to be the same for all CommoditySystems -> [0])
                self.calc_p_commodities(dtindex=dtindex,
                                        commodities=[self.apriori_commodities[key] for cs in self.cs_lm_dynamic
                                                     for key in cs.commodities.keys()],
                                        int_lvl=self.cs_lm_dynamic[0].int_lvl,
                                        p_csc_avail_total=None)

            # Execute external charging of commodities based on the defined criteria
            for commodity in self.apriori_commodities.values():
                commodity.ext_charging(dtindex)
                commodity.calc_new_soc(dtindex, self.scenario)

    def calc_p_commodities(self, dtindex, commodities, int_lvl, p_csc_avail_total):
        # get all commodities which are ready for charging
        commodities = [commodity for commodity in commodities if commodity.block.data_ph.loc[dtindex, 'atbase']]
        p_csc_assigned = 0

        if not commodities:
            return p_csc_assigned  # no commodities to charge at base within the current timestep

        if int_lvl == 'equal':
            # get maximum available power of dynamic load management if not limited by static load management
            if not p_csc_avail_total:
                # define bus priority and non-priority
                bus_prio = commodities[0].system
                bus_non_prio = get_bus(bus_prio, 'other')
                # calculate available power for dynamic load management
                p_csc_avail_total = self.p_esm_avail.loc[dtindex, bus_prio] + min(self.p_esm_avail.loc[dtindex, bus_non_prio],
                                                                           self.p_syscore_conv_avail[bus_non_prio]) * \
                             self.get_conv_eff(bus_non_prio, bus_prio)

            while (p_csc_avail_total - p_csc_assigned) > 0 and len(commodities) > 0:
                # calculate possible power for each commodity
                p_share = (p_csc_avail_total - p_csc_assigned) / len(commodities)
                # necessary to avoid floating point errors and endless loops
                if p_share < 1.00E-10:
                    return p_csc_assigned
                # slicing creates a copy of the list. This is necessary to remove commodities from the list without an
                # error in the for loop. Otherwise, the next element after the removed one is skipped.
                # ToDo: copy()??
                for commodity in commodities[:]:
                    # get maximum possible charging power for commodity, consider the power already assigned to the
                    # commodity in previous iterations
                    p_chg = min(p_share, commodity.calc_p_chg(dtindex, mode='int_ac') -
                                commodity.block.apriori_data.loc[dtindex, 'p_int_ac'] / commodity.block.parent.eff_chg)
                    # set charging power -> consider the power already assigned to the commodity in previous iterations
                    commodity.set_p(dtindex=dtindex, power=p_chg + commodity.block.apriori_data.loc[
                        dtindex, 'p_int_ac'] / commodity.block.parent.eff_chg, mode='int_ac')
                    p_csc_assigned += p_chg
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
                # get limitations of the system (available power on buses and converter or static load managment)
                if int_lvl == 'uc':
                    # no limitation for charging power on CommoditySystem level for uncoordinated charging
                    p_csc_avail_left = np.inf
                elif p_csc_avail_total:
                    # limitation for static load management is based on max power and already assigned power
                    p_csc_avail_left = p_csc_avail_total - p_csc_assigned
                else:
                    # define bus priority and non-priority for the commodity
                    bus_prio = get_bus(commodity.system, 'same')
                    bus_non_prio = get_bus(commodity.system, 'other')

                    # limitation for dynamic load management is based on available power on buses and converter
                    p_csc_avail_left = self.p_esm_avail.loc[dtindex, bus_prio] + \
                                min(self.p_esm_avail.loc[dtindex, bus_non_prio],
                                    self.p_syscore_conv_avail[bus_non_prio]) * self.get_conv_eff(bus_non_prio, bus_prio)

                # power for commodity considering limitations of the commodity and the system
                pwr_chg = min(commodity.calc_p_chg(dtindex=dtindex, mode='int_ac'), p_csc_avail_left)

                # update available power of the system for dynamic load management
                if not p_csc_avail_total and int_lvl != 'uc':
                    self.draw_power(bus_connected=bus_prio, pwr=pwr_chg, dtindex=dtindex)

                # update power already assigned to commodities
                p_csc_assigned += pwr_chg

                # assign charging power to commodity
                commodity.set_p(dtindex=dtindex, power=pwr_chg, mode='int_ac')

        return p_csc_assigned

    def draw_power(self, bus_connected, pwr, dtindex):
        # deduct power from available power on the corresponding bus(es) and the converter
        pwr_prio = min(self.p_esm_avail.loc[dtindex, bus_connected], pwr)
        self.p_esm_avail.loc[dtindex, bus_connected] -= pwr_prio

        pwr_non_prio = (pwr - pwr_prio) / self.get_conv_eff(get_bus(bus_connected, 'other'), bus_connected)
        self.p_esm_avail.loc[dtindex, get_bus(bus_connected, 'other')] -= pwr_non_prio
        self.p_syscore_conv_avail[get_bus(bus_connected, 'other')] -= pwr_non_prio

        for location, value in {'AC bus': self.p_esm_avail.loc[dtindex, 'ac'],
                                'DC bus': self.p_esm_avail.loc[dtindex, 'dc'],
                                'AC/DC converter': self.p_syscore_conv_avail['ac'],
                                'DC/AC converter': self.p_syscore_conv_avail['dc']}.items():
            if value < 0:
                self.scenario.logger.warning(f'Scenario \"{self.scenario.name}\": Power shortage of {-1 * value:.2E} W on'
                                             f' {location} occurred in AprioriScheduler at {dtindex}!'
                                             f' This shortage may lead to infeasibility during optimization.')

    def get_conv_eff(self, source, target):
        return {'ac': {'ac': 1,
                       'dc': self.sys_core.eff_acdc},
                'dc': {'ac': self.sys_core.eff_dcac,
                       'dc': 1}}[source][target]


class AprioriCommodity:
    def __init__(self, block, scenario):
        self.block = block
        self.scenario = scenario
        # get the system to which the block is connected
        self.system = self.block.parent.system

        # placeholder for current soh
        self.soh = None

        # define loss_rate
        self.loss_rate = 1 - (1 - self.block.parent.loss_rate) ** (self.scenario.timestep_td / pd.Timedelta('1h'))

        # get the indices of all nonzero minsoc rows in the data
        self.minsoc_inz = self.block.data.index[self.block.data['minsoc'] != 0]

        # get first timesteps, where vehicle has left the base
        self.dep_base_inz = self.block.data.index[~self.block.data['atbase'] & self.block.data['atbase'].shift(1, fill_value=False)]
        # get first timesteps, where vehicle has left the destination
        self.dep_dest_inz = self.block.data.index[~self.block.data['atac'] & self.block.data['atac'].shift(1, fill_value=False)]

        # get first timesteps, where vehicle is at base again
        self.arr_base_inz = self.block.data.index[self.block.data['atbase'] & ~self.block.data['atbase'].shift(fill_value=False)]
        # get first timesteps, where vehicle is parking at destination
        self.arr_dest_inz = self.block.data.index[self.block.data['atac'] & ~self.block.data['atac'].shift(fill_value=False)]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        self.chg_avail_inz = self.block.data.index[self.block.data[['atbase', 'atac', 'atdc']].any(axis=1)]

        # initialize variable for charging during single parking process
        self.parking_charging = False

    def init_ph(self, dti_ph, *_):
        # Initialize apriori_data DataFrame and set initial soc for horizon taking aging results into account
        self.block.apriori_data = pd.DataFrame(0,
                                               index=dti_ph,
                                               columns=['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption', 'soc'],
                                               dtype=float)

        self.block.apriori_data.loc[:, 'p_consumption'] = -1 * self.block.data_ph['consumption']

        self.block.apriori_data.loc[dti_ph.min(), 'soc'] = statistics.median([self.block.soc_min,
                                                                              self.block.soc_init_ph,
                                                                              self.block.soc_max])

        # get soh for current prediction horizon
        self.soh = self.block.soh.dropna().loc[self.block.soh.dropna().index.max()]

    def get_latest_arr(self, dtindex):
        # get latest arrival before current timestep
        return self.arr_base_inz[self.arr_base_inz <= dtindex].max()

    def set_p(self, dtindex, power, mode='int_ac'):
        p = {'int_ac': self.block.parent.eff_chg,
             'ext_ac': 1,
             'ext_dc': 1}[mode] * power

        self.block.apriori_data.loc[dtindex, f'p_{mode}'] = p

    def convert_soc_ui2internal(self, soc_ui):
        # convert the soc the displayed soc (UI) to the soc of the oemof storage taking aging effects into account
        return soc_ui * self.soh + self.block.soc_min

    def calc_p_chg(self, dtindex, mode='int_ac', soc_target=None, *_):
        # p_maxchg: maximum charging power in W, measured at storage, NOT at bus
        soc_target = self.get_soc_target(dtindex) if not soc_target else soc_target
        # apply aging boundaries to soc_target
        soc_target = self.convert_soc_ui2internal(soc_target)
        soc_threshold = 0
        p_maxchg = {'int_ac': self.block.pwr_chg * self.block.parent.eff_chg,
                    'ext_ac': self.block.parent.pwr_ext_ac,
                    'ext_dc': self.block.parent.pwr_ext_dc}[mode]
        eff = {'int_ac': self.block.parent.eff_chg,
               'ext_ac': 1,
               'ext_dc': 1}[mode]

        # Only charge if SOC falls below threshold (soc_max - soc_threshold)
        if (soc_current := self.block.apriori_data.loc[dtindex, 'soc']) > soc_target - soc_threshold * self.soh:
            return 0

        # STORAGE: power to be charged to target SOC in Wh in one timestep using SOC delta
        p_tosoc = ((soc_target - soc_current * (1 - self.loss_rate)) * self.block.size /
                   self.scenario.timestep_hours) - self.block.apriori_data.loc[dtindex, 'p_consumption']

        # BUS: charging power measured at connection to DC bus; reduce power in final step to just reach target SOC
        if (p_chg := min(p_maxchg, p_tosoc) / eff) < 0:
            self.scenario.logger.warning('Charging power below 0 W for commodity {self.block.name} at {dtindex}!')

        # avoid negative powers which can be caused by aging at the start of a new PH, if soc_target is already reached
        # but new soc_target which is decreased compared to the old soc_target due to aging.
        return max(p_chg, 0)

    def calc_new_soc(self, dtindex, scenario):
        columns = ['p_int_ac', 'p_ext_ac', 'p_ext_dc', 'p_consumption']
        power = self.block.apriori_data.loc[dtindex, columns].sum()
        # calculate new soc value
        new_soc = power / self.block.size * self.scenario.timestep_hours + \
                  self.block.apriori_data.loc[dtindex, 'soc'] * (1 - self.loss_rate)

        if new_soc < self.block.soc_min:
            self.scenario.logger.warning(f'SOC of commodity {self.block.name} falls below minimum SOC of'
                                         f' {self.block.soc_min * 100:.2f} % at {dtindex}!')

        # Assign new SOC to apriori_data DataFrame of block
        self.block.apriori_data.loc[dtindex + self.scenario.timestep_td, 'soc'] = new_soc

    def get_soc_target(self, dtindex):
        # check if there are any departures after current timestep within forecast period
        if (departures := self.dep_base_inz[(self.dep_base_inz >= dtindex) &
                                            (self.dep_base_inz <= dtindex + pd.Timedelta(hours=self.block.parent.forecast_hours)
                                            if self.block.parent.forecast_hours else True)]).empty:
            return self.block.parent.soc_target

        #  get start and end of next trip
        dep_nxt = departures.min()
        arr_nxt = self.arr_base_inz[self.arr_base_inz >= dtindex].min()
        #  sum up energy between trip start and end
        if arr_nxt <= dep_nxt:
            # Destination charging -> sum up remaining energy of ongoing trip until end of trip
            e_con = self.block.data.loc[dtindex:arr_nxt - self.scenario.timestep_td, 'consumption'].sum() * self.scenario.timestep_hours
        else:
            # Charging at base -> sum up energy for next trip
            e_con = self.block.data.loc[dep_nxt:arr_nxt - self.scenario.timestep_td, 'consumption'].sum() * self.scenario.timestep_hours
        #  Convert energy consumption to delta soc taking the current soh into account
        soc_delta = e_con / (self.block.size * self.soh)
        #  Set soc_target dependent on soc_delta of trip and settings of the MobileCommodity
        soc_target = 1 if soc_delta > (self.block.parent.soc_target - self.block.parent.soc_return) else self.block.parent.soc_target
        return soc_target

    def ext_charging(self, dtindex):
        # determine whether destination charging is necessary
        if self.block.data_ph.loc[dtindex, 'atac'] == 1:
            if dtindex in self.arr_dest_inz:  # plugging in only happens when parking starts
                # use current time and next arrival index to calculate consumption and convert to SOC
                arr_nxt = arrivals.min() if not (arrivals := self.arr_base_inz[self.arr_base_inz >= dtindex]).empty else self.block.data_ph.index.max()
                e_trip_remaining = self.block.data_ph.loc[dtindex:arr_nxt, 'consumption'].sum() * self.scenario.timestep_hours

                # set charging to True, if charging is necessary
                if e_trip_remaining > ((self.block.apriori_data.loc[dtindex, 'soc'] - self.block.parent.soc_return) *
                                       self.block.size * self.soh):
                    self.parking_charging = True
                else:
                    self.parking_charging = False

            if self.parking_charging is True:
                self.set_p(dtindex=dtindex, power=self.calc_p_chg(dtindex=dtindex, mode='ext_ac'), mode='ext_ac')

        # determine whether on-route charging is necessary
        elif self.block.data_ph.loc[dtindex, 'atdc'] == 1:
            # activate charging, if SOC will fall below threshold, before next possibility to charge
            chg_nxt = self.chg_avail_inz[self.chg_avail_inz > dtindex].min()
            soc_chg_nxt = self.block.apriori_data.loc[dtindex, 'soc'] - \
                          self.block.data_ph.loc[dtindex:chg_nxt - self.scenario.timestep_td,
                          'consumption'].sum() * self.scenario.timestep_hours / self.block.size
            if soc_chg_nxt < self.convert_soc_ui2internal(0.05):
                # fast-charging only up to SOC of 80 %
                self.set_p(dtindex=dtindex,
                           power=self.calc_p_chg(dtindex=dtindex, mode='ext_dc', soc_target=0.8),
                           mode='ext_dc')
