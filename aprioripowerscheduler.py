import pandas as pd
import blocks


class AprioriPowerScheduler:
    def __init__(self, scenario, horizon):

        self.data = None

        self.scenario = scenario
        self.horizon = horizon
        # get list of commodity systems with apriori integration level
        self.apriori_cs = [cs for cs in self.scenario.commodity_systems.values() if cs.int_lvl in cs.apriori_lvls]
        # initialize dataframe for available power for both the AC and the DC bus
        self.p_available = pd.DataFrame({'p_ac': 0, 'p_dc': 0}, index=self.horizon.ph_dti)
        self.func_map = {'uc': self.calc_power_uc,
                         'fcfs': self.calc_power_fcfs,
                         'equal': self.calc_power_equal,
                         'soc': self.calc_power_soc}
        self.ses_available = any(
            [isinstance(block, blocks.StationaryEnergyStorage) for block in self.scenario.blocks.values()])

        # define list of elements controlled by rule based algorithm (all commodity systems)
        self.commodities = {commodity.name: CommodityData(commodity, self.scenario) for block in self.apriori_cs for
                            commodity in block.commodities.values()}

        pass

    def calc_schedule(self):
        if self.scenario.strategy != 'uc':
            self.calc_available_power()

        for dtindex in self.scenario.sim_dti:
            # Schedule at base charging
            self.func_map[self.apriori_cs[0].int_lvl](dtindex)

            # ToDo: Schedule energy storage power
            pass

            # Schedule external charging
            for com in self.commodities.values():
                com.ext_charging(dtindex)

                # Update soc
                soc_delta = (com.commodity.apriori_data.loc[dtindex, 'p_int_ac'] + \
                             com.commodity.apriori_data.loc[dtindex, 'p_ext_ac'] + \
                             com.commodity.apriori_data.loc[dtindex, 'p_ext_dc'] - \
                             com.commodity.ph_data.loc[dtindex, 'consumption']) * self.scenario.timestep_hours / com.commodity.size
                new_soc = com.commodity.apriori_data.loc[dtindex, 'soc'] + soc_delta
                # ToDo: check whether SOC indexing fits optimization output -> depending on soc shifting in calc_results()
                com.commodity.apriori_data.loc[dtindex + self.scenario.timestep_td, 'soc'] = new_soc
                # ToDo: include min and max soc set by battery aging
                if (new_soc < 0) or (new_soc > 1):
                    # ToDo: Raise exception
                    print('Error! Calculation of UC charging profile failed. SOC out of bounds')

            # ToDo: Update soc for stationary energy storages
            pass

            #     # update SOC
            #     soc_delta = (self.uc_data.loc[dtindex, 'p_int_ac'] + \
            #                  self.uc_data.loc[dtindex, 'p_ext_ac'] + \
            #                  self.uc_data.loc[dtindex, 'p_ext_dc'] - \
            #                  row['consumption']) * self.scenario.timestep_hours / self.size
            #     new_soc = self.uc_data.loc[dtindex, 'soc'] + soc_delta
            #     # ToDo: check whether SOC indexing fits optimization output -> depending on soc shifting in calc_results()
            #     self.uc_data.loc[dtindex + self.scenario.timestep_td, 'soc'] = new_soc
            #     if (new_soc < 0) or (new_soc > 1):
            #         # ToDo: Raise exception
            #         print('Error! Calculation of UC charging profile failed. SOC out of bounds')

        # ToDo: assign charging power schedule to flows of commodities
        pass

    def calc_available_power(self):
        # compute available power resulting from different blocks at the AC and DC bus
        for block_name, block in self.scenario.blocks.items():
            if isinstance(block, blocks.WindSource):
                # ToDo: How to compute wind power - specific power * size or total power?
                p_wind = block.ph_data['P'] * block.eff  # measured at AC bus
                self.p_available['p_ac'] += p_wind
            elif isinstance(block, blocks.PVSource):
                p_pv = block.ph_data['p_spec'] * block.size * block.eff  # measured at DC bus
                self.p_available['p_dc'] += p_pv
            elif isinstance(block, blocks.ControllableSource):
                p_controllable_source = block.size * block.eff  # measured at AC bus
                self.p_available['p_ac'] += p_controllable_source
            elif isinstance(block, blocks.GridConnection):
                p_grid = block.size * block.eff  # measured at AC bus
                self.p_available['p_ac'] += p_grid
            elif isinstance(block, blocks.FixedDemand):
                p_dem = block.ph_data['power']
                self.p_available['p_ac'] -= p_dem

    def calc_power_uc(self, dtindex):
        for com in self.commodities.values():
            """Converting availability and consumption data of commodities into a power timeseries for uncoordinated
             (i.e. unoptimized and starting at full power after return) charging of commodity"""

            # Heuristic:
            # - Vehicle is charged immediately, if atbase is True
            # - Vehicle gets charged during driving, if soc in next timestep is going to fall below threshold value
            # - Vehicle is charged during parking at destination, if current SOC is not enough for trip back to base
            # ToDo: influence of min soc for atbase charging: min soc should not have any influence:
            #  vehicle starts charging, when returning and charges at full speed, until battery is full
            #  min soc doesn't accelerate things
            #  -> remove minsoc in its current function from atbase
            row = com.commodity.ph_data.loc[dtindex]
            if row['atbase'] == 1:  # commodity is at base and chargeable
                try:
                    minsoc_inxt = com.minsoc_inz[com.minsoc_inz >= dtindex][0]  # find next nonzero min SOC
                    dep_inxt = com.dep_inz[com.dep_inz >= dtindex][0]  # find next departure
                    if minsoc_inxt > dep_inxt:  # Next min soc is not defined before next departure
                        dep_soc = 1  # ToDo: implement input dependent target soc (e.g. 80%)
                    else:  # next min_soc is defined before departure and therefore valid for this charging session
                        dep_soc = com.commodity.ph_data.loc[minsoc_inxt, 'minsoc']  # get the SOC to recharge to
                except:  # when there is no further departure
                    dep_soc = 1  # ToDo: implement input dependent target soc (e.g. 80%)

                # Execute AC charging at base
                com.commodity.apriori_data.loc[dtindex, 'p_int_ac'] = com.charge(soc_target=dep_soc,
                                                                                 soc_current=com.commodity.apriori_data.loc[dtindex, 'soc'],
                                                                                 p_maxchg=com.commodity.chg_pwr,
                                                                                 chg_eff=com.commodity.parent.chg_eff)

    def calc_power_fcfs(self):
        return

    def calc_power_equal(self):
        return

    def calc_power_soc(self):
        return

    def calc_power_ses(self):
        return

    def calc_available_ses_power(self, ses, current_soc):
        p_ses = ses.size * min(ses.dis_crate,
                               current_soc / self.scenario.timestep_hours) * ses.dis_eff  # measured at DC bus
        return p_ses


class CommodityData:
    def __init__(self, commodity, scenario):
        self.commodity = commodity
        self.scenario = scenario
        self.commodity.apriori_data = pd.DataFrame({'p_int_ac': 0,  # charging power of AC charger at base
                                                    'p_ext_ac': 0,  # charging power of external AC charger
                                                    'p_ext_dc': 0,  # charging power of external DC charger
                                                    'soc': 0},  # soc of storage at beginning of timestep
                                                   index=self.commodity.ph_data.index)

        # initialize soc tracking
        self.commodity.apriori_data.loc[self.commodity.apriori_data.index[0], 'soc'] = self.commodity.init_soc

        # get the indices of all nonzero minsoc rows in the data
        self.minsoc_inz = self.commodity.ph_data.index[self.commodity.ph_data['minsoc'] != 0]

        # get first timesteps, where vehicle has left the base
        self.dep_inz = self.commodity.ph_data.index[self.commodity.ph_data['atbase'] & ~self.commodity.ph_data['atbase'].shift(-1, fill_value=False)]

        # get first timesteps, where vehicle is at base again
        self.arr_inz = self.commodity.ph_data.index[self.commodity.ph_data['atbase'] & ~self.commodity.ph_data['atbase'].shift(fill_value=False)][1:]

        # get first timesteps, where vehicle is parking at destination
        self.arr_parking_inz = self.commodity.ph_data.index[self.commodity.ph_data['atac'] & ~self.commodity.ph_data['atac'].shift(fill_value=False)]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        self.chg_inz = self.commodity.ph_data.index[self.commodity.ph_data[['atbase', 'atac', 'atdc']].any(axis=1)]

        # initialize variable for charging during single parking process
        self.parking_charging = False

    def ext_charging(self, dtindex):
        row = self.commodity.ph_data.loc[dtindex]
        if row['atac'] == 1:  # parking at destination
            if dtindex in self.arr_parking_inz:  # plugging in only happens when parking starts
                # use current int-index and next arrival index to calculate consumption and convert to SOC
                try:  # Fails, if current trip is last trip and doesn't end within prediction horizon
                    arr_inxt = self.arr_inz[self.arr_inz >= dtindex][0]
                except:
                    arr_inxt = self.commodity.ph_data.index[-1]
                consumption_remaining = self.commodity.ph_data.loc[dtindex:arr_inxt,
                                        'consumption'].sum() * self.scenario.timestep_hours
                # set charging to True, if charging is necessary
                if consumption_remaining > (
                        (self.commodity.apriori_data.loc[dtindex, 'soc'] + self.commodity.ph_data.loc[arr_inxt, 'minsoc']) * self.commodity.size):
                    self.parking_charging = True
                else:
                    self.parking_charging = False

            if self.parking_charging is True:
                dep_soc = 1  # ToDo: implement input dependent target soc (e.g. 80%)

                # Execute AC charging at destination parking
                self.commodity.apriori_data.loc[dtindex, 'p_ext_ac'] = self.charge(soc_target=dep_soc,
                                                                                   soc_current=self.commodity.apriori_data.loc[
                                                                                       dtindex, 'soc'],
                                                                                   p_maxchg=self.commodity.parent.ext_ac_power,
                                                                                   chg_eff=1)

        elif row['atdc'] == 1:  # vehicle is driving with possibility to charge on-route
            # activate charging, if SOC will fall below threshold, before next possibility to charge
            chg_inxt = self.chg_inz[self.chg_inz > dtindex][0]
            chg_soc = self.commodity.apriori_data.loc[dtindex, 'soc'] - self.commodity.ph_data.loc[dtindex:chg_inxt,
                                                                        'consumption'].sum() * self.scenario.timestep_hours / self.commodity.size
            if chg_soc < 0.05:
                dep_soc = 0.8  # fast-charging only up to SOC of 80 %

                # Execute DC charging on-route
                self.commodity.apriori_data.loc[dtindex, 'p_ext_dc'] = self.charge(soc_target=dep_soc,
                                                                                   soc_current=self.commodity.apriori_data.loc[
                                                                                       dtindex, 'soc'],
                                                                                   p_maxchg=self.commodity.parent.ext_dc_power,
                                                                                   chg_eff=1)

    def charge(self, soc_target, soc_current, p_maxchg, chg_eff):
        soc_target = min(soc_target, 1)  # soc must not get bigger than 1
        e_tominsoc = max(0, (soc_target - soc_current) * self.commodity.size)  # energy to be recharged to departure SOC in Wh
        p_tominsoc = e_tominsoc / self.scenario.timestep_hours  # power to recharge to departure SOC in one step
        p_act = min(p_maxchg * chg_eff, p_tominsoc)  # reduce chg power in final step to just reach departure SOC
        return p_act
