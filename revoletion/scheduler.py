import numpy as np
import pandas as pd

from revoletion import blocks


def get_mode_scheduling(fleet_units: dict,
                        block: blocks.BaseBlock) -> str | None:
    mode_scheduling = list({fu.block.mode_scheduling for fu in fleet_units.values()})
    if len(mode_scheduling) > 1:
        raise ValueError(f'Fleet units in fleet "{block.name}" have different scheduling modes: '
                         f'{mode_scheduling}')
    elif len(mode_scheduling) == 1:
        return mode_scheduling[0]
    else:
        return None


class AprioriPowerScheduler:
    def __init__(self, scenario):
        self.scenario = scenario

        self.core = AprioriCore(block=self.scenario.block_registry.get('TopLevelBlock', {})['core'],
                                scheduler=self)

        pass





    def calc_ph_schedule(self,
                         horizon: 'PredictionHorizon') -> None:

        self.core.init_ph(horizon=horizon)

        for ts in horizon.dti_ph:
            self.core.simulate_ts(ts=ts,
                                  horizon=horizon)

        for fu in {**self.core.fu_uc, **self.core.fu_stat, **self.core.fu_dyn}.values():
            fu.write_power_to_flows_apriori(horizon=horizon)


class AprioriCore:
    def __init__(self,
                 block,
                 scheduler):
        self.block = block
        self.scheduler = scheduler
        self.scenario = self.scheduler.scenario

        self.p_sys_avail = pd.DataFrame(columns=['ac', 'dc'])
        self.p_conv_avail = pd.DataFrame(columns=['ac', 'dc'])
        self.p_sys_fix = pd.DataFrame(columns=['ac', 'dc'])

        self.fleets = {fleet.name: AprioriFleet(block=fleet,
                                                scheduler=self.scheduler)
                       for fleet in self.scenario.block_registry.get('Fleet', {}).values()}

        self.fu_uc = {k: v for fleet in self.fleets.values() for k, v in fleet.fu_uc.items()}
        self.fu_stat = {k: v for fleet in self.fleets.values() for k, v in fleet.fu_stat.items()}
        self.fu_dyn = {k: v for fleet in self.fleets.values() for k, v in fleet.fu_dyn.items()}

        self.mode_scheduling_dyn = get_mode_scheduling(fleet_units=self.fu_dyn,
                                                       block=self.block)

        if self.mode_scheduling_dyn == 'equal':
            raise ValueError('Fleet units with dynamic load management are not allowed to use scheduling mode "equal"')

    def init_ph(self,
                horizon: 'PredictionHorizon'):

        # initialize power availability (system and converter) and fixed power consumption
        self.p_sys_avail = self.p_sys_avail.reindex(horizon.dti_ph)
        self.p_sys_avail[:] = 0

        self.p_conv_avail = self.p_conv_avail.reindex(horizon.dti_ph)
        self.p_conv_avail['ac'] = self.block.sizes.loc['acdc', 'preexisting']
        self.p_conv_avail['dc'] = self.block.sizes.loc['dcac', 'preexisting']

        self.p_sys_fix = self.p_sys_avail.reindex(horizon.dti_ph)
        self.p_sys_fix[:] = 0

        # get power production and consumption for each non-fleet top level block
        for block in self.scenario.block_registry.get('TopLevelBlock', {}).values():
            if isinstance(block, blocks.GridConnection):
                self.p_sys_avail.loc[:, block.system] += block.sizes.loc['g2s', 'preexisting'] * block.eff['block']
            elif isinstance(block, blocks.RenewableSource):
                self.p_sys_avail.loc[:, block.system] += block.data.loc[horizon.dti_ph, 'power_spec'] * \
                                                         block.sizes.loc['block', 'preexisting'] * block.eff['block']
            elif isinstance(block, blocks.ControllableSource):
                self.p_sys_avail.loc[:, block.system] += block.sizes.loc['block', 'preexisting'] * block.eff['block']
            elif isinstance(block, blocks.FixedDemand):
                self.p_sys_fix.loc[:, block.system] += block.flows_apriori.loc[horizon.dti_ph, 'demand']

        for fleet in self.fleets.values():
            fleet.init_ph(horizon=horizon)

    def simulate_ts(self,
                    ts: pd.Timestamp,
                    horizon: 'PredictionHorizon') -> None:

        for fu in {**self.fu_uc, **self.fu_stat, **self.fu_dyn}.values():
            fu.calc_p_bat_chg_max(ts=ts)

        # calculate atbase charging

        # calculate UC charging
        for fu in self.fu_uc.values():
            fu.calc_p_chg_atbase(ts=ts,
                                 p_max_system=np.inf)

        # calculate charging for fleets with static load management
        for fleet in self.fleets.values():
            if fleet.lm == 'stat':
                fleet.charge_static_lm(ts=ts)

        if len(self.fu_dyn) > 0:
            # add the charging power already assigned to fleet units to the fixed power consumption
            for fleet in self.fleets.values():
                self.p_sys_fix.loc[ts, fleet.block.system] += fleet.p_fix.loc[ts]

            # calculate new power availability for the system
            for system in ['ac', 'dc']:
                self._subtract_demand(ts=ts,
                                      system=system,
                                      p_demand=self.p_sys_fix.loc[ts, system])

            sort_key_funcs = {
                'fcfs': lambda x: x.arr_base_dti[x.arr_base_dti <= ts].max(),
                'soc': lambda x: x.data_battery.loc[ts, 'soc']
            }

            # sort the fleet units based on the scheduling mode
            for fu in sorted(self.fu_dyn.values(), key=sort_key_funcs[self.mode_scheduling_dyn]):
                p_chg = fu.calc_p_chg_atbase(ts=ts,
                                             p_max_system=self._get_p_avail(ts=ts,
                                                                            system=fu.fleet.block.system))

                self._subtract_demand(ts=ts,
                                      system=fu.fleet.block.system,
                                      p_demand=p_chg)

        for fu in {**self.fu_uc, **self.fu_stat, **self.fu_dyn}.values():
            fu.calc_p_chg_external(ts=ts,
                                   horizon=horizon)
            fu.calc_soc(ts=ts)

        pass

    def _get_p_avail(self, ts: pd.Timestamp,
                     system: str):
        """
        get maximum available power at a bus "system" at a timestamp "ts"
        """
        bus_same = self._get_bus(system, 'same')
        bus_other = self._get_bus(system, 'other')

        p_bus_same = self.p_sys_avail.loc[ts, bus_same]
        p_bus_other = (min(self.p_sys_avail.loc[ts, bus_other],
                           self.p_conv_avail.loc[ts, bus_other]) *
                       self._get_conv_eff(source=bus_other,
                                          target=bus_same)
                       )

        return p_bus_same + p_bus_other

    def _subtract_demand(self,
                         ts: pd.Timestamp,
                         system: str,
                         p_demand: float) -> None:
        """
        subtract power demand "p_demand" from the available power at a bus "system" at a timestamp "ts"
        consider produced power as well as limitations caused by the SystemCore's converter capacity
        """

        bus_same = self._get_bus(system, 'same')
        bus_other = self._get_bus(system, 'other')

        # calculate maximum power which can be drawn from the bus the fleet unit is connected to
        p_bus_same = min(self.p_sys_avail.loc[ts, system], p_demand)
        self.p_sys_avail.loc[ts, bus_same] -= p_bus_same

        # draw power exceeding the maximum power on the connected bus from other bus
        p_bus_other = (p_demand - p_bus_same) / self._get_conv_eff(source=bus_same,
                                                                   target=bus_other)
        if p_bus_other <= self.p_sys_avail.loc[ts, bus_other] and p_bus_other <= self.p_conv_avail.loc[ts, bus_other]:
            self.p_sys_avail.loc[ts, bus_other] -= p_bus_other
            self.p_conv_avail.loc[ts, bus_other] -= p_bus_other
        else:
            raise ValueError(f'Power limit of {self.block.name}\'s {system.upper()}/'
                             f'{self._get_bus(system, "other").upper()} converter exceeded at {ts}!')

    @staticmethod
    def _get_bus(bus: str,
                 target: str) -> str:

        # Takes the bus the block is connected to and returns the specified bus
        return {'ac': {'same': 'ac', 'other': 'dc'},
                'dc': {'same': 'dc', 'other': 'ac'}}[bus][target]

    def _get_conv_eff(self, source, target):
        return {'ac': {'ac': 1,
                       'dc': self.block.eff['acdc']},
                'dc': {'ac': self.block.eff['dcac'],
                       'dc': 1}}[source][target]


class AprioriFleet:
    def __init__(self,
                 block: blocks.BaseBlock,
                 scheduler: AprioriPowerScheduler):
        self.block=block
        self.scheduler=scheduler
        self.scenario=self.scheduler.scenario

        self.lm = 'stat' if pd.notna(self.block.sizes.loc['s2f', 'preexisting']) else 'dyn'

        self.fleet_units = {fu_name: AprioriFleetUnit(block=fu_block,
                                                      fleet=self,
                                                      scheduler=self.scheduler)
                            for subfleet in self.block.subblocks.values()
                            for fu_name, fu_block in subfleet.subblocks.items()
                            if (isinstance(fu_block, blocks.ElectricFleetUnit) and
                                fu_block.mode_scheduling in self.scenario.run.apriori_lvls)}

        self.fu_uc = {fu_name: fu_block
                      for fu_name, fu_block in self.fleet_units.items()
                      if fu_block.block.mode_scheduling == 'uc'}

        self.fu_stat = {fu_name: fu_block
                        for fu_name, fu_block in self.fleet_units.items()
                        if fu_block.block.mode_scheduling in self.scenario.run.apriori_lvls and
                        fu_block.block.mode_scheduling != 'uc' and
                        self.lm == 'stat'}

        self.mode_scheduling_stat = get_mode_scheduling(fleet_units=self.fu_stat,
                                                        block=self.block)

        self.fu_dyn = {fu_name: fu_block
                       for fu_name, fu_block in self.fleet_units.items()
                       if fu_block.block.mode_scheduling in self.scenario.run.apriori_lvls and
                       fu_block.block.mode_scheduling != 'uc' and
                       self.lm == 'dyn'}

        self.p_avail = pd.Series()
        self.p_fix = pd.Series()

    def init_ph(self,
                horizon: 'PredictionHorizon'):

        # initialize power availability and fixed power consumption
        self.p_avail = self.p_avail.reindex(horizon.dti_ph)
        self.p_avail[:] = np.inf if self.lm == 'dyn' else self.block.sizes.loc['s2f', 'preexisting']

        self.p_fix = self.p_fix.reindex(horizon.dti_ph)
        self.p_fix[:] = 0

        for fu in self.fleet_units.values():
            fu.init_ph(horizon=horizon)

    def charge_static_lm(self,
                         ts: pd.Timestamp) -> None:
        if len(self.fu_stat) == 0:
            return
        if self.mode_scheduling_stat == 'equal':
            # create list of fleet_units
            fu_to_charge = list(self.fu_stat.values())

            # iterate of all fleet units until available power is  charging demand is satisfied
            while self.p_fix.loc[ts] < self.p_avail.loc[ts] and len(fu_to_charge) > 0:
                p_max_per_fu = (self.p_avail.loc[ts] - self.p_fix.loc[ts]) / len(fu_to_charge)
                for fu in fu_to_charge:
                    p_chg_fu = fu.calc_p_chg_atbase(ts=ts,
                                                    p_max_system=p_max_per_fu)
                    # if fu doesn't charge (not atbase, reached target soc, reached charging power limit)
                    if p_chg_fu == 0:
                        fu_to_charge.remove(fu)

        else:
            sort_key_funcs = {
                'fcfs': lambda x: x.arr_base_dti[x.arr_base_dti <= ts].max(),
                'soc': lambda x: x.data_battery.loc[ts, 'soc']
            }

            # sort the fleet units based on the scheduling mode
            fu_prio_list = sorted(self.fu_stat.values(), key=sort_key_funcs[self.mode_scheduling_stat])

            for fu in fu_prio_list:
                fu.calc_p_chg_atbase(ts=ts,
                                     p_max_system=self.p_avail.loc[ts] - self.p_fix.loc[ts])


class AprioriFleetUnit:
    def __init__(self,
                 block,
                 fleet,
                 scheduler):
        self.block = block
        self.scenario = self.block.scenario
        self.scheduler = scheduler

        self.fleet = fleet

        # states and powers measured at the battery
        self.data_battery = pd.DataFrame(columns=['p_consumption', 'p_sd', 'p_max', 'p_chg', 'soc', 'soc_target'],
                                         dtype='float64')

        # powers measured at the bus connection / connection to external charger
        self.data_charging = pd.DataFrame(columns=['p_int', 'p_ext_ac', 'p_ext_dc'],
                                          dtype='float64')

        self.soh = None

        # get the indices of all nonzero target soc rows in the data
        self.dsoc_dti = self.block.log.index[self.block.log['dsoc'] != 0]

        # get first timesteps, where vehicle has left the base
        self.dep_base_dti = self.block.log.index[~self.block.log['atbase'] &
                                                 self.block.log['atbase'].shift(periods=1, fill_value=False)]

        # get first timesteps, where vehicle is at base again
        self.arr_base_dti = self.block.log.index[self.block.log['atbase'] &
                                                 ~self.block.log['atbase'].shift(periods=1, fill_value=False)]

        # get first timesteps, where vehicle has left the destination
        self.dep_dest_dti = self.block.log.index[~self.block.log['atac'] &
                                                 self.block.log['atac'].shift(periods=1, fill_value=False)]

        # get first timesteps, where vehicle is parking at destination
        self.arr_dest_dti = self.block.log.index[self.block.log['atac'] &
                                                 ~self.block.log['atac'].shift(periods=1, fill_value=False)]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        self.chg_avail_dti = self.block.log.index[self.block.log[['atbase', 'atac', 'atdc']].any(axis=1)]

        # initialize variable for charging during single parking process
        self.parking_charging = False

    def init_ph(self,
                horizon: 'PredictionHorizon'):

        # apply new index to data
        self.data_battery = self.data_battery.reindex(horizon.dti_ph)
        self.data_battery[:] = 0

        # add power consumption to data
        self.data_battery.loc[:, 'p_consumption'] = (-1 * self.block.log.loc[horizon.dti_ph, 'consumption'] /
                                                     self._get_eff('consumption')).astype('float64')

        # get current SOC
        self.data_battery.loc[horizon.dti_ph.min(), 'soc'] = self.block.states.loc[horizon.starttime, ['soc', 'soc_min', 'soc_max']].median()

        self.data_charging = self.data_charging.reindex(horizon.dti_ph)
        self.data_charging[:] = 0

        # get soh for current prediction horizon
        self.soh = self.block.states.loc[horizon.starttime, 'soh']

    def calc_soc_target(self,
                        ts: pd.Timestamp) -> float:
        # ToDo: add input parameter to specify target SOCs
        if self.block.log.loc[ts, 'atdc']:
            return 0.8

        soc_target_low = min(0.8, self.block.states.loc[ts, 'soc_max'])
        soc_target_high = min(1.0, self.block.states.loc[ts, 'soc_max'])

        # check if there are any departures after current timestep within forecast period
        departures = self.dep_base_dti[(self.dep_base_dti >= ts) &
                                       (self.dep_base_dti <= ts + pd.Timedelta(hours=self.block.forecast_hours)
                                        if self.block.forecast_hours else True)]

        arrivals = self.arr_base_dti[self.arr_base_dti >= ts]

        if departures.empty or arrivals.empty:
            return soc_target_low

        #  sum up energy between trip start and end
        arr_nxt = arrivals.min()
        dep_nxt = departures.min()
        if arr_nxt <= dep_nxt:  # trip currently ongoing
            # Destination charging -> sum up remaining energy of ongoing trip until end of trip
            e_con = (-1) * (self.data_battery.loc[ts:arr_nxt - self.scenario.timestep_td, 'p_consumption'].sum()
                            * self.scenario.timestep_hours)
        else:  # vehicle currently rechargeable at base
            e_con = (-1) * (self.data_battery.loc[dep_nxt:arr_nxt - self.scenario.timestep_td, 'p_consumption'].sum()
                            * self.scenario.timestep_hours)

        #  Convert energy consumption to delta soc taking the current soh into account
        soc_delta = e_con / self.block.sizes.loc['storage', 'preexisting']
        #  Set soc_target dependent on soc_delta of trip and settings of the MobileCommodity
        if soc_delta > (soc_target_low - self.block.soc_return):
            soc_target = soc_target_high
        else:
            soc_target = soc_target_low
        return soc_target

    def calc_p_bat_chg_max(self,
                           ts: pd.Timestamp) -> None:
        """
        calculate the required charging power at the battery in the current timestep to reach the specified
        target SOC (soc_target) within the current timestep
        """

        self.data_battery.loc[ts, 'soc_target'] = self.calc_soc_target(ts=ts)

        # calculate current energy content of battery
        e_bat = self.data_battery.loc[ts, 'soc'] * self.block.sizes.loc['storage', 'preexisting']

        # calculate self discharge power in current timestep based on the current energy content
        self.data_battery.loc[ts, 'p_sd'] = -1 * e_bat * self.block.loss_rate_per_ts / self.scenario.timestep_hours

        # calculate target energy content of battery
        e_target = self.data_battery.loc[ts, 'soc_target'] * self.block.sizes.loc['storage', 'preexisting']

        # calculate maximum charging power at battery (avoid p_max < 0 caused by changing soc_target)
        self.data_battery.loc[ts, 'p_max'] = max(((e_target - e_bat) / self.scenario.timestep_hours +
                                                  (-1) * self.data_battery.loc[ts, 'p_sd'] +
                                                  (-1) * self.data_battery.loc[ts, 'p_consumption']), 0)

    def calc_p_chg_atbase(self,
                          ts: pd.Timestamp,
                          p_max_system: float) -> float:
        """
        calculate the charging power at the base in the current timestep with respect to the maximum power provided
        by the local energy system's load management (static or dynamic) if applicable
        """
        # calculate charging power at base (measurement point at connection to SystemCore bus)
        p_max_fleet_unit_battery = (self.data_battery.loc[ts, 'p_max'] -
                                    self.data_battery.loc[ts, 'p_chg']) / self._get_eff(self.fleet.block.system)
        p_max_fleet_unit_connection = self.block.pwr_chg_max * self.block.log.loc[ts, 'atbase']

        p_chg_atbase = min(p_max_fleet_unit_battery,
                           p_max_fleet_unit_connection,
                           p_max_system)

        self.data_charging.loc[ts, 'p_int'] += p_chg_atbase

        # calculate charging power observed at battery
        self.data_battery.loc[ts, 'p_chg'] += p_chg_atbase * self._get_eff(self.fleet.block.system)

        # write charging power to fleet
        self.fleet.p_fix[ts] += p_chg_atbase

        if self.fleet.lm == 'stat' and self.fleet.p_fix[ts] > self.fleet.p_avail[ts]:
            raise ValueError(f'Power limit of {self.fleet.block.name}\'s static load management exceeded at {ts}!')

        return p_chg_atbase


    def calc_p_chg_external(self,
                            ts: pd.Timestamp,
                            horizon: 'PredictionHorizon'):
        """
        calculate charging power from external sources
        """

        # determine whether destination charging is necessary
        if self.block.log.loc[ts, 'atac'] == 1:
            if ts in self.arr_dest_dti:  # plugging in only happens when parking starts
                # calculate all upcoming arrival times at the base
                arrivals = self.arr_base_dti[self.arr_base_dti >= ts]
                # use current time and next arrival index to calculate consumption and convert to SOC
                arr_nxt = arrivals.min() if not arrivals.empty else horizon.dti_ph.max()
                e_trip_remaining = (-1) * self.data_battery.loc[ts:arr_nxt, 'p_consumption'].sum() * self.scenario.timestep_hours

                # set charging to True, if charging is necessary
                if e_trip_remaining > ((self.data_battery.loc[ts, 'soc'] - self.block.soc_return) *
                                       self.block.sizes.loc['storage', 'preexisting']):  # ToDo: add soh/aging
                    self.parking_charging = True
                else:
                    self.parking_charging = False

            if self.parking_charging is True:
                # calculate charging power at external AC charger (measurement point at connection to charger)
                p_max_fleet_unit_battery = self.data_battery.loc[ts, 'p_max'] / self._get_eff('ac')
                p_max_fleet_unit_connection = self.block.pwr_ext_ac_max

                self.data_charging.loc[ts, 'p_ext_ac'] += min(p_max_fleet_unit_battery,
                                                           p_max_fleet_unit_connection)

                # calculate charging power observed at battery
                self.data_battery.loc[ts, 'p_chg'] += self.data_charging.loc[ts, 'p_ext_ac'] * self._get_eff('ac')

        # determine whether on-route charging is necessary
        elif self.block.log.loc[ts, 'atdc'] == 1:
            # activate charging, if SOC will fall below threshold, before next possibility to charge
            chg_nxt = self.chg_avail_dti[self.chg_avail_dti > ts].min()
            soc_chg_nxt = (self.data_battery.loc[ts, 'soc'] -
                          (-1) * self.data_battery.loc[ts:chg_nxt - self.scenario.timestep_td, 'p_consumption'].sum() *
                           self.scenario.timestep_hours / self.block.sizes.loc['storage', 'preexisting'])
            # ToDo: add soh/aging: if soc_chg_nxt < self.convert_soc_ui2internal(0.05):
            if soc_chg_nxt < 0.05:
                # calculate charging power at external DC charger (measurement point at connection to charger)
                p_max_fleet_unit_battery = self.data_battery.loc[ts, 'p_max'] / self._get_eff('dc')
                p_max_fleet_unit_connection = self.block.pwr_ext_dc_max

                self.data_charging.loc[ts, 'p_ext_dc'] += min(p_max_fleet_unit_battery,
                                                              p_max_fleet_unit_connection)

                # calculate charging power observed at battery
                self.data_battery.loc[ts, 'p_chg'] += self.data_charging.loc[ts, 'p_ext_dc'] * self._get_eff('dc')

    def calc_soc(self,
                 ts: pd.Timestamp):
        # calculate state of charge based on calculated charging powers, consumption and self discharge
        soc_delta = (self.data_battery.loc[ts, ['p_consumption', 'p_sd', 'p_chg']].sum() *
                     self.scenario.timestep_hours / self.block.sizes.loc['storage', 'preexisting'])

        self.data_battery.loc[(ts + self.scenario.timestep_td), 'soc'] = self.data_battery.loc[ts, 'soc'] + soc_delta

    def write_power_to_flows_apriori(self,
                                     horizon: 'PredictionHorizon') -> None:

        self.block.flows_apriori.update({'p_int_chg': (self.data_charging['p_int'].clip(lower=0) /
                                                          self.block.pwr_chg_max),
                                         'p_int_dis': ((-1) * self.data_charging['p_int'].clip(upper=0) /
                                                          self.block.pwr_dis_max),
                                         'p_ext_ac_chg': (self.data_charging['p_ext_ac'].clip(lower=0) /
                                                          self.block.pwr_ext_ac_max),
                                         'p_ext_ac_dis': ((-1) * self.data_charging['p_ext_ac'].clip(upper=0) /
                                                          self.block.pwr_ext_ac_max),
                                         'p_ext_dc_chg': (self.data_charging['p_ext_dc'].clip(lower=0) /
                                                          self.block.pwr_ext_dc_max),
                                         'p_ext_dc_dis': ((-1) * self.data_charging['p_ext_dc'].clip(upper=0) /
                                                          self.block.pwr_ext_dc_max),
                                         })

        # fix NaN values caused by max_power = 0 leading and therefore division by 0
        self.block.flows_apriori.loc[horizon.dti_ph, :] = self.block.flows_apriori.loc[horizon.dti_ph, :].fillna(0.0)

    def _get_eff(self,
                 mode: str):
        # get charging efficiency for the selected mode: ac, dc, consumption
        if mode not in ['ac', 'dc', 'consumption']:
            raise ValueError(f'Invalid mode "{mode}" selected. Valid modes are "ac", "dc" and "consumption')

        eff = {'ac': self.block.eff['chg_ac'] * np.sqrt(self.block.eff['storage_roundtrip']),
               'dc': self.block.eff['chg_dc'] * np.sqrt(self.block.eff['storage_roundtrip']),
               'consumption': np.sqrt(self.block.eff['storage_roundtrip'])}[mode]
        return eff