import numpy as np
import pandas as pd
import statistics

from revoletion import blocks


class AprioriPowerScheduler:
    def __init__(self, scenario):
        self.scenario = scenario

        # get all ElectricFleetUnits with apriori charging scheduling (uc, fcfs, equal, soc)
        self.fleet_units = {fu_name: AprioriFleetUnit(block=fu_block,
                                                      scheduler=self)
                            for fleet in self.scenario.fleets.values()
                            for subfleet in fleet.subblocks.values()
                            for fu_name, fu_block in subfleet.subblocks.items()
                            if (isinstance(fu_block, blocks.ElectricFleetUnit) and
                                fu_block.mode_scheduling in self.scenario.run.apriori_lvls)}

    def calc_ph_schedule(self,
                         horizon: 'PredictionHorizon') -> None:
        for fu in self.fleet_units.values():
            fu.init_ph(horizon=horizon)

        for ts in horizon.dti_ph:
            for fu in self.fleet_units.values():
                fu.calc_p_bat_chg_max(ts=ts,
                                      soc_target=1.0)
                fu.calc_p_chg_atbase(ts=ts)
                fu.calc_p_chg_external(ts=ts,
                                       horizon=horizon)
                fu.calc_soc(ts=ts)

        for fu in self.fleet_units.values():
            fu.write_power_to_flows_apriori(horizon=horizon)


class AprioriFleetUnit:
    def __init__(self,
                 block,
                 scheduler):
        self.block = block
        self.scenario = self.block.scenario
        self.scheduler = scheduler

        self.fleet = self.block.parent.parent

        # states and powers measured at the battery
        self.data_battery = pd.DataFrame(columns=['p_consumption', 'p_sd', 'p_max', 'p_chg', 'soc'],
                                         dtype='float64')

        # powers measured at the bus connection / connection to external charger
        self.data_charging = pd.DataFrame(columns=['p_int', 'p_ext_ac', 'p_ext_dc'],
                                          dtype='float64')

        self.soh = None

        # get the indices of all nonzero target soc rows in the data
        self.dsoc_dti = self.block.log.index[self.block.log['dsoc'] != 0]

        # ToDo: check whether "1" in shift() is necessary
        # get first timesteps, where vehicle has left the base
        self.dep_base_dti = self.block.log.index[~self.block.log['atbase'] & self.block.log['atbase'].shift(1, fill_value=False)]

        # get first timesteps, where vehicle is at base again
        self.arr_base_dti = self.block.log.index[self.block.log['atbase'] & ~self.block.log['atbase'].shift(fill_value=False)]

        # get first timesteps, where vehicle has left the destination
        self.dep_dest_dti = self.block.log.index[~self.block.log['atac'] & self.block.log['atac'].shift(1, fill_value=False)]

        # get first timesteps, where vehicle is parking at destination
        self.arr_dest_dti = self.block.log.index[self.block.log['atac'] & ~self.block.log['atac'].shift(fill_value=False)]

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
        self.data_battery.loc[horizon.dti_ph.min(), 'soc'] = statistics.median(
            [self.block.soc_min,
             self.block.states.loc[horizon.starttime, 'soc'],
             self.block.soc_max])

        self.data_charging = self.data_charging.reindex(horizon.dti_ph)
        self.data_charging[:] = 0

        # get soh for current prediction horizon
        self.soh = self.block.states.loc[horizon.starttime, 'soh']

    def calc_p_bat_chg_max(self,
                           ts: pd.Timestamp,
                           soc_target: float = 1.0) -> None:
        """
        calculate the required charging power at the battery in the current timestep to reach the specified
        target SOC (soc_target) within the current timestep
        """

        # calculate current energy content of battery
        e_bat = self.data_battery.loc[ts, 'soc'] * self.block.sizes.loc['block', 'preexisting']

        # calculate self discharge power in current timestep based on the current energy content
        self.data_battery.loc[ts, 'p_sd'] = -1 * e_bat * self.block.loss_rate_per_ts / self.scenario.timestep_hours

        # calculate target energy content of battery
        e_target = soc_target * self.block.sizes.loc['block', 'preexisting']

        # calculate maximum charging power at battery (avoid p_max < 0 caused by changing soc_target)
        self.data_battery.loc[ts, 'p_max'] = max(((e_target - e_bat) / self.scenario.timestep_hours +
                                                  (-1) * self.data_battery.loc[ts, 'p_sd'] +
                                                  (-1) * self.data_battery.loc[ts, 'p_consumption']), 0)

    def calc_p_chg_atbase(self,
                          ts: pd.Timestamp):
        """
        calculate the charging power at the base in the current timestep with respect to the maximum power provided
        by the local energy system's load management (static or dynamic) if applicable
        """
        # calculate charging power at base (measurement point at connection to SystemCore bus)
        p_max_fleet_unit_battery = self.data_battery.loc[ts, 'p_max'] / self._get_eff(self.fleet.system)
        p_max_fleet_unit_connection = self.block.pwr_chg_max * self.block.log.loc[ts, 'atbase']
        p_max_system = np.inf  # ToDo: for "uc" only -> adjust for other mode_scheduling options

        self.data_charging.loc[ts, 'p_int'] = min(p_max_fleet_unit_battery,
                                                  p_max_fleet_unit_connection,
                                                  p_max_system)

        # calculate charging power observed at battery
        self.data_battery.loc[ts, 'p_chg'] = self.data_charging.loc[ts, 'p_int'] * self._get_eff(self.fleet.system)

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
                                       self.block.sizes.loc['block', 'preexisting']):  # ToDo: add soh/aging
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
                           self.scenario.timestep_hours / self.block.sizes.loc['block', 'preexisting'])
            # ToDo: add soh/aging: if soc_chg_nxt < self.convert_soc_ui2internal(0.05):
            if soc_chg_nxt < 0.05:
                # ToDo: fast-charging only up to SOC of 80 %
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
                     self.scenario.timestep_hours / self.block.sizes.loc['block', 'preexisting'])

        self.data_battery.loc[(ts + self.scenario.timestep_td), 'soc'] = self.data_battery.loc[ts, 'soc'] + soc_delta

    def write_power_to_flows_apriori(self,
                                     horizon: 'PredictionHorizon') -> None:
        cols2transfer = ['p_int', 'p_ext_ac', 'p_ext_dc']
        self.block.flows_apriori[cols2transfer] = self.data_charging[cols2transfer]

    def _get_eff(self,
                 mode: str):
        # get charging efficiency for the selected mode: ac, dc, consumption
        if mode not in ['ac', 'dc', 'consumption']:
            raise ValueError(f'Invalid mode "{mode}" selected. Valid modes are "ac", "dc" and "consumption')

        eff = {'ac': self.block.eff_chg_ac * np.sqrt(self.block.eff_storage_roundtrip),
               'dc': self.block.eff_chg_dc * np.sqrt(self.block.eff_storage_roundtrip),
               'consumption': np.sqrt(self.block.eff_storage_roundtrip)}[mode]
        return eff