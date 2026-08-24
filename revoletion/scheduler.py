#!/usr/bin/env python3

import numpy as np
import pandas as pd

from . import blocks, time

_SYSTEMS = ("ac", "dc")
_SYS_IDX = {"ac": 0, "dc": 1}

# Timestamps are handled as int64 nanoseconds inside the timestep loop: scalar indexing of a
# DatetimeIndex boxes a Timestamp and costs more than the searchsorted it feeds.
# Fleet units without a recorded arrival at base rank first under "fcfs": they have been waiting
# at the base since before the simulation started, so they came first.
_NEVER_ARRIVED = np.iinfo(np.int64).min


def get_mode_scheduling(fleet_units: dict, block: blocks.BaseBlock) -> str | None:
    mode_scheduling = list({fu.block.mode_scheduling for fu in fleet_units.values()})
    if len(mode_scheduling) > 1:
        raise ValueError(f'Fleet units in fleet "{block.name}" have different scheduling modes: {mode_scheduling}')
    elif len(mode_scheduling) == 1:
        return mode_scheduling[0]
    else:
        return None


def get_sort_key_func(mode_scheduling: str, ts_i8: int, idx: int):
    """
    Build the priority key used to order fleet units within one timestep.

    "fcfs" ranks by the last arrival at base at or before the current timestep, earliest first.
    Fleet units without a recorded arrival yet are ranked first via ``_NEVER_ARRIVED``. They
    previously produced ``NaT``, which compares ``False`` in both directions and therefore made
    the key an inconsistent ordering: a single ``NaT`` could also push two properly arrived fleet
    units into the wrong relative order, because the sort never compares them against each other.
    """
    if mode_scheduling == "fcfs":

        def key(fu: "AprioriFleetUnit") -> int:
            pos = np.searchsorted(fu.arr_base_i8, ts_i8, side="right")
            return fu.arr_base_i8[pos - 1] if pos else _NEVER_ARRIVED

        return key
    elif mode_scheduling == "soc":
        return lambda fu: fu.soc[idx]
    else:
        raise ValueError(f'No sorting defined for scheduling mode "{mode_scheduling}"')


def get_resort_mask(mode_scheduling: str | None, fleet_units: dict, dti: pd.DatetimeIndex) -> np.ndarray:
    """
    Determine the timesteps at which the priority order of ``fleet_units`` has to be rebuilt.

    Under "fcfs" the order only changes when one of the fleet units arrives at the base, so the
    sorted list can be reused in between. Under "soc" the keys change in every timestep.
    """
    if mode_scheduling == "fcfs":
        mask = np.zeros(len(dti), dtype=bool)
        mask[0] = True
        for fu in fleet_units.values():
            mask |= dti.isin(fu.arr_base_dti)
        return mask
    return np.ones(len(dti), dtype=bool)


class AprioriPowerScheduler:
    def __init__(self, scenario):
        self.scenario = scenario

        self.core = AprioriCore(block=self.scenario.block_registry.get("TopLevelBlock", {})["core"], scheduler=self)

    def calc_ph_schedule(self, horizon: time.TimeFrame) -> None:
        self.core.init_ph(horizon=horizon)

        for idx, ts in enumerate(horizon.dti):
            self.core.simulate_ts(idx=idx, ts=ts)

        self.core.finalize_ph(horizon=horizon)

        for fu in {**self.core.fu_uc, **self.core.fu_stat, **self.core.fu_dyn}.values():
            fu.write_power_to_flows_apriori(horizon=horizon)


class AprioriCore:
    def __init__(self, block, scheduler):
        self.block = block
        self.scheduler = scheduler
        self.scenario = self.scheduler.scenario

        self.p_sys_avail = pd.DataFrame(columns=["ac", "dc"])
        self.p_conv_avail = pd.DataFrame(columns=["ac", "dc"])
        self.p_sys_fix = pd.DataFrame(columns=["ac", "dc"])

        # per-horizon working copies of the frames above, shape (n_timesteps, 2), see _SYS_IDX
        self._p_sys_avail = np.empty((0, 2))
        self._p_conv_avail = np.empty((0, 2))
        self._p_sys_fix = np.empty((0, 2))

        self.fleets = {
            fleet.name: AprioriFleet(block=fleet, scheduler=self.scheduler)
            for fleet in self.scenario.block_registry.get("Fleet", {}).values()
        }

        self.fu_uc = {k: v for fleet in self.fleets.values() for k, v in fleet.fu_uc.items()}
        self.fu_stat = {k: v for fleet in self.fleets.values() for k, v in fleet.fu_stat.items()}
        self.fu_dyn = {k: v for fleet in self.fleets.values() for k, v in fleet.fu_dyn.items()}

        self.fu_all = {**self.fu_uc, **self.fu_stat, **self.fu_dyn}

        self.mode_scheduling_dyn = get_mode_scheduling(fleet_units=self.fu_dyn, block=self.block)

        if self.mode_scheduling_dyn == "equal":
            raise ValueError('Fleet units with dynamic load management are not allowed to use scheduling mode "equal"')

        self._resort_dyn = np.empty(0, dtype=bool)
        self._order_dyn = []

    def init_ph(self, horizon: time.TimeFrame):
        n_ts = len(horizon.dti)

        # initialize power availability (system and converter) and fixed power consumption
        self._p_sys_avail = np.zeros((n_ts, 2))

        self._p_conv_avail = np.empty((n_ts, 2))
        self._p_conv_avail[:, _SYS_IDX["ac"]] = self.block.sizes["acdc"].preexisting
        self._p_conv_avail[:, _SYS_IDX["dc"]] = self.block.sizes["dcac"].preexisting

        self._p_sys_fix = np.zeros((n_ts, 2))

        # get power production and consumption for each non-fleet top level block
        for block in self.scenario.block_registry.get("TopLevelBlock", {}).values():
            if isinstance(block, blocks.GridConnection):
                self._p_sys_avail[:, _SYS_IDX[block.system]] += block.sizes["g2s"].preexisting * block.eff["block"]
            elif isinstance(block, blocks.RenewableSource):
                self._p_sys_avail[:, _SYS_IDX[block.system]] += (
                    block.data.loc[horizon.dti, "power_spec"].to_numpy(dtype="float64")
                    * block.sizes["block"].preexisting
                    * block.eff["block"]
                )
            elif isinstance(block, blocks.ControllableSource):
                self._p_sys_avail[:, _SYS_IDX[block.system]] += block.sizes["block"].preexisting * block.eff["block"]
            elif isinstance(block, blocks.FixedDemand):
                self._p_sys_fix[:, _SYS_IDX[block.system]] += block.flows_apriori.loc[horizon.dti, "demand"].to_numpy(
                    dtype="float64"
                )

        for fleet in self.fleets.values():
            fleet.init_ph(horizon=horizon)

        # precompute the timesteps at which the dynamic-load-management priority order can change
        self._dti_i8 = horizon.dti.as_unit("ns").asi8
        self._resort_dyn = get_resort_mask(self.mode_scheduling_dyn, self.fu_dyn, horizon.dti)
        self._order_dyn = list(self.fu_dyn.values())

    def finalize_ph(self, horizon: time.TimeFrame) -> None:
        """Rebuild the timestamp-indexed views from the per-horizon working arrays."""
        self.p_sys_avail = pd.DataFrame(self._p_sys_avail, index=horizon.dti, columns=list(_SYSTEMS))
        self.p_conv_avail = pd.DataFrame(self._p_conv_avail, index=horizon.dti, columns=list(_SYSTEMS))
        self.p_sys_fix = pd.DataFrame(self._p_sys_fix, index=horizon.dti, columns=list(_SYSTEMS))

        for fleet in self.fleets.values():
            fleet.finalize_ph(horizon=horizon)

    def simulate_ts(self, idx: int, ts: pd.Timestamp) -> None:
        for fu in self.fu_all.values():
            fu.calc_p_bat_chg_max(idx=idx)

        # calculate atbase charging

        # calculate UC charging
        for fu in self.fu_uc.values():
            fu.calc_p_chg_atbase(idx=idx, ts=ts, p_max_system=np.inf)

        # calculate charging for fleets with static load management
        for fleet in self.fleets.values():
            if fleet.lm == "stat":
                fleet.charge_static_lm(idx=idx, ts=ts)

        if len(self.fu_dyn) > 0:
            # add the charging power already assigned to fleet units to the fixed power consumption
            for fleet in self.fleets.values():
                self._p_sys_fix[idx, _SYS_IDX[fleet.block.system]] += fleet._p_fix[idx]

            # calculate new power availability for the system
            for system in _SYSTEMS:
                self._subtract_demand(idx=idx, ts=ts, system=system, p_demand=self._p_sys_fix[idx, _SYS_IDX[system]])

            # sort the fleet units based on the scheduling mode (only when the order can change)
            if self._resort_dyn[idx]:
                self._order_dyn = sorted(
                    self.fu_dyn.values(),
                    key=get_sort_key_func(self.mode_scheduling_dyn, ts_i8=self._dti_i8[idx], idx=idx),
                )

            for fu in self._order_dyn:
                p_chg = fu.calc_p_chg_atbase(
                    idx=idx, ts=ts, p_max_system=self._get_p_avail(idx=idx, system=fu.fleet.block.system)
                )

                self._subtract_demand(idx=idx, ts=ts, system=fu.fleet.block.system, p_demand=p_chg)

        for fu in self.fu_all.values():
            fu.calc_p_chg_external(idx=idx)
            fu.calc_soc(idx=idx)

    def _get_p_avail(self, idx: int, system: str) -> float:
        """
        get maximum available power at a bus "system" at horizon position "idx"
        """
        i_same = _SYS_IDX[system]
        i_other = 1 - i_same

        p_bus_same = self._p_sys_avail[idx, i_same]
        p_bus_other = min(self._p_sys_avail[idx, i_other], self._p_conv_avail[idx, i_other]) * self._get_conv_eff(
            source=_SYSTEMS[i_other], target=system
        )

        return p_bus_same + p_bus_other

    def _subtract_demand(self, idx: int, ts: pd.Timestamp, system: str, p_demand: float) -> None:
        """
        subtract power demand "p_demand" from the available power at a bus "system" at horizon position "idx"
        consider produced power as well as limitations caused by the SystemCore's converter capacity
        """

        i_same = _SYS_IDX[system]
        i_other = 1 - i_same

        # calculate maximum power which can be drawn from the bus the fleet unit is connected to
        p_bus_same = min(self._p_sys_avail[idx, i_same], p_demand)
        self._p_sys_avail[idx, i_same] -= p_bus_same

        # draw power exceeding the maximum power on the connected bus from other bus
        p_bus_other = (p_demand - p_bus_same) / self._get_conv_eff(source=system, target=_SYSTEMS[i_other])
        if p_bus_other <= self._p_sys_avail[idx, i_other] and p_bus_other <= self._p_conv_avail[idx, i_other]:
            self._p_sys_avail[idx, i_other] -= p_bus_other
            self._p_conv_avail[idx, i_other] -= p_bus_other
        else:
            raise ValueError(
                f"Power limit of {self.block.name}'s {system.upper()}/"
                f"{_SYSTEMS[i_other].upper()} converter exceeded at {ts}!"
            )

    def _get_conv_eff(self, source, target):
        return {"ac": {"ac": 1, "dc": self.block.eff["acdc"]}, "dc": {"ac": self.block.eff["dcac"], "dc": 1}}[source][
            target
        ]


class AprioriFleet:
    def __init__(self, block: blocks.BaseBlock, scheduler: AprioriPowerScheduler):
        self.block = block
        self.scheduler = scheduler
        self.scenario = self.scheduler.scenario

        self.lm = "stat" if pd.notna(self.block.pwr_lim_s2f) else "dyn"

        self.fleet_units = {
            fu_name: AprioriFleetUnit(block=fu_block, fleet=self, scheduler=self.scheduler)
            for fu_name, fu_block in self.scenario.block_registry["ElectricFleetUnit"].items()
            if (fu_block.parent.parent == self.block and fu_block.mode_scheduling in self.scenario.apriori_lvls)
        }

        self.fu_uc = {
            fu_name: fu_block
            for fu_name, fu_block in self.fleet_units.items()
            if fu_block.block.mode_scheduling == "uc"
        }

        self.fu_stat = {
            fu_name: fu_block
            for fu_name, fu_block in self.fleet_units.items()
            if fu_block.block.mode_scheduling in self.scenario.apriori_lvls
            and fu_block.block.mode_scheduling != "uc"
            and self.lm == "stat"
        }

        self.mode_scheduling_stat = get_mode_scheduling(fleet_units=self.fu_stat, block=self.block)

        self.fu_dyn = {
            fu_name: fu_block
            for fu_name, fu_block in self.fleet_units.items()
            if fu_block.block.mode_scheduling in self.scenario.apriori_lvls
            and fu_block.block.mode_scheduling != "uc"
            and self.lm == "dyn"
        }

        self.p_avail = pd.Series()
        self.p_fix = pd.Series()

        # per-horizon working copies of the series above
        self._p_avail = np.empty(0)
        self._p_fix = np.empty(0)

        self._resort_stat = np.empty(0, dtype=bool)
        self._order_stat = []

    def init_ph(self, horizon: time.TimeFrame):
        n_ts = len(horizon.dti)

        # initialize power availability and fixed power consumption
        self._p_avail = np.full(n_ts, np.inf if self.lm == "dyn" else self.block.pwr_lim_s2f, dtype="float64")
        self._p_fix = np.zeros(n_ts)

        for fu in self.fleet_units.values():
            fu.init_ph(horizon=horizon)

        # precompute the timesteps at which the static-load-management priority order can change
        if self.mode_scheduling_stat not in (None, "equal"):
            self._dti_i8 = horizon.dti.as_unit("ns").asi8
            self._resort_stat = get_resort_mask(self.mode_scheduling_stat, self.fu_stat, horizon.dti)
            self._order_stat = list(self.fu_stat.values())

    def finalize_ph(self, horizon: time.TimeFrame) -> None:
        self.p_avail = pd.Series(self._p_avail, index=horizon.dti)
        self.p_fix = pd.Series(self._p_fix, index=horizon.dti)

    def charge_static_lm(self, idx: int, ts: pd.Timestamp) -> None:
        if len(self.fu_stat) == 0:
            return
        if self.mode_scheduling_stat == "equal":
            # create list of fleet_units
            fu_to_charge = list(self.fu_stat.values())

            # iterate of all fleet units until available power is  charging demand is satisfied
            while self._p_fix[idx] < self._p_avail[idx] and len(fu_to_charge) > 0:
                p_max_per_fu = (self._p_avail[idx] - self._p_fix[idx]) / len(fu_to_charge)
                for fu in fu_to_charge:
                    p_chg_fu = fu.calc_p_chg_atbase(idx=idx, ts=ts, p_max_system=p_max_per_fu)
                    # if fu doesn't charge (not atbase, reached target soc, reached charging power limit)
                    if p_chg_fu == 0:
                        fu_to_charge.remove(fu)

        else:
            # sort the fleet units based on the scheduling mode (only when the order can change)
            if self._resort_stat[idx]:
                self._order_stat = sorted(
                    self.fu_stat.values(),
                    key=get_sort_key_func(self.mode_scheduling_stat, ts_i8=self._dti_i8[idx], idx=idx),
                )

            for fu in self._order_stat:
                fu.calc_p_chg_atbase(idx=idx, ts=ts, p_max_system=self._p_avail[idx] - self._p_fix[idx])


class AprioriFleetUnit:
    _COLUMNS_BATTERY = ["p_consumption", "p_sd", "p_max", "p_chg", "soc", "soc_target"]
    _COLUMNS_CHARGING = ["p_int", "p_ext_ac", "p_ext_dc"]

    def __init__(self, block, fleet, scheduler):
        self.block = block
        self.scenario = self.block.scenario
        self.scheduler = scheduler

        self.fleet = fleet

        # states and powers measured at the battery
        self.data_battery = pd.DataFrame(columns=self._COLUMNS_BATTERY, dtype="float64")

        # powers measured at the bus connection / connection to external charger
        self.data_charging = pd.DataFrame(columns=self._COLUMNS_CHARGING, dtype="float64")

        self.soh = None

        # Departures, arrivals and target SOCs are taken from the dispatch events instead of being
        # recovered from the log: the log only records occupancy, which merges two rentals following
        # each other without an idle timestep in between into a single absence.
        # The indices are sorted so that they can be searched with searchsorted instead of masked.
        self.dep_base_dti = (
            pd.DatetimeIndex(self.block.events["time_dep"]).intersection(self.block.log.index).sort_values()
        )
        self.arr_base_dti = (
            pd.DatetimeIndex(self.block.events["time_return"]).intersection(self.block.log.index).sort_values()
        )

        # event timestamps as int64 nanoseconds - comparable across fleet units regardless of the
        # datetime resolution of the individual logs, searched instead of the DatetimeIndex itself
        self.dep_base_i8 = self.dep_base_dti.as_unit("ns").asi8
        self.arr_base_i8 = self.arr_base_dti.as_unit("ns").asi8

        # get first timesteps, where vehicle has left the destination
        self.dep_dest_dti = self.block.log.index[
            ~self.block.log["atac"] & self.block.log["atac"].shift(periods=1, fill_value=False)
        ]

        # get first timesteps, where vehicle is parking at destination
        self.arr_dest_dti = self.block.log.index[
            self.block.log["atac"] & ~self.block.log["atac"].shift(periods=1, fill_value=False)
        ]

        # get all timesteps, where charging is available (internal AC, external AC, external DC)
        self.chg_avail_dti = self.block.log.index[self.block.log[["atbase", "atac", "atdc"]].any(axis=1)]
        self.chg_avail_i8 = self.chg_avail_dti.as_unit("ns").asi8

        # initialize variable for charging during single parking process
        self.parking_charging = False

        # per-horizon working arrays, see module docstring
        self.p_consumption = self.p_sd = self.p_max = self.p_chg = self.soc_target = np.empty(0)
        self.soc = np.empty(0)
        self.p_int = self.p_ext_ac = self.p_ext_dc = np.empty(0)

    def init_ph(self, horizon: time.TimeFrame):
        n_ts = len(horizon.dti)
        self._dti_i8 = horizon.dti.as_unit("ns").asi8
        self._n_ts = n_ts

        # cache the scenario/block constants that were previously looked up in every timestep
        self._hours = self.scenario.timestep.hours
        self._td = self.scenario.timestep.td
        self._e_storage = self.block.sizes["storage"].preexisting
        self._eff_sys = self._get_eff(self.fleet.block.system)
        self._eff_ac = self._get_eff("ac")
        self._eff_dc_ext = self._get_eff("dc_ext")
        forecast_hours = self.block.forecast_hours
        # A NaN forecast horizon is truthy but produces NaT, against which every comparison is
        # False - i.e. no departure ever qualifies, which is what the boolean mask did as well.
        self._forecast_ns = None
        self._forecast_never = False
        if forecast_hours:
            if pd.isna(forecast_hours):
                self._forecast_never = True
            else:
                self._forecast_ns = pd.Timedelta(hours=forecast_hours).value

        # power consumption at the battery
        self.p_consumption = (
            (-1)
            * self.block.log.loc[horizon.dti, "consumption"].to_numpy(dtype="float64")
            / self._get_eff("consumption")
        )

        self.p_sd = np.zeros(n_ts)
        self.p_max = np.zeros(n_ts)
        self.p_chg = np.zeros(n_ts)
        self.soc_target = np.zeros(n_ts)

        # one entry longer than the horizon: calc_soc writes the SOC at the end of the last timestep
        self.soc = np.zeros(n_ts + 1)
        self.soc[0] = self.block.states.loc[horizon.start, ["soc", "soc_min", "soc_max"]].median()

        self.p_int = np.zeros(n_ts)
        self.p_ext_ac = np.zeros(n_ts)
        self.p_ext_dc = np.zeros(n_ts)

        # per-timestep block data, resolved once per horizon instead of once per timestep
        log_ph = self.block.log.loc[horizon.dti]
        self.atbase = log_ph["atbase"].to_numpy()
        self.atac = log_ph["atac"].to_numpy()
        self.atdc = log_ph["atdc"].to_numpy()
        self.soc_max = self.block.states.loc[horizon.dti, "soc_max"].to_numpy(dtype="float64")
        self.is_arr_dest = horizon.dti.isin(self.arr_dest_dti)

        # get soh for current prediction horizon
        self.soh = self.block.states.loc[horizon.start, "soh"]

    def _sum_consumption(self, start: int, end: int) -> float:
        """Sum of p_consumption over the horizon positions [start, end), clipped to the horizon."""
        if end <= start:
            return 0.0
        return self.p_consumption[start:end].sum()

    def calc_soc_target(self, idx: int) -> float:
        # ToDo: add input parameter to specify target SOCs
        if self._e_storage == 0:  # no onboard battery to schedule (e.g. pure swap/rex unit)
            return 0.0

        if self.atdc[idx]:
            return 0.8

        soc_max = self.soc_max[idx]
        soc_target_low = min(self.block.soc_target, soc_max)
        soc_target_high = min(1.0, soc_max)

        ts_i8 = self._dti_i8[idx]

        # check if there are any departures after current timestep within forecast period
        pos_dep = np.searchsorted(self.dep_base_i8, ts_i8, side="left")
        if pos_dep == len(self.dep_base_i8):
            return soc_target_low
        dep_nxt = self.dep_base_i8[pos_dep]
        if self._forecast_never or (self._forecast_ns is not None and dep_nxt > ts_i8 + self._forecast_ns):
            return soc_target_low

        pos_arr = np.searchsorted(self.arr_base_i8, ts_i8, side="left")
        if pos_arr == len(self.arr_base_i8):
            return soc_target_low
        arr_nxt = self.arr_base_i8[pos_arr]

        #  sum up energy between trip start and end
        # (exclusive end position of the label slice [..., arr_nxt - timestep])
        idx_end = np.searchsorted(self._dti_i8, arr_nxt, side="left")
        if arr_nxt <= dep_nxt:  # trip currently ongoing
            # Destination charging -> sum up remaining energy of ongoing trip until end of trip
            idx_start = idx
        else:  # vehicle currently rechargeable at base
            idx_start = np.searchsorted(self._dti_i8, dep_nxt, side="left")
        e_con = (-1) * (self._sum_consumption(idx_start, idx_end) * self._hours)

        #  Convert energy consumption to delta soc taking the current soh into account
        soc_delta = e_con / self._e_storage
        #  Set soc_target dependent on soc_delta of trip and settings of the MobileCommodity
        if soc_delta > (soc_target_low - self.block.soc_return):
            soc_target = soc_target_high
        else:
            soc_target = soc_target_low
        return soc_target

    def calc_p_bat_chg_max(self, idx: int) -> None:
        """
        calculate the required charging power at the battery in the current timestep to reach the specified
        target SOC (soc_target) within the current timestep
        """

        soc_target = self.calc_soc_target(idx=idx)
        self.soc_target[idx] = soc_target

        # calculate current energy content of battery
        e_bat = self.soc[idx] * self._e_storage

        # calculate self discharge power in current timestep based on the current energy content
        p_sd = -1 * e_bat * self.block.loss_rate_per_ts / self._hours
        self.p_sd[idx] = p_sd

        # calculate target energy content of battery
        e_target = soc_target * self._e_storage

        # calculate maximum charging power at battery (avoid p_max < 0 caused by changing soc_target)
        self.p_max[idx] = max(
            ((e_target - e_bat) / self._hours + (-1) * p_sd + (-1) * self.p_consumption[idx]),
            0,
        )

    def calc_p_chg_atbase(self, idx: int, ts: pd.Timestamp, p_max_system: float) -> float:
        """
        calculate the charging power at the base in the current timestep with respect to the maximum power provided
        by the local energy system's load management (static or dynamic) if applicable
        """
        # calculate charging power at base (measurement point at connection to SystemCore bus)
        p_max_fleet_unit_battery = (self.p_max[idx] - self.p_chg[idx]) / self._eff_sys
        p_max_fleet_unit_connection = self.block.pwr_chg_max * self.atbase[idx]

        p_chg_atbase = min(p_max_fleet_unit_battery, p_max_fleet_unit_connection, p_max_system)

        self.p_int[idx] += p_chg_atbase

        # calculate charging power observed at battery
        self.p_chg[idx] += p_chg_atbase * self._eff_sys

        # write charging power to fleet
        self.fleet._p_fix[idx] += p_chg_atbase

        if self.fleet.lm == "stat" and self.fleet._p_fix[idx] > self.fleet._p_avail[idx]:
            raise ValueError(f"Power limit of {self.fleet.block.name}'s static load management exceeded at {ts}!")

        return p_chg_atbase

    def calc_p_chg_external(self, idx: int) -> None:
        """
        calculate charging power from external sources
        """
        if self._e_storage == 0:  # no onboard battery to charge externally
            return

        ts_i8 = self._dti_i8[idx]

        # determine whether destination charging is necessary
        if self.atac[idx] == 1:
            if self.is_arr_dest[idx]:  # plugging in only happens when parking starts
                # calculate all upcoming arrival times at the base
                pos_arr = np.searchsorted(self.arr_base_i8, ts_i8, side="left")
                # use current time and next arrival index to calculate consumption and convert to SOC
                # (inclusive label slice [ts, arr_nxt]; without an arrival the horizon end is used)
                if pos_arr < len(self.arr_base_i8):
                    idx_end = np.searchsorted(self._dti_i8, self.arr_base_i8[pos_arr], side="right")
                else:
                    idx_end = self._n_ts
                e_trip_remaining = (-1) * self._sum_consumption(idx, idx_end) * self._hours

                # set charging to True, if charging is necessary
                if e_trip_remaining > (
                    (self.soc[idx] - self.block.soc_return) * self._e_storage
                ):  # ToDo: add soh/aging
                    self.parking_charging = True
                else:
                    self.parking_charging = False

            if self.parking_charging is True:
                # calculate charging power at external AC charger (measurement point at connection to charger)
                p_max_fleet_unit_battery = self.p_max[idx] / self._eff_ac
                p_max_fleet_unit_connection = self.block.pwr_ext_ac_max

                self.p_ext_ac[idx] += min(p_max_fleet_unit_battery, p_max_fleet_unit_connection)

                # calculate charging power observed at battery
                self.p_chg[idx] += self.p_ext_ac[idx] * self._eff_ac

        # determine whether on-route charging is necessary
        elif self.atdc[idx] == 1:
            # activate charging, if SOC will fall below threshold, before next possibility to charge
            pos_chg = np.searchsorted(self.chg_avail_i8, ts_i8, side="right")
            if pos_chg < len(self.chg_avail_i8):
                # exclusive end position of the label slice [ts, chg_nxt - timestep]
                idx_end = np.searchsorted(self._dti_i8, self.chg_avail_i8[pos_chg], side="left")
                soc_chg_nxt = self.soc[idx] - (-1) * self._sum_consumption(idx, idx_end) * self._hours / self._e_storage

                # ToDo: add soh/aging: if soc_chg_nxt < self.convert_soc_ui2internal(0.05):
                if soc_chg_nxt < 0.05:
                    # calculate charging power at external DC charger (measurement point at connection to charger)
                    p_max_fleet_unit_battery = self.p_max[idx] / self._eff_dc_ext
                    p_max_fleet_unit_connection = self.block.pwr_ext_dc_max

                    self.p_ext_dc[idx] += min(p_max_fleet_unit_battery, p_max_fleet_unit_connection)

                    # calculate charging power observed at battery
                    self.p_chg[idx] += self.p_ext_dc[idx] * self._eff_dc_ext

    def calc_soc(self, idx: int) -> None:
        # calculate state of charge based on calculated charging powers, consumption and self discharge
        if self._e_storage == 0:  # no onboard battery: soc is not meaningful, keep it at 0
            self.soc[idx + 1] = 0.0
            return

        soc_delta = (self.p_consumption[idx] + self.p_sd[idx] + self.p_chg[idx]) * self._hours / self._e_storage

        self.soc[idx + 1] = self.soc[idx] + soc_delta

    def write_power_to_flows_apriori(self, horizon: time.TimeFrame) -> None:
        self.finalize_ph(horizon=horizon)

        self.block.flows_apriori.update(
            {
                "p_int_chg": (self.data_charging["p_int"].clip(lower=0) / self.block.pwr_chg_max),
                "p_int_dis": ((-1) * self.data_charging["p_int"].clip(upper=0) / self.block.pwr_dis_max),
                "p_ext_ac_chg": (self.data_charging["p_ext_ac"].clip(lower=0) / self.block.pwr_ext_ac_max),
                "p_ext_ac_dis": ((-1) * self.data_charging["p_ext_ac"].clip(upper=0) / self.block.pwr_ext_ac_max),
                "p_ext_dc_chg": (self.data_charging["p_ext_dc"].clip(lower=0) / self.block.pwr_ext_dc_max),
                "p_ext_dc_dis": ((-1) * self.data_charging["p_ext_dc"].clip(upper=0) / self.block.pwr_ext_dc_max),
            }
        )

        # fix NaN values caused by max_power = 0 leading and therefore division by 0
        self.block.flows_apriori.loc[horizon.dti, :] = self.block.flows_apriori.loc[horizon.dti, :].fillna(0.0)

    def finalize_ph(self, horizon: time.TimeFrame) -> None:
        """Rebuild the timestamp-indexed views from the per-horizon working arrays."""
        # data_battery carries one extra row holding the SOC at the end of the horizon
        index_battery = horizon.dti.append(pd.DatetimeIndex([horizon.dti[-1] + self._td]))
        self.data_battery = pd.DataFrame(index=index_battery, columns=self._COLUMNS_BATTERY, dtype="float64")
        self.data_battery.iloc[: self._n_ts, self._COLUMNS_BATTERY.index("p_consumption")] = self.p_consumption
        self.data_battery.iloc[: self._n_ts, self._COLUMNS_BATTERY.index("p_sd")] = self.p_sd
        self.data_battery.iloc[: self._n_ts, self._COLUMNS_BATTERY.index("p_max")] = self.p_max
        self.data_battery.iloc[: self._n_ts, self._COLUMNS_BATTERY.index("p_chg")] = self.p_chg
        self.data_battery.iloc[:, self._COLUMNS_BATTERY.index("soc")] = self.soc
        self.data_battery.iloc[: self._n_ts, self._COLUMNS_BATTERY.index("soc_target")] = self.soc_target

        self.data_charging = pd.DataFrame(
            {"p_int": self.p_int, "p_ext_ac": self.p_ext_ac, "p_ext_dc": self.p_ext_dc},
            index=horizon.dti,
        )

    def _get_eff(self, mode: str):
        # get charging efficiency for the selected mode: ac, dc, consumption
        if mode not in ["ac", "dc", "dc_ext", "consumption"]:
            raise ValueError(f'Invalid mode "{mode}" selected. Valid modes are "ac", "dc" , "dc_ext", and "consumption')

        eff = {
            "ac": self.block.eff["chg_ac"] * np.sqrt(self.block.eff["storage_roundtrip"]),
            "dc": self.block.eff["chg_dc"] * np.sqrt(self.block.eff["storage_roundtrip"]),
            # 100% efficiency for external DC charger as measurement point is behind power electronics
            "dc_ext": np.sqrt(self.block.eff["storage_roundtrip"]),
            "consumption": np.sqrt(self.block.eff["storage_roundtrip"]),
        }[mode]
        return eff
