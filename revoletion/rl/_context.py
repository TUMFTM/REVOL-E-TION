import collections

import pandas as pd

from revoletion import blocks, utils
from revoletion import scenario as scn


class TimeContext:
    def __init__(self, horizon: utils.TimeSettings) -> None:
        self._step_idx = 0
        self.horizon = horizon

    @property
    def step_idx(self) -> int:
        return self._step_idx

    @property
    def current_time_step(self) -> pd.DatetimeIndex:
        return self.horizon.dti_extd[self._step_idx]

    @property
    def previous_time_step(self) -> pd.DatetimeIndex:
        return self.horizon.dti[max(self._step_idx - 1, 0)]

    def reset(self, new_horizon: utils.TimeSettings | None = None) -> None:
        self._step_idx = 0
        if new_horizon is not None:
            self.horizon = new_horizon

    def step(self) -> None:
        self._step_idx += 1


class Context:
    def __init__(self, scenario: scn.Scenario, horizon: utils.TimeSettings) -> None:
        self._block_registry = scenario.block_registry
        self._time_ctx = TimeContext(horizon)

        self.electric_fleets = _collect_electric_fleets(scenario)
        self.electric_fleet_unit_blocks = [efu for efus in self.electric_fleets.values() for efu in efus]

        self._efu_index = {}
        offset = 0
        for efus in self.electric_fleets.values():
            for i, efu in enumerate(efus):
                self._efu_index[efu] = offset + i
            offset += len(efus)

        self.grid_market_blocks = list(self._block_registry.get("GridMarket", {}).values())
        self.grid_connection_blocks = list(self._block_registry.get("GridConnection", {}).values())
        self.has_grid_connection = len(self.grid_connection_blocks) > 0

        self.renewable_source_blocks = list(self._block_registry.get("RenewableSource", {}).values())
        self.has_renewable_sources = len(self.renewable_source_blocks) > 0

        self.controllable_source_blocks = list(self._block_registry.get("ControllableSource", {}).values())
        self.has_controllable_sources = len(self.controllable_source_blocks) > 0

        self.stationary_battery_blocks = list(self._block_registry.get("StationaryBattery", {}).values())
        self.has_stationary_batteries = len(self.stationary_battery_blocks) > 0

        self.fixed_demand_blocks = list(self._block_registry.get("FixedDemand", {}).values())
        self.has_fixed_demands = len(self.fixed_demand_blocks) > 0

    def get_efu_index(self, efu: blocks.ElectricFleetUnit) -> int:
        return self._efu_index[efu]

    @property
    def time(self) -> TimeContext:
        return self._time_ctx

    @property
    def horizon(self) -> utils.TimeSettings:
        return self._time_ctx.horizon

    @property
    def current_time_step(self) -> pd.DatetimeIndex:
        return self._time_ctx.current_time_step

    @property
    def previous_time_step(self) -> pd.DatetimeIndex:
        return self._time_ctx.previous_time_step

    @property
    def step_idx(self) -> int:
        return self._time_ctx.step_idx

    def reset(self, new_horizon: utils.TimeSettings | None = None) -> None:
        self._time_ctx.reset(new_horizon)

    def step(self) -> None:
        self._time_ctx.step()


def _collect_electric_fleets(scenario: scn.Scenario) -> dict[blocks.Fleet, list[blocks.ElectricFleetUnit]]:
    electric_fleet_mappings = collections.defaultdict(list)
    fleets = scenario.block_registry.get("Fleet", {}).values()
    for fleet in fleets:
        for subfleet in fleet.subblocks.values():
            for fleet_unit in subfleet.subblocks.values():
                if not isinstance(fleet_unit, blocks.ElectricFleetUnit):
                    continue

                electric_fleet_mappings[fleet].append(fleet_unit)

    return electric_fleet_mappings
