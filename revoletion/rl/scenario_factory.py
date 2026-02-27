import abc
import logging
from typing import Iterable

import numpy as np

from revoletion import scenario as scn
from revoletion import utils

from . import utils as rl_utils

_LOGGER = logging.getLogger(__name__)


class HorizonInitializerInterface(abc.ABC):
    def __init__(self): ...

    @abc.abstractmethod
    def initialze_scenario_for_horizon(self, scenario: scn.Scenario, horizon: utils.TimeSettings) -> None: ...


class MinSocHorizonInitializer(HorizonInitializerInterface):
    def __init__(self, soc_min: float = 0.0) -> None:
        self._soc_min = soc_min

    def initialze_scenario_for_horizon(self, scenario: scn.Scenario, horizon: utils.TimeSettings) -> None:
        for electric_fleet_unit_block in scenario.block_registry.get("ElectricFleetUnit", {}).values():
            prev_soc_min = electric_fleet_unit_block.states.loc[horizon.dti, "soc_min"]
            new_soc_min = np.clip(prev_soc_min + self._soc_min, 0.0, 1.0)
            electric_fleet_unit_block.states.loc[horizon.dti, "soc_min"] = new_soc_min

        for stationary_battery_block in scenario.block_registry.get("StationaryBattery", {}).values():
            prev_soc_min = electric_fleet_unit_block.states.loc[horizon.dti, "soc_min"]
            new_soc_min = np.clip(prev_soc_min + self._soc_min, 0.0, 1.0)
            stationary_battery_block.states.loc[horizon.dti, "soc_min"] = new_soc_min


class SocEnvelopeHorizonInitialzer(HorizonInitializerInterface):
    def __init__(self, max_charge_power_frac: float = 1.0, envelope_target_soc: float = 0.0) -> None:
        self._max_charge_power_frac = max_charge_power_frac
        self._envelope_target_soc = envelope_target_soc

    def initialze_scenario_for_horizon(self, scenario: scn.Scenario, horizon: utils.TimeSettings) -> None:
        for electric_fleet_unit_block in scenario.block_registry.get("ElectricFleetUnit", {}).values():
            nom_capacity_wh = electric_fleet_unit_block.sizes["storage"].preexisting
            eff_charge = electric_fleet_unit_block.eff["chg_int"]
            eff_storage_roundtrip = np.sqrt(electric_fleet_unit_block.eff["storage_roundtrip"])
            max_charge_power_w = electric_fleet_unit_block.pwr_chg_max * eff_charge * eff_storage_roundtrip

            buffered_max_charge_power_w = max_charge_power_w * self._max_charge_power_frac
            # Determine a smoothed out dsoc step that is used for the SoC envelope computation.
            # This gives the optimizer a bit more freedom and reduces the number of infeasibilities.
            dsoc_step_max = (buffered_max_charge_power_w * horizon.timestep.hours) / nom_capacity_wh

            soc_envelope = rl_utils.get_soc_envelope(
                electric_fleet_unit_block, horizon, dsoc_step_max=dsoc_step_max, target_soc=self._envelope_target_soc
            )
            soc_min = electric_fleet_unit_block.states.loc[horizon.dti, "soc_min"]

            buffered_soc_envelope = np.clip(soc_envelope + soc_min, 0.0, 1.0)

            electric_fleet_unit_block.states.loc[horizon.dti, "soc_min"] = buffered_soc_envelope


class AtBaseHorizonInitializer(HorizonInitializerInterface):
    def initialze_scenario_for_horizon(self, scenario: scn.Scenario, horizon: utils.TimeSettings) -> None:
        for electric_fleet_unit_block in scenario.block_registry.get("ElectricFleetUnit", {}).values():
            atbase = electric_fleet_unit_block.log.loc[horizon.dti, "atbase"]
            ext_available = np.invert(atbase.astype(bool))

            electric_fleet_unit_block.log.loc[horizon.dti, "atac"] = ext_available
            electric_fleet_unit_block.log.loc[horizon.dti, "atdc"] = ext_available


class InitialSocHorizonInitializer(HorizonInitializerInterface):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def initialze_scenario_for_horizon(self, scenario: scn.Scenario, horizon: utils.TimeSettings) -> None:
        for electric_fleet_unit_block in scenario.block_registry.get("ElectricFleetUnit", {}).values():
            soc_envelope = rl_utils.get_soc_envelope(electric_fleet_unit_block, horizon)
            soc_min = electric_fleet_unit_block.states.loc[horizon.start, "soc_min"]
            buffered_soc_envelope = np.clip(soc_envelope + soc_min, 0.0, 1.0)

            initial_soc_min = buffered_soc_envelope.loc[horizon.start]

            initial_soc = self._rng.uniform(initial_soc_min, 1.0)
            electric_fleet_unit_block.states.loc[horizon.start, "soc"] = initial_soc

        for stationary_battery_blocks in scenario.block_registry.get("StationaryBattery", {}).values():
            soc_min = electric_fleet_unit_block.states.loc[horizon.start, "soc_min"]
            initial_soc = self._rng.uniform(low=soc_min, high=1.0)
            stationary_battery_blocks.states.loc[horizon.start, "soc"] = initial_soc


class HorizonInitializer:
    def __init__(self, horizon_initializers: Iterable[HorizonInitializerInterface]) -> None:
        self._horizon_initializers = horizon_initializers

    def initialize(self, scenario: scn.Scenario, horizon: utils.TimeSettings) -> None:
        for horizon_initializer in self._horizon_initializers:
            horizon_initializer.initialze_scenario_for_horizon(scenario, horizon)


class ScenarioFactory:
    """Create new scenario objects on demand for multi-processing setups"""

    def __init__(self, paths: scn.SimulationPaths) -> None:
        self._paths = paths
        self.scenario_name = self._paths.scenario.stem

        scenario_parameters = utils.read_scenario_from_file(self._paths.scenario)

        self._scenario_parameters = scenario_parameters[self.scenario_name]

        self._location = utils.Location.create_from_lat_lon(
            latitude=self._scenario_parameters.loc["scenario", "latitude"],
            longitude=self._scenario_parameters.loc["scenario", "longitude"],
            logger=_LOGGER,
        )

    def create_scenario(self) -> scn.Scenario:
        scenario = scn.Scenario(
            self._paths,
            scn.ScenarioSettings(),
            name=self.scenario_name,
            parameters=self._scenario_parameters,
            location=self._location,
        )

        full_horizon = utils.TimeSettings.create_from_start_timestamp(
            start=scenario.times.sim.start,
            timestep=scenario.timestep,
            end=scenario.times.sim.start + scenario.len_ph + scenario.len_ch,
        )

        AtBaseHorizonInitializer().initialze_scenario_for_horizon(scenario, full_horizon)

        return scenario
