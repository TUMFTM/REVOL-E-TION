import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np
import pandas as pd
from typing_extensions import override

from revoletion import blocks, optimization, utils
from revoletion import scenario as scn

from . import _utils as rl_utils

_LOGGER = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    penalty_factor_grid_opex: float = 1.0
    """Factor applied to the costs of importing/exporting energy to the grid."""

    penalty_factor_charge_opex: float = 0.0
    """Weight applied to the costs of charging/discharging the EVs."""

    penalty_factor_gen_opex: float = 1.0
    """Weight applied to the costs of charging/discharging the EVs."""

    penalty_factor_dsoc: float = 16.0
    """Weight for the penalty if the agent does not met the SoC requirements. In all those cases, the scenario will become infeasible in the future time steps and an infeasibility penalty will also be applied."""

    reward_factor_dsoc: float = 4.0
    """Weight of the reward for meeting a SoC requirement."""

    penalty_factor_infeasible: float = 1.0
    """Weight for the penatly if the energy system is determined to be infeasible and cannot be optimizated."""

    penalty_factor_power_diff: float = 0.5
    """Weight for the penalty if the agent tries to charge with a power that would exceed the maximimal/minimum capacity of an EV."""

    penalty_factor_atbase_violation: float = 1.0

    reward_factor_step: float = 0.01


@dataclass
class RevoletionEnvironmentConfig:
    min_duration: pd.Timedelta = field(default_factory=lambda: pd.to_timedelta("6h"))
    max_duration: pd.Timedelta = field(default_factory=lambda: pd.to_timedelta("48h"))

    episode_length: int | None = None

    forecast_horizon: int = 8
    """The length of the forecast horizon that is provided in the observations to the client."""

    power_precision: int = 1

    power_unit_buffer: float = 1e-6

    min_soc: float = 0.05

    reward_config: RewardConfig = field(default_factory=lambda: RewardConfig())


@dataclass
class RewardComponents:
    config: RewardConfig

    grid_opex: float = 0.0
    charge_opex: float = 0.0
    gen_opex: float = 0.0
    power_diffs: list[float] = field(default_factory=list)
    atbase_violation: int = 0
    soc_diffs: list[float] = field(default_factory=list)
    infeasible: float = 0.0
    step: int = 0

    @property
    def grid_opex_reward(self) -> float:
        if self.grid_opex > 0.0:
            return -self.grid_opex * self.config.penalty_factor_grid_opex
        return self.grid_opex

    @property
    def charge_opex_reward(self) -> float:
        return self.charge_opex * self.config.penalty_factor_charge_opex

    @property
    def gen_opex_reward(self) -> float:
        if self.gen_opex > 0.0:
            return -self.gen_opex * self.config.penalty_factor_gen_opex
        return self.gen_opex

    @property
    def power_diff_reward(self) -> float:
        if len(self.power_diffs) == 0:
            return 0
        avg_power_diff = sum(self.power_diffs) / len(self.power_diffs)
        return -abs(avg_power_diff) * self.config.penalty_factor_power_diff

    @property
    def atbase_violation_reward(self) -> float:
        return self.atbase_violation * self.config.penalty_factor_atbase_violation

    @property
    def soc_diff_reward(self) -> float:
        reward = 0.0
        for soc_diff in self.soc_diffs:
            if soc_diff < 0.0:
                reward += soc_diff * self.config.penalty_factor_dsoc
            else:
                reward += soc_diff * self.config.reward_factor_dsoc

        return reward

    @property
    def infeasibility_reward(self) -> float:
        return -self.infeasible * self.config.penalty_factor_infeasible

    @property
    def step_reward(self) -> float:
        return self.config.reward_factor_step

    @property
    def total_reward(self) -> float:
        return (
            self.grid_opex_reward
            + self.charge_opex_reward
            + self.gen_opex_reward
            + self.power_diff_reward
            + self.atbase_violation_reward
            + self.soc_diff_reward
            + self.infeasibility_reward
            + self.step_reward
        )

    def __str__(self) -> str:
        return f"{self.total_reward:.2f} (grid={self.grid_opex_reward:.2f}; gen={self.gen_opex_reward:.2f}; power_diff={self.power_diff_reward:.2f}; atbase={self.atbase_violation_reward:.2f}; soc_diff={self.soc_diff_reward:.2f}; infeasibility={self.infeasibility_reward:.2f}; step={self.step_reward:.2f})"


class EnvironmentStepStatus(Enum):
    OK = auto()
    INFEASIBLE = auto()


ActType: TypeAlias = np.ndarray
ObsType: TypeAlias = dict[str, np.ndarray]

# Keys in the observation space for consistent access in custom agents.
OBS_KEY_TIME_FEATURES = "time_of_day"
OBS_KEY_CARS_AVAILABLE = "cars_available"
OBS_KEY_CARS_SOC = "cars_soc"
OBS_KEY_CARS_REQUIRED_SOCS = "cars_required_socs"
OBS_KEY_RENEWABLES_POWER = "renewables_power"
OBS_KEY_RENEWABLES_SCHEDULE = "renewables_schedule"
OBS_KEY_FIXED_DEMANDS = "demands_schedule"
OBS_KEY_GRID_IMPORT_COSTS = "grid_import_costs"
OBS_KEY_GRID_IMPORT_POWER = "grid_import_power"
OBS_KEY_GRID_EXPORT_COSTS = "grid_export_costs"
OBS_KEY_GRID_EXPORT_POWER = "grid_export_power"
OBS_KEY_STATIONARY_BATTERIES_SOC = "stationary_batteries_soc"
OBS_KEY_CONTROLLABLE_SOURCES_POWER = "controllable_sources_power"

INFO_KEY_OPTIMIZATION_RESULT = "optimization_result"
INFO_KEY_STATUS = "status"
INFO_KEY_REWARD_COMPONENTS = "rewards"


class RevoletionEnvironment(gym.Env[ObsType, ActType]):
    def __init__(
        self,
        scenario: scn.Scenario,
        horizon: utils.TimeSettings,
        config: RevoletionEnvironmentConfig | None = None,
        logger: logging.Logger | None = None,
        train: bool = True,
    ) -> None:
        super().__init__()

        self._scenario = scenario
        self._block_registry = scenario.block_registry
        self._horizon = horizon
        self._config = config or RevoletionEnvironmentConfig()
        self._logger = logger or logging.getLogger(__name__)

        self._step_size = horizon._timestep

        self._electric_fleet_unit_blocks = list(self._block_registry.get("ElectricFleetUnit", {}).values())

        self._grid_market_blocks = list(self._block_registry.get("GridMarket", {}).values())
        self._grid_connection_blocks = list(self._block_registry.get("GridConnection", {}).values())
        self._has_grid_connection = len(self._grid_connection_blocks) > 0

        self._renewable_source_blocks = list(self._block_registry.get("RenewableSource", {}).values())
        self._has_renewable_sources = len(self._renewable_source_blocks) > 0

        self._controllable_source_blocks = list(self._block_registry.get("ControllableSource", {}).values())
        self._has_controllable_sources = len(self._controllable_source_blocks) > 0

        self._stationary_battery_blocks = list(self._block_registry.get("StationaryBattery", {}).values())
        self._has_stationary_batteries = len(self._stationary_battery_blocks) > 0

        self._fixed_demand_blocks = list(self._block_registry.get("FixedDemand", {}).values())
        self._has_fixed_demands = len(self._fixed_demand_blocks) > 0

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self._electric_fleet_unit_blocks),), dtype=np.float32
        )

        obs_dict = {
            # Time information like
            OBS_KEY_TIME_FEATURES: gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            # The SoCs of the vehicles at the current time step.
            OBS_KEY_CARS_SOC: gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._electric_fleet_unit_blocks),), dtype=np.float32
            ),
            # A forecast for each vehicle, if it is available for charging in the current and upcoming time steps.
            OBS_KEY_CARS_AVAILABLE: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._electric_fleet_unit_blocks), self._config.forecast_horizon),
                dtype=np.float32,
            ),
            # A forecast for each vehicle, of its required SoC.
            OBS_KEY_CARS_REQUIRED_SOCS: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._electric_fleet_unit_blocks), self._config.forecast_horizon),
                dtype=np.float32,
            ),
        }

        if self._has_grid_connection:
            obs_dict[OBS_KEY_GRID_IMPORT_COSTS] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._grid_market_blocks),), dtype=np.float32
            )

            obs_dict[OBS_KEY_GRID_IMPORT_POWER] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._grid_market_blocks),), dtype=np.float32
            )

            obs_dict[OBS_KEY_GRID_EXPORT_COSTS] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._grid_market_blocks),), dtype=np.float32
            )

            obs_dict[OBS_KEY_GRID_EXPORT_POWER] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._grid_market_blocks),), dtype=np.float32
            )

        if self._has_renewable_sources:
            obs_dict[OBS_KEY_RENEWABLES_SCHEDULE] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._renewable_source_blocks), self._config.forecast_horizon),
                dtype=np.float32,
            )
            obs_dict[OBS_KEY_RENEWABLES_POWER] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._renewable_source_blocks),),
                dtype=np.float32,
            )

        if self._has_stationary_batteries:
            obs_dict[OBS_KEY_STATIONARY_BATTERIES_SOC] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._stationary_battery_blocks),), dtype=np.float32
            )

        if self._has_controllable_sources:
            # The current generation of each controllable source.
            obs_dict[OBS_KEY_CONTROLLABLE_SOURCES_POWER] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._controllable_source_blocks),),
                dtype=np.float32,
            )

        if self._has_fixed_demands:
            # The current consumption of each load.
            obs_dict[OBS_KEY_FIXED_DEMANDS] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._fixed_demand_blocks),),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(obs_dict)

        self._step_idx = 0
        self._max_step_idx = 0

        self._reward_history = []

        self._prev_obs = None

        self._train = train

    @property
    def current_time_step(self) -> pd.DatetimeIndex:
        return self._curr_horizon.dti[self._step_idx]

    @property
    def previous_time_step(self) -> pd.DatetimeIndex:
        return self._curr_horizon.dti[max(self._step_idx - 1, 0)]

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        if self._config.episode_length is None:
            # Episode length randomization.
            min_steps = max(int(self._config.min_duration / self._step_size), 0)
            max_steps = min(int(self._config.max_duration / self._step_size), len(self._horizon) - 1)
            episode_length = self.np_random.integers(min_steps, max_steps)

            # Start time randomization.
            max_start_idx = len(self._horizon) - episode_length
            start_idx = self.np_random.integers(0, max_start_idx)
        else:
            start_idx = 0
            episode_length = self._config.episode_length

        self._curr_horizon = self._horizon.cut(start_idx, episode_length)

        self._step_idx = 0
        self._max_step_idx = episode_length - 1

        opt_problem_config = optimization.OptimizationProblemConfig(
            cost_eps=self._scenario.cost_eps,
            solver=optimization.Solver.HIGHS,
            invest=False,
        )

        self._optimization_problem = optimization.create_optimization_problem(
            backend=optimization.OptimizationBackend.PYPSA,
            scenario=self._scenario,
            horizon=self._curr_horizon,
            logger=self._logger,
            config=opt_problem_config,
        )

        self._logger.debug(f"Reset environment: episode_length={episode_length}; start={self._curr_horizon.start}")

        envelopes = {}
        for electric_fleet_unit_block in self._electric_fleet_unit_blocks:
            soc_envelope = rl_utils.get_soc_envelope(
                electric_fleet_unit_block, self._curr_horizon, min_soc=self._config.min_soc
            )
            envelopes[electric_fleet_unit_block] = soc_envelope
        self._soc_envelopes = envelopes

        return self._get_obs(), {}

    def _get_obs(self, optimization_result: optimization.OptimizationResult | None = None) -> ObsType:
        """Create the observation space for the current step.

        The observation space consists of several feature vectors which are individually collected.
        """
        state_dict = {}

        state_dict.update(self._get_time_features())

        state_dict.update(self._get_cars_features(optimization_result))

        if self._has_grid_connection:
            state_dict.update(self._get_grid_markets_features(optimization_result))

        if self._has_renewable_sources:
            state_dict.update(self._get_renewable_sources_features(optimization_result))

        if self._has_controllable_sources:
            state_dict.update(self._get_controllable_sources_features(optimization_result))

        if self._has_stationary_batteries:
            state_dict.update(self._get_stationary_batteries_features(optimization_result))

        if self._has_fixed_demands:
            state_dict.update(self._get_fixed_demands_features())

        _LOGGER.debug(f"Observation at {self._step_idx}: {state_dict}")

        self._prev_obs = state_dict

        return state_dict

    def _get_time_features(self) -> dict[str, np.ndarray]:
        # Hour of day (0-23)
        hour = self.current_time_step.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Day of week (0-6)
        day_of_week = self.current_time_step.dayofweek
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        day_of_year = self.current_time_step.dayofyear
        doy_sin = np.sin(2 * np.pi * day_of_year / 365)
        doy_cos = np.cos(2 * np.pi * day_of_year / 365)

        return {
            OBS_KEY_TIME_FEATURES: np.array([hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos], dtype=np.float32)
        }

    def _get_cars_features(
        self, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        cars_soc = []
        cars_required_socs = []
        cars_available = []
        for electric_fleet_unit_block in self._electric_fleet_unit_blocks:
            if optimization_result is None:
                soc = electric_fleet_unit_block.states["soc"].median()
            else:
                stored_energy = optimization_result.get_stored_energy(
                    electric_fleet_unit_block, self.previous_time_step
                )
                soc = stored_energy / electric_fleet_unit_block.sizes["storage"].preexisting
            cars_soc.append(soc)

            required_socs = self._get_forecast(electric_fleet_unit_block.log["dsoc"])
            cars_required_socs.append(required_socs)

            car_available = self._get_forecast(electric_fleet_unit_block.log["atbase"]).astype(np.float32)
            cars_available.append(car_available)
        return {
            OBS_KEY_CARS_SOC: np.array(cars_soc, dtype=np.float32),
            OBS_KEY_CARS_REQUIRED_SOCS: np.array(cars_required_socs, dtype=np.float32),
            OBS_KEY_CARS_AVAILABLE: np.array(cars_available, dtype=np.float32),
        }

    def _get_grid_markets_features(
        self, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        grid_markets_import_cost = []
        grid_markets_import_power_unit = []
        grid_markets_export_cost = []
        grid_markets_export_power_unit = []
        for grid_market_block in self._grid_market_blocks:
            import_cost = grid_market_block.evaluators["g2s"].opt.spec_ep_operation.loc[self.current_time_step]
            grid_markets_import_cost.append(import_cost)

            export_cost = grid_market_block.evaluators["s2g"].opt.spec_ep_operation.loc[self.current_time_step]
            grid_markets_export_cost.append(export_cost)

            if optimization_result is not None:
                power_flows = optimization_result.get_power_flow(grid_market_block, self.previous_time_step)

                import_power_max = grid_market_block.pwr_g2s
                import_power_unit = power_flows["in"] / import_power_max
                clipped_import_power_unit = np.clip(import_power_unit, 0.0, 1.0)
                grid_markets_import_power_unit.append(clipped_import_power_unit)

                export_power_max = grid_market_block.pwr_s2g
                export_power_unit = power_flows["out"] / export_power_max
                clipped_export_power_unit = np.clip(export_power_unit, 0.0, 1.0)
                grid_markets_export_power_unit.append(clipped_export_power_unit)
            else:
                grid_markets_import_power_unit.append(0.0)
                grid_markets_export_power_unit.append(0.0)

        return {
            OBS_KEY_GRID_IMPORT_COSTS: np.array(grid_markets_import_cost, dtype=np.float32),
            OBS_KEY_GRID_IMPORT_POWER: np.array(grid_markets_import_power_unit, dtype=np.float32),
            OBS_KEY_GRID_EXPORT_COSTS: np.array(grid_markets_export_cost, dtype=np.float32),
            OBS_KEY_GRID_EXPORT_POWER: np.array(grid_markets_export_power_unit, dtype=np.float32),
        }

    def _get_renewable_sources_features(
        self, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        renewable_gens_schedule = []
        renewable_gens_power = []
        for renewable_source_block in self._renewable_source_blocks:
            max_renewable_gen = renewable_source_block.sizes["block"].preexisting
            production_power_forecast = self._get_forecast(renewable_source_block.data["power_spec"])
            production_power_forecast_unit = production_power_forecast / max_renewable_gen
            clipped_production_power_unit_forecast = np.clip(production_power_forecast_unit, 0.0, 1.0)
            renewable_gens_schedule.append(clipped_production_power_unit_forecast)

            if optimization_result is not None:
                power_flow = optimization_result.get_power_flow(renewable_source_block, self.previous_time_step)
                production_power_unit = power_flow["out"] / max_renewable_gen
                clipped_production_power_unit = np.clip(production_power_unit, 0.0, 1.0)
                renewable_gens_power.append(clipped_production_power_unit)
            else:
                renewable_gens_power.append(0.0)

        return {
            OBS_KEY_RENEWABLES_SCHEDULE: np.array(renewable_gens_schedule, dtype=np.float32),
            OBS_KEY_RENEWABLES_POWER: np.array(renewable_gens_power, dtype=np.float32),
        }

    def _get_fixed_demands_features(self) -> dict[str, np.ndarray]:
        demands_powers = []
        for demand_block in self._fixed_demand_blocks:
            demand_load = demand_block.flows_apriori.loc["demand", self.current_time_step]
            demands_powers.append(demand_load)
        return {OBS_KEY_FIXED_DEMANDS: np.array(demands_powers, dtype=np.float32)}

    def _get_controllable_sources_features(
        self, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        power_units = []
        for controllable_source_block in self._controllable_source_blocks:
            max_power_wh = controllable_source_block.sizes["block"].preexisting
            if optimization_result is not None:
                power_flow = optimization_result.get_power_flow(controllable_source_block, self.previous_time_step)
                power_unit = power_flow["out"] / max_power_wh
                clipped_power_unit = np.clip(power_unit, 0.0, 1.0)
                power_units.append(clipped_power_unit)
            else:
                power_units.append(0.0)

        return {OBS_KEY_CONTROLLABLE_SOURCES_POWER: np.array(power_units, dtype=np.float32)}

    def _get_stationary_batteries_features(
        self, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        socs = []
        for stationary_battery_block in self._stationary_battery_blocks:
            max_energy_wh = stationary_battery_block.sizes["storage"].preexisting
            if optimization_result is not None:
                curr_energy_wh = optimization_result.get_stored_energy(
                    stationary_battery_block, self.previous_time_step
                )
                soc = curr_energy_wh / max_energy_wh
                clipped_soc = np.clip(soc, 0.0, 1.0)
                socs.append(clipped_soc)
            else:
                # TODO: SoC is currently always initialized to 100%, which is not realistic.
                socs.append(1.0)

        return {OBS_KEY_STATIONARY_BATTERIES_SOC: np.array(socs, dtype=np.float32)}

    def _get_forecast(self, time_series: pd.DataFrame) -> np.ndarray:
        forecast_horizon_dti = self._curr_horizon.dti[self._step_idx : self._step_idx + self._config.forecast_horizon]
        forecast_values = time_series.loc[forecast_horizon_dti].values

        return np.pad(forecast_values, (0, self._config.forecast_horizon - len(forecast_values)), constant_values=0.0)

    @override
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        _LOGGER.debug(f"Action at {self._step_idx}: {action}")
        reward = RewardComponents(self._config.reward_config)
        reward.step = self._step_idx
        for i, charge_power_frac in enumerate(action):
            block = self._electric_fleet_unit_blocks[i]
            try:
                normalized_charge_power_frac = self._normalize_charge_power(block, charge_power_frac, reward)
                _LOGGER.debug(f"Normalized action at {self._step_idx} for vehicle {i}: {normalized_charge_power_frac}")
                # By default the input power is adjusted. This is necessary to ensure that actions around 0 do not lead to unintended infeasibilities.
                # The charging power is always set to a power range. While this ensures that PyPSA
                # does not run into numerical issues, it hands over some control to PyPSA.
                # So if the agent sets the charge power to around 0 and output power is the default,
                # PyPSA might decide to discharge the batteries inside the given power range, which could lead to infeasibility.
                # By setting the input power by default instead, PyPSA can use the power range instead to compensate for any standing loss.
                if normalized_charge_power_frac >= 0:
                    self._optimization_problem.set_input_power_unit(
                        block,
                        normalized_charge_power_frac,
                        self.current_time_step,
                        power_unit_buffer=self._config.power_unit_buffer,
                    )
                else:
                    self._optimization_problem.set_output_power_unit(
                        block,
                        abs(normalized_charge_power_frac),
                        self.current_time_step,
                        power_unit_buffer=self._config.power_unit_buffer,
                    )
            except ValueError as e:
                _LOGGER.debug(f"Tried to charge vehicle {block.name} which is not available: {e}")

        optimization_status, optimization_result = self._optimization_problem.solve_time_step(self.current_time_step)
        infos = {INFO_KEY_REWARD_COMPONENTS: reward}
        if not self._train:
            infos = {INFO_KEY_OPTIMIZATION_RESULT: optimization_result}

        if optimization_status != optimization.OptimizationStatus.OPTIMAL or optimization_result is None:
            previous_profit = sum(filter(lambda x: x > 0, map(lambda x: x.total_reward, self._reward_history)))
            reward.infeasible = previous_profit
            _LOGGER.debug(
                f"Optimization failed at {self._step_idx}. Infeasibility penalty: {reward.infeasibility_reward}"
            )

            infos[INFO_KEY_STATUS] = EnvironmentStepStatus.INFEASIBLE

            terminated = False
            truncated = True
            return self._get_obs(), reward.total_reward, terminated, truncated, infos

        infos[INFO_KEY_STATUS] = EnvironmentStepStatus.OK

        self._compute_rewards(reward, optimization_result)
        _LOGGER.debug(f"Reward at {self._step_idx}: {reward}")

        self._step_idx += 1
        terminated, truncated = self._is_done()

        obs = self._get_obs(optimization_result)

        return obs, reward.total_reward, terminated, truncated, infos

    def _normalize_charge_power(self, block: blocks.ElectricFleetUnit, power: float, reward: RewardComponents) -> float:
        """Normalize the charge power to ensure it stays within the bounds of the energy system"""
        power = np.round(power, self._config.power_precision)
        # If the EV is not present at the charger it cannot be charged.
        # However, the RL agent might still try to charge the EVs. To avoid an increased amount of infeasible scenarios,
        # the agent just receives a penalty and can continue.
        is_at_base = block.log.loc[self.current_time_step, "atbase"]
        if not is_at_base:
            if np.round(power, self._config.power_precision) != 0.0:
                reward.atbase_violation += 1
            return 0.0

        # TODO: this is not clean.
        timestep_h = self._horizon._timestep.seconds / 3600.0

        nominal_battery_capacity_wh = block.sizes["storage"].preexisting
        current_battery_capacity_wh = (
            nominal_battery_capacity_wh
            * self._prev_obs[OBS_KEY_CARS_SOC][self._electric_fleet_unit_blocks.index(block)]
        )

        eff_charge = block.eff["chg_int"]
        eff_discharge = block.eff["dis_int"]
        max_charge_power_w = block.pwr_chg_max * eff_charge
        max_discharge_power_w = block.pwr_dis_max * eff_discharge

        soc_envelope = self._soc_envelopes[block]
        required_soc = soc_envelope[self.current_time_step]
        min_capacity_wh = max(
            0.0,
            (nominal_battery_capacity_wh * (required_soc + self._config.power_unit_buffer))
            - current_battery_capacity_wh,
        )

        max_energy_in_wh = max(0.0, nominal_battery_capacity_wh - current_battery_capacity_wh)
        # Always leave some remainder in battery for standing loss and numerical errors.
        max_energy_out_wh = max(0.0, current_battery_capacity_wh - (nominal_battery_capacity_wh * self._config.min_soc))

        soc_limited_charge_power_w = max_energy_in_wh / timestep_h
        soc_limited_discharge_power_w = max_energy_out_wh / timestep_h
        min_capacity_limited_power_w = (min_capacity_wh / timestep_h) / eff_charge

        if max_charge_power_w <= 0.0:
            upper_limit = 0.0
        else:
            upper_limit = min(1.0, soc_limited_charge_power_w / max_charge_power_w)

        if max_discharge_power_w <= 0.0:
            lower_limit = 0.0
        else:
            if min_capacity_limited_power_w > 0.0:
                lower_limit = np.clip(
                    (min_capacity_limited_power_w / max_charge_power_w) + self._config.power_unit_buffer, 0.0, 1.0
                )
            else:
                lower_limit = max(-1.0, -soc_limited_discharge_power_w / max_discharge_power_w)

        normalized_power = np.clip(power, lower_limit, upper_limit)

        power_diff = np.round(abs(abs(power) - abs(normalized_power)), 2)
        reward.power_diffs.append(power_diff)
        return normalized_power

    def _compute_rewards(self, reward: RewardComponents, optimization_result: optimization.OptimizationResult):
        reward.grid_opex = sum(
            [optimization_result.get_opex(block, self.current_time_step) for block in self._grid_connection_blocks]
        )
        reward.charge_opex = sum(
            [optimization_result.get_opex(block, self.current_time_step) for block in self._electric_fleet_unit_blocks]
        )

        if self._has_renewable_sources:
            reward.gen_opex = sum(
                [optimization_result.get_opex(block, self.current_time_step) for block in self._renewable_source_blocks]
            )

        soc_diffs = []
        for block in self._electric_fleet_unit_blocks:
            dsoc = block.log.loc[self.current_time_step, "dsoc"]
            if dsoc == 0.0:
                continue

            curr_stored_energy = optimization_result.get_stored_energy(block, self.current_time_step)
            curr_soc = curr_stored_energy / block.sizes["storage"].preexisting
            soc_diff = curr_soc - dsoc
            soc_diffs.append(soc_diff)

        reward.soc_diffs = soc_diffs

        self._reward_history.append(reward)

    def _is_done(self) -> tuple[bool, bool]:
        terminated = bool(self._step_idx >= self._max_step_idx)
        truncated = False
        return terminated, truncated
