import copy
import logging
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np
import pandas as pd
from typing_extensions import override

from revoletion import blocks, optimization, utils
from revoletion import scenario as scn

_LOGGER = logging.getLogger(__name__)


@dataclass
class RevoletionEnvironmentConfig:
    min_duration: pd.Timedelta = field(default_factory=lambda: pd.to_timedelta("6h"))
    max_duration: pd.Timedelta = field(default_factory=lambda: pd.to_timedelta("48h"))

    forecast_horizon: int = 8
    """The length of the forecast horizon that is provided in the observations to the client."""

    penalty_factor_grid_cost: float = 1.0
    """Factor applied to the costs of importing/exporting energy to the grid."""

    penalty_factor_charge_cost: float = 2.0
    """Weight applied to the costs of charging/discharging the EVs."""

    penalty_factor_dsoc: float = 3.0
    """Weight for the penalty if the agent does not met the SoC requirements. In all those cases, the scenario will become infeasible in the future time steps and an infeasibility penalty will also be applied."""

    penalty_factor_infeasible: float = 2.0
    """Weight for the penatly if the energy system is determined to be infeasible and cannot be optimizated."""

    penalty_factor_power_diff: float = 1.5
    """Weight for the penalty if the agent tries to charge with a power that would exceed the maximimal/minimum capacity of an EV."""

    penalty_factor_not_at_base: float = 2.0
    """Weight for the penalty if the agent tries to charge the car without an EV being present."""

    reward_factor_dsoc: float = 4.0
    """Weight of the reward for meeting a SoC requirement."""

    episode_length: int | None = None


ActType: TypeAlias = np.ndarray
ObsType: TypeAlias = dict[str, np.ndarray]

# Keys in the observation space for consistent access in custom agents.
OBS_KEY_TIME_FEATURES = "time_of_day"
OBS_KEY_CARS_AVAILABLE = "cars_available"
OBS_KEY_CARS_SOC = "cars_soc"
OBS_KEY_CARS_REQUIRED_SOCS = "cars_required_socs"
OBS_KEY_RENEWABLES_SCHEDULE = "renewables_schedule"
OBS_KEY_DEMANDS_SCHEDULE = "demands_schedule"
OBS_KEY_GRID_IMPORT_COSTS = "grid_import_costs"
OBS_KEY_GRID_EXPORT_COSTS = "grid_export_costs"


class RevoletionEnvironment(gym.Env[ObsType, ActType]):
    def __init__(
        self,
        scenario: scn.Scenario,
        horizon: utils.TimeSettings,
        config: RevoletionEnvironmentConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()

        self._scenario = scenario
        self._block_registry = scenario.block_registry
        self._horizon = horizon
        self._config = config or RevoletionEnvironmentConfig()
        self._logger = logger or logging.getLogger(__name__)

        self._step_size = pd.to_timedelta("15min")

        self._electric_fleet_unit_blocks = list(self._block_registry.get("ElectricFleetUnit", {}).values())

        self._grid_market_blocks = list(self._block_registry.get("GridMarket", {}).values())
        self._grid_connection_blocks = list(self._block_registry.get("GridConnection", {}).values())

        self._renewable_source_blocks = list(self._block_registry.get("RenewableSource", {}).values())
        self._has_renewable_gen = len(self._renewable_source_blocks) > 0

        self._fixed_demand_blocks = list(self._block_registry.get("FixedDemand", {}).values())
        self._has_fixed_demand = len(self._fixed_demand_blocks) > 0

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self._electric_fleet_unit_blocks),), dtype=np.float32
        )

        obs_dict = {
            OBS_KEY_TIME_FEATURES: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            OBS_KEY_CARS_SOC: gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._electric_fleet_unit_blocks),), dtype=np.float32
            ),
            OBS_KEY_CARS_AVAILABLE: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._electric_fleet_unit_blocks), self._config.forecast_horizon),
                dtype=np.float32,
            ),
            OBS_KEY_CARS_REQUIRED_SOCS: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._electric_fleet_unit_blocks), self._config.forecast_horizon),
                dtype=np.float32,
            ),
            OBS_KEY_GRID_IMPORT_COSTS: gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._grid_market_blocks),), dtype=np.float32
            ),
            OBS_KEY_GRID_EXPORT_COSTS: gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self._grid_market_blocks),), dtype=np.float32
            ),
        }

        if self._has_renewable_gen:
            obs_dict[OBS_KEY_RENEWABLES_SCHEDULE] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._renewable_source_blocks), self._config.forecast_horizon),
                dtype=np.float32,
            )

        if self._has_fixed_demand:
            obs_dict[OBS_KEY_DEMANDS_SCHEDULE] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self._fixed_demand_blocks), self._config.forecast_horizon),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(obs_dict)

        self._step_idx = 0
        self._max_step_idx = 0

        self._reward_history = []

        self._prev_obs = None

    @property
    def current_time_step(self) -> pd.DatetimeIndex:
        return self._curr_horizon.dti[self._step_idx]

    @property
    def previous_time_step(self) -> pd.DatetimeIndex:
        return self._curr_horizon.dti[max(self._step_idx - 1, 0)]

    @property
    def optimization_problem(self) -> optimization.OptimizationProblem:
        return self._optimization_problem

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
        self._max_step_idx = episode_length

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

        self._grid_cost_max, self._grid_cost_min = self._get_grid_cost_normalization_params()

        self._logger.debug(f"Reset environment: episode_length={episode_length}; start={self._curr_horizon.start}")

        return self._get_obs(), {}

    def _get_obs(self, optimization_result: optimization.OptimizationResult | None = None) -> ObsType:
        """Create the observation space for the current step.

        The observation space consists of several feature vectors which are individually collected.
        """
        state_dict = {}

        state_dict.update(self._get_time_features())

        state_dict.update(self._get_cars_features(optimization_result))

        state_dict.update(self._get_grid_markets_features())

        if self._has_renewable_gen:
            state_dict.update(self._get_renewable_sources_features())

        if self._has_fixed_demand:
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

        return {OBS_KEY_TIME_FEATURES: np.array([hour_sin, hour_cos, dow_sin, dow_cos], dtype=np.float32)}

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
                stored_energy = optimization_result.get_stored_energy(electric_fleet_unit_block, self.current_time_step)
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

    def _get_grid_markets_features(self) -> dict[str, np.ndarray]:
        grid_markets_import_cost = []
        grid_markets_export_cost = []
        for grid_market_block in self._grid_market_blocks:
            import_cost_max = grid_market_block.evaluators["g2s"].opt.spec_ep_operation.max()
            import_cost_min = grid_market_block.evaluators["g2s"].opt.spec_ep_operation.min()
            import_cost = grid_market_block.evaluators["g2s"].opt.spec_ep_operation.loc[self.current_time_step]
            import_cost_norm = np.clip(
                (import_cost - import_cost_min) / ((import_cost_max - import_cost_min) + 1e-8), 0.0, 1.0
            )
            grid_markets_import_cost.append(import_cost_norm)

            export_cost_max = grid_market_block.evaluators["s2g"].opt.spec_ep_operation.max()
            export_cost_min = grid_market_block.evaluators["s2g"].opt.spec_ep_operation.min()
            export_cost = grid_market_block.evaluators["s2g"].opt.spec_ep_operation.loc[self.current_time_step]
            export_cost_norm = np.clip(
                (export_cost - export_cost_min) / ((export_cost_max - export_cost_min) + 1e-8), 0.0, 1.0
            )
            grid_markets_export_cost.append(export_cost_norm)

        return {
            OBS_KEY_GRID_IMPORT_COSTS: np.array(grid_markets_export_cost, dtype=np.float32),
            OBS_KEY_GRID_EXPORT_COSTS: np.array(grid_markets_export_cost, dtype=np.float32),
        }

    def _get_renewable_sources_features(self) -> dict[str, np.ndarray]:
        renewable_gens_powers = []
        for renewable_source_block in self._renewable_source_blocks:
            max_renewable_gen = renewable_source_block.sizes["block"].preexisting
            production_power_forecast = (
                self._get_forecast(renewable_source_block.data["power_spec"]) / max_renewable_gen
            )
            clipped_production_power_forecast = np.clip(production_power_forecast, 0.0, 1.0)
            renewable_gens_powers.append(clipped_production_power_forecast)
        return {OBS_KEY_RENEWABLES_SCHEDULE: np.array(renewable_gens_powers, dtype=np.float32)}

    def _get_fixed_demands_features(self) -> dict[str, np.ndarray]:
        demands_powers = []
        for demand_block in self._fixed_demand_blocks:
            max_demand_load = demand_block.flows_apriori.loc["demand"].max()
            consumption_power_forecast = self._get_forecast(demand_block.flows_apriori["demand"]) / max_demand_load
            clipped_consumption_power_forecast = np.clip(consumption_power_forecast.values, 0.0, 1.0)
            demands_powers.append(clipped_consumption_power_forecast)
        return {OBS_KEY_DEMANDS_SCHEDULE: np.array(demands_powers, dtype=np.float32)}

    def _get_forecast(self, time_series: pd.DataFrame) -> np.ndarray:
        forecast_horizon_dti = self._curr_horizon.dti[self._step_idx : self._step_idx + self._config.forecast_horizon]
        forecast_values = time_series.loc[forecast_horizon_dti].values

        return np.pad(forecast_values, (0, self._config.forecast_horizon - len(forecast_values)), constant_values=0.0)

    @override
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        _LOGGER.debug(f"Action at {self._step_idx}: {action}")
        charge_power_penalties = []
        for i, charge_power_frac in enumerate(action):
            block = self._electric_fleet_unit_blocks[i]
            try:
                normalized_charge_power_frac, charge_power_penalty = self._normalize_charge_power(
                    block, charge_power_frac
                )
                _LOGGER.debug(f"Normalized action at {self._step_idx}: {normalized_charge_power_frac}")
                charge_power_penalties.append(charge_power_penalty)
                # By default the input power is adjusted. This is necessary to ensure that actions around 0 do not lead to unintended infeasibilities.
                # The charging power is always set to a power range. While this ensures that PyPSA
                # does not run into numerical issues, it hands over some control to PyPSA.
                # So if the agent sets the charge power to around 0 and output power is the default,
                # PyPSA might decide to discharge the batteries inside the given power range, which could lead to infeasibility.
                # By setting the input power by default instead, PyPSA can use the power range instead to compensate for any standing loss.
                if normalized_charge_power_frac >= 0:
                    self._optimization_problem.set_input_power_unit(
                        block, normalized_charge_power_frac, self.current_time_step
                    )
                else:
                    self._optimization_problem.set_output_power_unit(
                        block, abs(normalized_charge_power_frac), self.current_time_step
                    )
            except ValueError as e:
                _LOGGER.debug(f"Tried to charge vehicle {block.name} which is not available: {e}")

        optimization_status, optimization_result = self._optimization_problem.solve_time_step(self.current_time_step)
        if optimization_status != optimization.OptimizationStatus.OPTIMAL or optimization_result is None:
            previous_profit = max(sum(filter(lambda x: x >= 0.0, self._reward_history)), 1.0)
            reward = -self._config.penalty_factor_infeasible * previous_profit
            _LOGGER.debug(f"Optimization failed at {self._step_idx}. Infeasibility penalty: {reward}")
            return self._get_obs(), reward, False, True, {}

        self.last_optimization_result = optimization_result

        obs = self._get_obs(optimization_result)

        reward = self._compute_reward(optimization_result)
        reward -= max(charge_power_penalties)
        _LOGGER.debug(f"Reward at {self._step_idx}: {reward}")

        self._step_idx += 1
        terminated, truncated = self._is_done()

        return obs, reward, terminated, truncated, {}

    def _normalize_charge_power(self, block: blocks.ElectricFleetUnit, power: float) -> tuple[float, float]:
        """Normalize the charge power to ensure it stays within the bounds of the energy system"""
        power = np.round(power, 1)
        # If the EV is not present at the charger it cannot be charged.
        # However, the RL agent might still try to charge the EVs. To avoid an increased amount of infeasible scenarios,
        # the agent just receives a penalty and can continue.
        is_at_base = block.log.loc[self.current_time_step, "atbase"]
        if not is_at_base:
            return 0.0, abs(power) * self._config.penalty_factor_not_at_base

        nominal_battery_capacity_wh = block.sizes["storage"].preexisting
        current_battery_capacity_wh = (
            nominal_battery_capacity_wh
            * self._prev_obs[OBS_KEY_CARS_SOC][self._electric_fleet_unit_blocks.index(block)]
        )

        max_energy_in_wh = nominal_battery_capacity_wh - current_battery_capacity_wh
        # Always leave some remainder in battery for standing loss and numerical errors.
        max_energy_out_wh = current_battery_capacity_wh * 0.95

        # TODO: Either get the timestep from the dti or accept as param to env.
        timestep_hours = 0.25

        soc_limited_charge_power_w = max_energy_in_wh / timestep_hours
        soc_limited_discharge_power_w = max_energy_out_wh / timestep_hours

        max_charge_power_w = block.pwr_chg_max
        max_discharge_power_w = block.pwr_dis_max * block.eff["dis_int"]

        if max_charge_power_w <= 0.0:
            upper_limit = 0.0
        else:
            upper_limit = min(1.0, soc_limited_charge_power_w / max_charge_power_w)

        if max_discharge_power_w <= 0.0:
            lower_limit = 0.0
        else:
            lower_limit = max(-1.0, -soc_limited_discharge_power_w / max_discharge_power_w)

        normalized_power = np.round(np.clip(power, lower_limit, upper_limit), 1)

        power_diff = round(abs(abs(power) - abs(normalized_power)), 2)
        return normalized_power, power_diff * self._config.penalty_factor_power_diff

    def _compute_reward(self, optimization_result: optimization.OptimizationResult) -> float:
        total_costs = 0.0

        total_costs += (
            sum(
                [
                    self._normalize_grid_cost(optimization_result.get_opex(block, self.current_time_step))
                    for block in self._grid_connection_blocks
                ]
            )
            * self._config.penalty_factor_grid_cost
        )
        total_costs += (
            sum(
                [
                    self._normalize_charge_cost(optimization_result.get_opex(block, self.current_time_step))
                    for block in self._electric_fleet_unit_blocks
                ]
            )
            * self._config.penalty_factor_charge_cost
        )

        if self._has_renewable_gen:
            total_costs += sum(
                [optimization_result.get_opex(block, self.current_time_step) for block in self._renewable_source_blocks]
            )

        reward = -total_costs
        _LOGGER.debug(f"cost reward {reward}")

        for block in self._electric_fleet_unit_blocks:
            dsoc = block.log.loc[self.current_time_step, "dsoc"]
            if dsoc == 0.0:
                continue

            curr_stored_energy = optimization_result.get_stored_energy(block, self.current_time_step)
            curr_soc = curr_stored_energy / block.sizes["storage"].preexisting
            soc_diff = curr_soc - dsoc

            if soc_diff < 0.0:
                # Not enough SoC
                soc_penalty = abs(soc_diff) * self._config.penalty_factor_dsoc
            else:
                soc_penalty = -((1.0 + soc_diff) * self._config.reward_factor_dsoc)

            reward -= soc_penalty
            _LOGGER.debug(f"SoC penalty {soc_penalty}")

        self._reward_history.append(reward)

        return reward

    def _is_done(self) -> tuple[bool, bool]:
        terminated = False
        truncated = self._step_idx >= self._max_step_idx
        return terminated, truncated

    def _get_grid_cost_normalization_params(self) -> tuple[float, float]:
        max_grid_cost = 0.0
        min_grid_cost = 0.0
        for grid_conn_block in self._grid_connection_blocks:
            max_import_capacity = grid_conn_block.sizes["g2s"].preexisting
            max_export_capacity = grid_conn_block.sizes["s2g"].preexisting

            for grid_market_block in grid_conn_block.subblocks.values():
                grid_market_import_cost = (
                    grid_market_block.evaluators["g2s"].opt.spec_ep_operation[self._curr_horizon.dti].max()
                )
                max_grid_cost = max(max_import_capacity * grid_market_import_cost, max_grid_cost)

                grid_market_export_cost = (
                    grid_market_block.evaluators["s2g"].opt.spec_ep_operation[self._curr_horizon.dti].max()
                )
                min_grid_cost = min(max_export_capacity * grid_market_export_cost, min_grid_cost)

        return (max_grid_cost, min_grid_cost)

    def _normalize_grid_cost(self, grid_cost: float) -> float:
        return (grid_cost - self._grid_cost_min) / (self._grid_cost_max - self._grid_cost_min)

    def _normalize_charge_cost(self, ev_cost: float) -> float:
        return ev_cost
