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
from revoletion.rl.scenario_factory import (
    AtBaseHorizonInitializer,
    HorizonInitializer,
    InitialSocHorizonInitializer,
    SocEnvelopeHorizonInitialzer,
)

from . import _context as context
from . import _features as features
from . import _forecast_provider as forecast_provider
from . import _normalization as normalization
from . import utils as rl_utils

_LOGGER = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    penalty_factor_grid_opex: float = 10.0
    """Factor applied to the costs of importing/exporting energy to the grid."""

    penalty_factor_charge_opex: float = 0.0
    """Weight applied to the costs of charging/discharging the EVs."""

    penalty_factor_gen_opex: float = 1.0
    """Weight applied to the costs of charging/discharging the EVs."""

    penalty_factor_ext_charge_opex: float = 0.0

    penalty_base_dsoc: float = -0.5

    penalty_factor_dsoc: float = 1.0
    """Weight for the penalty if the agent does not met the SoC requirements."""

    reward_base_dsoc: float = 0.1

    reward_factor_dsoc: float = 0.0
    """Weight of the reward for meeting a SoC requirement."""

    penalty_continous_dsoc: bool = False

    penalty_factor_infeasible: float = 25.0
    """Weight for the penatly if the energy system is determined to be infeasible and cannot be optimizated."""

    penalty_scaling_infeasible: bool = False

    penalty_base_power_diff: float = -0.05

    penalty_factor_power_diff: float = -0.05
    """Weight for the penalty if the agent tries to charge with a power that would exceed the maximimal/minimum capacity of an EV."""

    penalty_base_atbase_violation: float = -0.05

    penalty_factor_atbase_violation: float = -0.05
    """Weight for the penalty if the agent tries to charge an EV even though the EV is currently not available at the charger."""

    reward_factor_step: float = 0.0
    """Small reward applied each step to encourage the agent to progress."""


@dataclass
class RevoletionEnvironmentConfig:
    min_steps: int = 48
    max_steps: int = 200

    episode_length: int | None = None
    """Length of one training/evaluation episode in time steps. If not given, episode length randomization is enabled."""

    forecast_horizon: int = 16
    """The length of the forecast horizon that is provided in the observations to the agent."""

    power_precision: int = 1

    power_unit_buffer: float = 1e-6
    """Buffer in both direction applied to the charge/discharge power unit. Used to give the optimizer some room for numerical tie breaking."""

    soc_min: float = 0.05

    dsoc_correction: bool = False

    reward_config: RewardConfig = field(default_factory=lambda: RewardConfig())


@dataclass
class RewardComponents:
    config: RewardConfig

    grid_opex: float = 0.0
    charge_opex: float = 0.0
    ext_charge_opex: float = 0.0
    gen_opex: float = 0.0
    power_diffs: list[float] = field(default_factory=list)
    atbase_violations: list[float] = field(default_factory=list)
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
        return -self.charge_opex * self.config.penalty_factor_charge_opex

    @property
    def ext_charge_opex_reward(self) -> float:
        return -self.ext_charge_opex * self.config.penalty_factor_ext_charge_opex

    @property
    def gen_opex_reward(self) -> float:
        if self.gen_opex > 0.0:
            return -self.gen_opex * self.config.penalty_factor_gen_opex
        return self.gen_opex

    @property
    def power_diff_reward(self) -> float:
        if len(self.power_diffs) == 0:
            return 0.0

        num_power_diffs = len(self.power_diffs)
        power_diffs_base_penalty = self.config.penalty_base_power_diff * num_power_diffs

        power_diff_sum = sum([abs(power_diff) for power_diff in self.power_diffs])
        scaled_power_diffs = power_diff_sum * self.config.penalty_factor_power_diff

        return power_diffs_base_penalty + scaled_power_diffs

    @property
    def atbase_violation_reward(self) -> float:
        if len(self.atbase_violations) == 0:
            return 0.0

        num_violations = len(self.atbase_violations)
        atbase_violations_base_penalty = self.config.penalty_base_atbase_violation * num_violations

        atbase_violations_sum = sum([abs(violation) for violation in self.atbase_violations])
        scaled_atbase_violations = atbase_violations_sum * self.config.penalty_factor_atbase_violation

        return atbase_violations_base_penalty + scaled_atbase_violations

    @property
    def soc_diff_reward(self) -> float:
        if len(self.soc_diffs) == 0:
            return 0.0

        reward = 0.0
        for soc_diff in self.soc_diffs:
            if soc_diff < 0.0:
                reward += self.config.penalty_base_dsoc + (soc_diff * self.config.penalty_factor_dsoc)
            else:
                reward += self.config.reward_base_dsoc + (soc_diff * self.config.reward_factor_dsoc)

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
        return f"{self.total_reward:.2f} (grid={self.grid_opex_reward:.2f}; gen={self.gen_opex_reward:.2f}; charge={self.charge_opex_reward}; power_diff={self.power_diff_reward:.2f}; atbase={self.atbase_violation_reward:.2f}; soc_diff={self.soc_diff_reward:.2f}; infeasibility={self.infeasibility_reward:.2f}; step={self.step_reward:.2f})"


class EnvironmentStepStatus(Enum):
    OK = auto()
    INFEASIBLE = auto()


ActType: TypeAlias = np.ndarray
ObsType: TypeAlias = dict[str, np.ndarray]

# Keys in the observation space for consistent access in custom agents.
OBS_KEY_TIME_FEATURES = "time_of_day"
OBS_KEY_EFUS_AVAILABLE = "efus_available"
OBS_KEY_EFUS_SOC = "efus_soc"
OBS_KEY_EFUS_REQUIRED_SOCS = "efus_required_socs"
OBS_KEY_RENEWABLES_POWER = "renewables_power"
OBS_KEY_RENEWABLES_SCHEDULE = "renewables_schedule"
OBS_KEY_FIXED_DEMANDS = "demands_schedule"
OBS_KEY_GRID_IMPORT_COSTS = "grid_import_costs"
OBS_KEY_GRID_IMPORT_POWER = "grid_import_power"
OBS_KEY_GRID_EXPORT_COSTS = "grid_export_costs"
OBS_KEY_GRID_EXPORT_POWER = "grid_export_power"
OBS_KEY_STATIONARY_BATTERIES_SOC = "stationary_batteries_soc"
OBS_KEY_CONTROLLABLE_SOURCES_POWER = "controllable_sources_power"
OBS_KEY_FLEETS_IN_POWER = "fleets_in_power"
OBS_KEY_FLEETS_OUT_POWER = "fleets_out_power"

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

        self._ctx = context.Context(scenario=scenario, horizon=horizon)
        limited_forecast_provider = forecast_provider.LimitedForecastProvider(
            forecast_horizon=self._config.forecast_horizon
        )
        self._normalization_provider = normalization.NormalizationProvider.from_ctx(self._ctx)
        self._feature_extractor = features.EnvironmentFeatureExtractor(
            forecast_provider=limited_forecast_provider,
            soc_min=self._config.soc_min,
            normalization_provider=self._normalization_provider,
        )

        self._horizon_initializer = HorizonInitializer(
            horizon_initializers=[
                SocEnvelopeHorizonInitialzer(),
                InitialSocHorizonInitializer(),
                AtBaseHorizonInitializer(),
            ]
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self._ctx.electric_fleet_unit_blocks),), dtype=np.float32
        )

        obs_dict = self._feature_extractor.get_observation_dict(self._ctx)
        self.observation_space = gym.spaces.Dict(obs_dict)

        self._step_idx = 0
        self._max_step_idx = 0

        self._reward_history = []

        self._prev_obs = None

        self._train = train

    @property
    def current_time_step(self) -> pd.DatetimeIndex:
        return self._ctx.current_time_step

    @property
    def previous_time_step(self) -> pd.DatetimeIndex:
        return self._ctx.previous_time_step

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        if self._config.episode_length is None:
            # Episode length randomization.
            min_steps = max(int(self._config.min_steps), 0)
            max_steps = min(int(self._config.max_steps), len(self._horizon) - 1)
            episode_length = self.np_random.integers(min_steps, max_steps)

            # Start time randomization.
            max_start_idx = len(self._horizon) - episode_length
            start_idx = self.np_random.integers(0, max_start_idx)
        else:
            start_idx = 0
            episode_length = self._config.episode_length

        self._step_idx = 0
        self._max_step_idx = episode_length

        # Instead of creating a new scenario with the new horizon and associated constraints (e.g., SoC envelope), we reset and reinitialize.
        new_horizon = self._horizon.cut(start_idx, episode_length)
        self._ctx.reset(new_horizon)

        self._horizon_initializer.initialize(self._scenario, new_horizon)

        opt_problem_config = optimization.OptimizationProblemConfig(
            cost_eps=self._scenario.cost_eps,
            solver=optimization.Solver.HIGHS,
            invest=False,
            warmstart=True,
            enforce_soc_constraints=False,
        )

        self._optimization_problem = optimization.create_optimization_problem(
            backend=optimization.OptimizationBackend.PYPSA,
            scenario=self._scenario,
            horizon=self._ctx.horizon,
            logger=self._logger,
            config=opt_problem_config,
        )

        self._reward_history = []
        self._prev_obs = None

        self._logger.debug(f"Reset environment: episode_length={episode_length}; start={self._ctx.horizon.start}")

        return self._get_obs(), {}

    def _get_obs(self, optimization_result: optimization.OptimizationResult | None = None) -> ObsType:
        """Create the observation space for the current step.

        The observation space consists of several feature vectors which are individually collected.
        """

        state_dict = self._feature_extractor.extract_all_features(self._ctx, optimization_result)

        _LOGGER.debug(f"Observation at {self._step_idx}: {state_dict}")

        self._prev_obs = state_dict

        return state_dict

    @override
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        _LOGGER.debug(f"Action at {self._step_idx} ({self.current_time_step}): {action}")
        reward = RewardComponents(self._config.reward_config)
        reward.step = self._step_idx

        initial_charge_power_fracs = {
            electric_fleet_unit: float(action[self._ctx.get_efu_index(electric_fleet_unit)])
            for electric_fleet_unit in self._ctx.electric_fleet_unit_blocks
        }
        normalized_charge_powers = {}

        for fleet, efus in self._ctx.electric_fleets.items():
            fleet_charge_power_fracs = {}
            for efu in efus:
                charge_power_frac = initial_charge_power_fracs[efu]
                normalized_charge_power_frac = self._normalize_charge_power(efu, charge_power_frac, reward)
                fleet_charge_power_fracs[efu] = normalized_charge_power_frac

            fleet_normalized_charge_power_fracs = self._distribute_charge_power_in_fleet(
                fleet, fleet_charge_power_fracs
            )
            normalized_charge_powers.update(fleet_normalized_charge_power_fracs)

        reward.charge_opex += sum(abs(charge_power_frac) for charge_power_frac in normalized_charge_powers.values())

        for electric_fleet_unit, normalized_charge_power_frac in normalized_charge_powers.items():
            original_power_frac = initial_charge_power_fracs[electric_fleet_unit]
            power_frac_diff = abs(original_power_frac - normalized_charge_power_frac)
            if power_frac_diff > 0.1:
                reward.power_diffs.append(power_frac_diff)

            try:
                # By default the input power is adjusted. This is necessary to ensure that actions around 0 do not lead to unintended infeasibilities.
                # The charging power is always set to a power range. While this ensures that PyPSA
                # does not run into numerical issues, it hands over some control to PyPSA.
                # So if the agent sets the charge power to around 0 and output power is the default,
                # PyPSA might decide to discharge the batteries inside the given power range, which could lead to infeasibility.
                # By setting the input power by default instead, PyPSA can use the power range instead to compensate for any loss or numerical issues.
                if normalized_charge_power_frac >= 0:
                    self._optimization_problem.set_input_power_unit(
                        electric_fleet_unit,
                        normalized_charge_power_frac,
                        self.current_time_step,
                        power_unit_buffer=self._config.power_unit_buffer,
                    )
                else:
                    self._optimization_problem.set_output_power_unit(
                        electric_fleet_unit,
                        abs(normalized_charge_power_frac),
                        self.current_time_step,
                        power_unit_buffer=self._config.power_unit_buffer,
                    )
            except ValueError as e:
                _LOGGER.debug(f"Failed to charge vehicle {electric_fleet_unit.name}: {e}")

        format_charge_power_fracs = {
            efu.name: round(norm_frac, self._config.power_precision)
            for efu, norm_frac in normalized_charge_powers.items()
        }
        _LOGGER.debug(f"Normalized action at {self._step_idx}: {format_charge_power_fracs}")

        optimization_status, optimization_result = self._optimization_problem.solve_time_step(self.current_time_step)
        infos: dict[str, Any] = {INFO_KEY_REWARD_COMPONENTS: reward}
        if not self._train:
            infos[INFO_KEY_OPTIMIZATION_RESULT] = optimization_result

        if optimization_status != optimization.OptimizationStatus.OPTIMAL or optimization_result is None:
            if self._config.reward_config.penalty_scaling_infeasible:
                previous_profit = sum(filter(lambda x: x > 0, map(lambda x: x.total_reward, self._reward_history)))
                reward.infeasible = previous_profit
            else:
                reward.infeasible = 1.0
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
        self._ctx.step()
        terminated, truncated = self._is_done()

        obs = self._get_obs(optimization_result)

        return obs, reward.total_reward, terminated, truncated, infos

    def _distribute_charge_power_in_fleet(
        self, fleet: blocks.Fleet, charge_power_fracs: dict[blocks.ElectricFleetUnit, float]
    ) -> dict[blocks.ElectricFleetUnit, float]:
        """
        Distributes charging power across fleet units respecting fleet-level constraints.

        Strategy:
        1. Check if aggregate power exceeds fleet limits
        2. If yes, prioritize vehicles below target SoC (deficit-first)
        3. Distribute remaining capacity equally among other vehicles

        Returns:
            Adjusted charge power fractions respecting fleet constraints
        """
        max_fleet_charge_power = fleet.pwr_lim_s2f
        max_fleet_discharge_power = fleet.pwr_lim_f2s

        # Calculate aggregate power demand at unit level (before efficiency losses)
        gross_charge_power = 0.0
        gross_discharge_power = 0.0

        for efu, frac in charge_power_fracs.items():
            if frac >= 0.0:
                charge_power = efu.pwr_chg_max * frac
                gross_charge_power += charge_power
                gross_discharge_power -= charge_power * efu.eff["chg_int"]
            else:
                charge_power = efu.pwr_dis_max * abs(frac)
                gross_discharge_power += charge_power
                gross_charge_power -= charge_power * efu.eff["chg_int"]

        gross_charge_power = max(gross_charge_power, 0.0)
        gross_discharge_power = max(gross_discharge_power, 0.0)

        adjusted_charge_power_fracs = charge_power_fracs.copy()

        if gross_charge_power >= max_fleet_charge_power:
            adjustments = self._allocate_limited_charge_power(fleet, charge_power_fracs)
            adjusted_charge_power_fracs.update(adjustments)

        if gross_discharge_power >= max_fleet_discharge_power:
            adjustments = self._allocate_limited_discharge_power(fleet, charge_power_fracs)
            adjusted_charge_power_fracs.update(adjustments)

        return adjusted_charge_power_fracs

    def _allocate_limited_charge_power(
        self,
        fleet: blocks.Fleet,
        charge_power_fracs: dict[blocks.ElectricFleetUnit, float],
    ) -> dict[blocks.ElectricFleetUnit, float]:
        max_fleet_charge_power = fleet.pwr_lim_s2f

        # Only consider vehicles requesting charging and currently available
        efu_charging_requests = {efu: frac for efu, frac in charge_power_fracs.items() if frac > 0.0}

        if not efu_charging_requests:
            return {}

        # Calculate SoC deficits
        priority_list = []
        remainder_list = []

        for efu in efu_charging_requests.keys():
            current_soc = self._prev_obs[OBS_KEY_EFUS_SOC][self._ctx.get_efu_index(efu)]
            soc_envelope_horizon = self._ctx.horizon.cut(
                self._ctx.step_idx,
                min(self._config.forecast_horizon, len(self._ctx.horizon) - self._ctx.step_idx - 1),
            )
            soc_envelope = rl_utils.get_soc_envelope(efu, soc_envelope_horizon)
            target_soc = soc_envelope[self.current_time_step]
            soc_deficit = target_soc - current_soc

            if soc_deficit > 0:
                priority_list.append((soc_deficit, efu))
            else:
                remainder_list.append(efu)

        # Sort by deficit (largest first)
        priority_list.sort(key=lambda x: x[0], reverse=True)

        adjusted_fracs = {}
        remaining_power = max_fleet_charge_power

        # Allocate to priority vehicles first (full power if possible)
        for _, efu in priority_list:
            allocatable_power = min(efu.pwr_chg_max, remaining_power)
            allocatable_power_frac = allocatable_power / efu.pwr_chg_max

            allocated_power_frac = min(efu_charging_requests[efu], allocatable_power_frac)
            adjusted_fracs[efu] = allocated_power_frac
            remaining_power = max(remaining_power - (allocated_power_frac * efu.pwr_chg_max), 0.0)

        # Distribute remaining power equally among remainder vehicles
        if remainder_list and remaining_power > 0:
            avg_power_per_vehicle = remaining_power / len(remainder_list)
            for efu in remainder_list:
                allocatable_power = min(avg_power_per_vehicle, efu.pwr_chg_max)
                adjusted_fracs[efu] = allocatable_power / efu.pwr_chg_max

        for efu in efu_charging_requests.keys():
            if efu not in adjusted_fracs:
                adjusted_fracs[efu] = 0.0

        return adjusted_fracs

    def _allocate_limited_discharge_power(
        self,
        fleet: blocks.Fleet,
        charge_power_fracs: dict[blocks.ElectricFleetUnit, float],
    ) -> dict[blocks.ElectricFleetUnit, float]:
        max_fleet_discharge_power = fleet.pwr_lim_f2s

        discharging_efus = {efu: abs(frac) for efu, frac in charge_power_fracs.items() if frac < 0.0}

        if not discharging_efus:
            return {}

        # Calculate total requested discharge
        total_discharge_request = sum(efu.pwr_dis_max * frac for efu, frac in discharging_efus.items())

        # Proportional scaling factor
        scaling_factor = max_fleet_discharge_power / total_discharge_request

        adjusted_fracs = {}
        for efu, frac_abs in discharging_efus.items():
            adjusted_fracs[efu] = -frac_abs * scaling_factor

        return adjusted_fracs

    def _normalize_charge_power(
        self, block: blocks.ElectricFleetUnit, power_frac: float, reward: RewardComponents
    ) -> float:
        """Normalize the charge power to ensure it stays within the bounds of the energy system"""
        # power_frac = np.round(power_frac, self._config.power_precision)
        # If the EV is not present at the charger it cannot be charged.
        # However, the RL agent might still try to charge the EVs. To avoid an increased amount of infeasible scenarios,
        # the agent just receives a penalty and can continue.
        is_at_base = block.log.loc[self.current_time_step, "atbase"]
        if not is_at_base:
            if abs(power_frac) >= 0.1:
                reward.atbase_violations.append(power_frac)
            return 0.0

        timestep_h = self._ctx.horizon.timestep.hours

        nominal_battery_capacity_wh = block.sizes["storage"].preexisting
        current_battery_capacity_wh = (
            nominal_battery_capacity_wh
            * self._prev_obs[OBS_KEY_EFUS_SOC][self._ctx.electric_fleet_unit_blocks.index(block)]
        )

        eff_charge = block.eff["chg_int"]
        eff_discharge = block.eff["dis_int"]
        max_charge_power_w = block.pwr_chg_max * eff_charge
        max_discharge_power_w = block.pwr_dis_max * eff_discharge

        if self._config.dsoc_correction:
            soc_envelope_horizon = self._ctx.horizon.cut(
                self._ctx.step_idx, min(self._config.forecast_horizon, len(self._ctx.horizon) - self._ctx.step_idx - 1)
            )
            soc_envelope = rl_utils.get_soc_envelope(block, soc_envelope_horizon)
            required_soc = soc_envelope[self.current_time_step] + self._config.soc_min
            min_capacity_wh = max(
                0.0,
                (nominal_battery_capacity_wh * required_soc) - current_battery_capacity_wh,
            )
        else:
            min_capacity_wh = 0.0

        max_energy_in_wh = max(0.0, nominal_battery_capacity_wh - current_battery_capacity_wh)
        # Always leave some remainder in battery for standing loss and numerical errors.
        max_energy_out_wh = max(0.0, current_battery_capacity_wh - (nominal_battery_capacity_wh * self._config.soc_min))

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

        normalized_power = np.clip(power_frac, lower_limit, upper_limit)

        return normalized_power

    def _compute_rewards(self, reward: RewardComponents, optimization_result: optimization.OptimizationResult):
        reward.grid_opex = self._compute_grid_opex(optimization_result)
        reward.gen_opex = self._compute_generator_opex(optimization_result)

        if self._config.reward_config.penalty_continous_dsoc:
            self._compute_continous_soc_diff_reward(reward, optimization_result)
        else:
            self._compute_discrete_soc_diff_reward(reward, optimization_result)

        ext_charge_opex = 0
        for block in self._ctx.electric_fleet_unit_blocks:
            power_flows = optimization_result.get_power_flow(block, self.current_time_step)
            ext_charge_opex += power_flows["ext_ac"]
            ext_charge_opex += power_flows["ext_dc"]

        reward.ext_charge_opex = ext_charge_opex

        self._reward_history.append(reward)

    def _compute_continous_soc_diff_reward(
        self, reward: RewardComponents, optimization_result: optimization.OptimizationResult
    ) -> None:
        soc_diffs = []
        for block in self._ctx.electric_fleet_unit_blocks:
            soc_min = block.states.loc[self.current_time_step, "soc_min"]

            curr_stored_energy = optimization_result.get_stored_energy(block, self.current_time_step)
            curr_soc = curr_stored_energy / block.sizes["storage"].preexisting
            soc_diff = curr_soc - soc_min
            soc_diffs.append(soc_diff)

        reward.soc_diffs = soc_diffs

    def _compute_discrete_soc_diff_reward(
        self, reward: RewardComponents, optimization_result: optimization.OptimizationResult
    ) -> None:
        soc_diffs = []
        for block in self._ctx.electric_fleet_unit_blocks:
            required_soc = block.log.loc[self.current_time_step, "dsoc"]
            if required_soc == 0.0:
                continue

            curr_stored_energy = optimization_result.get_stored_energy(block, self.current_time_step)
            curr_soc = curr_stored_energy / block.sizes["storage"].preexisting

            if curr_soc >= required_soc:
                soc_diffs.append(1.0)
            else:
                soc_diffs.append(curr_soc - required_soc)

        reward.soc_diffs = soc_diffs

    def _compute_grid_opex(self, optimization_result: optimization.OptimizationResult) -> float:
        grid_opex = 0.0
        for grid_market_block in self._ctx.grid_market_blocks:
            power_flows = optimization_result.get_power_flow(grid_market_block, self.current_time_step)

            grid_import_costs_per_unit = grid_market_block.evaluators["g2s"].opt.spec_ep_operation[
                self.current_time_step
            ]
            raw_grid_import_costs = power_flows["in"] * grid_import_costs_per_unit
            grid_opex += self._normalization_provider.normalize_input_opex(raw_grid_import_costs, grid_market_block)

            grid_export_profit_per_unit = grid_market_block.evaluators["s2g"].opt.spec_ep_operation[
                self.current_time_step
            ]
            raw_grid_export_profit = power_flows["out"] * grid_export_profit_per_unit
            grid_opex += self._normalization_provider.normalize_output_opex(raw_grid_export_profit, grid_market_block)
        return grid_opex

    def _compute_generator_opex(self, optimization_result: optimization.OptimizationResult) -> float:
        gen_opex = 0.0
        for source_block in self._ctx.renewable_source_blocks + self._ctx.controllable_source_blocks:
            power_flows = optimization_result.get_power_flow(source_block, self.current_time_step)
            variable_costs = source_block.evaluators["block"].opt.spec_ep_operation[self.current_time_step]

            raw_opex = power_flows["out"] * variable_costs
            normalized_opex = self._normalization_provider.normalize_output_opex(raw_opex, source_block)

            gen_opex += normalized_opex

        return gen_opex

    def _is_done(self) -> tuple[bool, bool]:
        terminated = bool(self._step_idx >= self._max_step_idx)
        truncated = False
        return terminated, truncated
