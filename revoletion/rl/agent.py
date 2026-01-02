import abc
import collections
import enum
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
import torch.nn as nn
import typing_extensions
from stable_baselines3.common import policies
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from typing_extensions import Self

from revoletion import optimization, utils
from revoletion import scenario as scn

from ._features import (
    OBS_KEY_CONTROLLABLE_SOURCES_POWER,
    OBS_KEY_EFUS_AVAILABLE,
    OBS_KEY_EFUS_REQUIRED_SOCS,
    OBS_KEY_EFUS_SOC,
    OBS_KEY_FIXED_DEMANDS,
    OBS_KEY_FLEETS_IN_POWER,
    OBS_KEY_FLEETS_OUT_POWER,
    OBS_KEY_GRID_EXPORT_COSTS,
    OBS_KEY_GRID_EXPORT_POWER,
    OBS_KEY_GRID_IMPORT_COSTS,
    OBS_KEY_GRID_IMPORT_POWER,
    OBS_KEY_RENEWABLES_POWER,
    OBS_KEY_RENEWABLES_SCHEDULE,
    OBS_KEY_STATIONARY_BATTERIES_SOC,
    OBS_KEY_TIME_FEATURES,
)
from .environment import (
    INFO_KEY_OPTIMIZATION_RESULT,
    INFO_KEY_REWARD_COMPONENTS,
    INFO_KEY_STATUS,
    ActType,
    EnvironmentStepStatus,
    ObsType,
    RevoletionEnvironment,
    RevoletionEnvironmentConfig,
    RewardComponents,
    RewardConfig,
)

_LOGGER = logging.getLogger(__name__)


class AgentAlgorithm(enum.Enum):
    RANDOM = "random"
    FULL_CHARGING = "full-charge"
    FULL_DISCHARGE = "full-discharge"
    IDLE = "idle"
    BASIC = "basic"
    OPTIMAL = "optimal"

    PPO = "ppo"
    TD3 = "td3"
    A2C = "a2c"
    SAC = "sac"
    DDPG = "ddpg"

    def needs_training(self) -> bool:
        return self not in {
            AgentAlgorithm.RANDOM,
            AgentAlgorithm.FULL_CHARGING,
            AgentAlgorithm.FULL_DISCHARGE,
            AgentAlgorithm.BASIC,
            AgentAlgorithm.IDLE,
        }

    def __str__(self) -> str:
        return self.value


@dataclass
class AgentConfig:
    learning_rate: float = 0.001
    gamma: float = 0.99
    seed: int = 42
    n_steps: int = 1
    tensorboard_log: str | None = "/tmp/revol"
    gradient_steps: int | None = None
    target_policy_noise: float | None = None
    target_noise_clip: float | None = None
    train_freq: int | tuple[int, str] | None = None
    batch_size: int | None = None
    learning_starts: int | None = None
    buffer_size: int | None = None

    @classmethod
    def default_for_algorithm(cls, algorithm: AgentAlgorithm) -> Self:
        match algorithm:
            case AgentAlgorithm.PPO:
                return cls(learning_rate=0.0003, gamma=0.99, n_steps=128, batch_size=64)
            case AgentAlgorithm.TD3:
                return cls(
                    learning_rate=0.0001,
                    gamma=0.99,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    batch_size=256,
                    learning_starts=10_000,
                    buffer_size=50_000,
                )
            case AgentAlgorithm.A2C:
                return cls(learning_rate=0.0007, n_steps=5)
            case AgentAlgorithm.SAC:
                return cls(
                    learning_rate=0.0003,
                    gamma=0.99,
                    batch_size=256,
                    learning_starts=10_000,
                    buffer_size=50_000,
                )
            case AgentAlgorithm.DDPG:
                return cls(learning_rate=0.0001)
            case _:
                return cls()


class RevoletionAgent(abc.ABC):
    def __init__(self, algorithm: AgentAlgorithm) -> None:
        self._algorithm = algorithm

    @property
    def algorithm(self) -> AgentAlgorithm:
        return self._algorithm

    def learn(self, total_timesteps: int) -> None:
        pass

    @abc.abstractmethod
    def predict(
        self,
        obs: ObsType,
        deterministic: bool = False,
    ) -> ActType: ...


class RevoletionSB3Agent(RevoletionAgent):
    def __init__(self, algorithm: AgentAlgorithm, sb3_agent) -> None:
        super().__init__(algorithm)
        self._sb3_agent = sb3_agent

    def save(self, model_path: Path) -> None:
        self._sb3_agent.save(model_path)

    def learn(self, total_timesteps: int) -> None:
        return self._sb3_agent.learn(
            total_timesteps,
            callback=_TracingCallback(),
        )

    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        return self._sb3_agent.predict(obs, deterministic=deterministic)


_ScenarioFactoryT = typing.Callable[[], scn.Scenario]


def build_rl_environment(
    scenario_or_scenario_factory: scn.Scenario | _ScenarioFactoryT,
    horizon: utils.TimeSettings,
    train: bool = True,
    env_config: RevoletionEnvironmentConfig | None = None,
) -> gym.Env[ObsType, ActType]:
    if isinstance(scenario_or_scenario_factory, scn.Scenario):
        scenario = scenario_or_scenario_factory
    else:
        scenario = scenario_or_scenario_factory()

    env_config = env_config or RevoletionEnvironmentConfig(
        reward_config=RewardConfig(),
        episode_length=100 if train else len(horizon),
    )
    env = RevoletionEnvironment(scenario, horizon, config=env_config, train=train)
    return env


def evaluate_with_agent(
    scenario: scn.Scenario,
    agent: RevoletionAgent,
    horizon: utils.TimeSettings,
) -> tuple[float, dict[str, list[float]], optimization.OptimizationResult | None]:
    base_env = build_rl_environment(scenario, horizon, train=False)
    vec_env = DummyVecEnv([lambda: base_env])
    vec_norm_env = VecNormalize(vec_env, training=False)
    obs = vec_norm_env.reset()
    total_reward = 0.0
    info_dict = {}

    action_trace = []

    for _ in horizon.dti:
        action = agent.predict(obs, deterministic=True)
        if isinstance(action, tuple):
            action = action[0]
        action_trace.append(action)

        obs, reward, done, infos = vec_norm_env.step(action)
        total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        done_flag = done[0] if isinstance(done, np.ndarray) else done
        info_dict = infos[0] if isinstance(infos, list) and len(infos) > 0 else infos

        if done_flag:
            break

    actions_map = collections.defaultdict(list)
    for actions_at_timestep in action_trace:
        for efu in base_env._ctx.electric_fleet_unit_blocks:
            action = actions_at_timestep[base_env._ctx.get_efu_index(efu)]
            actions_map[efu].append(action)

    if info_dict[INFO_KEY_STATUS] == EnvironmentStepStatus.INFEASIBLE:
        return total_reward, None

    return total_reward, actions_map, info_dict.get(INFO_KEY_OPTIMIZATION_RESULT)


def _get_path_for_algorithm(algorithm: AgentAlgorithm, name: str, models_path: Path) -> Path:
    return (models_path / algorithm.value / f"{name}-{algorithm.value}").with_suffix(".zip")


def save_agent(agent: RevoletionAgent, name: str, models_path: Path) -> None:
    # Only save trained agents
    if not agent.algorithm.needs_training():
        return

    if not isinstance(agent, RevoletionSB3Agent):
        raise ValueError("Only agents based on stable-baselines3 can be saved")

    save_path = _get_path_for_algorithm(agent.algorithm, name, models_path)
    agent.save(save_path)
    _LOGGER.info(f"Saved agent {agent.algorithm} to {save_path}")


def load_agent(algorithm: AgentAlgorithm, name: str, models_path: Path) -> RevoletionAgent | None:
    if not algorithm.needs_training():
        return _create_non_trainable_agent(algorithm)

    load_path = _get_path_for_algorithm(algorithm, name, models_path)

    if not load_path.exists():
        return None

    sb3_type = _get_sb3_type(algorithm)
    sb3_agent = _load_sb3_agent(sb3_type, load_path)

    agent = RevoletionSB3Agent(algorithm=algorithm, sb3_agent=sb3_agent)

    _LOGGER.info(f"Loaded agent {algorithm} from {load_path}")
    return agent


_SCALAR_FEATURES = {
    OBS_KEY_TIME_FEATURES,
    OBS_KEY_FLEETS_IN_POWER,
    OBS_KEY_FLEETS_OUT_POWER,
    OBS_KEY_RENEWABLES_POWER,
    OBS_KEY_STATIONARY_BATTERIES_SOC,
    OBS_KEY_CONTROLLABLE_SOURCES_POWER,
    OBS_KEY_FIXED_DEMANDS,
}


class StructuredEnergyExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        forecast_horizon: int = 16,
        features_dim: int = 256,
        vehicle_embed_dim: int = 64,
        forecast_embed_dim: int = 32,
        use_attention: bool = True,
        num_attention_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        self._observation_space = observation_space.spaces
        self._forecast_horizon = forecast_horizon

        self._use_attention = use_attention
        self._vehicle_embed_dim = vehicle_embed_dim
        self._forecast_embed_dim = forecast_embed_dim

        # Each vehicle has: SoC (1) + availability forecast (H) + required SoC forecast (H)
        vehicle_input_dim = 1 + 2 * self._forecast_horizon

        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_input_dim, vehicle_embed_dim * 2),
            nn.LayerNorm(vehicle_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(vehicle_embed_dim * 2, vehicle_embed_dim),
            nn.LayerNorm(vehicle_embed_dim),
            nn.ReLU(),
        )

        # Attention-based pooling across vehicles
        if use_attention:
            self.vehicle_attention = nn.MultiheadAttention(
                embed_dim=vehicle_embed_dim,
                num_heads=num_attention_heads,
                batch_first=True,
                dropout=0.1,
            )
            # Learnable query token for pooling
            self.vehicle_query = nn.Parameter(torch.randn(1, 1, vehicle_embed_dim))

        vehicle_output_dim = vehicle_embed_dim

        # ============================================
        # 2. Renewable Generation Forecast Processing
        # ============================================
        if OBS_KEY_RENEWABLES_SCHEDULE in self._observation_space:
            self.n_renewables = self._observation_space[OBS_KEY_RENEWABLES_SCHEDULE].shape[0]

            # 1D CNN to extract temporal patterns from forecasts
            self.renewable_encoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.n_renewables,
                    out_channels=forecast_embed_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=forecast_embed_dim,
                    out_channels=forecast_embed_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # Pool to single value per channel
            )
            renewable_output_dim = forecast_embed_dim
        else:
            renewable_output_dim = 0

        # ============================================
        # 3. Grid Features Processing
        # ============================================
        if OBS_KEY_GRID_IMPORT_COSTS in self._observation_space:
            # import_costs, import_power, export_costs, export_power
            grid_input_dim = self._observation_space[OBS_KEY_GRID_IMPORT_COSTS].shape[0] * 4
            grid_output_dim = 32
            self.grid_encoder = nn.Sequential(
                nn.Linear(grid_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, grid_output_dim),
                nn.ReLU(),
            )
        else:
            grid_output_dim = 0

        # ============================================
        # 4. Scalar Features Processing
        # ============================================
        scalar_dim = 0

        for scalar_feature in _SCALAR_FEATURES:
            if scalar_feature not in self._observation_space:
                continue
            scalar_dim += self._observation_space[scalar_feature].shape[0]

        if scalar_dim > 0:
            scalar_output_dim = 64
            self.scalar_encoder = nn.Sequential(
                nn.Linear(scalar_dim, 128),
                nn.ReLU(),
                nn.Linear(128, scalar_output_dim),
                nn.ReLU(),
            )
        else:
            scalar_output_dim = 0

        # ============================================
        # 5. Fusion Layer
        # ============================================
        total_dim = vehicle_output_dim + renewable_output_dim + grid_output_dim + scalar_output_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Dictionary of observations from the environment

        Returns:
            Encoded features of shape (batch_size, features_dim)
        """
        encoded_parts = []

        # Concatenate per-vehicle features
        vehicle_features = torch.cat(
            [
                observations[OBS_KEY_EFUS_SOC].unsqueeze(-1),  # (B, N, 1)
                observations[OBS_KEY_EFUS_AVAILABLE],  # (B, N, H)
                observations[OBS_KEY_EFUS_REQUIRED_SOCS],  # (B, N, H)
            ],
            dim=-1,
        )  # (B, N, 1+2H)

        # Encode each vehicle
        B, N, F = vehicle_features.shape
        vehicle_features_flat = vehicle_features.view(B * N, F)
        vehicle_encoded = self.vehicle_encoder(vehicle_features_flat)
        vehicle_encoded = vehicle_encoded.view(B, N, self._vehicle_embed_dim)

        # Pool across vehicles
        if self._use_attention:
            # Use learnable query for attention pooling
            query = self.vehicle_query.expand(B, -1, -1)  # (B, 1, D)
            pooled_vehicle, _ = self.vehicle_attention(query, vehicle_encoded, vehicle_encoded)
            pooled_vehicle = pooled_vehicle.squeeze(1)  # (B, D)
        else:
            # Simple mean pooling
            pooled_vehicle = vehicle_encoded.mean(dim=1)  # (B, D)

        encoded_parts.append(pooled_vehicle)

        # ============================================
        # 2. Process Renewable Forecasts
        # ============================================
        if OBS_KEY_RENEWABLES_SCHEDULE in observations:
            renewable_forecast = observations[OBS_KEY_RENEWABLES_SCHEDULE]  # (B, N, H)
            renewable_encoded = self.renewable_encoder(renewable_forecast)  # (B, D, 1)
            renewable_encoded = renewable_encoded.squeeze(-1)  # (B, D)
            encoded_parts.append(renewable_encoded)

        # ============================================
        # 3. Process Grid Features
        # ============================================
        if OBS_KEY_GRID_IMPORT_COSTS in observations:
            grid_features = torch.cat(
                [
                    observations[OBS_KEY_GRID_IMPORT_COSTS],
                    observations[OBS_KEY_GRID_IMPORT_POWER],
                    observations[OBS_KEY_GRID_EXPORT_COSTS],
                    observations[OBS_KEY_GRID_EXPORT_POWER],
                ],
                dim=-1,
            )
            grid_encoded = self.grid_encoder(grid_features)
            encoded_parts.append(grid_encoded)

        # ============================================
        # 4. Process Scalar Features
        # ============================================
        scalar_features = []

        if OBS_KEY_TIME_FEATURES in observations:
            scalar_features.append(observations[OBS_KEY_TIME_FEATURES])
        if OBS_KEY_FLEETS_IN_POWER in observations:
            scalar_features.append(observations[OBS_KEY_FLEETS_IN_POWER])
            scalar_features.append(observations[OBS_KEY_FLEETS_OUT_POWER])
        if OBS_KEY_RENEWABLES_POWER in observations:
            scalar_features.append(observations[OBS_KEY_RENEWABLES_POWER])
        if OBS_KEY_STATIONARY_BATTERIES_SOC in observations:
            scalar_features.append(observations[OBS_KEY_STATIONARY_BATTERIES_SOC])
        if OBS_KEY_CONTROLLABLE_SOURCES_POWER in observations:
            scalar_features.append(observations[OBS_KEY_CONTROLLABLE_SOURCES_POWER])
        if OBS_KEY_FIXED_DEMANDS in observations:
            scalar_features.append(observations[OBS_KEY_FIXED_DEMANDS])

        if scalar_features:
            scalar_cat = torch.cat(scalar_features, dim=-1)
            scalar_encoded = self.scalar_encoder(scalar_cat)
            encoded_parts.append(scalar_encoded)

        combined = torch.cat(encoded_parts, dim=-1)
        output = self.fusion(combined)

        return output


_DEFAULT_FEATURE_EXTRACTOR_KWARGS = dict(
    features_dim=256,
    vehicle_embed_dim=64,
    forecast_embed_dim=32,
    use_attention=True,
    num_attention_heads=4,
)
_DEFAULT_POLICY_KWARGS = dict(
    features_extractor_class=StructuredEnergyExtractor,
    features_extractor_kwargs=_DEFAULT_FEATURE_EXTRACTOR_KWARGS,
)


def _load_sb3_agent(sb3_type: type[BaseAlgorithm], model_path: Path, env=None) -> BaseAlgorithm:
    sb3_agent = sb3_type.load(
        model_path,
        env=env,
    )
    return sb3_agent


def train(
    algorithm: AgentAlgorithm,
    scenario_factory: _ScenarioFactoryT,
    train_horizon: utils.TimeSettings,
    config: AgentConfig | None = None,
    n_proc: int | None = None,
    total_timesteps: int = 10000,
    base_policy_path: Path | None = None,
) -> RevoletionAgent:
    if not algorithm.needs_training():
        return _create_non_trainable_agent(algorithm)

    if n_proc is None or n_proc < 2:
        env = make_vec_env(lambda: build_rl_environment(scenario_factory, train_horizon), n_envs=1)
    else:
        env = make_vec_env(
            lambda: build_rl_environment(scenario_factory, train_horizon), n_envs=n_proc, vec_env_cls=SubprocVecEnv
        )
    env = VecNormalize(env, training=True)

    agent = create_trainable_agent(algorithm, env, config, base_policy_path)

    _ = agent.learn(total_timesteps=total_timesteps)
    return agent


def _create_non_trainable_agent(algorithm: AgentAlgorithm) -> RevoletionAgent:
    match algorithm:
        case AgentAlgorithm.RANDOM:
            return RandomChargingAgent(algorithm)
        case AgentAlgorithm.FULL_CHARGING:
            return FullChargingAgent(algorithm)
        case AgentAlgorithm.FULL_DISCHARGE:
            return FullDischargingAgent(algorithm)
        case AgentAlgorithm.BASIC:
            return BasicChargingAgent(algorithm)
        case AgentAlgorithm.IDLE:
            return IdleAgent(algorithm)
        case _:
            raise ValueError()


def create_trainable_agent(
    algorithm: AgentAlgorithm,
    env: gym.Env[ObsType, ActType] | SubprocVecEnv,
    config: AgentConfig | None = None,
    base_policy_path: Path | None = None,
) -> RevoletionSB3Agent:
    if config is None:
        config = AgentConfig.default_for_algorithm(algorithm)

    kwargs: dict[str, typing.Any] = {}
    kwargs["policy_kwargs"] = _DEFAULT_POLICY_KWARGS.copy()
    if algorithm in {AgentAlgorithm.TD3, AgentAlgorithm.DDPG}:
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise

    if algorithm in {AgentAlgorithm.SAC}:
        kwargs["policy_kwargs"]["share_features_extractor"] = True

    if config.gradient_steps is not None:
        kwargs["gradient_steps"] = config.gradient_steps

    if config.train_freq is not None:
        kwargs["train_freq"] = config.train_freq

    if config.target_noise_clip is not None:
        kwargs["target_noise_clip"] = config.target_noise_clip

    if config.target_policy_noise is not None:
        kwargs["target_policy_noise"] = config.target_policy_noise

    if config.batch_size is not None:
        kwargs["batch_size"] = config.batch_size

    sb3_type = _get_sb3_type(algorithm)
    sb3_agent = sb3_type(
        "MultiInputPolicy",
        env=env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        seed=config.seed,
        n_steps=config.n_steps,
        tensorboard_log=config.tensorboard_log,
        verbose=1,
        **kwargs,
    )

    if base_policy_path is not None:
        sb3_agent.policy = type(sb3_agent.policy).load(str(base_policy_path))

    return RevoletionSB3Agent(
        algorithm,
        sb3_agent=sb3_agent,
    )


def _get_sb3_type(algorithm: AgentAlgorithm):
    match algorithm:
        case AgentAlgorithm.PPO:
            return stable_baselines3.PPO
        case AgentAlgorithm.TD3:
            return stable_baselines3.TD3
        case AgentAlgorithm.A2C:
            return stable_baselines3.A2C
        case AgentAlgorithm.SAC:
            return stable_baselines3.SAC
        case AgentAlgorithm.DDPG:
            return stable_baselines3.DDPG
        case _:
            raise ValueError(f"Unkown agent algorithm: {algorithm}")


class RandomChargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_EFUS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = 2 * np.random.sample(num_cars) - 1
        # Random charge pattern masked by car availability
        return charge_pattern * cars_available[:, 0]


class FullChargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        actions = []
        for cars_available in obs[OBS_KEY_EFUS_AVAILABLE]:
            num_cars = len(cars_available)

            charge_pattern = np.ones(num_cars)
            action = charge_pattern * cars_available[:, 0]
            actions.append(action)
        return np.array(actions, dtype=np.float32)


class FullDischargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_EFUS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = np.ones(num_cars) * -1
        return charge_pattern * cars_available[:, 0]


class BasicChargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_EFUS_AVAILABLE]
        cars_socs = obs[OBS_KEY_EFUS_SOC]
        cars_target_socs = obs[OBS_KEY_EFUS_REQUIRED_SOCS]

        num_cars = len(cars_available)

        charge_pattern = np.zeros(num_cars)
        for i in range(num_cars):
            car_available = cars_available[i, 0] == 1.0

            if not car_available:
                continue

            current_soc = cars_socs[i]
            target_soc = cars_target_socs[i].sum()

            if current_soc < target_soc + 0.1:
                charge_pattern[i] = 1.0

        return charge_pattern


class IdleAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_EFUS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = np.zeros(num_cars)
        return charge_pattern * cars_available[:, 0]


_BASE_TRACE_KEY = "revoletion"
# _TRACE_KEY_DONE_COUNT = f"{_BASE_TRACE_KEY}/01_done_count"
_TRACE_KEY_INFEASIBILITY_COUNT = f"{_BASE_TRACE_KEY}/02_infeasibility_count"
# _TRACE_KEY_INFEASIBILITY_RATE = f"{_BASE_TRACE_KEY}/03_infeasibility_rate"
# _TRACE_KEY_INFEASIBILITY = f"{_BASE_TRACE_KEY}/04_infeasibility_reward"
_TRACE_KEY_REWARD = f"{_BASE_TRACE_KEY}/01_mean_step_reward"
_TRACE_KEY_GRID_COST = f"{_BASE_TRACE_KEY}/02_grid_opex_reward"
_TRACE_KEY_GEN_COST = f"{_BASE_TRACE_KEY}/03_gen_opex_reward"
_TRACE_KEY_CHARGE_COST = f"{_BASE_TRACE_KEY}/04_charge_opex_reward"
_TRACE_KEY_EXT_CHARGE_COST = f"{_BASE_TRACE_KEY}/05_ext_charge_opex_reward"
_TRACE_KEY_SOC_DIFF = f"{_BASE_TRACE_KEY}/06_soc_diff_reward"
_TRACE_KEY_SOC_VIOLATIONS_COUNT = f"{_BASE_TRACE_KEY}/07_soc_violations_count"
_TRACE_KEY_SOC_VIOLATIONS_RATE = f"{_BASE_TRACE_KEY}/08_soc_violations_rate"
_TRACE_KEY_MEAN_SOC_VIOLATIONS = f"{_BASE_TRACE_KEY}/09_soc_violations_mean"
_TRACE_KEY_POWER_DIFF = f"{_BASE_TRACE_KEY}/10_power_diff_reward"
_TRACE_KEY_ATBASE_VIOLATION = f"{_BASE_TRACE_KEY}/11_atbase_violation_reward"


_MOVING_AVERAGE_HORIZON = 500


class _TracingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        # self._done_count = 0
        # self._infeasible_count = 0
        self._soc_count = 0
        self._soc_violation_count = 0

        # self._status = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._reward = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._grid_opex = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._gen_opex = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._charge_opex = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._ext_charge_opex = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._soc_diff = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._power_diff = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._infeasibility = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._soc_violation = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._atbase_violation = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)

    def _on_step(self) -> bool:
        vec_infos = self.locals["infos"]
        vec_dones = self.locals["dones"]
        for infos, done in zip(vec_infos, vec_dones):
            reward: RewardComponents = infos[INFO_KEY_REWARD_COMPONENTS]

            self._reward.append(reward.total_reward)
            self._grid_opex.append(reward.grid_opex_reward)
            self._gen_opex.append(reward.gen_opex_reward)
            self._charge_opex.append(reward.charge_opex_reward)
            self._ext_charge_opex.append(reward.ext_charge_opex_reward)
            self._soc_diff.append(reward.soc_diff_reward)
            self._power_diff.append(reward.power_diff_reward)
            self._atbase_violation.append(reward.atbase_violation_reward)

            self.logger.record(_TRACE_KEY_REWARD, sum(self._reward) / len(self._reward))
            self.logger.record(_TRACE_KEY_GRID_COST, sum(self._grid_opex) / len(self._grid_opex))
            self.logger.record(_TRACE_KEY_GEN_COST, sum(self._gen_opex) / len(self._gen_opex))
            self.logger.record(_TRACE_KEY_CHARGE_COST, sum(self._charge_opex) / len(self._charge_opex))
            self.logger.record(_TRACE_KEY_EXT_CHARGE_COST, sum(self._ext_charge_opex) / len(self._ext_charge_opex))
            self.logger.record(_TRACE_KEY_SOC_DIFF, sum(self._soc_diff) / len(self._soc_diff))
            self.logger.record(_TRACE_KEY_POWER_DIFF, sum(self._power_diff) / len(self._power_diff))
            self.logger.record(_TRACE_KEY_ATBASE_VIOLATION, sum(self._atbase_violation) / len(self._atbase_violation))

            for soc_diff in reward.soc_diffs:
                self._soc_count += 1
                if soc_diff < 0.0:
                    self._soc_violation.append(soc_diff)
                    self.logger.record(
                        _TRACE_KEY_MEAN_SOC_VIOLATIONS, sum(self._soc_violation) / len(self._soc_violation)
                    )
                    self._soc_violation_count += 1
                    self.logger.record(_TRACE_KEY_SOC_VIOLATIONS_COUNT, self._soc_violation_count)
                self.logger.record(_TRACE_KEY_SOC_VIOLATIONS_RATE, self._soc_violation_count / self._soc_count)

            if not done:
                continue

            # self._done_count += 1

            if infos[INFO_KEY_STATUS] == EnvironmentStepStatus.INFEASIBLE:
                # self._status.append(1)
                self._infeasible_count += 1
                self.logger.record(_TRACE_KEY_INFEASIBILITY_COUNT, self._infeasible_count)
            #     self._infeasibility.append(reward.infeasibility_reward)
            #     self.logger.record(_TRACE_KEY_INFEASIBILITY, sum(self._infeasibility) / len(self._infeasibility))
            # else:
            #     self._status.append(0)

            # self.logger.record(_TRACE_KEY_DONE_COUNT, self._done_count)
            # self.logger.record(_TRACE_KEY_INFEASIBILITY_RATE, sum(self._status) / len(self._status))

        return True
