import abc
import collections
import enum
import logging
import typing
from dataclasses import dataclass, fields
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3
import typing_extensions
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from revoletion import optimization, utils
from revoletion import scenario as scn

from ._features import (
    OBS_KEY_EFUS_AVAILABLE,
    OBS_KEY_EFUS_REQUIRED_SOCS,
    OBS_KEY_EFUS_SOC,
)
from ._structured_feature_extractor import StructuredFeatureExtractor
from ._training_callback import TrainingCallback
from .environment import (
    INFO_KEY_OPTIMIZATION_RESULT,
    INFO_KEY_STATUS,
    ActType,
    EnvironmentStepStatus,
    ObsType,
    RevoletionEnvironment,
    RevoletionEnvironmentConfig,
    RewardConfig,
)

_LOGGER = logging.getLogger(__name__)

_DEFAULT_FEATURE_EXTRACTOR_KWARGS = dict(
    embed_dim=128,
    num_attention_heads=4,
)
_DEFAULT_POLICY_KWARGS = dict(
    features_extractor_class=StructuredFeatureExtractor,
    features_extractor_kwargs=_DEFAULT_FEATURE_EXTRACTOR_KWARGS,
)


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
    tensorboard_log: str | None = "/tmp/tb-revoletion-rl"
    gradient_steps: int | None = None
    train_freq: int | tuple[int, str] | None = None
    batch_size: int | None = None
    stats_window_size: int | None = 100

    def as_dict(self) -> dict[str, typing.Any]:
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                result[field.name] = value

        return result


@dataclass
class PPOAgentConfig(AgentConfig):
    ent_coef: float | None = None
    use_sde: bool | None = None
    sde_sample_freq: int | None = None


@dataclass
class OffPolicyAgentConfig(AgentConfig):
    target_policy_noise: float | None = None
    target_noise_clip: float | None = None
    learning_starts: int | None = None
    buffer_size: int | None = None


@dataclass
class SACPolicyAgentConfig(OffPolicyAgentConfig):
    use_sde: bool | None = None
    sde_sample_freq: int | None = None
    use_sde_at_warmup: bool = False
    ent_coef: float | typing.Literal["auto"] | None = None
    target_entropy: float | typing.Literal["auto"] | None = None


DEFAULT_PPO_AGENT_CONFIG = PPOAgentConfig(
    learning_rate=0.0003,  # sb3: 0.0003
    gamma=0.99,  # sb3: 0.99
    n_steps=256,  # sb3: 2028
    batch_size=64,  # sb3: 64
    use_sde=True,  # sb3: False
    sde_sample_freq=4,  # sb3: None
)
DEFAULT_TD3_AGENT_CONFIG = OffPolicyAgentConfig(
    learning_rate=0.0001,
    gamma=0.99,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    batch_size=256,
    learning_starts=10_000,
    buffer_size=50_000,
)
DEFAULT_SAC_AGENT_CONFIG = SACPolicyAgentConfig(
    learning_rate=0.0003,
    gamma=0.99,
    batch_size=256,
    learning_starts=1_000,
    buffer_size=50_000,
    use_sde=True,
    sde_sample_freq=4,
    ent_coef="auto",
    target_entropy="auto",
)


def get_default_agent_config_for_algorithm(algorithm: AgentAlgorithm):
    match algorithm:
        case AgentAlgorithm.PPO:
            return DEFAULT_PPO_AGENT_CONFIG
        case AgentAlgorithm.TD3:
            return DEFAULT_TD3_AGENT_CONFIG
        case AgentAlgorithm.SAC:
            return DEFAULT_SAC_AGENT_CONFIG
        case _:
            return AgentConfig()


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
            callback=TrainingCallback(),
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
        episode_length=None if train else len(horizon),
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
        action_trace.append(action[0])

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

    sb3_type = get_sb3_type(algorithm)
    sb3_agent = sb3_type.load(load_path)

    agent = RevoletionSB3Agent(algorithm=algorithm, sb3_agent=sb3_agent)

    _LOGGER.info(f"Loaded agent {algorithm} from {load_path}")
    return agent


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
        env = make_vec_env(
            lambda: build_rl_environment(scenario_factory, train_horizon), n_envs=1, vec_env_cls=DummyVecEnv
        )
    else:
        env = make_vec_env(
            lambda: build_rl_environment(scenario_factory, train_horizon), n_envs=n_proc, vec_env_cls=SubprocVecEnv
        )
    # env = VecNormalize(env, training=True)

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
        config = get_default_agent_config_for_algorithm(algorithm)

    kwargs: dict[str, typing.Any] = config.as_dict()
    kwargs["policy_kwargs"] = _DEFAULT_POLICY_KWARGS.copy()

    if algorithm in {AgentAlgorithm.TD3, AgentAlgorithm.DDPG}:
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise

    sb3_type = get_sb3_type(algorithm)
    sb3_agent = sb3_type(
        "MultiInputPolicy",
        env=env,
        verbose=1,
        **kwargs,
    )

    if base_policy_path is not None:
        sb3_agent.policy = type(sb3_agent.policy).load(str(base_policy_path))

    return RevoletionSB3Agent(
        algorithm,
        sb3_agent=sb3_agent,
    )


def get_sb3_type(algorithm: AgentAlgorithm):
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
