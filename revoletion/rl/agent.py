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
import typing_extensions
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from typing_extensions import Self

from revoletion import optimization, utils
from revoletion import scenario as scn

from .environment import (
    INFO_KEY_OPTIMIZATION_RESULT,
    INFO_KEY_REWARD_COMPONENTS,
    INFO_KEY_STATUS,
    OBS_KEY_CARS_AVAILABLE,
    OBS_KEY_CARS_REQUIRED_SOCS,
    OBS_KEY_CARS_SOC,
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
    tensorboard_log: str | None = None
    gradient_steps: int | None = None
    target_policy_noise: float | None = None
    target_noise_clip: float | None = None
    train_freq: int | tuple[int, str] | None = None
    batch_size: int | None = None

    @classmethod
    def default_for_algorithm(cls, algorithm: AgentAlgorithm) -> Self:
        match algorithm:
            case AgentAlgorithm.PPO:
                return cls(learning_rate=0.0003, gamma=0.99, n_steps=100)
            case AgentAlgorithm.TD3:
                return cls(
                    learning_rate=0.0001,
                    gamma=0.995,
                    n_steps=1,
                    # gradient_steps=-1,
                    # train_freq=100,
                    target_policy_noise=0.03,
                    target_noise_clip=0.1,
                    batch_size=512,
                )
            case AgentAlgorithm.A2C:
                return cls(learning_rate=0.0007, n_steps=5)
            case AgentAlgorithm.SAC:
                return cls(
                    learning_rate=0.0003,
                    gamma=0.99,
                    # train_freq=100,
                    # gradient_steps=-1,
                    batch_size=512,
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
        return self._sb3_agent.learn(total_timesteps, callback=_TracingCallback())

    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        return self._sb3_agent.predict(obs, deterministic=deterministic)


_ScenarioFactoryT = typing.Callable[[], scn.Scenario]


def _build_rl_environment(
    scenario_or_scenario_factory: scn.Scenario | _ScenarioFactoryT, horizon: utils.TimeSettings, train: bool = True
) -> gym.Env[ObsType, ActType]:
    if isinstance(scenario_or_scenario_factory, scn.Scenario):
        scenario = scenario_or_scenario_factory
    else:
        scenario = scenario_or_scenario_factory()

    env_config = RevoletionEnvironmentConfig(
        reward_config=RewardConfig(),
        episode_length=None if train else len(horizon),
    )
    env = RevoletionEnvironment(scenario, horizon, config=env_config, train=train)
    return env


def train(
    algorithm: AgentAlgorithm,
    scenario_factory: _ScenarioFactoryT,
    horizon: utils.TimeSettings,
    config: AgentConfig | None = None,
    n_proc: int | None = None,
    total_timesteps: int = 10000,
) -> RevoletionAgent:
    if not algorithm.needs_training():
        return _create_non_trainable_agent(algorithm)

    if n_proc is None or n_proc < 2:
        env = make_vec_env(lambda: _build_rl_environment(scenario_factory, horizon), n_envs=1)
    else:
        env = make_vec_env(
            lambda: _build_rl_environment(scenario_factory, horizon), n_envs=n_proc, vec_env_cls=SubprocVecEnv
        )

    agent = _create_trainable_agent(algorithm, env, config)

    _ = agent.learn(total_timesteps=total_timesteps)
    return agent


def evaluate_with_agent(
    scenario: scn.Scenario,
    agent: RevoletionAgent,
    horizon: utils.TimeSettings,
) -> tuple[float, optimization.OptimizationResult | None]:
    env = _build_rl_environment(scenario, horizon, train=False)
    obs, _ = env.reset()
    total_reward = 0.0
    infos = {}

    for _ in horizon.dti:
        action = agent.predict(obs, deterministic=True)
        if isinstance(action, tuple):
            action = action[0]
        obs, reward, terminated, truncated, infos = env.step(action)
        total_reward += reward
        if truncated:
            return total_reward, None
        elif terminated:
            break

    return total_reward, infos.get(INFO_KEY_OPTIMIZATION_RESULT)


def _get_path_for_algorithm(algorithm: AgentAlgorithm, name: str, models_path: Path) -> Path:
    return (models_path / f"{name}-{algorithm.value}").with_suffix(".zip")


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
    sb3_agent = sb3_type.load(load_path)

    agent = RevoletionSB3Agent(algorithm=algorithm, sb3_agent=sb3_agent)

    _LOGGER.info(f"Loaded agent {algorithm} from {load_path}")
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


def _create_trainable_agent(
    algorithm: AgentAlgorithm, env: gym.Env[ObsType, ActType] | SubprocVecEnv, config: AgentConfig | None = None
) -> RevoletionSB3Agent:
    if config is None:
        config = AgentConfig.default_for_algorithm(algorithm)

    sb3_type = _get_sb3_type(algorithm)

    kwargs = {}
    if algorithm in {AgentAlgorithm.TD3, AgentAlgorithm.DDPG, AgentAlgorithm.SAC}:
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.03 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise

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

    return RevoletionSB3Agent(
        algorithm,
        sb3_agent=sb3_type(
            "MultiInputPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            seed=config.seed,
            n_steps=config.n_steps,
            tensorboard_log=config.tensorboard_log,
            verbose=1,
            **kwargs,
        ),
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
        cars_available = obs[OBS_KEY_CARS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = 2 * np.random.sample(num_cars) - 1
        # Random charge pattern masked by car availability
        return charge_pattern * cars_available[:, 0]


class FullChargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_CARS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = np.ones(num_cars)
        return charge_pattern * cars_available[:, 0]


class FullDischargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_CARS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = np.ones(num_cars) * -1
        return charge_pattern * cars_available[:, 0]


class BasicChargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_CARS_AVAILABLE]
        cars_socs = obs[OBS_KEY_CARS_SOC]
        cars_target_socs = obs[OBS_KEY_CARS_REQUIRED_SOCS]

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
        cars_available = obs[OBS_KEY_CARS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = np.zeros(num_cars)
        return charge_pattern * cars_available[:, 0]


_BASE_TRACE_KEY = "revoletion"
_TRACE_KEY_DONE_COUNT = f"{_BASE_TRACE_KEY}/01_done_count"
_TRACE_KEY_INFEASIBILITY_COUNT = f"{_BASE_TRACE_KEY}/02_infeasibility_count"
_TRACE_KEY_INFEASIBILITY_RATE = f"{_BASE_TRACE_KEY}/03_infeasibility_rate"
_TRACE_KEY_GRID_COST = f"{_BASE_TRACE_KEY}/04_grid_opex_reward"
_TRACE_KEY_CHARGE_COST = f"{_BASE_TRACE_KEY}/05_charge_opex_reward"
_TRACE_KEY_SOC_DIFF = f"{_BASE_TRACE_KEY}/06_soc_diff_reward"
_TRACE_KEY_SOC_VIOLATIONS_COUNT = f"{_BASE_TRACE_KEY}/07_soc_violations_count"
_TRACE_KEY_MEAN_SOC_VIOLATIONS = f"{_BASE_TRACE_KEY}/08_soc_violations_mean"
_TRACE_KEY_SOC_VIOLATIONS_RATE = f"{_BASE_TRACE_KEY}/09_soc_violations_rate"
_TRACE_KEY_POWER_DIFF = f"{_BASE_TRACE_KEY}/09_power_diff_reward"
_TRACE_KEY_INFEASIBILITY = f"{_BASE_TRACE_KEY}/10_infeasibility_reward"
_TRACE_KEY_REWARD = f"{_BASE_TRACE_KEY}/11_mean_step_reward"


_MOVING_AVERAGE_HORIZON = 500


class _TracingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        self._done_count = 0
        self._infeasible_count = 0
        self._soc_count = 0
        self._soc_violation_count = 0

        self._reward = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._grid_opex = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._charge_opex = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._soc_diff = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._power_diff = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._infeasibility = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)
        self._soc_violation = collections.deque(maxlen=_MOVING_AVERAGE_HORIZON)

    def _on_step(self) -> bool:
        vec_infos = self.locals["infos"]
        vec_dones = self.locals["dones"]
        for infos, done in zip(vec_infos, vec_dones):
            reward: RewardComponents = infos[INFO_KEY_REWARD_COMPONENTS]

            self._reward.append(reward.total_reward)
            self._grid_opex.append(reward.grid_opex_reward)
            self._charge_opex.append(reward.charge_opex_reward)
            self._soc_diff.append(reward.soc_diff_reward)
            self._power_diff.append(reward.power_diff_reward)

            self.logger.record(_TRACE_KEY_REWARD, sum(self._reward) / len(self._reward))
            self.logger.record(_TRACE_KEY_GRID_COST, sum(self._grid_opex) / len(self._grid_opex))
            self.logger.record(_TRACE_KEY_CHARGE_COST, sum(self._charge_opex) / len(self._charge_opex))
            self.logger.record(_TRACE_KEY_SOC_DIFF, sum(self._soc_diff) / len(self._soc_diff))
            self.logger.record(_TRACE_KEY_POWER_DIFF, sum(self._power_diff) / len(self._power_diff))

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

            self._done_count += 1

            if infos[INFO_KEY_STATUS] == EnvironmentStepStatus.INFEASIBLE:
                self._infeasible_count += 1
                self.logger.record(_TRACE_KEY_INFEASIBILITY_COUNT, self._infeasible_count)
                self._infeasibility.append(reward.infeasibility_reward)
                self.logger.record(_TRACE_KEY_INFEASIBILITY, sum(self._infeasibility) / len(self._infeasibility))

            self.logger.record(_TRACE_KEY_DONE_COUNT, self._done_count)
            self.logger.record(_TRACE_KEY_INFEASIBILITY_RATE, self._infeasible_count / self._done_count)

        return True
