import abc
import enum
import logging
import typing
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import stable_baselines3
import typing_extensions
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from typing_extensions import Self

from revoletion import optimization
from revoletion import scenario as scn

from .environment import (
    OBS_KEY_CARS_AVAILABLE,
    OBS_KEY_CARS_REQUIRED_SOCS,
    OBS_KEY_CARS_SOC,
    ActType,
    ObsType,
    RevoletionEnvironment,
    RevoletionEnvironmentConfig,
)

_LOGGER = logging.getLogger(__name__)


class AgentAlgorithm(enum.Enum):
    RANDOM = "random"
    FULL_CHARGING = "full-charge"
    FULL_DISCHARGE = "full-discharge"
    BASIC = "basic"

    PPO = "ppo"
    TD3 = "td3"
    A2C = "a2c"
    SAC = "sac"

    def needs_training(self) -> bool:
        return self not in {
            AgentAlgorithm.RANDOM,
            AgentAlgorithm.FULL_CHARGING,
            AgentAlgorithm.FULL_DISCHARGE,
            AgentAlgorithm.BASIC,
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

    @classmethod
    def default_for_algorithm(cls, algorithm: AgentAlgorithm) -> Self:
        match algorithm:
            case AgentAlgorithm.PPO:
                # Parameters based on master thesis
                return cls(learning_rate=0.0005, gamma=0.987, n_steps=100)
            case AgentAlgorithm.TD3:
                return cls(learning_rate=0.001, gamma=0.99, n_steps=1)
            case AgentAlgorithm.A2C:
                return cls(learning_rate=0.0007, n_steps=5)
            case AgentAlgorithm.SAC:
                return cls(learning_rate=0.0003)
            case _:
                return cls()


class RevoletionAgent(abc.ABC):
    def learn(self, total_timesteps: int) -> None:
        pass

    @abc.abstractmethod
    def predict(
        self,
        obs: ObsType,
        deterministic: bool = False,
    ) -> ActType: ...


class RevoletionSB3Agent(RevoletionAgent):
    def __init__(self, sb3_agent) -> None:
        self._sb3_agent = sb3_agent

    def learn(self, total_timesteps: int) -> None:
        return self._sb3_agent.learn(total_timesteps)

    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        return self._sb3_agent.predict(obs, deterministic=deterministic)


_ScenarioFactoryT = typing.Callable[[], scn.Scenario]


def _build_rl_environment(
    scenario_or_scenario_factory: scn.Scenario | _ScenarioFactoryT, train: bool = True
) -> gym.Env[ObsType, ActType]:
    if isinstance(scenario_or_scenario_factory, scn.Scenario):
        scenario = scenario_or_scenario_factory
    else:
        scenario = scenario_or_scenario_factory()

    if scenario.scheduler:
        scenario.scheduler.calc_ph_schedule(scenario.times.sim)
    env_config = RevoletionEnvironmentConfig(
        penalty_factor_charge_cost=0.0,
        penalty_factor_grid_cost=0.0,
        episode_length=None if train else len(scenario.times.sim),
    )
    env = RevoletionEnvironment(scenario, scenario.times.sim, config=env_config)
    return env


def train(
    algorithm: AgentAlgorithm,
    scenario_factory: _ScenarioFactoryT,
    config: AgentConfig | None = None,
    n_proc: int | None = None,
    total_timesteps: int = 10000,
) -> RevoletionAgent:
    if not algorithm.needs_training():
        return _create_non_trainable_agent(algorithm)

    if n_proc is None or n_proc < 2:
        env = _build_rl_environment(scenario_factory)
    else:
        env = SubprocVecEnv([lambda: _build_rl_environment(scenario_factory) for _ in range(n_proc)])

    agent = _create_trainable_agent(algorithm, env, config)

    _ = agent.learn(total_timesteps=total_timesteps)
    return agent


def evaluate_with_agent(
    scenario: scn.Scenario, agent: RevoletionAgent
) -> tuple[float, optimization.OptimizationResult]:
    env = _build_rl_environment(scenario, train=False)
    obs, _ = env.reset()
    total_reward = 0.0

    for _ in scenario.times.sim.dti:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward, env.last_optimization_result


def _create_non_trainable_agent(algorithm: AgentAlgorithm) -> RevoletionAgent:
    match algorithm:
        case AgentAlgorithm.RANDOM:
            return RandomChargingAgent()
        case AgentAlgorithm.FULL_CHARGING:
            return FullChargingAgent()
        case AgentAlgorithm.FULL_DISCHARGE:
            return FullDischargingAgent()
        case AgentAlgorithm.BASIC:
            return BasicChargingAgent()
        case _:
            raise ValueError()


def _create_trainable_agent(
    algorithm: AgentAlgorithm, env: gym.Env[ObsType, ActType] | SubprocVecEnv, config: AgentConfig | None = None
) -> RevoletionAgent:
    if config is None:
        config = AgentConfig.default_for_algorithm(algorithm)

    match algorithm:
        case AgentAlgorithm.PPO:
            return RevoletionSB3Agent(
                stable_baselines3.PPO(
                    "MultiInputPolicy",
                    env=env,
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                    seed=config.seed,
                    n_steps=config.n_steps,
                    tensorboard_log=config.tensorboard_log,
                )
            )
        case AgentAlgorithm.TD3:
            return RevoletionSB3Agent(
                stable_baselines3.TD3(
                    "MultiInputPolicy",
                    env=env,
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                    seed=config.seed,
                    n_steps=config.n_steps,
                    tensorboard_log=config.tensorboard_log,
                )
            )
        case AgentAlgorithm.A2C:
            return RevoletionSB3Agent(
                stable_baselines3.A2C(
                    "MultiInputPolicy",
                    env=env,
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                    seed=config.seed,
                    n_steps=config.n_steps,
                    tensorboard_log=config.tensorboard_log,
                )
            )
        case AgentAlgorithm.SAC:
            return RevoletionSB3Agent(
                stable_baselines3.SAC(
                    "MultiInputPolicy",
                    env=env,
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                    seed=config.seed,
                    n_steps=config.n_steps,
                    tensorboard_log=config.tensorboard_log,
                )
            )
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

        charge_pattern = np.zeros(num_cars)
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
