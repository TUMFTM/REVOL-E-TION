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
)

_LOGGER = logging.getLogger(__name__)


class AgentAlgorithm(enum.Enum):
    RANDOM = enum.auto()
    FULL_CHARGING = enum.auto()
    FULL_DISCHARGE = enum.auto()
    BASIC = enum.auto()

    PPO = enum.auto()
    TD3 = enum.auto()
    A2C = enum.auto()
    SAC = enum.auto()

    def needs_training(self) -> bool:
        return self not in {
            AgentAlgorithm.RANDOM,
            AgentAlgorithm.FULL_CHARGING,
            AgentAlgorithm.FULL_DISCHARGE,
            AgentAlgorithm.BASIC,
        }


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


def _build_rl_environment(scenario_factory: typing.Callable[[], scn.Scenario]) -> gym.Env[ObsType, ActType]:
    scenario = scenario_factory()
    opt_config = optimization.OptimizationProblemConfig(
        cost_eps=scenario.cost_eps, debug=False, solver=optimization.Solver.HIGHS, invest=False
    )
    opt_model = optimization.create_optimization_problem(
        backend=optimization.OptimizationBackend.PYPSA,
        scenario=scenario,
        horizon=scenario.times.sim,
        logger=scenario.logger,
        config=opt_config,
    )

    # TODO: compute number time steps from horizon length and time step size
    env = RevoletionEnvironment(opt_model, scenario.block_registry, scenario.times.sim.dti, time_steps=96)
    return env


def train(
    algorithm: AgentAlgorithm,
    scenario_factory: typing.Callable[[], scn.Scenario],
    config: AgentConfig | None = None,
    n_proc: int | None = None,
) -> RevoletionAgent:
    if not algorithm.needs_training():
        return _create_non_trainable_agent(algorithm)

    if n_proc is None:
        env = _build_rl_environment(scenario_factory)
    else:
        env = SubprocVecEnv([lambda: _build_rl_environment(scenario_factory) for _ in range(n_proc)])

    agent = _create_trainable_agent(algorithm, env, config)

    _ = agent.learn(total_timesteps=10000)
    return agent


def evaluate_with_model(scenario_factory: typing.Callable[[], scn.Scenario], agent: RevoletionAgent) -> float:
    env = _build_rl_environment(scenario_factory)
    obs, _ = env.reset()
    total_reward = 0.0

    for _ in range(10):
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


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
        return charge_pattern * cars_available


class FullChargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_CARS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = np.ones(num_cars)
        return charge_pattern * cars_available


class FullDischargingAgent(RevoletionAgent):
    @typing_extensions.override
    def predict(self, obs: ObsType, deterministic: bool = False) -> ActType:
        cars_available = obs[OBS_KEY_CARS_AVAILABLE]
        num_cars = len(cars_available)

        charge_pattern = np.zeros(num_cars)
        return charge_pattern * cars_available


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
