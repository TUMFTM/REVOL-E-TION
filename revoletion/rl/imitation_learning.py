import logging
from collections.abc import Sequence
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from imitation.algorithms import base as algo_base
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout, types
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import util
from imitation.util.networks import RunningNorm
from stable_baselines3.common import (
    buffers,
    type_aliases,
    vec_env,
)
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from revoletion import optimization, utils
from revoletion import scenario as scn
from revoletion.rl import agent

from . import _context as context
from . import _features as features
from . import _utils as rl_utils

_LOGGER = logging.getLogger(__name__)


def compute_imitation_trajectories(
    scenario_or_scenario_factory: scn.Scenario, rollout_horizon: utils.TimeSettings, episode_length: int = 300
) -> list[types.Trajectory]:
    if isinstance(scenario_or_scenario_factory, scn.Scenario):
        scenario = scenario_or_scenario_factory
    else:
        scenario = scenario_or_scenario_factory()

    _LOGGER.debug(
        f"Computing rollout buffer for scenario {scenario.name} over horizon {rollout_horizon.dti[0]} - {rollout_horizon.dti_extd[-1]}"
    )

    len_horizon = len(rollout_horizon)
    num_episodes = len_horizon // episode_length

    trajectories = []
    feature_extractor = features.FeatureExtractor(forecast_horizon=16, soc_min=0.05)

    _LOGGER.debug("Collecting rollout buffer")
    for i in range(0, num_episodes):
        episode_horizon = rollout_horizon.cut(i * episode_length, episode_length)
        ctx = context.Context(scenario, episode_horizon)
        _LOGGER.debug(f"Processing episode {i} from {episode_horizon.start} - {episode_horizon.end}")

        optimization_problem_config = optimization.OptimizationProblemConfig(
            solver=optimization.Solver.HIGHS,
            invest=False,
            enforce_soc_constraints=True,
            warmstart=False,
        )
        _LOGGER.debug("Solving optimization problem over rollout horizon")
        optimization_problem = optimization.PypsaOptimizationProblem.from_revoletion_scenario(
            scenario,
            episode_horizon,
            logger=_LOGGER,
            config=optimization_problem_config,
        )

        _fix_power_envelope_for_optimization_problem(scenario, episode_horizon, optimization_problem)
        optimization_status, optimization_result = optimization_problem.solve()
        if optimization_status != optimization.OptimizationStatus.OPTIMAL:
            _LOGGER.warning(f"Optimization for episode {episode_horizon.dti[0]} failed")
            continue
        assert optimization_result is not None

        observations_buffer = []
        actions_buffer = []

        for time_step in episode_horizon.dti:
            obs_dict = feature_extractor.extract_all_features(ctx, optimization_result)
            obs = types.maybe_wrap_in_dictobs(obs_dict)
            observations_buffer.append(obs)

            unorded_actions = []
            for efu in ctx.electric_fleet_unit_blocks:
                power_flow = optimization_result.get_power_flow(efu, time_step)

                out_flow = power_flow["out"]
                out_power_frac = out_flow / efu.pwr_dis_max

                in_flow = power_flow["in"]
                in_power_frac = in_flow / efu.pwr_chg_max

                action = in_power_frac if in_power_frac > 0 else -out_power_frac
                action_idx = ctx.get_efu_index(efu)
                unorded_actions.append((action_idx, action))

            action_vec = np.array(list(map(lambda a: a[1], sorted(unorded_actions, key=lambda a: a[0]))))

            actions_buffer.append(action_vec)
            ctx.step()

        trajectory = types.Trajectory(
            obs=np.array(observations_buffer),
            # For each episode, we must collect one more observation than actions (to also have an observation for the last action).
            acts=np.array(actions_buffer[:-1]),
            infos=None,
            terminal=True,
        )
        trajectories.append(trajectory)

    _LOGGER.debug(f"Collected {len(trajectories)} trajectories")

    return trajectories


def _fix_power_envelope_for_optimization_problem(
    scenario: scn.Scenario, episode_horizon: utils.TimeSettings, optimization_problem: optimization.OptimizationProblem
) -> None:
    for block in scenario.block_registry.get("ElectricFleetUnit", {}).values():
        nom_capacity_wh = block.sizes["storage"].preexisting
        eff_charge = block.eff["chg_int"]
        max_charge_power_w = block.pwr_chg_max * eff_charge
        buffered_max_charge_power_w = max_charge_power_w * 0.9

        dsoc_step = (buffered_max_charge_power_w * episode_horizon.timestep.hours) / nom_capacity_wh

        soc_envelope = rl_utils.get_soc_envelope(block, episode_horizon, dsoc_step)
        padded_soc_envelope = np.clip(soc_envelope + 0.1, 0.0, 1.0)
        power_envelope = rl_utils.get_power_envelope(block, episode_horizon, padded_soc_envelope)

        # Create masks for charging and discharging
        is_charging = power_envelope >= 0.0

        target_power_unit_charge = pd.Series(0.0, index=episode_horizon.dti)

        target_power_unit_charge[is_charging] = np.clip(
            power_envelope[is_charging] / max_charge_power_w,
            0.0,
            1.0,
        )

        for time_step in episode_horizon.dti:
            optimization_problem.set_input_power_unit(
                block, target_power_unit_charge[time_step], time_step, power_unit_buffer=0.1
            )


def train_imitation_policy_bc(
    trajectories: Sequence[types.Trajectory], env, base_policy: ActorCriticPolicy, seed: int = 42, n_epochs: int = 15
) -> None:
    transitions = rollout.flatten_trajectories(trajectories)

    rng = np.random.default_rng(seed)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=base_policy,
        demonstrations=transitions,
        rng=rng,
        ent_weight=1e-2,
    )
    bc_trainer.train(n_epochs=n_epochs)


def train_imitation_policy_gail(
    trajectories: list[types.Trajectory], env, base_agent: BaseAlgorithm, seed: int = 42, train_timesteps: int = 20_000
) -> None:
    transitions = rollout.flatten_trajectories(trajectories)

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=base_agent,
        reward_net=reward_net,
    )
    env.seed(seed)
    gail_trainer.train(total_timesteps=train_timesteps)


class SQILReplayBuffer(buffers.DictReplayBuffer):
    """A replay buffer that injects 50% expert demonstrations when sampling.

    This buffer is fundamentally the same as ReplayBuffer,
    but it includes an expert demonstration internal buffer.
    When sampling a batch of data, it will be 50/50 expert and collected data.

    It can be used in off-policy algorithms like DQN/SAC/TD3.

    Here it is used as part of SQIL, where it is used to train a DQN.

    Supports both standard and dictionary observation spaces.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        demonstrations: algo_base.AnyTransitions,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        """Create a SQILReplayBuffer instance.

        Args:
            buffer_size: Max number of elements in the buffer
            observation_space: Observation space (can be Dict or standard space)
            action_space: Action space
            demonstrations: Expert demonstrations.
            device: PyTorch device.
            n_envs: Number of parallel environments. Defaults to 1.
            optimize_memory_usage: Enable a memory efficient variant
                of the replay buffer which reduces by almost a factor two
                the memory used, at a cost of more complexity.
        """

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=False,
        )

        # Create expert buffer with same type
        self.expert_buffer = buffers.DictReplayBuffer(
            buffer_size=0,
            observation_space=observation_space,
            action_space=action_space,
            handle_timeout_termination=False,
            n_envs=1,
        )
        self.set_demonstrations(demonstrations)

    def set_demonstrations(
        self,
        demonstrations: algo_base.AnyTransitions,
    ) -> None:
        """Set the expert demonstrations to be injected when sampling from the buffer.

        Args:
            demonstrations (algo_base.AnyTransitions): Expert demonstrations.

        Raises:
            NotImplementedError: If `demonstrations` is not a transitions object
                or a list of trajectories.
        """
        # If demonstrations is a list of trajectories,
        # flatten it into a list of transitions
        if not isinstance(demonstrations, types.Transitions):
            (
                item,
                demonstrations,
            ) = util.get_first_iter_element(  # type: ignore[assignment]
                demonstrations,  # type: ignore[assignment]
            )
            if isinstance(item, types.Trajectory):
                demonstrations = rollout.flatten_trajectories(
                    demonstrations,  # type: ignore[arg-type]
                )

        if not isinstance(demonstrations, types.Transitions):
            raise NotImplementedError(
                f"Unsupported demonstrations type: {demonstrations}",
            )

        n_samples = len(demonstrations)

        self.expert_buffer = buffers.DictReplayBuffer(
            buffer_size=n_samples,
            observation_space=self.observation_space,
            action_space=self.action_space,
            handle_timeout_termination=False,
            n_envs=1,
        )

        for transition in demonstrations:
            obs = {k: v for k, v in transition["obs"].items()}
            next_obs = {k: v for k, v in transition["next_obs"].items()}
            self.expert_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=transition["acts"],
                done=transition["dones"],
                reward=np.array(1.0),
                infos=[{}],
            )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=np.array(0.0),
            done=done,
            infos=infos,
        )

    def sample(self, batch_size: int, env: vec_env.VecNormalize | None = None):
        learner_bs, expert_bs = util.split_in_half(batch_size)

        learner = super().sample(learner_bs, env)
        expert = self.expert_buffer.sample(expert_bs, env)

        observations = {
            k: th.cat([learner.observations[k], expert.observations[k]], dim=0) for k in learner.observations
        }
        next_observations = {
            k: th.cat([learner.next_observations[k], expert.next_observations[k]], dim=0)
            for k in learner.next_observations
        }

        return type_aliases.DictReplayBufferSamples(
            observations=observations,
            actions=th.cat([learner.actions, expert.actions], dim=0),
            rewards=th.cat([learner.rewards, expert.rewards], dim=0),
            dones=th.cat([learner.dones, expert.dones], dim=0),
            next_observations=next_observations,
        )


def train_imitation_policy_sqil(
    trajectories: list[types.Trajectory], env, base_agent: BaseAlgorithm, seed: int = 42, train_timesteps: int = 20_000
) -> None:
    transitions = rollout.flatten_trajectories(trajectories)

    replay_buffer = SQILReplayBuffer(
        buffer_size=50_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        n_envs=base_agent.n_envs,
    )
    base_agent.replay_buffer = replay_buffer

    base_agent.learn(total_timesteps=train_timesteps, callback=agent._TracingCallback())
