import functools
import logging
import multiprocessing
from collections.abc import Sequence
from dataclasses import dataclass
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
from revoletion.rl.scenario_factory import (
    AtBaseHorizonInitializer,
    HorizonInitializer,
    InitialSocHorizonInitializer,
    MinSocHorizonInitializer,
)

from . import _context as context
from . import _features as features
from . import _forecast_provider as forecast_provider
from . import _normalization as normalization
from . import utils as rl_utils

_LOGGER = logging.getLogger(__name__)


@dataclass
class ImitationTrajectoryComputerConfig:
    seed: float = 42

    episode_length: int = 200
    """The length of each episode that is optimized."""

    forecast_horizon: int = 16
    """Length of the forecast horizon, e.g., car available."""

    soc_min: float = 0.05
    """Minimum SoC that is added in the observations."""

    envelope_split: float = 0.5
    """Determines the distribution of power envelope vehicles and optimal vehicles."""

    envelope_power_unit_buffer: float = 0.1
    """The buffer allowed for vehicles with defined power envelope."""

    envelope_max_charge_buffer: float = 0.8
    """Reduction of max charge power for vehicles with defined power envelope to smooth out charging."""

    envelope_target_soc: float = 0.0
    """Increase the target SoC at the end of a episode for vehicles with defined power envelope to encourage buffered charging."""

    envelope_soc_padding: float = 0.2
    """Increase the base SoC for all vehicles with defined power envelope to encourage the agent to keep some buffer in the vehicles."""


class ImitationTrajectoryComputer:
    """
    Computes imitation learning trajectories by solving optimization problems
    and extracting observations and actions from the optimal solutions.
    """

    def __init__(self, config: ImitationTrajectoryComputerConfig | None = None, logger: logging.Logger | None = None):
        """
        Initialize the trajectory computer.

        Args:
            config: Configuration object for trajectory computation
            logger: Logger instance for debugging output
        """
        self._config = config or ImitationTrajectoryComputerConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._horizon_initializer = HorizonInitializer(
            horizon_initializers=[
                MinSocHorizonInitializer(soc_min=self._config.soc_min),
                # SocEnvelopeHorizonInitialzer(
                #     soc_min=self._config.soc_min + self._config.envelope_soc_padding,
                #     max_charge_power_frac=self._config.envelope_max_charge_buffer,
                #     envelope_target_soc=self._config.envelope_target_soc,
                # ),
                InitialSocHorizonInitializer(),
                AtBaseHorizonInitializer(),
            ]
        )

    def compute_trajectories(
        self, scenario_or_factory, rollout_horizon: utils.TimeSettings, n_procs: int = 1
    ) -> list[types.Trajectory]:
        """
        Compute imitation trajectories for the given scenario and time horizon.

        Automatically uses multiprocessing if configured with n_processes > 1.

        Args:
            scenario_or_factory: Either a Scenario instance or a callable that returns one
            rollout_horizon: Time settings defining the rollout period

        Returns:
            List of computed trajectories
        """
        if n_procs > 1:
            trajectories = self._compute_trajectories_parallel(scenario_or_factory, rollout_horizon, n_procs)
        else:
            trajectories = self._compute_imitation_trajectories(scenario_or_factory, rollout_horizon)
        return trajectories

    def _compute_imitation_trajectories(
        self, scenario_or_factory, rollout_horizon: utils.TimeSettings
    ) -> list[types.Trajectory]:
        scenario = self._get_scenario(scenario_or_factory)

        self.logger.debug(
            f"Computing rollout buffer for scenario {scenario.name} "
            + f"over horizon {rollout_horizon.dti[0]} - {rollout_horizon.dti_extd[-1]}"
        )

        num_episodes = len(rollout_horizon) // self._config.episode_length
        trajectories = []

        self.logger.debug("Collecting rollout buffer")
        for episode_idx in range(num_episodes):
            episode_horizon = rollout_horizon.cut(
                episode_idx * self._config.episode_length, self._config.episode_length
            )
            trajectory = self._compute_episode_trajectory(scenario, episode_horizon)
            if trajectory is not None:
                trajectories.append(trajectory)

        self.logger.debug(
            f"Collected {len(trajectories)} trajectories for rollout horizon {rollout_horizon.start} - {rollout_horizon.end}"
        )
        return trajectories

    def _compute_trajectories_parallel(
        self, scenario_or_factory, rollout_horizon: utils.TimeSettings, n_procs: int
    ) -> list[types.Trajectory]:
        # Split the full rollout horizon into multiple sub-horizons which are then processed by each worker.
        sub_horizons = self._split_horizon(rollout_horizon, n_procs)
        worker_fn = functools.partial(self._compute_imitation_trajectories, scenario_or_factory)

        with multiprocessing.Manager() as manager:
            # self._total = manager.Value(int, 0)
            # self._num_power_envelope = manager.Value(int, 0)

            with manager.Pool(processes=n_procs) as pool:
                trajectories_nested = pool.map(worker_fn, sub_horizons)

            # total = self._total.get()
            # num_power_envelope = self._num_power_envelope.get()

        # self.logger.info(f"Real envelope split: {num_power_envelope / total:.2f} ({total=}; {num_power_envelope=})")

        # Each worker returns a list of trajectories. These must be flattened.
        trajectories = [traj for sublist in trajectories_nested for traj in sublist]

        return trajectories

    def _split_horizon(self, horizon: utils.TimeSettings, n_splits: int) -> list[utils.TimeSettings]:
        """
        Split a time horizon into equal sub-horizons for parallel processing.

        Args:
            horizon: The full time horizon to split
            n_splits: Number of sub-horizons to create

        Returns:
            List of sub-horizons
        """
        sub_horizon_length = len(horizon) // n_splits
        sub_horizons = []

        for i in range(n_splits):
            sub_horizon = horizon.cut(i * sub_horizon_length, length=sub_horizon_length)
            sub_horizons.append(sub_horizon)

        return sub_horizons

    def _get_scenario(self, scenario_or_factory) -> "scn.Scenario":
        """Extract scenario from either instance or factory."""
        if callable(scenario_or_factory):
            return scenario_or_factory()
        return scenario_or_factory

    def _compute_episode_trajectory(
        self, scenario: scn.Scenario, episode_horizon: utils.TimeSettings
    ) -> types.Trajectory | None:
        """
        Compute trajectory for a single episode.

        Args:
            scenario: The scenario to simulate
            rollout_horizon: Overall time horizon
            episode_idx: Index of the current episode

        Returns:
            Computed trajectory or None if optimization failed
        """

        self.logger.debug(f"Processing episode from {episode_horizon.start} to {episode_horizon.end}")

        # This is not really safe, since we are modifying a shared scenario.
        # But since this should be only called for non-overlapping episodes we should be good.
        # Otherwise we would need to create new scenarios for each episode which might
        # result in a heavy performance overhead.
        self._horizon_initializer.initialize(scenario, episode_horizon)

        optimization_result = self._solve_optimization(scenario, episode_horizon)
        if optimization_result is None:
            return None

        # Collect observations and actions
        ctx = context.Context(scenario, episode_horizon)
        observations, actions = self._collect_observations_and_actions(ctx, episode_horizon, optimization_result)

        self.logger.debug(f"Collected trajectory for episode {episode_horizon.start}")

        return types.Trajectory(
            obs=np.array(observations),
            acts=np.array(actions[:-1]),  # One less action than observation
            infos=None,
            terminal=True,
        )

    def _solve_optimization(
        self, scenario: scn.Scenario, episode_horizon: utils.TimeSettings
    ) -> optimization.OptimizationResult | None:
        """
        Solve the optimization problem for the episode.

        Args:
            scenario: The scenario to optimize
            episode_horizon: Time horizon for this episode

        Returns:
            Optimization result or None if optimization failed
        """
        self.logger.debug("Solving optimization problem over rollout horizon")

        problem_config = optimization.OptimizationProblemConfig(
            solver=optimization.Solver.HIGHS,
            invest=False,
            enforce_soc_constraints=True,
            warmstart=False,
            committment=True,
        )

        problem = optimization.PypsaOptimizationProblem.from_revoletion_scenario(
            scenario,
            episode_horizon,
            logger=self.logger,
            config=problem_config,
        )

        self._apply_power_envelope_constraints(scenario, episode_horizon, problem)

        status, result = problem.solve()
        if status != optimization.OptimizationStatus.OPTIMAL:
            self.logger.warning(f"Optimization for episode {episode_horizon.start} failed")
            return None

        return result

    def _apply_power_envelope_constraints(
        self,
        scenario: scn.Scenario,
        episode_horizon: utils.TimeSettings,
        problem: optimization.OptimizationProblem,
    ) -> None:
        """
        Apply power envelope constraints to electric fleet units.

        Args:
            scenario: The scenario containing fleet units
            episode_horizon: Time horizon for constraints
            problem: Optimization problem to constrain
        """
        total = 0
        num_power_envelope = 0
        for block in scenario.block_registry.get("ElectricFleetUnit", {}).values():
            total += 1

            if not self._should_apply_power_envelope():
                problem.set_minimum_output_power_unit(block, 0.1, episode_horizon.dti)
                problem.set_minimum_input_power_unit(block, 0.1, episode_horizon.dti)
                continue

            num_power_envelope += 1

            power_envelope = self._compute_power_envelope(block, episode_horizon)
            for time_step in episode_horizon.dti:
                target_power_unit = power_envelope[time_step]
                if target_power_unit > 0.0:
                    problem.set_maximum_output_power_unit(block, power_unit=0.0, dti=time_step)
                    problem.set_minimum_input_power_unit(block, target_power_unit, time_step)
                else:
                    problem.set_input_power_unit(
                        block,
                        0.0,
                        time_step,
                        power_unit_buffer=0.0,
                    )

        # self._total.set(self._total.get() + total)
        # self._num_power_envelope.set(self._num_power_envelope.get() + num_power_envelope)

    def _should_apply_power_envelope(self) -> bool:
        """Determine if envelope should be applied to this block (random sampling)."""
        return np.random.random() <= self._config.envelope_split

    def _compute_power_envelope(self, block, episode_horizon: utils.TimeSettings) -> pd.Series:
        """
        Compute the power envelope for a fleet unit block.

        Args:
            block: Electric fleet unit block
            episode_horizon: Time horizon for envelope

        Returns:
            Array of target power unit charge values
        """
        # Extract block parameters
        nom_capacity_wh = block.sizes["storage"].preexisting
        eff_charge = block.eff["chg_int"]
        max_charge_power_w = block.pwr_chg_max * eff_charge
        dsoc_step_max = (
            max_charge_power_w * self._config.envelope_max_charge_buffer * episode_horizon.timestep.hours
        ) / nom_capacity_wh

        soc_envelope = rl_utils.get_soc_envelope(block, episode_horizon, dsoc_step_max=dsoc_step_max)

        soc_min = block.states.loc[episode_horizon.dti, "soc_min"]
        buffered_soc_envelope = soc_envelope + soc_min

        power_envelope = rl_utils.get_power_envelope(block, episode_horizon, buffered_soc_envelope)

        target_power_unit = np.clip(power_envelope / max_charge_power_w, 0.0, 1.0)
        return target_power_unit

    def _collect_observations_and_actions(
        self,
        ctx: context.Context,
        episode_horizon: utils.TimeSettings,
        optimization_result: optimization.OptimizationResult,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """
        Collect observations and actions from the optimization result.

        Args:
            ctx: Context for the episode
            episode_horizon: Time horizon
            optimization_result: Result from optimization

        Returns:
            Tuple of (observations_buffer, actions_buffer)
        """
        perfect_forecast_provider = forecast_provider.PerfectForesightForecastProvider(self._config.forecast_horizon)
        # limited_forecast_provider = forecast_provider.LimitedForecastProvider(self._config.forecast_horizon)

        normalization_provider = normalization.NormalizationProvider.from_ctx(ctx)
        feature_extractor = features.EnvironmentFeatureExtractor(
            perfect_forecast_provider, soc_min=self._config.soc_min, normalization_provider=normalization_provider
        )

        observations_buffer = []
        actions_buffer = []

        for time_step in episode_horizon.dti:
            # Extract actions
            action_vec = self._extract_actions(ctx, optimization_result, time_step)
            actions_buffer.append(action_vec)

            # Extract observation
            obs_dict = feature_extractor.extract_all_features(ctx, optimization_result)
            obs = types.maybe_wrap_in_dictobs(obs_dict)
            observations_buffer.append(obs)

            ctx.step()

        return observations_buffer, actions_buffer

    def _extract_actions(
        self, ctx: context.Context, optimization_result: optimization.OptimizationResult, time_step: int
    ) -> np.ndarray:
        """
        Extract action vector for all electric fleet units at a time step.

        Args:
            ctx: Context containing fleet units
            optimization_result: Optimization result
            time_step: Current time step

        Returns:
            Array of actions for all fleet units
        """
        unordered_actions = []

        for efu in ctx.electric_fleet_unit_blocks:
            power_flow = optimization_result.get_power_flow(efu, time_step)

            # Calculate fractional power for charging/discharging
            out_power_frac = power_flow["out"] / efu.pwr_dis_max
            in_power_frac = power_flow["in"] / efu.pwr_chg_max

            # Combine into single action (positive=charge, negative=discharge)
            action = in_power_frac if in_power_frac > 0 else -out_power_frac
            action_idx = ctx.get_efu_index(efu)

            unordered_actions.append((action_idx, action))

        # Sort by index and extract action values
        sorted_actions = sorted(unordered_actions, key=lambda x: x[0])
        return np.array([action for _, action in sorted_actions])


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
                reward=np.array([1.0]),
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
            reward=np.array([0.0]),
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

    base_agent.learn(total_timesteps=train_timesteps, callback=agent.TrainingCallback(verbose=1, stats_window_size=100))
