import logging

import numpy as np
from imitation.algorithms import bc
from imitation.data import rollout, types

from revoletion import optimization, utils
from revoletion import scenario as scn

from . import _context as context
from . import _features as features

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

        optimization_result = _optimize_over_rollout_horizon(scenario, episode_horizon)
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


def _optimize_over_rollout_horizon(
    scenario: scn.Scenario, rollout_horizon: utils.TimeSettings
) -> optimization.OptimizationResult:
    optimization_problem_config = optimization.OptimizationProblemConfig(
        solver=optimization.Solver.HIGHS,
        invest=False,
        enforce_soc_constraints=True,
    )
    _LOGGER.debug("Solving optimization problem over rollout horizon")
    optimization_problem = optimization.PypsaOptimizationProblem.from_revoletion_scenario(
        scenario,
        rollout_horizon,
        logger=_LOGGER,
        config=optimization_problem_config,
    )
    _, optimization_result = optimization_problem.solve()
    assert optimization_result is not None

    return optimization_result


def train_imitation_policy(
    trajectories: list[types.Trajectory], env, base_policy, seed: int = 42, n_epochs: int = 15
) -> None:
    transitions = rollout.flatten_trajectories(trajectories)

    rng = np.random.default_rng(seed)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=base_policy,
        demonstrations=transitions,
        rng=rng,
    )
    bc_trainer.train(n_epochs=n_epochs)
