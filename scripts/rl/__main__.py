import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from revoletion import logger, rl, utils
from revoletion import scenario as scn
from revoletion.rl import agent, imitation_learning

_LOGGER = logging.getLogger(__name__)


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train_rl(
    scenario_path: Path,
    algorithm: agent.AgentAlgorithm,
    output_folder: Path,
    n_proc: int = 1,
    train_timesteps: int = 50_000,
    base_policy_path: Path | None = None,
    debug: bool = False,
    seed: int = 42,
    custom_feature_extractor: bool = False,
) -> None:
    np.random.seed(seed)
    logger.configure_root_logger(debugmode=debug)
    paths = scn.SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )
    scenario_factory = rl.ScenarioFactory(paths)
    scenario = scenario_factory.create_scenario()

    agent_config = agent.get_default_agent_config_for_algorithm(algorithm)

    train_horizon = utils.TimeSettings.create_from_start_timestamp(
        start=scenario.times.sim.start,
        timestep=scenario.timestep,
        end=scenario.times.sim.start + scenario.len_ph,
    )

    revoletion_agent = agent.train(
        algorithm,
        scenario_factory.create_scenario,
        train_horizon=train_horizon,
        n_proc=n_proc,
        config=agent_config,
        total_timesteps=train_timesteps,
        base_policy_path=base_policy_path,
        custom_feature_extractor=custom_feature_extractor,
    )

    hyperparameters_id = revoletion_agent.hyperparameters.generate_hyperparameters_id()

    model_save_path: Path = output_folder / scenario.name / algorithm.value / hyperparameters_id
    model_save_path.mkdir(exist_ok=True)
    print(f"Trained new agent with algorithm {algorithm.value}")
    agent.save_agent(revoletion_agent, scenario.name, model_save_path)


@app.command()
def generate_trajectories(
    scenario_path: Path,
    output_folder: Path,
    start: str = "01.01.2020",
    duration: str = "365d",
    episode_length: int = 200,
    debug: bool = False,
    envelope_split: float = 0.5,
    seed: int = 42,
    n_proc: int = 4,
) -> None:
    np.random.seed(seed)
    logger.configure_root_logger(debugmode=debug)

    paths = scn.SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )

    scenario_factory = rl.ScenarioFactory(paths)
    scenario = scenario_factory.create_scenario()

    normalized_start = start.replace(".", "_")
    trajectories_configuration_str = (
        f"trajectories-{scenario.name}-{normalized_start}-{duration}-{episode_length}-{seed}-{envelope_split}"
    )
    trajectories_configuration_id = hashlib.sha256(trajectories_configuration_str.encode()).hexdigest()[0:8]
    trajectories_file_name = f"{trajectories_configuration_str}-{trajectories_configuration_id}.pb"
    trajectories_save_path = output_folder / trajectories_file_name

    imitation_trajectory_computer_config = imitation_learning.ImitationTrajectoryComputerConfig(
        seed=seed,
        episode_length=episode_length,
        forecast_horizon=16,
        soc_min=0.05,
        envelope_split=envelope_split,
    )

    imitation_trajectory_computer = imitation_learning.ImitationTrajectoryComputer(
        config=imitation_trajectory_computer_config
    )

    imitation_horizon = utils.TimeSettings.create_from_start_timestamp(
        start=pd.Timestamp(start, tz=scenario.times.sim.start.tz),
        timestep=scenario.times.sim.timestep,
        duration=pd.Timedelta(duration),
    )

    print(
        f"Generating trajectories over horizon {imitation_horizon.start} - {imitation_horizon.end} with {n_proc} process(es)"
    )

    trajectories = imitation_trajectory_computer.compute_trajectories(
        scenario_or_factory=scenario_factory.create_scenario, rollout_horizon=imitation_horizon, n_procs=n_proc
    )

    print(f"Generated {len(trajectories)} trajectories")
    if len(trajectories) == 0:
        return

    with open(trajectories_save_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Saved to {trajectories_save_path}")


@app.command()
def train_imitation_bc(
    scenario_path: Path,
    trajectories_path: Path,
    output_folder: Path,
    algorithm: agent.AgentAlgorithm,
    start: str = "01.01.2020",
    duration: str = "365d",
    seed: int = 42,
    n_epochs: int = 10,
    debug: bool = False,
    custom_feature_extractor: bool = False,
) -> None:
    np.random.seed(seed)

    logger.configure_root_logger(debugmode=debug)
    paths = scn.SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )
    scenario_factory = rl.ScenarioFactory(paths)
    scenario = scenario_factory.create_scenario()
    imitation_horizon = utils.TimeSettings.create_from_start_timestamp(
        start=pd.Timestamp(start, tz=scenario.times.sim.start.tz),
        timestep=scenario.times.sim.timestep,
        duration=pd.Timedelta(duration),
    )

    with open(trajectories_path, "rb") as f:
        trajectories = pickle.load(f)

    trajectories_id = trajectories_path.stem.split("-")[-1]
    print(f"Training BC policy: {seed=}; {n_epochs=}; {len(trajectories)=}; {trajectories_id=}")

    env = agent.build_rl_environment(scenario, imitation_horizon)
    agent_config = agent.get_default_agent_config_for_algorithm(algorithm)
    base_agent = agent.create_trainable_agent(
        algorithm, env, agent_config, custom_feature_extractor=custom_feature_extractor
    )
    base_policy = base_agent._sb3_agent.policy
    imitation_learning.train_imitation_policy_bc(trajectories, env, base_policy, seed, n_epochs)

    policy_name = f"{scenario.name}-{algorithm.value}-bc-seed_{seed}-epochs_{n_epochs}-features_{'custom' if custom_feature_extractor else 'default'}-{trajectories_id}.zip"
    output_path = output_folder / policy_name
    base_policy.save(output_path)

    print(f"BC policy was saved to {output_path}")


@app.command()
def train_imitation_sqil(
    scenario_path: Path,
    trajectories_path: Path,
    output_folder: Path,
    start: str = "01.01.2020",
    duration: str = "365d",
    seed: int = 42,
    train_timesteps: int = 20_000,
    n_proc: int = 4,
    debug: bool = False,
) -> None:
    np.random.seed(seed)

    algorithm = agent.AgentAlgorithm.SAC

    logger.configure_root_logger(debugmode=debug)
    paths = scn.SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )
    scenario_factory = rl.ScenarioFactory(paths)
    scenario = scenario_factory.create_scenario()
    train_horizon = utils.TimeSettings.create_from_start_timestamp(
        start=pd.Timestamp(start, tz=scenario.times.sim.start.tz),
        timestep=scenario.times.sim.timestep,
        duration=pd.Timedelta(duration),
    )

    with open(trajectories_path, "rb") as f:
        trajectories = pickle.load(f)

    trajectories_id = trajectories_path.stem.split("-")[-1]
    print(f"Training SQIL policy: {seed=}; {train_timesteps=}; {n_proc=}; {len(trajectories)=}; {trajectories_id=}")

    # env = agent.build_rl_environment(scenario, train_horizon)
    env = make_vec_env(
        lambda: agent.build_rl_environment(scenario_factory.create_scenario, train_horizon),
        n_envs=n_proc,
        vec_env_cls=SubprocVecEnv,
    )

    agent_config = agent.DEFAULT_SAC_AGENT_CONFIG
    # For imitation learning the entropy should be reduced quite significantly.
    agent_config.use_sde = False
    agent_config.sde_sample_freq = None
    agent_config.ent_coef = 0.1
    agent_config.target_entropy = -1
    base_agent = agent.create_trainable_agent(algorithm, env, agent_config)

    imitation_learning.train_imitation_policy_sqil(trajectories, env, base_agent._sb3_agent, seed, train_timesteps)

    policy_name = (
        f"{scenario.name}-{algorithm.value}-sqil-seed_{seed}-timesteps_{train_timesteps}-{trajectories_id}.zip"
    )
    output_path = output_folder / policy_name
    base_agent._sb3_agent.policy.save(output_path)

    print(f"SQIL policy was saved to {output_path}")


if __name__ == "__main__":
    app()
