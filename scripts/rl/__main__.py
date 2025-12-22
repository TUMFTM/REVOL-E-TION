import functools
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from revoletion import logger, simulation, utils
from revoletion import scenario as scn
from revoletion.rl import agent, imitation_learning

_LOGGER = logging.getLogger(__name__)


app = typer.Typer(pretty_exceptions_enable=False)

np.random.seed(42)


@app.command()
def train_rl(
    scenario_path: Path,
    algorithm: agent.AgentAlgorithm,
    output_folder: Path,
    n_proc: int = 1,
    train_timesteps: int = 50_000,
    base_policy_path: Path | None = None,
    debug: bool = False,
) -> None:
    logger.configure_root_logger(debugmode=debug)
    paths = scn.SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )
    scenario_factory = simulation.ControlScenarioFactory(paths)
    scenario = scenario_factory.create_scenario()

    agent_config = agent.AgentConfig.default_for_algorithm(algorithm)
    agent_config.tensorboard_log = "/tmp/revol"

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
    )

    print(f"Trained new agent with algorithm {algorithm.value}")
    agent.save_agent(revoletion_agent, scenario.name, output_folder)


@app.command()
def train_imitation_policy(
    scenario_path: Path,
    output_folder: Path,
    algorithm: agent.AgentAlgorithm,
    start: str = "01.01.2022",
    duration: str = "365d",
    seed: int = 42,
    n_epochs: int = 10,
    debug: bool = False,
    n_proc: int = 4,
) -> None:
    logger.configure_root_logger(debugmode=debug)
    paths = scn.SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )
    scenario_factory = simulation.ControlScenarioFactory(paths)
    scenario = scenario_factory.create_scenario()

    imitation_horizon = utils.TimeSettings.create_from_start_timestamp(
        start=pd.Timestamp(start, tz=scenario.times.sim.start.tz),
        timestep=scenario.times.sim.timestep,
        duration=pd.Timedelta(duration),
    )

    env = agent.build_rl_environment(scenario, imitation_horizon)
    agent_config = agent.AgentConfig.default_for_algorithm(algorithm)
    base_agent = agent.create_trainable_agent(algorithm, env, agent_config)

    print(f"Generating trajectories over horizon {imitation_horizon.start} - {imitation_horizon.end}")
    if n_proc <= 1:
        trajectories = imitation_learning.compute_imitation_trajectories(scenario, imitation_horizon)
    else:
        sub_horizon_length = len(imitation_horizon) // n_proc
        sub_horizons = []
        for i in range(0, n_proc):
            sub_horizons.append(imitation_horizon.cut(i * sub_horizon_length, length=sub_horizon_length))

        worker_fn = functools.partial(
            imitation_learning.compute_imitation_trajectories, scenario_factory.create_scenario
        )

        with mp.Pool(processes=n_proc) as pool:
            trajectories_nested = pool.map(worker_fn, sub_horizons)
        trajectories = [xs for xss in trajectories_nested for xs in xss]
    print(f"Training base imitation policy: {seed=}; {n_epochs=}; {len(trajectories)=}")
    imitation_learning.train_imitation_policy(trajectories, env, base_agent._sb3_agent.policy, seed, n_epochs)

    scenario_fingerprint = hash(scenario)
    policy_name = (
        f"{scenario.name}-{algorithm.value}-base-seed_{seed}-epochs_{n_epochs}-{str(scenario_fingerprint)[0:8]}.zip"
    )
    output_path = output_folder / policy_name
    base_agent._sb3_agent.policy.save(output_path)
    print(f"Base imitation policy was saved to {output_path}")


if __name__ == "__main__":
    app()
