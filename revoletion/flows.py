from prefect import flow, get_run_logger

# from pathlib import Path
# from .logger import configure_root_logger
# from .run import SimulationRun
# from .scenario import SimulationPaths
# from .simulation import SimulationSettings


@flow(name="scenario-run")
def scenario_run_flow(
    scenario_path: str,
    input_dir: str,
    output_dir: str,
) -> None:
    logger = get_run_logger()
    logger.info("scenario_path=%s", scenario_path)
    logger.info("input_dir=%s", input_dir)
    logger.info("output_dir=%s", output_dir)
    logger.info("Dummy flow completed successfully!")
    logger.info("Finally! I am allowed to rest 🪦")


# @flow(name="scenario-run")
# def scenario_run_flow(
#     scenario_path: str,
#     input_dir: str,
#     output_dir: str,
# ) -> None:
#     paths = SimulationPaths.from_plain_paths(
#         scenario=Path(scenario_path),
#         input=Path(input_dir),
#         output=Path(output_dir),
#     )
#     configure_root_logger(paths.log, debugmode=False)
#     run = SimulationRun(paths=paths, settings=SimulationSettings())
#     run.execute()
