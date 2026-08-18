import asyncio
import logging
from pathlib import Path
from typing import override

from prefect import flow, get_run_logger
from prefect.logging.loggers import LoggingAdapter
from revoletion_core.model import ScenarioModel
from revoletion_core.types import CollectedRef, RunParameters, collect_refs

from revoletion.backend.client import Client
from revoletion.backend.convert import convert_to_csv
from revoletion.logger import configure_root_logger
from revoletion.optimization import Solver
from revoletion.run import SimulationRun
from revoletion.scenario import SimulationPaths
from revoletion.simulation import SimulationSettings

# Remove warnings of experimental PrefectBridgeHandler
# pyright: reportUnknownMemberType=false, reportUnusedClass=false, reportMissingParameterType=false, reportUnknownParameterType=false

Logger = logging.Logger | LoggingAdapter


class _PrefectBridgeHandler(logging.Handler):
    """Routes root-logger records into the active Prefect flow-run log.

    Prefect's Docker worker only shows logs that go through its API-based
    PrefectHandler. The simulation logs via the Python root logger, which
    only reaches stdout/file — not the Prefect UI. This handler bridges the
    two systems so simulation output appears in the Prefect flow run log.
    """

    def __init__(self, prefect_logger):
        super().__init__()
        self._log = prefect_logger  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def emit(self, record):
        # Avoid re-routing Prefect's own records back into itself.
        if record.name.startswith("prefect"):
            return
        try:
            msg = record.getMessage()
            if record.levelno >= logging.ERROR:
                self._log.error(msg)
            elif record.levelno >= logging.WARNING:
                self._log.warning(msg)
            elif record.levelno >= logging.INFO:
                self._log.info(msg)
            else:
                self._log.debug(msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)


# TODO: Make this a dynamic value (at least through config).
BACKEND_URL = "http://172.17.0.1:8000"
# TODO: Rotate key!
WORKER_TOKEN = "cm91dGVraWRzc3VjaGNvbGRwZW9wbGVwYXJ0bHlmb3J3YXJkc3dpbmdleHBlcmllbmM="


async def download_scenarios_and_remote_objects(hashes: list[str], logger: Logger) -> list[ScenarioModel]:
    logger.info("Downloading scenarios { %s } from %s", str.join(", ", hashes), BACKEND_URL)
    async with Client(BACKEND_URL, WORKER_TOKEN) as client:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(client.get_scenario(h)) for h in hashes]
        scenario_models: list[ScenarioModel] = [t.result() for t in tasks]
        all_refs = []
        for scenario in scenario_models:
            collected_refs: list[CollectedRef] = collect_refs(scenario)
            all_refs.append(collected_refs)

        return scenario_models


def save_as_csv(scenarios: list[ScenarioModel], parent_dir: Path, logger: Logger) -> Path:
    scenario: ScenarioModel = scenarios[0]
    path: Path = convert_to_csv(scenario, parent_dir)
    return path


@flow(name="scenario-run")
async def scenario_run_flow(parameters: RunParameters) -> None:
    logger: Logger = get_run_logger()

    logger.info(
        "solver=%s  debugmode=%s, hashes=%s",
        parameters.solver,
        parameters.debugmode,
        parameters.scenario_hashes,
    )

    download_dir: Path = Path(Path.cwd(), "downloads").resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    results_dir: Path = Path(Path.cwd(), "results").resolve()

    scenarios: list[ScenarioModel] = await download_scenarios_and_remote_objects(parameters.scenario_hashes, logger)

    scenario_path: Path = save_as_csv(scenarios, download_dir, logger)

    paths = SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )
    logger.info("Starting scenario run: scenario_path=%s", scenario_path)

    # Required to get output on the prefect-server.
    # The _PrefectBridgeHandler essentially reroutes stdout to the prefect logger.
    configure_root_logger(
        Path(results_dir, "run.log").resolve(),
        debugmode=parameters.debugmode,
        stdout=False,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger().addHandler(_PrefectBridgeHandler(logger))

    settings = SimulationSettings(
        solver=Solver[parameters.solver.upper()],
        debugmode=parameters.debugmode,
        n_processes=1,
    )

    _ = SimulationRun(paths=paths, settings=settings)
    # run = SimulationRun(paths=paths, settings=settings)
    # run.execute()

    logger.info("Scenario run completed: results written to %s", paths.output)


# TODO:: Remove, this is only for debugging/dev
if __name__ == "__main__":
    # Runs the flow locally in an ephemeral test state
    parameters: RunParameters = RunParameters(
        project_id="",
        blueprint_version=1,
        scenario_hashes=["1d84c87aaca93888df689eb594e2e12606f777faac07d54d942509f1b190b719"],
        solver="gurobi",
        debugmode=True,
    )
    asyncio.run(scenario_run_flow(parameters=parameters))
