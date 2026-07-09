import logging
from pathlib import Path

import requests
from prefect import flow, get_run_logger
from prefect.blocks.system import Secret
from prefect.variables import Variable
from revoletion_core.model import ScenarioModel

from .logger import configure_root_logger
from .optimization import Solver
from .run import SimulationRun
from .scenario import SimulationPaths
from .simulation import SimulationSettings

worker_token: str = "cm91dGVraWRzc3VjaGNvbGRwZW9wbGVwYXJ0bHlmb3J3YXJkc3dpbmdleHBlcmllbmM="


class _PrefectBridgeHandler(logging.Handler):
    """Routes root-logger records into the active Prefect flow-run log.

    Prefect's Docker worker only shows logs that go through its API-based
    PrefectHandler. The simulation logs via the Python root logger, which
    only reaches stdout/file — not the Prefect UI. This handler bridges the
    two systems so simulation output appears in the Prefect flow run log.
    """

    def __init__(self, prefect_logger):
        super().__init__()
        self._log = prefect_logger

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
        except Exception:
            self.handleError(record)


async def download_scenarios(ids: list[str]) -> list[ScenarioModel]:
    backend_url: str | None = Variable.get("REVOLETION_BACKEND_URL")
    worker_token: str | None = Secret.load("REVOLETION_WORKER_TOKEN").get()

    if not backend_url:
        raise Exception(
            "Worker does not know a backend url. Make sure `REVOLETION_BACKEND_URL` is set as a Prefect variable."
        )

    if not worker_token:
        raise Exception(
            "Worker can't authenticate. Worker token missing. Make sure `REVOLETION_WORKER_TOKEN` is set as Prefect secret block."
        )

    headers = {"Authorization": f"Bearer {worker_token}", "Accept": "application/json"}

    # For now just assume there is only a single id.
    scenario_id = ids[0]

    return [requests.get(f"{backend_url}/api/scenarios/{scenario_id}", headers=headers)]


def save_as_csv(scenarios: list[ScenarioModel]) -> Path:
    return Path()


@flow(name="scenario-run")
def scenario_run_flow(
    scenario_ids: list[str] = "",
    solver: str = "GUROBI",
    debugmode: bool = False,
) -> None:
    # TODO: define the correct input schema
    # TODO: input validation

    logger = get_run_logger()

    logger.info("solver=%s  debugmode=%s", solver, debugmode)

    scenarios: list[ScenarioModel] = download_scenarios(scenario_ids)
    scenario_path: Path = save_as_csv(scenarios)
    paths = SimulationPaths.from_plain_paths(scenario_path)
    logger.info("Starting scenario run: scenario_path=%s", scenario_path)

    # Required to get output on the prefect-server.
    # The _PrefectBridgeHandler essentially reroutes stdout to the prefect logger.
    configure_root_logger("/example/log.log", debugmode=debugmode, stdout=False)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger().addHandler(_PrefectBridgeHandler(logger))

    settings = SimulationSettings(solver=Solver[solver.upper()], debugmode=debugmode, n_processes=1)

    run = SimulationRun(paths=paths, settings=settings)
    run.execute()

    logger.info("Scenario run completed: results written to %s", paths.output)
