import logging
from pathlib import Path

from prefect import flow, get_run_logger

from .logger import configure_root_logger
from .optimization import Solver
from .run import SimulationRun
from .scenario import SimulationPaths
from .simulation import SimulationSettings

_EXAMPLE_SCENARIO = Path(__file__).parent.parent / "example" / "scenarios.csv"


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


@flow(name="scenario-run")
def scenario_run_flow(
    scenario_path: str = str(_EXAMPLE_SCENARIO),
    input_dir: str = "",
    output_dir: str = "",
    solver: str = "GUROBI",
    debugmode: bool = False,
) -> None:
    logger = get_run_logger()

    # DEBUG: Check if env vars are present
    import os

    logger.info("WLSACCESSID: %s", os.getenv("WLSACCESSID", "NOT SET"))
    logger.info("WLSSECRET: %s", os.getenv("WLSSECRET", "NOT SET"))
    logger.info("LICENSEID: %s", os.getenv("LICENSEID", "NOT SET"))

    logger.info("Starting scenario run: scenario_path=%s", scenario_path)
    logger.info("solver=%s  debugmode=%s", solver, debugmode)

    input_path: Path | None = Path(input_dir) if input_dir else None
    output_path: Path | None = Path(output_dir) if output_dir else None

    paths = SimulationPaths.from_plain_paths(
        scenario=Path(scenario_path),
        input=input_path,
        output=output_path,
    )

    configure_root_logger(paths.log, debugmode=debugmode, stdout=False)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger().addHandler(_PrefectBridgeHandler(logger))

    settings = SimulationSettings(
        solver=Solver[solver.upper()],
        debugmode=debugmode,
    )

    run = SimulationRun(paths=paths, settings=settings)
    run.execute()

    logger.info("Scenario run completed: results written to %s", paths.output)
