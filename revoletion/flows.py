import asyncio
import base64
import io
import logging
import pathlib
import zipfile
from pathlib import Path
from typing import override

import aiocsv
import aiofiles
from prefect import flow, get_run_logger
from prefect.logging.loggers import LoggingAdapter
from revoletion_core.model import ScenarioModel
from revoletion_core.types import CollectedRef, RemoteObject, RunParameters, TimeSeries, collect_refs

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


# Upper bound for simultaneously in-flight remote object downloads,
# so large batches do not hammer the backend (or blow up worker RAM).
_MAX_CONCURRENT_DOWNLOADS = 4


async def _fetch_and_save_remote_object(
    client: Client, collected: CollectedRef, destination_dir: Path
) -> tuple[str, Path]:
    """
    Downloads a single remote object, decompresses it and writes it to
    `destination_dir` under the object's label.
    """
    object: RemoteObject = await client.get_remote_object(remote_object_id=collected.ref.id)
    zip_bytes: bytes = base64.b64decode(object.payload)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        contents = zip_file.read(zip_file.namelist()[0])

    filename = pathlib.Path(object.label).name  # .name strips any path components
    if (destination_dir / filename).exists():
        raise ValueError(f"Duplicate remote object filename: {filename}")

    # No await between the collision check above and this write, so two tasks
    # sharing a filename cannot interleave here (single-threaded event loop).
    out_path = destination_dir / filename
    _ = out_path.write_bytes(contents)

    return (collected.ref.id, out_path)


async def _download_remote_objects_and_save(
    collected_refs: list[CollectedRef], destination_dir: Path, logger: Logger
) -> dict[str, Path]:
    downloadable_refs = [ref for ref in collected_refs if ref.target_type is TimeSeries]
    async with Client(BACKEND_URL, WORKER_TOKEN) as client:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)

        async def bounded_download(collected_ref: CollectedRef) -> tuple[str, Path]:
            async with semaphore:
                try:
                    return await _fetch_and_save_remote_object(client, collected_ref, destination_dir)
                except Exception as exc:
                    path_str = "/".join(map(str, collected_ref.path))
                    raise RuntimeError(
                        f"Failed to download remote object '{collected_ref.ref.id}' at path '{path_str}': {exc}"
                    ) from exc

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(bounded_download(ref)) for ref in downloadable_refs]

    tuples: list[tuple[str, Path]] = [t.result() for t in tasks]

    mapping: dict[str, Path] = {}
    for key, value in tuples:
        mapping[key] = value

    return mapping


async def _replace_ids_in_csv(file_path: Path, mapping: dict[str, Path]) -> Path:
    """
    Replaces ID values across all cells in a CSV file and safely overwrites it.
    """
    resolved_path: Path = file_path.resolve()

    # Note: Added newline="" to infile to prevent CSV newline translation issues
    async with (
        aiofiles.open(resolved_path, "r", encoding="utf-8", newline="") as infile,
        aiofiles.tempfile.NamedTemporaryFile(
            "w", dir=resolved_path.parent, delete=False, newline="", encoding="utf-8"
        ) as outfile,
    ):
        temp_path: Path = Path(str(outfile.name))

        # Standard csv.reader/writer are synchronous and will crash with aiofiles
        reader = aiocsv.AsyncReader(infile)
        writer = aiocsv.AsyncWriter(outfile)

        row: list[str]
        async for row in reader:
            # Cast to str() because mapping values are Path objects
            await writer.writerow([str(mapping.get(cell, cell)) for cell in row])

    # Atomically overwrite the original file
    _ = temp_path.replace(resolved_path)

    return resolved_path


def _identifier(x: CollectedRef) -> str:
    """Returns a hashible unique identifier for refs."""
    return str(x.target_type) + x.ref.id


def _unique(all_refs: list[CollectedRef]) -> list[CollectedRef]:
    """Returns a list of collected refs without duplicates."""
    used: set[str] = set()
    return [x for x in all_refs if _identifier(x) not in used and (used.add(_identifier(x)) or True)]


async def _download_scenarios_and_collect_refs(
    hashes: list[str], logger: Logger
) -> tuple[list[ScenarioModel], list[CollectedRef]]:
    """
    Downloads the scenarios identified by their hashes as well as
    the remote objects identified by their ids and
    ref_types from the backend.
    """
    logger.info("Downloading scenarios { %s } from %s", str.join(", ", hashes), BACKEND_URL)
    async with Client(BACKEND_URL, WORKER_TOKEN) as client:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(client.get_scenario(h)) for h in hashes]
        scenario_models: list[ScenarioModel] = [t.result() for t in tasks]
        all_refs: list[CollectedRef] = []
        for scenario in scenario_models:
            collected_refs: list[CollectedRef] = collect_refs(scenario)
            all_refs += collected_refs

        unique_refs = _unique(all_refs)

        return scenario_models, unique_refs


def _save_as_flipped_csv(scenarios: list[ScenarioModel], parent_dir: Path, logger: Logger) -> Path:
    """
    Glue code function as long as scientific uses paths of csv files.
    """
    scenario: ScenarioModel = scenarios[0]
    logger.info("Saving scenario: %s", scenario)
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
    results_dir: Path = Path(Path.cwd(), "results").resolve()

    scenarios, collected_refs = await _download_scenarios_and_collect_refs(parameters.scenario_hashes, logger)

    scenario_path: Path = _save_as_flipped_csv(scenarios, download_dir, logger)
    remote_objects_id_path_mapping: dict[str, Path] = await _download_remote_objects_and_save(
        collected_refs, download_dir, logger
    )

    logger.info("mapping: %s", remote_objects_id_path_mapping)

    scenario_path: Path = await _replace_ids_in_csv(scenario_path, remote_objects_id_path_mapping)

    async with aiofiles.open(scenario_path, mode="r", encoding="utf-8") as file:
        content: str = await file.read()
        logger.info("Scenario csv after replacement: \n%s", content)

    paths: SimulationPaths = SimulationPaths.from_plain_paths(
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

    run = SimulationRun(paths=paths, settings=settings)
    run.execute()

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
