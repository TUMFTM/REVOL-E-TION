import asyncio
import base64
import io
import logging
import pathlib
import re
import zipfile
from pathlib import Path

import aiocsv
import aiofiles
from prefect import flow, get_run_logger
from prefect.logging.loggers import LoggingAdapter
from revoletion_core.model import PolymorphicBlock, ScenarioModel
from revoletion_core.types import (
    CollectedRef,
    RemoteObject,
    Result,
    RunParameters,
    collect_refs,
)

from revoletion.backend.client import Client
from revoletion.backend.convert import convert_to_csv
from revoletion.logger import configure_root_logger
from revoletion.optimization import Solver
from revoletion.prefect.logging import PrefectBridgeHandler
from revoletion.run import SimulationRun
from revoletion.scenario import SimulationPaths
from revoletion.simulation import SimulationSettings

Logger = logging.Logger | LoggingAdapter

# TODO: Make this a dynamic value (at least through config).
# Currently points at docker host bridge
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
    object: RemoteObject = await client.get_remote_object(remote_object_id=collected.identifier)
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

    return (collected.identifier, out_path)


async def _download_remote_objects_and_save(
    collected_refs: list[CollectedRef], destination_dir: Path
) -> dict[str, str]:
    """
    Downloads the remote objects and saves them to disk. The dictionary returns a mapping
    from remote object identifier to file path string, used for later replacement in the scenario csv.
    """
    downloadable_refs = [ref for ref in collected_refs if ref.target_type == "TimeSeries"]
    async with Client(BACKEND_URL, WORKER_TOKEN) as client:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)

        async def bounded_download(collected_ref: CollectedRef) -> tuple[str, Path]:
            async with semaphore:
                try:
                    return await _fetch_and_save_remote_object(client, collected_ref, destination_dir)
                except Exception as exc:
                    path_str = "/".join(map(str, collected_ref.path))
                    raise RuntimeError(
                        f"Failed to download remote object '{collected_ref.identifier}' at path '{path_str}': {exc}"
                    ) from exc

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(bounded_download(ref)) for ref in downloadable_refs]

    tuples: list[tuple[str, Path]] = [t.result() for t in tasks]

    mapping: dict[str, str] = {}
    for key, value in tuples:
        mapping[key] = str(value)

    return mapping


def _get_block_id_name_mapping(scenarios: list[ScenarioModel]) -> dict[str, str]:
    """
    Returns a mapping from block identifier to block name. Since revoletion currently requires that block identifiers
    are human readable/understandable [namingly in the mtf.py], the mapping is later used to fulfill that requirement.

    Beware: this implicitly constrains the block names to be unique identifiers aswell! Later this behavior can be dropped.
    """
    mapping: dict[str, str] = {}
    for scenario in scenarios:
        block_configs: dict[str, PolymorphicBlock] = scenario.block_configs
        for key, value in block_configs.items():
            mapping[key] = value.name

    return mapping


async def _replace_ids_in_csv(file_path: Path, mapping: dict[str, str]) -> Path:
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

        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))

        async for row in reader:
            # Replace mapped IDs in every cell of the row
            new_row = [pattern.sub(lambda m: mapping[m.group(0)], cell) for cell in row]
            await writer.writerow(new_row)

    # Atomically overwrite the original file
    _ = temp_path.replace(resolved_path)

    return resolved_path


def _unique(all_refs: list[CollectedRef]) -> list[CollectedRef]:
    """Returns a list of collected refs without duplicates."""

    def _identifier(x: CollectedRef) -> str:
        """Returns a hashible unique identifier for refs."""
        return (x.target_type or "") + x.identifier

    used: set[str] = set()
    return [x for x in all_refs if _identifier(x) not in used and (used.add(_identifier(x)) or True)]


async def _download_scenarios_and_collect_refs(
    hashes: list[str], logger: Logger
) -> tuple[list[ScenarioModel], list[CollectedRef]]:
    """
    Downloads the scenarios identified by their hashes as well as
    the remote objects identified by their ids and types from the backend.
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


def _zip_directory_to_base64(directory: Path) -> str:
    """
    Zips all files contained within *directory* (recursively) into an
    in-memory archive and returns the archive bytes as a base64-encoded string.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in directory.rglob("*"):
            if file.is_file():
                zip_file.write(file, arcname=str(file.relative_to(directory)))
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@flow(name="scenario-run")
async def scenario_run_flow(parameters: RunParameters) -> None:
    """
    The main function which the prefect worker calls when spinning up the deployment container.
    """

    logger: Logger = get_run_logger()
    logger.info("Starting a simulation with parameters: %s", parameters.model_dump_json())

    download_dir: Path = Path(Path.cwd(), "downloads").resolve()
    results_dir: Path = Path(Path.cwd(), "results").resolve()

    scenarios: list[ScenarioModel]
    collected_refs: list[CollectedRef]

    scenarios, collected_refs = await _download_scenarios_and_collect_refs(parameters.scenario_hashes, logger)

    """
    We call it flipped csv file, since the input csv to revoletion
    follows more of a dictionary definition than standard csv.
    """
    scenario_path_with_identifiers: Path = _save_as_flipped_csv(scenarios, download_dir, logger)

    remote_objects_id_path_mapping: dict[str, str] = await _download_remote_objects_and_save(
        collected_refs, download_dir
    )

    block_name_id_mapping: dict[str, str] = _get_block_id_name_mapping(scenarios)

    identifier_replacement_mapping: dict[str, str] = {
        **remote_objects_id_path_mapping,
        **block_name_id_mapping,
    }
    logger.info("Replacing identifiers with values: %s", identifier_replacement_mapping)
    scenario_path: Path = await _replace_ids_in_csv(scenario_path_with_identifiers, identifier_replacement_mapping)

    async with aiofiles.open(scenario_path, mode="r", encoding="utf-8") as file:
        content: str = await file.read()
        logger.info("Scenario csv after replacement: %s", content)

    paths: SimulationPaths = SimulationPaths.from_plain_paths(
        scenario=scenario_path,
    )

    logger.info("Starting revoletion with scenario_path: %s", scenario_path)

    # Required to get output on the prefect-server.
    # The _PrefectBridgeHandler essentially reroutes stdout to the prefect logger.
    configure_root_logger(
        Path(results_dir, "run.log").resolve(),
        debugmode=parameters.debugmode,
        stdout=False,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger().addHandler(PrefectBridgeHandler(logger))

    settings = SimulationSettings(
        solver=Solver[parameters.solver.upper()],
        debugmode=parameters.debugmode,
        n_processes=1,
    )

    run = SimulationRun(paths=paths, settings=settings)
    run.execute()

    logger.info("Scenario run completed: results written to %s", paths.output)

    result_b64 = _zip_directory_to_base64(paths.output)
    result = Result(
        scenario_hash=parameters.scenario_hashes[0],
        payload=result_b64,
    )
    async with Client(BACKEND_URL, WORKER_TOKEN) as client:
        await client.upload_result(result)

    logger.info("Uploaded results archive for scenario %s", result.scenario_hash)
