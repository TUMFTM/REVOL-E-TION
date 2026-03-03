import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from revoletion import run, simulation, utils

_LOGGER = logging.getLogger(__name__)
_NPV_TOLERANCE = 0.1


@pytest.mark.parametrize("scenario_name", ["icev"])
def test_process_example_scenarios(scenario_name: str):
    example_scenarios_path = Path(__file__).resolve().parents[2] / "example" / "scenarios.csv"
    with tempfile.TemporaryDirectory() as tempdir_raw:
        tempdir_path = Path(tempdir_raw)
        simulation_paths = simulation.SimulationPaths.from_plain_paths(
            scenario=example_scenarios_path,
            output=tempdir_path,
        )
        simulation_settings = simulation.SimulationSettings(solver="cbc")
        scenario_parameters = utils.read_scenario_from_file(simulation_paths.scenario)

        single_scenario_parameters = scenario_parameters[scenario_name]

        worker = run.ScenarioWorker(
            simulation_paths,
            simulation_settings,
            name=scenario_name,
            parameters=single_scenario_parameters,
            logger=_LOGGER,
            status_update=lambda status, queue: None,
        )
        worker.execute()

        result_dir_entries = list(tempdir_path.iterdir())
        assert len(result_dir_entries) == 1, (
            f"REVOL-E-TION produced invalid number of results: {len(result_dir_entries)}"
        )

        specific_result_dir = result_dir_entries[0]
        assert specific_result_dir.is_dir(), (
            f"REVOL-E-TION result does not have the expected format: {specific_result_dir.absolute()} is not a directory"
        )

        summary_files = list(specific_result_dir.rglob("*_summary_temp.pkl"))
        assert len(summary_files) == 1

        summary_file = summary_files[0]

        df = pd.read_pickle(summary_file)
        assert df.at[("scenario", "npv"), "icev"] == pytest.approx(-315545, rel=_NPV_TOLERANCE)
