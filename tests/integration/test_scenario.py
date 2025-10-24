import importlib.resources
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import revoletion.example
from revoletion.simulation import Scenario, SimulationPaths, SimulationSettings
from revoletion.utils import read_scenario_from_file

_POWER_TOLERANCE = 0.1


def test_process_example_scenarios():
    example_scenarios_traversable = importlib.resources.files(revoletion.example).joinpath("scenarios_example.csv")
    with importlib.resources.as_file(example_scenarios_traversable) as example_scenarios_path:
        with tempfile.TemporaryDirectory() as tempdir_raw:
            tempdir_path = Path(tempdir_raw)
            simulation_paths = SimulationPaths.from_plain_paths(
                scenario=example_scenarios_path,
                output=tempdir_path,
            )
            simulation_settings = SimulationSettings(solver="cbc")
            scenario_parameters = read_scenario_from_file(simulation_paths.scenario)

            single_scenario_parameters = scenario_parameters["icev"]

            scenario = Scenario(
                simulation_paths,
                simulation_settings,
                name="icev",
                parameters=single_scenario_parameters,
            )
            scenario.execute()

            result_dir_entries = list(tempdir_path.iterdir())
            assert len(result_dir_entries) == 1, (
                f"REVOL-E-TION produced invalid number of results: {len(result_dir_entries)}"
            )

            specific_result_dir = result_dir_entries[0]
            assert specific_result_dir.is_dir(), (
                f"REVOL-E-TION result does not have the expected format: {specific_result_dir.absolute()} is not a directory"
            )

            results_ts_files = list(specific_result_dir.rglob("*_results_ts.csv"))
            assert len(results_ts_files) == 1

            results_ts_file = results_ts_files[0]

            df = pd.read_csv(results_ts_file, header=[0, 1])
            assert df["core"]["acdc"][1] == pytest.approx(600, rel=_POWER_TOLERANCE)
