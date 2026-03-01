from pathlib import Path

from revoletion import utils


def test_get_current_project_git_commit_hash():
    commit_hash = utils.get_current_project_git_commit_hash()
    assert commit_hash != utils.UNKNOWN_VERSION


def test_get_revoletion_python_package_version():
    version = utils.get_revoletion_python_package_version()
    assert version != utils.UNKNOWN_VERSION


def test_read_scenario_from_file():
    example_scenarios_path = Path(__file__).resolve().parents[2] / "examples" / "scenarios.csv"
    scenario_parameters = utils.read_scenario_from_file(example_scenarios_path)
    assert "icev" in scenario_parameters
