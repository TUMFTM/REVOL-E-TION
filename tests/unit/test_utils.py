import importlib.resources

import revoletion.example
from revoletion import utils


def test_get_current_project_git_commit_hash():
    commit_hash = utils.get_current_project_git_commit_hash()
    assert commit_hash != utils.UNKNOWN_VERSION


def test_get_revoletion_python_package_version():
    version = utils.get_revoletion_python_package_version()
    assert version != utils.UNKNOWN_VERSION


def test_read_scenario_from_file():
    with importlib.resources.path(revoletion.example, "scenarios_example.csv") as example_scenarios_path:
        scenario_parameters = utils.read_scenario_from_file(example_scenarios_path)
        assert "icev" in scenario_parameters
