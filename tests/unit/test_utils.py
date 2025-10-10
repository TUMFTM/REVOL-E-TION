import importlib.resources
import revoletion.example
from revoletion.utils import (
    UNKNOWN_VERSION,
    get_current_project_git_commit_hash,
    get_revoletion_python_package_version,
    read_scenario_from_file,
)


def test_get_current_project_git_commit_hash():
    commit_hash = get_current_project_git_commit_hash()
    assert commit_hash != UNKNOWN_VERSION


def test_get_revoletion_python_package_version():
    version = get_revoletion_python_package_version()
    assert version != UNKNOWN_VERSION


def test_read_scenario_from_file():
    with importlib.resources.path(revoletion.example, "scenarios_example.csv") as example_scenarios_path:
        scenario_parameters = read_scenario_from_file(example_scenarios_path)
        assert "icev" in scenario_parameters
