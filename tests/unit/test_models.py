import importlib.resources

import revoletion.example
from revoletion.models import ScenarioModel, validate_scenario_csv_file


def test_example_scenario_is_valid():
    example_scenarios_traversable = importlib.resources.files(revoletion.example).joinpath("scenarios_example.csv")
    with importlib.resources.as_file(example_scenarios_traversable) as example_scenarios_path:
        scenario_is_valid = validate_scenario_csv_file(example_scenarios_path)
        assert scenario_is_valid


def test_correctly_convertes_raw_values():
    test_scenario_data = {
        "starttime": "01.01.2025",
        "timestep": "15min",
        "sim_duration": "1 day",
        "sim_endtime": None,
        "prj_duration": 1,
        "compensate_sim_prj": "true",
        "strategy": "Go",
        "len_ph": None,
        "len_ch": None,
        "truncate_ph": False,
        "latitude": 50.1,
        "longitude": "43.89",
        "country": "DE",
        "state": "BY",
        "consider_holidays": "True",
        "temp_air": None,
        "cost_eps": "1e-3",
        "blocks": "{'foo': 'Foo'}",
    }

    scenario_model = ScenarioModel.model_validate(test_scenario_data)
    assert isinstance(scenario_model.blocks, dict)
    assert isinstance(scenario_model.latitude, float)
    assert isinstance(scenario_model.longitude, float)
    assert isinstance(scenario_model.consider_holidays, bool)
    assert scenario_model.strategy == "go"
    assert isinstance(scenario_model.compensate_sim_prj, bool)
    assert isinstance(scenario_model.cost_eps, float)
