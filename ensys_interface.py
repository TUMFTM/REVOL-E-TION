import datetime
import os
import warnings

import pandas as pd

from blocks import VehicleCommoditySystem, StationaryEnergyStorage
from scenario import Scenario
from revoletion_rl_interface import predict_with_existing_model, build_model_and_train, map_flows_to_apriori_data, \
    get_blocks

SCALE_FACTOR = 1.0

def call_ensys_interface(scenario: Scenario,
                         forecast_length: int,
                         algorithm: str = "PPO",
                         model_dir: str = None) -> pd.DataFrame:
    """
    Creates an EnSyS config from the given Revoletion input files and predicts the energy flows with an existing model.
    :param input_config_dir: The input directory of the Revoletion configuration.
    :param model_dir: The model directory.
    :param forecast_length: The forecast length.
    :param scenario_name: The name of the scenario within the scenarios.csv file. Defaults to first scenario.
    :param tick_length: The tick length.
    :param algorithm: The algorithm.
    :param env_name: The environment name.
    :param entry_point: The entry point.
    :return: The energy flow history.
    """
    # temporarily allow UserWarnings
    original_filters = warnings.filters[:]
    warnings.simplefilter('default', category=UserWarning)

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    tick_length = datetime.timedelta(minutes=15)

    input_config_dir = scenario.run.path_input_data
    scenarios_file_name = scenario.run.scenario_file_name + '.csv'
    if model_dir is None:
        energy_flows, summary = build_model_and_train(input_config_dir=input_config_dir,
                                                      forecast_length=forecast_length,
                                                      training_steps=200000,
                                                      scenario_name=scenario.name,
                                                      scenarios_file_name=scenarios_file_name,
                                                      output_config_dir=f'ensys_interface/configs/{timestamp}',
                                                      tick_length=str(tick_length),
                                                      algorithm=algorithm,
                                                      env_name='EssEnvGridInfo-v0',
                                                      entry_point='rl.env.ess_env_grid_info:EssEnvGridInfo')
    else:
        energy_flows, summary = predict_with_existing_model(input_config_dir=input_config_dir,
                                                            model_dir=model_dir,
                                                            forecast_length=forecast_length,
                                                            scenarios_file_name=scenarios_file_name,
                                                            scenario_name=scenario.name,
                                                            output_config_dir=f'ensys_interface/configs/{timestamp}',
                                                            tick_length=str(tick_length),
                                                            algorithm=algorithm,
                                                            env_name='EssEnvGridInfo-v0',
                                                            entry_point='rl.env.ess_env_grid_info:EssEnvGridInfo')
    apriori_data = map_flows_to_apriori_data(energy_flow_history=energy_flows,
                                             blocks=get_blocks(input_dir=input_config_dir,
                                                               scenarios_file_name=scenarios_file_name,
                                                               scenario_name=scenario.name),
                                             tick_length=datetime.timedelta(minutes=15))

    vehicle_commodity_block: VehicleCommoditySystem | None = None
    ess_block: StationaryEnergyStorage | None = None
    for block in scenario.blocks.values():
        if isinstance(block, VehicleCommoditySystem):
            vehicle_commodity_block = block
        elif isinstance(block, StationaryEnergyStorage):
            ess_block = block

    if vehicle_commodity_block is not None:
        set_vehicles_apriori_data(vehicle_commodity_block, apriori_data)
    if ess_block is not None:
        set_ess_apriori_data(ess_block, apriori_data['ess'])
    warnings.filters = original_filters


def set_vehicles_apriori_data(vehicle: VehicleCommoditySystem, apriori_data: dict[str, pd.DataFrame]):
    """
    Sets the apriori data for all vehicles in the given vehicle commodity system.
    :param vehicle: The vehicle commodity system.
    :param apriori_data: The apriori data dict containing all apriori data DataFrames.
    """
    for name, bev in vehicle.commodities.items():
        bev.apriori_data = apriori_data[name] * SCALE_FACTOR
        # add two columns of zeros: p_ext_ac and p_ext_dc
        bev.apriori_data['p_ext_ac'] = 0
        bev.apriori_data['p_ext_dc'] = 0


def set_ess_apriori_data(ess: StationaryEnergyStorage, apriori_data: pd.DataFrame):
    """
    Sets the apriori data for the given ESS.
    :param ess: The ess.
    :param apriori_data: The ESS apriori data DataFrame.
    """
    ess.apriori_data = apriori_data * SCALE_FACTOR


def load_apriori_data(input_dir: str, start_time: datetime.datetime, tick_length: pd.Timedelta):
    """
    Loads the apriori data from the given input directory. The apriori Data is made of several DataFrames serialized as
    csv files. The DataFrames are stored in a dictionary with the key being the name of the DataFrame.
    :param input_dir: The input directory.
    :param start_time: The start time.
    :param tick_length: The tick length.
    :return: The apriori data dict (of DataFrames)
    """
    apriori_data = {}
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            apriori_data[file.split('.')[0]] = pd.read_csv(os.path.join(input_dir, file))

    data_length = len(next(iter(apriori_data.values())))
    date_range = pd.date_range(start_time, periods=data_length, freq=tick_length)
    for value in apriori_data.values():
        value.index = date_range
    return apriori_data
