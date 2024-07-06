import datetime
import os

import pandas as pd

from blocks import VehicleCommoditySystem, StationaryEnergyStorage


# from revoletion_ensys_interface import predict_with_existing_model, build_model_and_train, map_flows_to_apriori_data

def call_ensys_interface(scenario,
                         input_config_dir: str,
                         forecast_length: int,
                         algorithm: str = "DQN",
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
    # energy_flows = build_model_and_train(input_dir=input_config_dir, forecast_length=forecast_length,
    #                                           scenario_name=scenario.name, algorithm=algorithm)
    # apriori_data = map_flows_to_apriori_data(energy_flows)

    apriori_data = load_apriori_data('apriori_data', scenario.starttime, scenario.timestep_td)

    vehicle_commodity_block: VehicleCommoditySystem = None
    ess_block: StationaryEnergyStorage = None
    for block in scenario.blocks.values():
        if isinstance(block, VehicleCommoditySystem):
            vehicle_commodity_block = block
        elif isinstance(block, StationaryEnergyStorage):
            ess_block = block

    if vehicle_commodity_block is not None:
        set_vehicles_apriori_data(vehicle_commodity_block, apriori_data)
    if ess_block is not None:
        set_ess_apriori_data(ess_block, apriori_data['ess'])


def set_vehicles_apriori_data(vehicle: VehicleCommoditySystem, apriori_data: dict[str, pd.DataFrame]):
    """
    Sets the apriori data for all vehicles in the given vehicle commodity system.
    :param vehicle: The vehicle commodity system.
    :param apriori_data: The apriori data dict containing all apriori data DataFrames.
    """
    for name, bev in vehicle.commodities.items():
        bev.apriori_data = apriori_data[name]
        # add two columns of zeros: p_ext_ac and p_ext_dc
        bev.apriori_data['p_ext_ac'] = 0
        bev.apriori_data['p_ext_dc'] = 0


def set_ess_apriori_data(ess: StationaryEnergyStorage, apriori_data: pd.DataFrame):
    """
    Sets the apriori data for the given ESS.
    :param ess: The ess.
    :param apriori_data: The ESS apriori data DataFrame.
    """
    ess.apriori_data = apriori_data


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


if __name__ == '__main__':
    apr = load_apriori_data("apriori_data", datetime.datetime(2018, 1, 1), pd.Timedelta('15min'))
    print(len(apr))
    # input_config_dir = os.path.join(os.getcwd(), 'input')
    # forecast_length = 8
    # algorithm = "DQN"
    # scenario_name = "dispatch_go"
    # call_ensys_interface(input_config_dir, forecast_length, algorithm, scenario_name)
