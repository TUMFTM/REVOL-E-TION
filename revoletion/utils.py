#!/usr/bin/env python3

import ast
import importlib.util
import numpy as np
import pandas as pd
import pandas.errors
import os

from revoletion import economics as eco


def infer_dtype(value):
    """
    infer the data type of a value from a string representation. To be used as a .map(infer_dtype) function.
    """

    # remove whitespace at beginning or end of string (convert to string, as nan already is of type float)
    value = str(value).strip()

    try:
        return int(value)
    except (ValueError or OverflowError):
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() in ['none', 'null', 'nan', '']:
        return None
    elif os.path.isdir(value):
        return value

    try:
        evaluated = ast.literal_eval(value)
        if isinstance(evaluated, dict):
            return evaluated
        elif isinstance(evaluated, list):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    return value.lower()


def create_results_from_dataframe(df: pd.DataFrame,
                                  name_prefix: str) -> pd.Series:
    """
    Convert results stored in a DataFrame to a Series for scenario.result_summary.
    """
    result_series = pd.Series(df.stack())
    # create MultiIndex. Use "_".join() to avoid problems if df already has MultiIndex
    result_series.index = result_series.index.map(lambda x: f'{name_prefix}_{"_".join(x)}')

    return result_series


def conv_nan2none(value):
    """
    Convert NaN values to None as oemof components require None instead of NaN.
    """
    return value if pd.notna(value) else None


def extend_dti(dti: pd.DatetimeIndex,
               freq: pd.DateOffset | pd.Timedelta | str) -> pd.DatetimeIndex:
    """
    Extend a datetime index by one timestep to include the last timestep of the simulation timeframe.
    """
    dti_ext = dti.union(dti.shift(periods=1, freq=freq)[-1:])
    return dti_ext


def import_module_from_path(module_name, file_path):
    """
    Import a Python module from a specific file path. Is used for timeframe mapper user input code.
    """
    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)
    # Load and execute the module
    spec.loader.exec_module(module)
    return module


def read_timeseries_csv(path_input_file: str,
                        block: 'Block',
                        scenario: 'Scenario',
                        multiheader: bool = False,
                        resampling: bool = True):
    """
    Properly read in timezone-aware example timeseries csv files and form correct datetimeindex
    """
    if multiheader:
        df = pd.read_csv(path_input_file, header=[0, 1])
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1)
        df.sort_index(axis=1, sort_remaining=True, inplace=True)
    else:
        df = pd.read_csv(path_input_file)
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1)

    # parser in to_csv does not create datetimeindex
    df = df.tz_convert(scenario.timezone)
    if not resampling:
        return df
    else:
        df = resample_to_timestep(df, scenario)
        if not (scenario.dti_eval.isin(df.index).all()):
            raise IndexError(f'Block "{block.name}":'
                             f'Input timeseries data in {path_input_file} does not cover simulation timeframe')
        return df.loc[scenario.dti_sim]


def resample_to_timestep(data: pd.DataFrame, scenario):
    """
    Resample the data to the timestep of the scenario, conserving the proper index end even in upsampling
    """

    # Add one element to the dataframe to include the last timestep
    data_extd = data.reindex(extend_dti(dti=data.index, freq=scenario.timestep_td)).ffill()

    def resample_column(column):
        if data_extd[column].dtype == bool:
            return data_extd[column].resample(scenario.timestep).ffill().bfill()
        else:
            return data_extd[column].resample(scenario.timestep).mean().ffill().bfill()

    resampled_data = pd.DataFrame({col: resample_column(col) for col in data_extd.columns})[:-1]
    return resampled_data


def transform_scalar_var(value, scenario, block=None):
    """
    Transform a value holding either the filename of a csv file containing a timeseries or a scalar
    to a pandas Series with the same DatetimeIndex as the simulation.
    """
    if isinstance(value, str):  # value contains filename
        filename = set_extension(filename=value, default_extension='.csv')
        df = read_timeseries_csv(path_input_file=os.path.join(scenario.run.paths['input'], filename),
                                 block=block,
                                 scenario=scenario,
                                 multiheader=False,
                                 resampling=True)
        if df.shape[1] != 1:
            scenario.logger.warning(f'Block "{block.name}": Input data in {filename} contains more than one column - '
                                    f'only first column is used.')

        return df.iloc[:, 0]  # return only first column

    else:  # value is given as scalar
        return pd.Series(data=value,
                         index=scenario.dti_sim)


def set_extension(filename, default_extension='.csv'):
    """
    Add a default extension to a filename if none is given. If the filename already has an extension, it is kept.
    """
    base, ext = os.path.splitext(filename)
    if not ext:
        filename = base + default_extension
    return filename