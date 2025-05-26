#!/usr/bin/env python3

import ast
import importlib.metadata
import importlib.util
import logging
import os
import re
import shutil
import subprocess

import pandas as pd

_LOGGER = logging.getLogger(__name__)

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
        df.sort_index(axis=1,
                      level=0,
                      key=lambda x: x.map(lambda s: int(m.group(1))
                                            # get the last continuous sequence of digits if possible
                                            if (m := re.search(r'(\d+)(?!.*\d)', s))
                                            else s
                                          ),
                      sort_remaining=True,
                      inplace=True)
    else:
        df = pd.read_csv(path_input_file)
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1)

    # parser in to_csv does not create datetimeindex
    df = df.tz_convert(scenario.timezone)
    if not resampling:
        return df
    else:
        df_extd = df.reindex(extend_dti(dti=df.index, freq=scenario.timestep_td)).ffill()

        def resample_column(column):
            if df_extd[column].dtype == bool:
                return df_extd[column].resample(scenario.timestep).ffill().bfill()
            else:
                return df_extd[column].resample(scenario.timestep).mean().ffill().bfill()

        df = pd.DataFrame({col: resample_column(col) for col in df_extd.columns})[:-1]

        if not (scenario.dti_eval.isin(df.index).all()):
            raise IndexError(f'Block "{block.name}":'
                             f'Input timeseries data in {path_input_file} does not cover simulation timeframe')
        return df.loc[scenario.dti_sim]


def set_extension(filename, default_extension='.csv'):
    """
    Add a default extension to a filename if none is given. If the filename already has an extension, it is kept.
    """
    base, ext = os.path.splitext(filename)
    if not ext:
        filename = base + default_extension
    return filename

UNKNOWN_VERSION = "unknown"

def get_current_project_git_commit_hash() -> str:
    """
    Retrieves the short git commit hash of the current repository.
    
    Returns:
        The first 6 characters of the current git commit hash if available, otherwise `UNKNOWN_VERSION`.             
    """
    git_binary = shutil.which("git")
    if git_binary is None:
        # Some environments (e.g. docker, pip distribution) might not have git available.
        return UNKNOWN_VERSION

    try:
        commit_hash = (
            subprocess.check_output([git_binary, "rev-parse", "HEAD"])
            .strip()
            .decode()[0:6]
        )
        return commit_hash
    except subprocess.CalledProcessError:
        return UNKNOWN_VERSION


def get_revoletion_python_package_version() -> str:
    """
    Retrieves the version of the installed 'revoletion' package.

    Returns: 
        The version string of the 'revoletion' package if installed, otherwise `UNKNOWN_VERSION`.
    """
    try:
        return importlib.metadata.version("revoletion")
    except importlib.metadata.PackageNotFoundError:
        # If REVOL-E-TION is executed as script, the module might not be available in the current context.
        # This is usually the case, if only the dependencies of the project were installed
        # but the project itself was not explicitly installed (e.g. docker, script).
        _LOGGER.warning(
            "Failed to query REVOL-E_TION package version. This probably means that the package is not correctly installed in your current python environment."
        )
        return UNKNOWN_VERSION
