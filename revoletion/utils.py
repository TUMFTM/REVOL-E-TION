#!/usr/bin/env python3

import ast
import importlib.metadata
import importlib.util
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)


def convert2timedelta(value: pd.Timedelta | str | float | int | None, unit: str = None) -> pd.Timedelta | None:
    if value is None:
        return None

    if not isinstance(value, (pd.Timedelta, str, float, int)):
        raise TypeError("Value must be of type pd.Timedelta, str, float or int.")

    if isinstance(value, str):
        value = pd.Timedelta(value)
    elif isinstance(value, (float, int)):
        if unit is None:
            raise ValueError("If value is a number, a unit must be provided.")
        value = pd.Timedelta(value, unit=unit)

    return value


@dataclass
class RunTime:
    start: float = field(default_factory=time.perf_counter, repr=False)
    end: float = field(default=np.nan, repr=False)
    duration: float = field(default=np.nan)

    def stop(self) -> None:
        self.end = time.perf_counter()
        self.duration = self.end - self.start

    @property
    def result_summary(self) -> pd.Series:
        # only export runtime duration -> start and end are not interpretable
        return pd.Series({"runtime_duration_s": round(self.duration, 2)})


def infer_dtype(value):
    """
    infer the data type of a value from a string representation. To be used as a .map(infer_dtype) function.
    """

    # remove whitespace at beginning or end of string (convert to string, as nan already is of type float)
    value = str(value).strip()

    try:
        return int(value)
    except ValueError or OverflowError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.lower() in ["none", "null", "nan", ""]:
        return None

    try:
        evaluated = ast.literal_eval(value)
        if isinstance(evaluated, dict):
            return evaluated
        elif isinstance(evaluated, list):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    return value.lower()


def create_results_from_dataframe(df: pd.DataFrame, name_prefix: str) -> pd.Series:
    """
    Convert results stored in a DataFrame to a Series for scenario.result_summary.
    """
    result_series = pd.Series(df.stack())
    # create MultiIndex. Use "_".join() to avoid problems if df already has MultiIndex
    result_series.index = result_series.index.map(lambda x: f"{name_prefix}_{'_'.join(x)}")

    return result_series


def conv_nan2none(value):
    """
    Convert NaN values to None as oemof components require None instead of NaN.
    """
    return value if pd.notna(value) else None


def extend_dti(dti: pd.DatetimeIndex, freq: pd.DateOffset | pd.Timedelta | str) -> pd.DatetimeIndex:
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


def read_timeseries_csv(
    path_input_file: str | Path, scenario: "simulation.Scenario", multiheader: bool = False, resampling: bool = True
):
    """
    Properly read in timezone-aware example timeseries csv files and form correct datetimeindex

    :raises IndexError: If timeseries data does not cover simulation timeframe.
    """
    if multiheader:
        df = pd.read_csv(path_input_file, header=[0, 1])
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1)
        df.sort_index(
            axis=1,
            level=0,
            key=lambda x: x.map(
                lambda s: int(m.group(1))
                # get the last continuous sequence of digits if possible
                if (m := re.search(r"(\d+)(?!.*\d)", s))
                else s
            ),
            sort_remaining=True,
            inplace=True,
        )
    else:
        df = pd.read_csv(path_input_file)
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1)

    # parser in to_csv does not create datetimeindex
    df = df.tz_convert(scenario.location.timezone)
    if not resampling:
        return df
    else:
        df_extd = df.reindex(extend_dti(dti=df.index, freq=scenario.timestep.td)).ffill()

        def resample_column(column):
            if df_extd[column].dtype == bool:
                return df_extd[column].resample(scenario.timestep.td).ffill().bfill()
            else:
                return df_extd[column].resample(scenario.timestep.td).mean().ffill().bfill()

        df = pd.DataFrame({col: resample_column(col) for col in df_extd.columns})[:-1]

        if not (scenario.times.sim.dti.isin(df.index).all()):
            raise IndexError(f"Input timeseries data in {path_input_file} does not cover simulation timeframe")
        return df.loc[scenario.times.sim.dti]


def set_extension(filename: Path | str, default_extension: str = ".csv") -> Path:
    """
    Add a default extension to a filename if none is given. If the filename already has an extension, it is kept.
    """
    return path.with_suffix(default_extension) if not (path := Path(filename)).suffix else path


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
        commit_hash = subprocess.check_output([git_binary, "rev-parse", "HEAD"]).strip().decode()[0:6]
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


def read_scenario_from_file(scenario_path: Path) -> pd.DataFrame:
    """
    Load the scenario from the scenario path. Valid formats are CSV and Python pickle.

    Args:
        scenario_path: Path to the scenario file.

    Returns:
        The pandas DataFrame containing the scenario parameters.

    Raises:
        FileNotFoundError: Either if `scenario_path` does not exist or if it isn't a valid file.
        TypeError: If the scenario file content is not a pandas DataFrame.
        ValueError: If the file is neither a CSV nor a pickle file.
    """
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file '{scenario_path}' does not exist")

    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario at '{scenario_path}' is not a file")

    if scenario_path.suffix == ".csv":
        parameters = pd.read_csv(scenario_path, index_col=[0, 1], keep_default_na=False)
    elif scenario_path.suffix == ".pkl":
        parameters = pd.read_pickle(scenario_path)
        if not isinstance(parameters, pd.DataFrame):
            raise TypeError(
                f"Scenario parameters from file '{scenario_path}' have wrong format:"
                + "Expected '{type(pd.DataFrame)} but got {type(parameters)}'"
            )
    else:
        raise ValueError(f"Scenario file '{scenario_path}' is neither CSV nor PKL file.")

    parameters = parameters.sort_index(sort_remaining=True).map(infer_dtype)
    return parameters
