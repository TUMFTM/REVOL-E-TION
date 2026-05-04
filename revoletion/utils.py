#!/usr/bin/env python3

import ast
import importlib.metadata
import importlib.util
import logging
import re
import zoneinfo
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import geopy
import geopy.geocoders
import holidays
import pandas as pd
import pytz
import timezonefinder
import typing_extensions
from typing_extensions import Self

from . import time
from .time import extend_dti

_LOGGER = logging.getLogger(__name__)


def convert2timedelta(value: pd.Timedelta | str | float | int | None, unit: str | None = None) -> pd.Timedelta | None:
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


def get_holiday_dates(
    dti: pd.DatetimeIndex, country: str, state: str | None, logger: logging.Logger
) -> list[pd.Timestamp]:
    years = range(min(dti).year, max(dti).year + 1)
    try:
        holiday_dates = sorted(getattr(holidays, country)(years=years, state=state))
    except NotImplementedError:  # not for all countries the states are available (e.g. France)
        holiday_dates = sorted(getattr(holidays, country)(years=years))
        logger.warning(
            f"Holidays for state {state} not available. Country-wide holidays for {country} are used instead."
        )
    except AttributeError:  # not all countries worldwide are available
        holiday_dates = []
        logger.warning(
            f"Holidays for country {country} not available. No public holidays are considered in this scenario."
        )
    return holiday_dates


def add_index_level(index: pd.Index, level_name: str, level_value) -> pd.MultiIndex:
    """
    Add a level to a pandas index. The new level will have the same value for all entries in the index.

    :param index: The index to which the level should be added.
    :param level_name: The name of the new level.
    :param level_value: The value of the new level for all entries in the index.

    :return: A new MultiIndex with the added level.
    """

    return pd.MultiIndex.from_tuples(
        tuples=[(level_value, *(i if isinstance(i, tuple) else (i,))) for i in index],
        names=[level_name] + list(index.names),
    )

    @typing_extensions.override
    def __str__(self) -> str:
        return f"{self.duration:.2f}s"

    def __enter__(self) -> typing_extensions.Self:
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.stop()


@dataclass
class Location:
    latitude: float
    longitude: float
    timezone: pytz.BaseTzInfo = field(default_factory=lambda: pytz.timezone("Europe/Berlin"))
    country: str = "DE"
    state: str = "BY"

    @classmethod
    def create_from_lat_lon(
        cls, latitude: float, longitude: float, logger: logging.Logger, geocode: bool = True
    ) -> Self:
        tzfinder = timezonefinder.TimezoneFinder()
        timezone_raw = tzfinder.certain_timezone_at(lat=latitude, lng=longitude)
        if timezone_raw is None:
            raise ValueError(f"Failed to determine timezone at {latitude}/{longitude}")

        timezone = pytz.timezone(timezone_raw)

        if geocode:
            location = cls._reverse_geocode_location(latitude, longitude)
        else:
            location = None

        if location is None:
            location = cls(
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
            )
            if geocode:
                # Warning is only necessary if geocoding was requested.
                logger.warning(
                    f"Connection to Geocoder failed. "
                    f"Using default country ({location.country}) and state ({location.state})."
                )

            return location

        address = location.raw.get("address", {})

        if "ISO3166-2-lvl4" in address:
            country, state = address["ISO3166-2-lvl4"].split("-")
        elif "ISO3166-2-lvl3" in address:
            country, state = address["ISO3166-2-lvl3"].split("-")
        else:
            # fallback: try country_code + state name
            country = address.get("country_code", "").upper()
            state = address.get("state", "")

        return cls(latitude=latitude, longitude=longitude, timezone=timezone, country=country, state=state)

    @staticmethod
    def _reverse_geocode_location(latitude: float, longitude: float) -> None | geopy.Location:
        geolocator = geopy.geocoders.Nominatim(user_agent="location_finder")
        try:
            return geolocator.reverse(query=(latitude, longitude), language="en", exactly_one=True)
        except geopy.exc.GeocoderUnavailable:
            return None


@dataclass(frozen=True)
class Timestep:
    hours: float
    td: pd.Timedelta

    @classmethod
    def from_str(cls, timestep_str: str) -> Self:
        td = pd.Timedelta(timestep_str)

        return cls(td=td, hours=td.total_seconds() / 3600)


@dataclass(frozen=True)
class TimeSettings:
    start: pd.Timestamp
    end: pd.Timestamp
    duration: pd.Timedelta

    timestep: Timestep

    @classmethod
    def create_from_start_timestamp(
        cls,
        start: pd.Timestamp,
        timestep: Timestep,
        end: pd.Timestamp | None = None,
        duration: pd.Timedelta | None = None,
    ) -> Self:
        if (end is None and duration is None) or (end is not None and duration is not None):
            raise ValueError('Exactly one of the parameters "end" or "duration" must be provided.')
        elif duration is None:
            duration = (end - start).floor(timestep.td)
        elif end is None:
            duration = duration.floor(timestep.td)
        # always recalculate end to ensure consistency
        end = start + duration

        return cls(start=start, end=end, duration=duration, timestep=timestep)

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self.timestep.td, inclusive="left")

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self.timestep.td, inclusive="both")

    def __len__(self) -> int:
        return len(self.dti)

    def cut(self, start_idx: int, length: int) -> "TimeSettings":
        start = self.dti_extd[start_idx]
        end = self.dti_extd[start_idx + length]

        return TimeSettings.create_from_start_timestamp(start=start, timestep=self.timestep, end=end)


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
    path_input_file: str | Path,
    timezone: zoneinfo.ZoneInfo,
    multiheader: bool = False,
    resampling_dti: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Properly read in timezone-aware example timeseries csv files and form correct datetimeindex

    :param path_input_file: Path to the CSV file containing the timeseries data.
    :param timezone: Timezone to which the timeseries data should be aligned to.
    :param multiheader: Whether the timeseries data is stored in CSV file with multiple headers.
    :param resampling_dti: If given, the timeseries data is resampled to the given datetimeindex.

    :raises IndexError: If timeseries data does not cover `resampling_dti` timeframe.
    """
    if multiheader:
        df = pd.read_csv(path_input_file, header=[0, 1])
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1)
        df.sort_index(
            axis=1,
            level=0,
            key=lambda x: x.map(
                lambda s: (
                    int(m.group(1))
                    # get the last continuous sequence of digits if possible
                    if (m := re.search(r"(\d+)(?!.*\d)", s))
                    else s
                )
            ),
            sort_remaining=True,
            inplace=True,
        )
    else:
        df = pd.read_csv(path_input_file)
        df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).drop(df.columns[0], axis=1)

    # parser in to_csv does not create datetimeindex
    df = df.tz_convert(timezone)
    if resampling_dti is None:
        return df

    timestep = time.Timestep.from_dti(resampling_dti)

    df_extd = df.reindex(extend_dti(dti=df.index)).ffill()

    def resample_column(column):
        if df_extd[column].dtype == bool:
            return df_extd[column].resample(timestep.td).ffill().bfill()
        else:
            return df_extd[column].resample(timestep.td).mean().ffill().bfill()

    df = pd.DataFrame({col: resample_column(col) for col in df_extd.columns})[:-1]

    if not (resampling_dti.isin(df.index).all()):
        raise IndexError(f"Input timeseries data in {path_input_file} does not cover resampling timeframe")
    return df.loc[resampling_dti]


def set_extension(filename: Path | str, default_extension: str = ".csv") -> Path:
    """
    Add a default extension to a filename if none is given. If the filename already has an extension, it is kept.
    """
    if filename is None:
        return None
    else:
        return path.with_suffix(default_extension) if not (path := Path(filename)).suffix else path


UNKNOWN_VERSION = "unknown"


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


class RevoletionError(Exception): ...
