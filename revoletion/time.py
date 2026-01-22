import time
from dataclasses import dataclass
from functools import cached_property

import pandas as pd
import pytz
import typing_extensions
from typing_extensions import Self


class RunTimer:
    """
    Helper to measure the runtime of scenarios or runs.

    Usage as context manager:

         with RunTimer() as run_time:
             do_stuff()
         print(run_time)


     Plain usage:

         run_time = RunTimer()
         do_stuff()
         run_time.stop()
         print(run_time)
    """

    start: float = float("nan")
    end: float = float("nan")
    duration: float = float("nan")

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def stop(self) -> None:
        self.end = time.perf_counter()
        self.duration = self.end - self.start

    @property
    def result_summary(self) -> pd.Series:
        # only export runtime duration -> start and end are not interpretable
        return pd.Series({"runtime_duration_s": round(self.duration, 2)})

    @typing_extensions.override
    def __str__(self) -> str:
        return f"{self.duration:.2f}s"

    def __enter__(self) -> typing_extensions.Self:
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.stop()


@dataclass
class Timestep:
    td: pd.Timedelta

    @property
    def hours(self) -> float:
        return self.td.total_seconds() / 3600

    @classmethod
    def from_dti(cls, dti: pd.DatetimeIndex) -> Self:
        return cls(td=pd.Timedelta(pd.infer_freq(dti)))

    @classmethod
    def from_str(cls, timestep_str: str) -> Self:
        return cls(td=pd.Timedelta(timestep_str))


@dataclass(frozen=True)
class TimeFrame:
    start: pd.Timestamp
    end: pd.Timestamp
    duration: pd.Timedelta

    _timestep: pd.Timedelta

    @classmethod
    def create_from_start_timestamp(
        cls,
        start: pd.Timestamp,
        timestep: pd.Timedelta,
        end: pd.Timestamp | None = None,
        duration: pd.Timedelta | None = None,
    ) -> Self:
        if (end is None and duration is None) or (end is not None and duration is not None):
            raise ValueError('Exactly one of the parameters "end" or "duration" must be provided.')
        elif duration is None:
            duration = (end - start).floor(timestep)
        elif end is None:
            duration = duration.floor(timestep)
        # always recalculate end to ensure consistency
        end = start + duration

        return cls(start=start, end=end, duration=duration, _timestep=timestep)

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep, inclusive="left")

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep, inclusive="both")

    @cached_property
    def end_extd(self) -> pd.Timestamp:
        return max(self.dti_extd)


@dataclass
class SimulationTimes:
    sim: TimeFrame
    eval: TimeFrame
    prj: TimeFrame

    @classmethod
    def create_from_plain(
        cls,
        timestep: str,
        timezone: pytz.BaseTzInfo,
        starttime: str,
        sim_endtime: str,
        sim_duration: str,
        prj_duration: str,
    ) -> Self:
        starttime_timestamp = cls._convert_time_str(starttime, timestep, timezone)
        if starttime_timestamp is None:
            raise ValueError(f"Failed to convert starttime ({starttime}) to pd.Timestamp")

        sim_endtime_timestamp = cls._convert_time_str(sim_endtime, timestep, timezone)

        timestep_timedelta = pd.Timedelta(timestep)
        if timestep_timedelta is None:
            raise ValueError(f"Failed to convert timestep '{timestep}' to pd.Timedelta")

        sim_duration_timedelta = pd.Timedelta(sim_duration, unit="day") if sim_duration is not None else None

        sim = TimeFrame.create_from_start_timestamp(
            start=starttime_timestamp,
            timestep=timestep_timedelta,
            end=sim_endtime_timestamp,
            duration=sim_duration_timedelta,
        )
        eval = TimeFrame.create_from_start_timestamp(
            start=starttime_timestamp,
            timestep=timestep_timedelta,
            end=sim_endtime_timestamp,
            duration=sim_duration_timedelta,
        )
        prj = TimeFrame.create_from_start_timestamp(
            start=starttime_timestamp,
            timestep=timestep_timedelta,
            end=starttime_timestamp + pd.DateOffset(years=prj_duration),
        )

        return cls(sim=sim, eval=eval, prj=prj)

    @staticmethod
    def _convert_time_str(time_str: str | None, timestep: str, timezone: pytz.BaseTzInfo) -> pd.Timestamp | None:
        if time_str is None:
            return None

        # ToDo: reformat time
        time_str = time_str if len(time_str) > 10 else time_str + " 00:00"
        value = pd.to_datetime(time_str, format="%d.%m.%Y %H:%M").floor(timestep).tz_localize(timezone)
        return value
