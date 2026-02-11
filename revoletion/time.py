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


@dataclass(slots=True, frozen=True)
class Timestep:
    td: pd.Timedelta
    _hours: float

    def __init__(self, td: pd.Timedelta):
        # use custom init to enforce consistency between td and hours
        # cached_property collides with slots=True -> store hours as a regular attribute
        object.__setattr__(self, "td", td)
        object.__setattr__(self, "_hours", td.total_seconds() / 3600)

    @property
    def hours(self) -> float:
        return self._hours

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
        timestep: pd.Timedelta | str,
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

        return cls(start=start, end=end, duration=duration, _timestep=pd.Timedelta(timestep))

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep, inclusive="left", name="time")

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep, inclusive="both", name="time")

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
        timestep: str | pd.Timedelta | Timestep,
        timezone: pytz.BaseTzInfo | str,
        starttime: str | pd.Timestamp,
        sim_endtime: str | pd.Timestamp | None,
        sim_duration: float | int | None,
        prj_duration: int,
    ) -> Self:
        timestep_td = pd.Timedelta(timestep) if not isinstance(timestep, Timestep) else timestep.td
        if timestep_td is None:
            raise ValueError(f"Failed to convert timestep '{timestep}' to pd.Timedelta")

        starttime_ts = cls._convert_time_str(starttime, timestep_td, timezone)
        if starttime_ts is None:
            raise ValueError(f"Failed to convert starttime ({starttime}) to pd.Timestamp")

        sim_endtime_ts = cls._convert_time_str(sim_endtime, timestep, timezone) if sim_endtime is not None else None
        sim_duration_td = pd.Timedelta(sim_duration, unit="day") if sim_duration is not None else None

        simulation = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep_td,
            end=sim_endtime_ts,
            duration=sim_duration_td,
        )
        evaluation = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep_td,
            end=sim_endtime_ts,
            duration=sim_duration_td,
        )
        project = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep_td,
            end=starttime_ts + pd.DateOffset(years=prj_duration),
        )

        return cls(sim=simulation, eval=evaluation, prj=project)

    @staticmethod
    def _convert_time_str(
        time_in: str | pd.Timestamp | None, timestep: pd.Timedelta | str, timezone: pytz.BaseTzInfo | str
    ) -> pd.Timestamp | None:
        if time_in is None:
            return None

        if isinstance(time_in, pd.Timestamp):
            if time_in.tz is None:
                return time_in.tz_localize(timezone)
            else:
                return time_in.tz_convert(timezone)

        # ToDo: reformat time
        time_in = time_in if len(time_in) > 10 else time_in + " 00:00"
        value = pd.to_datetime(time_in, format="%d.%m.%Y %H:%M").floor(timestep).tz_localize(timezone)
        return value
