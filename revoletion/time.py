from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import pandas as pd
import pytz
import typing_extensions


def timedelta_to_freqstr(td: pd.Timedelta) -> str:
    if td == pd.Timedelta(0):
        return "0ns"

    sign = "-" if td < pd.Timedelta(0) else ""
    td = abs(td)
    c = td.components

    parts = [
        (c.days, "D"),
        (c.hours, "h"),
        (c.minutes, "min"),
        (c.seconds, "s"),
        (c.milliseconds, "ms"),
        (c.microseconds, "us"),
        (c.nanoseconds, "ns"),
    ]

    freq = "".join(f"{value}{unit}" for value, unit in parts if value)
    return sign + freq


def parse_datetime_str(time_str: str) -> pd.Timestamp:
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return pd.to_datetime(time_str, format=fmt, errors="raise")
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {time_str!r}")


def ensure_timezone(ts: pd.Timestamp, timezone: pytz.BaseTzInfo | str) -> pd.Timestamp:
    if ts.tz is None:
        return ts.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return ts.tz_convert(timezone)


def convert_to_timestamp(
    time_in: str | pd.Timestamp | None,
    timestep: Timestep,
    timezone: pytz.BaseTzInfo | str,
) -> pd.Timestamp | None:
    if time_in is None:
        return None

    ts = time_in if isinstance(time_in, pd.Timestamp) else parse_datetime_str(time_in)

    ts = ensure_timezone(ts, timezone)

    return ts.floor(timestep.freqstr)


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


@dataclass(frozen=True)
class Timestep:
    td: pd.Timedelta

    @cached_property
    def hours(self) -> float:
        return self.td.total_seconds() / 3600

    @cached_property
    def freqstr(self) -> str:
        return timedelta_to_freqstr(self.td)

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

    _timestep_td: pd.Timedelta

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
            duration = (end - start).floor(timestep.freqstr)
        elif end is None:
            duration = duration.floor(timestep.freqstr)
        # always recalculate end to ensure consistency
        end = start + duration

        return cls(start=start, end=end, duration=duration, _timestep_td=timestep.td)

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep_td, inclusive="left", name="time")

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep_td, inclusive="both", name="time")

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
        timestep: Timestep,
        timezone: pytz.BaseTzInfo | str,
        starttime: str | pd.Timestamp,
        sim_endtime: str | pd.Timestamp | None,
        sim_duration: float | int | None,
        prj_duration: int,
    ) -> Self:
        starttime_ts = convert_to_timestamp(starttime, timestep, timezone)
        if starttime_ts is None:
            raise ValueError(f"Failed to convert starttime ({starttime}) to pd.Timestamp")

        sim_endtime_ts = convert_to_timestamp(sim_endtime, timestep, timezone) if sim_endtime is not None else None
        sim_duration_td = pd.Timedelta(sim_duration, unit="day") if sim_duration is not None else None

        simulation = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep,
            end=sim_endtime_ts,
            duration=sim_duration_td,
        )
        evaluation = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep,
            end=sim_endtime_ts,
            duration=sim_duration_td,
        )
        project = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep,
            end=starttime_ts + pd.DateOffset(years=prj_duration),
        )

        return cls(sim=simulation, eval=evaluation, prj=project)

    @staticmethod
    def _convert_time_str(
        time_in: str | pd.Timestamp | None, timestep: Timestep, timezone: pytz.BaseTzInfo | str
    ) -> pd.Timestamp | None:
        if time_in is None:
            return None

        if isinstance(time_in, pd.Timestamp):
            if time_in.tz is None:
                return time_in.tz_localize(timezone)
            else:
                return time_in.tz_convert(timezone)

        def parse_input_datetime(time_str: str) -> pd.Timestamp:
            for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y"):
                try:
                    return pd.to_datetime(time_str, format=fmt, errors="raise")
                except ValueError:
                    continue
            raise ValueError(f"Invalid date format: {time_str!r}")

        time_ts = parse_input_datetime(time_in)
        return time_ts.floor(timestep.freqstr).tz_localize(timezone)
