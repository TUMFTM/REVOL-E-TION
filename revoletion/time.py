from __future__ import annotations

import re
import time
import zoneinfo
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import pandas as pd
import typing_extensions


def extend_dti(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Extend a datetime index by one timestep to include the last timestep of the simulation timeframe.
    """
    # append() outperforms union() in terms of speed by approx. a factor of 10x
    return dti.append(pd.DatetimeIndex([dti[-1] + (dti[-1] - dti[-2])]))


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


def ensure_timezone(ts: pd.Timestamp, timezone: zoneinfo.ZoneInfo | str) -> pd.Timestamp:
    if ts.tz is None:
        return ts.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return ts.tz_convert(timezone)


def convert_to_timestamp(
    time_in: str | pd.Timestamp | None,
    timestep: Timestep,
    timezone: zoneinfo.ZoneInfo | str,
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
    _td: pd.Timedelta

    @cached_property
    def td(self) -> pd.Timedelta:
        return self._td

    @cached_property
    def hours(self) -> float:
        return self.td.total_seconds() / 3600

    @cached_property
    def freqstr(self) -> str:
        return timedelta_to_freqstr(self.td)

    @classmethod
    def from_dti(cls, dti: pd.DatetimeIndex) -> Self:
        return cls(_td=pd.Timedelta(pd.infer_freq(dti)))

    @classmethod
    def from_td(cls, td: pd.Timedelta) -> Self:
        return cls(_td=td)

    @classmethod
    def from_str(cls, timestep_str: str) -> Self:
        # Ensure that timestep_str starts with a digit
        # This may not be the case if it originates from DatetimeIndex.inferred_freq (e.g. "h")
        timestep_str = re.sub(r"^(?!\d)", "1", timestep_str)
        return cls(_td=pd.Timedelta(timestep_str))


@dataclass(frozen=True)
class TimeFrame:
    start: pd.Timestamp
    end: pd.Timestamp
    duration: pd.Timedelta
    timestep: Timestep
    timezone: zoneinfo.ZoneInfo

    @classmethod
    def create_from_start_timestamp(
        cls,
        start: pd.Timestamp,
        timestep: Timestep,
        timezone: zoneinfo.ZoneInfo,
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

        start = ensure_timezone(ts=start, timezone=timezone)
        end = ensure_timezone(ts=end, timezone=timezone)

        return cls(start=start, end=end, duration=duration, timestep=timestep, timezone=timezone)

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self.timestep.td, inclusive="left", name="time")

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self.timestep.td, inclusive="both", name="time")

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
        timezone: zoneinfo.ZoneInfo | str,
        starttime: str | pd.Timestamp,
        sim_endtime: str | pd.Timestamp | None,
        sim_duration: float | int | None,
        prj_duration: int,
    ) -> Self:
        # ensure timezone is a ZoneInfo object
        timezone = zoneinfo.ZoneInfo(timezone) if not isinstance(timezone, zoneinfo.ZoneInfo) else timezone

        starttime_ts = convert_to_timestamp(starttime, timestep, timezone)
        if starttime_ts is None:
            raise ValueError(f"Failed to convert starttime ({starttime}) to pd.Timestamp")

        sim_endtime_ts = convert_to_timestamp(sim_endtime, timestep, timezone)
        sim_duration_td = pd.Timedelta(sim_duration, unit="day") if sim_duration is not None else None

        simulation = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep,
            timezone=timezone,
            end=sim_endtime_ts,
            duration=sim_duration_td,
        )
        evaluation = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep,
            timezone=timezone,
            end=sim_endtime_ts,
            duration=sim_duration_td,
        )
        project = TimeFrame.create_from_start_timestamp(
            start=starttime_ts,
            timestep=timestep,
            timezone=timezone,
            end=starttime_ts + pd.DateOffset(years=prj_duration),
        )

        return cls(sim=simulation, eval=evaluation, prj=project)
