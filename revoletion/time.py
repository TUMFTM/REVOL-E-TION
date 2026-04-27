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
    Extend a DatetimeIndex by one additional timestep.

    The function infers the timestep from the difference between the last
    two entries of the index and appends a new timestamp accordingly.
    This is useful for ensuring that the final timestep of a simulation
    timeframe is explicitly included.

    Parameters
    ----------
    dti : pd.DatetimeIndex
        A datetime index with at least two entries and a consistent frequency.

    Returns
    -------
    pd.DatetimeIndex
        A new DatetimeIndex with one additional timestamp appended at the end.

    Notes
    -----
    This implementation uses `append()` instead of `union()` for performance
    reasons, as it is significantly faster for this use case.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2023-01-01", periods=3, freq="D")
    >>> extend_dti(idx)
    DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'], dtype='datetime64[us]', freq=None)
    """
    return dti.append(pd.DatetimeIndex([dti[-1] + (dti[-1] - dti[-2])]))


def convert_freqstr(freq: str) -> str:
    """
    Normalize a pandas frequency string so it is compatible with pd.Timedelta().

    If the input frequency string does not start with a numeric value,
    a leading "1" is inserted. This ensures that strings like "H", "D",
    or "min" are converted to "1H", "1D", and "1min", which are valid
    inputs for pd.Timedelta().

    Parameters
    ----------
    freq : str
        A frequency string, typically inferred from a pandas object
        (e.g., via `pd.DatetimeIndex.inferred_freq`).

    Returns
    -------
    str
        A normalized frequency string that starts with a numeric value
        and can be safely passed to pd.Timedelta().

    Examples
    --------
    >>> convert_freqstr("h")
    '1h'
    >>> convert_freqstr("1h")
    '1h'
    >>> convert_freqstr("15min")
    '15min'
    """
    return re.sub(r"^(?!\d)", "1", freq)


def timedelta_to_freqstr(td: pd.Timedelta) -> str:
    """
    Convert a pandas Timedelta to a frequency string representation.

    The function decomposes a ``pandas.Timedelta`` into its constituent
    components (days, hours, minutes, seconds, milliseconds, microseconds,
    and nanoseconds) and formats them into a compact frequency string.
    Components with zero values are omitted. Negative timedeltas are
    prefixed with a minus sign.

    Parameters
    ----------
    td : pandas.Timedelta
        The timedelta object to convert.

    Returns
    -------
    str
        A string representation of the timedelta in frequency format.
        For example, "1D2h30min", "-5s", or "0ns" for a zero timedelta.

    Notes
    -----
    - The output omits any time units with a value of zero.
    - The smallest unit represented is nanoseconds.
    - A zero timedelta is explicitly represented as "0ns".

    Examples
    --------
    >>> import pandas as pd
    >>> timedelta_to_freqstr(pd.Timedelta(days=1, hours=2, minutes=30))
    '1D2h30min'

    >>> timedelta_to_freqstr(pd.Timedelta(seconds=-5))
    '-5s'

    >>> timedelta_to_freqstr(pd.Timedelta(0))
    '0ns'
    """
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
    """
    Parse a datetime string into a pandas Timestamp.

    The function attempts to parse the input string using a predefined
    set of date and datetime formats. It tries each format in order and
    returns the first successfully parsed result. If none of the formats
    match, a ValueError is raised.

    Parameters
    ----------
    time_str : str
        The input string representing a date or datetime.

    Returns
    -------
    pandas.Timestamp
        The parsed timestamp object.

    Raises
    ------
    ValueError
        If the input string does not match any of the supported formats.

    Notes
    -----
    Supported formats include:
    - "DD.MM.YYYY HH:MM"
    - "DD.MM.YYYY"
    - "YYYY-MM-DD HH:MM"
    - "YYYY-MM-DD"

    The parsing is strict and does not allow deviations from these formats.

    Examples
    --------
    >>> parse_datetime_str("27.04.2026 14:30")
    Timestamp('2026-04-27 14:30:00')

    >>> parse_datetime_str("2026-04-27")
    Timestamp('2026-04-27 00:00:00')

    >>> parse_datetime_str("invalid")
    Traceback (most recent call last):
        ...
    ValueError: Invalid date format: 'invalid'
    """
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return pd.to_datetime(time_str, format=fmt, errors="raise")
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {time_str!r}")


def ensure_timezone(ts: pd.Timestamp, timezone: zoneinfo.ZoneInfo | str) -> pd.Timestamp:
    """
    Ensure that a pandas Timestamp has the specified timezone.

    If the input timestamp is timezone-naive, it is localized to the given
    timezone. If it is already timezone-aware, it is converted to the target
    timezone.

    Parameters
    ----------
    ts : pandas.Timestamp
        The timestamp to localize or convert.
    timezone : zoneinfo.ZoneInfo or str
        The target timezone. Can be a ``zoneinfo.ZoneInfo`` object or a
        string recognized by pandas (e.g., "UTC", "Europe/Berlin").

    Returns
    -------
    pandas.Timestamp
        A timezone-aware timestamp in the specified timezone.

    Raises
    ------
    pytz.exceptions.AmbiguousTimeError
        If the timestamp is ambiguous during localization (e.g., due to
        daylight saving time transitions) and cannot be resolved.
    pytz.exceptions.NonExistentTimeError
        If the timestamp does not exist in the target timezone (e.g., skipped
        during a daylight saving time transition).

    Notes
    -----
    - Naive timestamps are localized using ``tz_localize`` with
      ``ambiguous="raise"`` and ``nonexistent="raise"`` to enforce strict
      handling of daylight saving time edge cases.
    - Timezone-aware timestamps are converted using ``tz_convert``.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneinfo import ZoneInfo
    >>> ts = pd.Timestamp("2026-04-27 12:00")
    >>> ensure_timezone(ts, "UTC")
    Timestamp('2026-04-27 12:00:00+0000', tz='UTC')


    >>> ts_aware = pd.Timestamp("2026-04-27 12:00", tz="UTC")
    >>> ensure_timezone(ts_aware, "Europe/Berlin")
    Timestamp('2026-04-27 14:00:00+0200', tz='Europe/Berlin')
    """
    if ts.tz is None:
        return ts.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return ts.tz_convert(timezone)


def convert_to_timestamp(
    time_in: str | pd.Timestamp | None,
    timestep: Timestep,
    timezone: zoneinfo.ZoneInfo | str,
) -> pd.Timestamp | None:
    """
    Convert an input value to a timezone-aware pandas Timestamp aligned
    to a given timestep.

    The function accepts a string, a pandas ``Timestamp``, or ``None``.
    Strings are parsed into timestamps using ``parse_datetime_str``.
    The resulting timestamp is then ensured to have the specified timezone
    using ``ensure_timezone`` and finally floored to the frequency defined
    by the provided timestep.

    Parameters
    ----------
    time_in : str or pandas.Timestamp or None
        The input time value. If a string, it must match one of the supported
        formats in ``parse_datetime_str``. If ``None``, the function returns
        ``None``.
    timestep : Timestep
        An object defining the desired temporal resolution. It must provide
        a ``freqstr`` attribute compatible with pandas frequency strings.
    timezone : zoneinfo.ZoneInfo or str
        The target timezone for the resulting timestamp.

    Returns
    -------
    pandas.Timestamp or None
        A timezone-aware timestamp floored to the timestep frequency,
        or ``None`` if the input was ``None``.

    Raises
    ------
    ValueError
        If ``time_in`` is a string that cannot be parsed into a valid datetime.
    pytz.exceptions.AmbiguousTimeError
        If timezone localization encounters an ambiguous time.
    pytz.exceptions.NonExistentTimeError
        If timezone localization encounters a non-existent time.

    Notes
    -----
    - String inputs are parsed using ``parse_datetime_str``.
    - Timezone handling is delegated to ``ensure_timezone``.
    - The final timestamp is aligned using ``Timestamp.floor`` with
      ``timestep.freqstr``.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneinfo import ZoneInfo
    >>> class Timestep:
    ...     freqstr = "1h"
    ...
    >>> convert_to_timestamp("2026-04-27 14:45", Timestep(), "UTC")
    Timestamp('2026-04-27 14:00:00+0000', tz='UTC')

    >>> convert_to_timestamp(None, Timestep(), "UTC") is None
    True
    """
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
        self.start = time.perf_counter()
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
        return cls(_td=pd.Timedelta(convert_freqstr(dti.inferred_freq)))

    @classmethod
    def from_td(cls, td: pd.Timedelta) -> Self:
        return cls(_td=td)

    @classmethod
    def from_str(cls, timestep_str: str) -> Self:
        return cls(_td=pd.Timedelta(convert_freqstr(timestep_str)))


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
        return self.dti_extd[-1]


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
