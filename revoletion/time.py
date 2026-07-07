from __future__ import annotations

import re
import time
import zoneinfo
from dataclasses import dataclass
from functools import cached_property
from typing import Self, override

import numpy as np
import pandas as pd


def get_indices(dti_full: pd.DatetimeIndex, dti_subset: pd.DatetimeIndex) -> tuple[slice, slice]:
    """
    Get slice indices for a subset DatetimeIndex within a full DatetimeIndex.

    The function determines the integer slice positions of ``dti_subset`` in ``dti_full``.
    The subset must be a contiguous segment of the full index.

    Parameters
    ----------
    dti_full : pandas.DatetimeIndex
        Full reference datetime index.
    dti_subset : pandas.DatetimeIndex
        Datetime index representing a contiguous subset of ``dti_full``.

    Returns
    -------
    tuple[slice, slice]
        Tuple containing:
        
        - Full index slice spanning the complete ``dti_full`` range.
        - Subset slice selecting the corresponding range within ``dti_full``.

    Raises
    ------
    ValueError
        If ``dti_subset`` is not a contiguous subset of ``dti_full``.

    Notes
    -----
    The returned slices can be used directly for NumPy array indexing, for
    example::

        arr_full[idx_full]
        arr_full[idx_subset]
    """
    # check if idx_subset is real subset of idx_full
    locs = dti_full.get_indexer(dti_subset)
    if (np.diff(locs) != 1).any():
        raise ValueError("subset is not a contiguous slice of full index")
    
    idx_full = slice(0, len(dti_full))
    idx_subset = slice(int(locs[0]), int(locs[-1] + 1))

    return idx_full, idx_subset


def duration2steps(duration: pd.Timedelta, timestep: pd.Timedelta) -> int:
    """
    Convert a time duration into the number of discrete simulation steps
    defined by a given timestep.

    The function assumes that `duration` is an exact multiple of `timestep`.
    If this is not the case, a ValueError is raised.

    Parameters
    ----------
    duration : pd.Timedelta
        Total time span to be converted into steps.
    timestep : pd.Timedelta
        Duration of a single time step.

    Returns
    -------
    int
        Number of full timesteps contained in `duration`.

    Raises
    ------
    ValueError
        If `duration` is not exactly divisible by `timestep`.
    """
    
    ratio = duration / timestep
    if not ratio.is_integer():
        raise ValueError(f"Duration {duration} is not divisible by timestep {timestep}")
    
    return int(ratio)


def extend_dti(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Extend a DatetimeIndex by one additional timestep.

    The function infers the timestep from the difference between the last two entries of the index and appends a new
    timestamp accordingly. This is useful for ensuring that the final timestep of a simulation timeframe is explicitly
    included.

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
    This implementation uses `append()` instead of `union()` for performance reasons, as it is significantly faster
    (approx. by a factor of 10) for this use case.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2023-01-01", periods=3, freq="D")
    >>> extend_dti(idx)
    DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'], dtype='datetime64[us]', freq=None)
    """
    return pd.DatetimeIndex(dti.append(pd.DatetimeIndex([dti[-1] + (dti[-1] - dti[-2])])))


def convert_freqstr(freq: str) -> str:
    """
    Normalize a pandas frequency string so it is compatible with pd.Timedelta().

    If the input frequency string does not start with a numeric value, a leading "1" is inserted. This ensures that
    strings like "H", "D", or "min" are converted to "1H", "1D", and "1min", which are valid inputs for pd.Timedelta().

    Parameters
    ----------
    freq : str
        A frequency string, typically inferred from a pandas object (e.g., via `pd.DatetimeIndex.inferred_freq`).

    Returns
    -------
    str
        A normalized frequency string that starts with a numeric value and can be safely passed to pd.Timedelta().

    Raises
    ------
    TypeError
        If ``freq`` is not a string.

    Examples
    --------
    >>> convert_freqstr("h")
    '1h'
    >>> convert_freqstr("1h")
    '1h'
    >>> convert_freqstr("15min")
    '15min'
    """
    if not isinstance(freq, str):
        raise TypeError(
            f"convert_freqstr() expects argument freq to be of type string, "
            f"got type {type(freq).__name__} ({freq}) instead."
        )
    return re.sub(r"^(?!\d)", "1", freq)


def timedelta_to_freqstr(td: pd.Timedelta) -> str:
    """
    Convert a pandas Timedelta to a frequency string representation.

    The function decomposes a ``pandas.Timedelta`` into its constituent components (days, hours, minutes, seconds,
    milliseconds, microseconds, and nanoseconds) and formats them into a compact frequency string.
    Components with zero values are omitted. Negative timedeltas are prefixed with a minus sign.

    Parameters
    ----------
    td : pandas.Timedelta
        A valid pandas Timedelta to convert. Must not be NaT.

    Returns
    -------
    str
        A string representation of the timedelta in frequency format.
        For example, "1D2h30min", "-5s", or "0ns" for a zero timedelta.

    Raises
    ------
    TypeError
        If ``td`` is not a valid pandas Timedelta or is NaT.

    Notes
    -----
    - The output omits any time units with a value of zero.
    - The smallest unit represented is nanoseconds.
    - A zero timedelta is explicitly represented as "0ns".
    - NaT values are not accepted and will raise a TypeError.

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
    if not isinstance(td, pd.Timedelta) or pd.isna(td):
        raise TypeError("Expected a valid pandas Timedelta (not NaT)")

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

    The function attempts to parse the input string using a predefined set of date and datetime formats.
    It tries each format in order and returns the first successfully parsed result.
    If none of the formats match, a ValueError is raised.

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
    Supported formats are:
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


def convert_to_zoneinfo(timezone: zoneinfo.ZoneInfo | str) -> zoneinfo.ZoneInfo:
    """
    Convert a timezone representation to a ``zoneinfo.ZoneInfo`` object.

    Parameters
    ----------
    timezone : zoneinfo.ZoneInfo or str
        The timezone to convert. If a string is provided, it must be a valid IANA timezone name (e.g., "Europe/Berlin").
        If a ``ZoneInfo`` instance is provided, it is returned unchanged.

    Returns
    -------
    zoneinfo.ZoneInfo
        A ``ZoneInfo`` object corresponding to the given timezone.

    Raises
    ------
    TypeError
        If ``timezone`` is not a ``str`` or ``zoneinfo.ZoneInfo`` instance.
    zoneinfo.ZoneInfoNotFoundError
        If ``timezone`` is a string but not a valid IANA timezone name.

    Examples
    --------
    >>> convert_to_zoneinfo("Europe/Berlin")
    ZoneInfo(key='Europe/Berlin')

    >>> tz = zoneinfo.ZoneInfo("UTC")
    >>> convert_to_zoneinfo(tz) is tz
    True
    """
    if isinstance(timezone, zoneinfo.ZoneInfo):
        return timezone
    elif isinstance(timezone, str):
        return zoneinfo.ZoneInfo(timezone)
    else:
        raise TypeError(
            f"convert_to_zoneinfo(): 'timezone' must be a str or ZoneInfo instance, "
            f"got {type(timezone).__name__}: {timezone!r}"
        )


def ensure_timezone(ts: pd.Timestamp, timezone: zoneinfo.ZoneInfo | str) -> pd.Timestamp:
    """
    Ensure that a pandas Timestamp has the specified timezone.

    If the input timestamp is timezone-naive, it is localized to the given timezone.
    If it is already timezone-aware, it is converted to the target timezone.

    Parameters
    ----------
    ts : pandas.Timestamp
        The timestamp to localize or convert.
    timezone : zoneinfo.ZoneInfo or str
        The target timezone. Can be a ``zoneinfo.ZoneInfo`` object or a string recognized by pandas
        (e.g., "UTC", "Europe/Berlin").

    Returns
    -------
    pandas.Timestamp
        A timezone-aware timestamp in the specified timezone.

    Raises
    ------
    ValueError
        If the timestamp is ambiguous during localization or does not exist in the target timezone (e.g., due to DST
        transitions) and therefore cannot be resolved.

    Notes
    -----
    - Naive timestamps are localized using ``tz_localize`` with ``ambiguous="raise"`` and ``nonexistent="raise"``
      to enforce strict handling of daylight saving time edge cases.
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

    tz = convert_to_zoneinfo(timezone)

    if ts.tz is None:
        return ts.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    return ts.tz_convert(tz)


def convert_to_timestamp(
    time_in: str | pd.Timestamp | None,
    timestep: Timestep,
    timezone: zoneinfo.ZoneInfo | str,
) -> pd.Timestamp | None:
    """
    Convert an input value to a timezone-aware pandas Timestamp aligned to a given timestep.

    The function accepts a string, a pandas ``Timestamp``, or ``None``. Strings are parsed into timestamps using
    ``parse_datetime_str``. The resulting timestamp is then ensured to have the specified timezone using
    ``ensure_timezone`` and finally floored to the frequency defined by the provided timestep.

    Parameters
    ----------
    time_in : str or pandas.Timestamp or None
        The input time value. If a string, it must match one of the supported formats in ``parse_datetime_str``.
        If ``None``, the function returns ``None``.
    timestep : Timestep
        The target Timestep, defining the desired temporal resolution.
    timezone : zoneinfo.ZoneInfo or str
        The target timezone for the resulting timestamp.

    Returns
    -------
    pandas.Timestamp or None
        A timezone-aware timestamp floored to the timestep frequency, or ``None`` if the input was ``None``.

    Raises
    ------
    ValueError
        If ``time_in`` is a string that cannot be parsed into a valid datetime or if timezone localization encounters
        an ambiguous or non-existent time.

    Notes
    -----
    - String inputs are parsed using ``parse_datetime_str``.
    - Timezone handling is delegated to ``ensure_timezone``.
    - The final timestamp is aligned using ``Timestamp.floor`` with ``timestep.freqstr``.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneinfo import ZoneInfo
    >>> step = Timestep.from_str("1h")
    >>> convert_to_timestamp("2026-04-27 14:45", step, "UTC")
    Timestamp('2026-04-27 14:00:00+0000', tz='UTC')

    >>> convert_to_timestamp(None, step, "UTC") is None
    True
    """
    if time_in is None:
        return None

    ts = parse_datetime_str(time_in) if not isinstance(time_in, pd.Timestamp) else time_in

    ts = ensure_timezone(ts, timezone)

    return ts.floor(timestep.freqstr)


class RunTimer:
    """
    Simple utility for measuring execution time.

    This class provides a lightweight timer based on ``time.perf_counter``.
    It records the start time upon initialization and computes the elapsed duration when ``stop`` is called.
    It also supports usage as a context manager.

    Attributes
    ----------
    start : float
        The start time in seconds (from ``time.perf_counter``).
    end : float
        The end time in seconds.
    duration : float
        The elapsed time in seconds between ``start`` and ``end``.

    Notes
    -----
    - The timer starts automatically upon instantiation.
    - The ``start`` and ``end`` values are not absolute timestamps and should only be used to compute durations.
    - Duration is expressed in seconds with sub-second precision.

    Examples
    --------
    >>> import time
    >>> timer = RunTimer()
    >>> time.sleep(0.1)
    >>> timer.stop()
    >>> print(timer.duration)
    0.1

    Using as a context manager:

    >>> with RunTimer() as timer:
    ...     time.sleep(0.1)
    >>> print(timer)
    0.10s
    """

    start: float = float("nan")
    end: float = float("nan")
    duration: float = float("nan")

    def __init__(self) -> None:
        """
        Start the timer.
        """
        self.start = time.perf_counter()

    def stop(self) -> None:
        """
        Stop the timer.
        """
        self.end = time.perf_counter()
        self.duration = self.end - self.start

    @property
    def result_summary(self) -> pd.Series:
        """
        Return a summary of the timing result.

        Returns
        -------
        pandas.Series
            A series containing the runtime duration in seconds, rounded to two decimal places.
        """
        # only export runtime duration -> start and end are not interpretable
        return pd.Series({"runtime_duration_s": round(self.duration, 2)})

    @override
    def __str__(self) -> str:
        """
        Return a string representation of the duration.

        Returns
        -------
        str
            The elapsed time formatted in seconds (e.g., "0.10s").
        """
        return f"{self.duration:.2f}s"

    def __enter__(self) -> Self:
        """
        Enter the runtime context and restart the timer.

        Returns
        -------
        RunTimer
            The timer instance.
        """
        self.start = time.perf_counter()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        """
        Exit the runtime context and stop the timer.
        """
        self.stop()


@dataclass(frozen=True)
class Timestep:
    """
    Immutable representation of a time resolution based on ``pandas.Timedelta``.

    This class provides a convenient abstraction for working with time steps, exposing derived properties such as
    the duration in hours and a pandas-compatible frequency string.
    Instances are immutable and cache computed properties for efficiency.

    Parameters
    ----------
    _td : pandas.Timedelta
        The underlying timedelta defining the timestep.

    Attributes
    ----------
    td : pandas.Timedelta
        The underlying timedelta.
    hours : float
        The timestep expressed in hours.
    freqstr : str
        A frequency string representation of the timestep, compatible with pandas (e.g., "1h", "30min").

    Notes
    -----
    - The class is frozen (immutable), meaning its state cannot be modified after creation.
    - Properties are cached using ``functools.cached_property`` to avoid recomputation.
    - Frequency strings are generated via ``timedelta_to_freqstr``.

    Methods
    -------
    from_dti(dti)
        Create a ``Timestep`` from a ``pandas.DatetimeIndex`` by inferring its frequency.
    from_td(td)
        Create a ``Timestep`` directly from a ``pandas.Timedelta``.
    from_str(timestep_str)
        Create a ``Timestep`` from a frequency string.

    Examples
    --------
    >>> import pandas as pd
    >>> ts = Timestep.from_td(pd.Timedelta(hours=1))
    >>> ts.hours
    1.0
    >>> ts.freqstr
    '1h'

    >>> dti = pd.date_range("2026-01-01", periods=3, freq="30min")
    >>> Timestep.from_dti(dti).freqstr
    '30min'

    >>> Timestep.from_str("2h").td
    Timedelta('0 days 02:00:00')
    """

    _td: pd.Timedelta

    @cached_property
    def td(self) -> pd.Timedelta:
        """
        Return the underlying timedelta.

        Returns
        -------
        pandas.Timedelta
            The stored timedelta value.
        """
        return self._td

    @cached_property
    def hours(self) -> float:
        """
        Return the timestep duration in hours.

        Returns
        -------
        float
            The total duration of the timestep expressed in hours.
        """
        return self.td.total_seconds() / 3600

    @cached_property
    def freqstr(self) -> str:
        """
        Return a pandas-compatible frequency string.

        Returns
        -------
        str
            The frequency string representation of the timestep.
        """
        return timedelta_to_freqstr(self.td)

    @classmethod
    def from_dti(cls, dti: pd.DatetimeIndex) -> Self:
        """
        Create a ``Timestep`` from a pandas DatetimeIndex.

        The frequency is inferred from the index and converted into a ``pandas.Timedelta``.

        Parameters
        ----------
        dti : pandas.DatetimeIndex
            The datetime index from which to infer the timestep.

        Returns
        -------
        Timestep
            A new instance representing the inferred timestep.

        Raises
        ------
        ValueError
            If the frequency of the index cannot be inferred.

        Examples
        --------
        >>> import pandas as pd
        >>> dti = pd.date_range("2026-01-01", periods=3, freq="1h")
        >>> Timestep.from_dti(dti).freqstr
        '1h'
        """
        freq = dti.inferred_freq

        if freq is None:
            raise ValueError("Frequency cannot be inferred from provided datetime index.")

        return cls(_td=pd.Timedelta(convert_freqstr(freq)))

    @classmethod
    def from_td(cls, td: pd.Timedelta) -> Self:
        """
        Create a ``Timestep`` from a pandas Timedelta.

        Parameters
        ----------
        td : pandas.Timedelta
            The timedelta defining the timestep.

        Returns
        -------
        Timestep
            A new instance with the given timedelta.

        Examples
        --------
        >>> import pandas as pd
        >>> Timestep.from_td(pd.Timedelta(minutes=15)).freqstr
        '15min'
        """
        return cls(_td=td)

    @classmethod
    def from_str(cls, timestep_str: str) -> Self:
        """
        Create a ``Timestep`` from a frequency string.

        The string is first normalized using ``convert_freqstr`` and then converted into a ``pandas.Timedelta``.

        Parameters
        ----------
        timestep_str : str
            A frequency string (e.g., "1h", "30min").

        Returns
        -------
        Timestep
            A new instance representing the given timestep.

        Raises
        ------
        ValueError
            If the string cannot be converted into a valid frequency.

        Examples
        --------
        >>> Timestep.from_str("2h").td
        Timedelta('0 days 02:00:00')
        """
        return cls(_td=pd.Timedelta(convert_freqstr(timestep_str)))
    

@dataclass(frozen=True)
class TimeSlice:
    """
    Integer slice representation of a time frame.

    The class stores the start and end indices of a time interval on a discrete time grid.
    The end index is exclusive, following standard Python slicing semantics.

    Parameters
    ----------
    start : int
        Inclusive start index.
    end : int
        Exclusive end index.

    Parameters
    ----------
    end_extd : int
        End index incremented by one.
    dti : slice
        Slice equivalent to ``slice(start, end)``.
    dti_extd : slice
        Slice equivalent to ``slice(start, end + 1)``.
    """
    start: int
    end: int

    @classmethod
    def from_timeframe(cls, tf: TimeFrame, ref: bool = True):
        """
        Create a TimeSlice from a TimeFrame.

        Parameters
        ----------
        tf : TimeFrame
            Time frame to convert.
        ref : bool, default=True
            If True, compute the start index relative to ``tf.start_ref``.
            Otherwise, the returned slice starts at index 0.

        Returns
        -------
        TimeSlice
            Integer slice corresponding to the given time frame.

        Raises
        ------
        ValueError
            If the duration between timestamps is not an integer multiple of the time step.
        """
        timestep=tf.timestep.td
        start = duration2steps(
            duration=tf.start - tf.start_ref,
            timestep=timestep,
        ) if ref else 0
        end = start + duration2steps(
            duration=tf.end - tf.start,
            timestep=timestep,
        )
        return cls(start=start, end=end)

    @cached_property
    def end_extd(self) -> int:
        """
        Extended end index. Corresponds to ``TimeFrame.end_extd``

        Returns
        -------
        int
            End index incremented by one.
        """
        return self.end + 1

    @cached_property
    def dti(self) -> slice:
        """
        Slice corresponding to ``TimeFrame.dti``.

        Returns
        -------
        slice
            Slice equivalent to ``slice(start, end)``.
        """
        return slice(self.start, self.end)
    
    @cached_property
    def dti_extd(self) -> slice:
        """
        Slice corresponding to ``TimeFrame.dti_extd``.

        Returns
        -------
        slice
            Slice equivalent to ``slice(start, end + 1)``.
        """
        return slice(self.start, self.end_extd)


@dataclass(frozen=True)
class TimeFrame:
    """
    Immutable representation of a time interval with a fixed timestep.

    This class encapsulates a time range defined by start and end timestamp, duration, and resolution (``Timestep``).
    It provides convenient access to derived datetime indices aligned to the specified timestep.

    Parameters
    ----------
    start : pandas.Timestamp
        The start of the time frame (inclusive).
    end : pandas.Timestamp
        The end of the time frame (exclusive for most operations).
    duration : pandas.Timedelta
        The duration of the time frame.
    timestep : Timestep
        The temporal resolution of the time frame.
    timezone : zoneinfo.ZoneInfo
        The timezone associated with the timestamps.
    start_ref : pandas.Timestamp
        The reference start time, usually the start of the simulation TimeFrame.

    Attributes
    ----------
    dti : pandas.DatetimeIndex
        Datetime index spanning ``[start, end)`` with the given timestep.
    dti_extd : pandas.DatetimeIndex
        Datetime index spanning ``[start, end]`` (inclusive of both bounds).
    end_extd : pandas.Timestamp
        The last timestamp in ``dti_extd``.
    idx : TimeIndex
        TimeIndex relative to ``start_ref``.
    idx_tf : TimeIndex
        TimeIndex relative to ``start``.

    Notes
    -----
    - The class is immutable (frozen dataclass).
    - All timestamps are normalized to the specified timezone using ``ensure_timezone``.
    - The ``end`` timestamp is always recomputed from ``start + duration`` to ensure internal consistency.
    - The ``duration`` is aligned (floored) to the timestep frequency.

    Methods
    -------
    create_from_start_timestamp(start, timestep, timezone, end=None, duration=None)
        Construct a ``TimeFrame`` from a start timestamp and either an end timestamp or a duration.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneinfo import ZoneInfo
    >>> ts = pd.Timestamp("2026-01-01 00:00")
    >>> timestep = Timestep.from_str("1h")
    >>> tf = TimeFrame.create_from_start_timestamp(
    ...     start=ts,
    ...     timestep=timestep,
    ...     timezone=ZoneInfo("UTC"),
    ...     duration=pd.Timedelta(hours=3),
    ... )
    >>> tf.dti
    DatetimeIndex(['2026-01-01 00:00:00+00:00',
                   '2026-01-01 01:00:00+00:00',
                   '2026-01-01 02:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', name='time', freq='H')
    """

    start: pd.Timestamp
    end: pd.Timestamp
    duration: pd.Timedelta
    timestep: Timestep
    timezone: zoneinfo.ZoneInfo
    start_ref: pd.Timestamp

    @classmethod
    def create_from_start_timestamp(
        cls,
        start: pd.Timestamp,
        timestep: Timestep,
        timezone: zoneinfo.ZoneInfo | str,
        end: pd.Timestamp | None = None,
        duration: pd.Timedelta | None = None,
        start_ref: pd.Timestamp | None = None,
    ) -> Self:
        """
        Create a ``TimeFrame`` from a start timestamp.

        Exactly one of ``end`` or ``duration`` must be provided. The missing value is computed accordingly, and the
        resulting duration is aligned (floored) to the timestep frequency. The end timestamp is always recomputed as
        ``start + duration`` to ensure consistency.

        Parameters
        ----------
        start : pandas.Timestamp
            The start timestamp.
        timestep : Timestep
            The timestep defining the resolution.
        timezone : zoneinfo.ZoneInfo or str
            The timezone to enforce on the timestamps.
        end : pandas.Timestamp, optional
            The end timestamp. Must not be provided together with ``duration``.
        duration : pandas.Timedelta, optional
            The duration of the time frame. Must not be provided together with ``end``.
        start_ref : pd.Timestamp, optional
            The start time used to calculate ``idx``.

        Returns
        -------
        TimeFrame
            A new ``TimeFrame`` instance.

        Raises
        ------
        ValueError
            If neither or both of ``end`` and ``duration`` are provided.

        Notes
        -----
        - Duration is floored to ``timestep.freqstr``.
        - Both ``start`` and ``end`` are converted to the specified timezone
          using ``ensure_timezone``.

        Examples
        --------
        >>> import pandas as pd
        >>> from zoneinfo import ZoneInfo
        >>> ts = pd.Timestamp("2026-01-01 00:00")
        >>> timestep = Timestep.from_str("1h")
        >>> tf = TimeFrame.create_from_start_timestamp(
        ...     start=ts,
        ...     timestep=timestep,
        ...     timezone="UTC",
        ...     end=pd.Timestamp("2026-01-01 03:30"),
        ... )
        >>> tf.duration
        Timedelta('0 days 03:00:00')
        """

        tz = convert_to_zoneinfo(timezone)

        if (end is None and duration is None) or (end is not None and duration is not None):
            raise ValueError('Exactly one of the parameters "end" or "duration" must be provided.')
        elif duration is None:
            duration = (end - start).floor(timestep.freqstr)
        elif end is None:
            duration = duration.floor(timestep.freqstr)
        end = start + duration

        start = ensure_timezone(ts=start, timezone=tz)
        end = ensure_timezone(ts=end, timezone=tz)

        if start_ref is None:
            start_ref = start
        else:
            start_ref = ensure_timezone(ts=start_ref, timezone=tz)

        return cls(start=start, end=end, duration=duration, timestep=timestep, timezone=tz, start_ref=start_ref)

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        """
        Return a left-inclusive datetime index for the time frame.

        The index spans ``[start, end)`` with frequency defined by the timestep.

        Returns
        -------
        pandas.DatetimeIndex
            Datetime index excluding the end timestamp.
        """
        return pd.date_range(start=self.start, end=self.end, freq=self.timestep.td, inclusive="left", name="time")

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        """
        Return a fully inclusive datetime index for the time frame.

        The index spans ``[start, end]`` including both boundaries.

        Returns
        -------
        pandas.DatetimeIndex
            Datetime index including both start and end timestamps.
        """
        return pd.date_range(start=self.start, end=self.end, freq=self.timestep.td, inclusive="both", name="time")

    @cached_property
    def end_extd(self) -> pd.Timestamp:
        """
        Return the last timestamp in the extended datetime index.

        Returns
        -------
        pandas.Timestamp
            The last value of ``dti_extd``.
        """
        return self.dti_extd[-1]
    
    @cached_property
    def idx(self) -> TimeSlice:
        """
        Returns a TimeSlice. Indexing starts at ``TimeFrame.start_ref``.

        Returns
        -------
        TimeSlice
            Timeslice with indexing starting at ``self.start_ref``.
        """
        return TimeSlice.from_timeframe(tf=self, ref=True)
    
    @cached_property
    def idx_tf(self) -> TimeSlice:
        """
        Returns a TimeSlice. Indexing starts at ``TimeFrame.start``.

        Returns
        -------
        TimeSlice
            Timeslice with indexing starting at ``self.start``.
        """
        return TimeSlice.from_timeframe(tf=self, ref=False)


@dataclass
class SimulationTimes:
    """
    Container for simulation-related time frames.

    This class groups together three aligned ``TimeFrame`` objects:
    simulation, evaluation, and project time frames. It provides a convenient factory method to construct these from
    plain input values such as strings, durations, and timezone specifications.

    Parameters
    ----------
    sim : TimeFrame
        The simulation time frame.
    eval : TimeFrame
        The evaluation time frame (typically aligned with simulation).
    prj : TimeFrame
        The project time frame (usually longer-term).

    Notes
    -----
    - The ``sim`` and ``eval`` time frames are constructed identically in ``create_from_plain``.
    - The ``prj`` time frame extends from the same start time but spans a number of years defined by ``prj_duration``.
    - All timestamps are normalized to the specified timezone.

    Methods
    -------
    create_from_plain(timestep, timezone, starttime, sim_endtime, sim_duration, prj_duration)
        Construct ``SimulationTimes`` from basic input types.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneinfo import ZoneInfo
    >>> timestep = Timestep.from_str("1h")
    >>> sim_times = SimulationTimes.create_from_plain(
    ...     timestep=timestep,
    ...     timezone="UTC",
    ...     starttime="2026-01-01 00:00",
    ...     sim_endtime="2026-01-03 00:00",
    ...     sim_duration=None,
    ...     prj_duration=1,
    ... )
    >>> sim_times.sim.duration
    Timedelta('2 days 00:00:00')
    >>> sim_times.prj.duration
    Timedelta('365 days 00:00:00')
    """

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
        """
        Create ``SimulationTimes`` from plain input values.

        This method converts input values (strings, durations, timezone) into properly aligned ``TimeFrame``
        objects for simulation, evaluation, and project horizons.

        Parameters
        ----------
        timestep : Timestep
            The timestep defining temporal resolution.
        timezone : zoneinfo.ZoneInfo or str
            The timezone for all generated timestamps.
        starttime : str or pandas.Timestamp
            The start time of all time frames.
        sim_endtime : str or pandas.Timestamp or None
            The end time of the simulation/evaluation period.
        sim_duration : float or int or None
            The simulation duration in days. Used if ``sim_endtime`` is not provided.
        prj_duration : int
            The project duration in years.

        Returns
        -------
        SimulationTimes
            A new instance containing simulation, evaluation, and project time frames.

        Raises
        ------
        ValueError
            If the start time cannot be converted to a valid timestamp or both or neither of ``sim_endtime`` and
             ``sim_duration`` are provided (propagated from ``TimeFrame.create_from_start_timestamp``).

        Notes
        -----
        - ``starttime`` and ``sim_endtime`` are parsed using ``convert_to_timestamp``.
        - ``sim_duration`` is interpreted in days and converted to ``pandas.Timedelta``.
        - The project time frame always uses ``prj_duration`` in years via ``pandas.DateOffset``.

        Examples
        --------
        >>> import pandas as pd
        >>> timestep = Timestep.from_str("1h")
        >>> sim_times = SimulationTimes.create_from_plain(
        ...     timestep=timestep,
        ...     timezone="UTC",
        ...     starttime="2026-01-01",
        ...     sim_endtime=None,
        ...     sim_duration=2,
        ...     prj_duration=1,
        ... )
        >>> sim_times.sim.duration
        Timedelta('2 days 00:00:00')
        """

        starttime_ts = convert_to_timestamp(time_in=starttime, timestep=timestep, timezone=timezone)
        if starttime_ts is None:
            raise ValueError(f"Failed to convert starttime ({starttime}) to pd.Timestamp")

        sim_endtime_ts = convert_to_timestamp(time_in=sim_endtime, timestep=timestep, timezone=timezone)
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
