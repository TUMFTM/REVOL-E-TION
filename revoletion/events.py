#!/usr/bin/env python3
"""
Dispatch events are the exchange format between the Fleet blocks and their respective dispatchers.

An event holds the values of one rental of one unit (energy, distance, target SOC) instead of a value per timestep, and is therefore independent of the simulation timestep. Departure and return are resolved exactly, whereas a time based log can only express them as a change of occupancy and cannot separate two rentals following each other without an idle timestep in between.
The energy system model works per timestep, so the log is still needed - it is materialized from the events here. Everything in a dispatched log is derived from the events; the reverse only holds up to the reconstruction implemented in from_log().
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Event table: the demand list after dispatch. One row per rented unit and dispatch process, plus one
# row for every request that could not be served - those keep their pid, status and requested time, and
# have no unit and no dispatched values.
COLUMNS = [
    "pid",
    "unit",
    "status",
    "time_req",
    "time_dep",
    "time_return",
    "time_available",
    "dsoc",
    "energy",
    "dist",
    "rex",
]

STATUS_SUCCESS = "success"

TIME_COLUMNS = ["time_req", "time_dep", "time_return", "time_available"]
VALUE_COLUMNS = ["dsoc", "energy", "dist"]


def normalize(events: pd.DataFrame) -> pd.DataFrame:
    """
    Bring an event table into the common column order and dtypes.

    Rows of unserved requests leave most columns empty, which otherwise turns them into object columns
    that cannot be serialized.
    """
    events = events.reindex(columns=COLUMNS)

    for col in ["pid", "unit", "status"]:
        events[col] = events[col].astype("string")
    for col in VALUE_COLUMNS:
        events[col] = events[col].astype("float64")
    for col in TIME_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(events[col]):
            events[col] = pd.to_datetime(events[col], utc=True)
    # range extension is a property of the request, so it is known regardless of the outcome
    events["rex"] = events["rex"].astype(bool)

    return events.sort_values(by=["time_dep", "unit"], ignore_index=True)


def served(events: pd.DataFrame) -> pd.DataFrame:
    """
    Rows of requests that were actually served, i.e. that have a unit assigned to them.
    """
    return events[events["status"].eq(STATUS_SUCCESS)]


# time based log of one unit: column -> value while the unit is at base
LOG_DEFAULTS = {
    "atbase": True,
    "atac": False,
    "atdc": False,
    "dsoc": 0.0,
    "consumption": 0.0,
    "dist": 0.0,
    "rex": False,
}
LOG_COLUMNS = list(LOG_DEFAULTS)


def empty() -> pd.DataFrame:
    """
    Empty event table.
    """
    return normalize(pd.DataFrame(columns=COLUMNS))


def overlapping(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Served rentals reaching into the given timeframe. One starting before or ending after it still
    contributes the part that falls inside and must therefore not be dropped.
    """
    events = served(events)
    return events[events["time_dep"].le(index.max()) & events["time_return"].gt(index.min())]


def kpis(events: pd.DataFrame, index: pd.DatetimeIndex, step: pd.Timedelta, n_units: int) -> dict:
    """
    Dispatch KPIs.

    rate_blocked counts all time units are unavailable for other requests (rental plus subsequent
    recharging), rate_utilization only counts the rental time itself, i.e. the time units are not at
    base. Requests are counted per process, not per row, so a rental taking several units at once
    still counts as one request.
    """
    requests = events.drop_duplicates(subset="pid")
    rentals = served(events)
    time_total = (index.max() + step) - index.min()

    try:
        return {
            "rate_success": requests["status"].eq(STATUS_SUCCESS).mean(),
            "rate_blocked": (rentals["time_available"] - rentals["time_dep"]).sum() / time_total / n_units,
            "rate_utilization": (rentals["time_return"] - rentals["time_dep"]).sum() / time_total / n_units,
        }
    except (TypeError, ZeroDivisionError):
        return {"rate_success": np.nan, "rate_blocked": 0.0, "rate_utilization": 0.0}


def read(path: Path | str, timezone: str) -> pd.DataFrame:
    """
    Read an event table from file, e.g. one written by a previous run, to replay its dispatch.

    Times are converted to the timezone of the scenario, as they are used to address the simulation
    timeframe, which is localized.
    """
    path = Path(path).resolve()
    events = pd.read_feather(path) if path.suffix == ".feather" else pd.read_csv(path)

    missing = set(COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f'Event file "{path}" is missing the columns {sorted(missing)}')

    events = normalize(events)
    for col in TIME_COLUMNS:
        events[col] = events[col].dt.tz_convert(timezone)

    return events


def empty_log_unit(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Log of a single unit, at base with no consumption, distance or SOC change.
    """
    return pd.DataFrame(LOG_DEFAULTS, index=index)


def empty_log(index: pd.DatetimeIndex, units: list) -> pd.DataFrame:
    """
    Log of several units, at base with no consumption, distance or SOC change.
    """
    return materialize(events=empty(), index=index, units=units, step=pd.Timedelta(0), step_hours=0.0)


def materialize_unit(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    step: pd.Timedelta,
    step_hours: float,
) -> pd.DataFrame:
    """
    Build the time based log of a single unit from its events.

    Event values are distributed evenly over the rental period, as the dispatch resolves no
    intra-rental detail. The return timestep is not part of the rental - the unit is back at base.
    Rentals reaching beyond the given timeframe contribute the part that falls inside it.
    """
    log = empty_log_unit(index=index)

    for event in served(events).itertuples():
        time_end = event.time_return - step
        steps = (event.time_return - event.time_dep) / step

        log.loc[event.time_dep : time_end, "atbase"] = False
        log.loc[event.time_dep : time_end, "atac"] = False  # todo destination charging?
        log.loc[event.time_dep : time_end, "atdc"] = True
        log.loc[event.time_dep : time_end, "consumption"] = event.energy / (steps * step_hours)
        log.loc[event.time_dep : time_end, "dist"] = event.dist / steps
        # mark range extension for another fleet -> internal service, not an external rental
        log.loc[event.time_dep : time_end, "rex"] = event.rex

        # scalar assignment would insert a row for a departure before the start of the timeframe
        if event.time_dep in log.index:
            log.at[event.time_dep, "dsoc"] = event.dsoc

    return log


def materialize(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    units: list,
    step: pd.Timedelta,
    step_hours: float,
) -> pd.DataFrame:
    """
    Build the time based log of several units from the event table.
    """
    if not units:
        return pd.DataFrame(index=index)

    return pd.concat(
        [
            materialize_unit(events=events[events["unit"].eq(unit)], index=index, step=step, step_hours=step_hours)
            for unit in units
        ],
        axis=1,
        keys=units,
        names=["unit", "time"],
    )


def from_log(log: pd.DataFrame, step: pd.Timedelta) -> pd.DataFrame:
    """
    Reconstruct the event table from a log read from file.

    A log only records occupancy, so contiguous absences are the best available reconstruction: two
    rentals following each other without an intermediate timestep at base are indistinguishable and are
    recovered as a single event. Intra-rental detail a measured log may contain (destination charging,
    varying consumption) is deliberately left in the log rather than folded into the events.

    A log holds no record of requests that were never served and none of the recharging after a return,
    so the reconstruction contains served rentals only and reports no waiting time before departure.
    """
    events = []

    for unit in log.columns.get_level_values(0).unique():
        away = ~log[(unit, "atbase")].astype(bool)
        if not away.any():
            continue

        # number the contiguous absences to group the timesteps belonging to one rental
        rentals = (away & ~away.shift(periods=1, fill_value=False)).cumsum()[away]
        consumption = log[(unit, "consumption")].astype(float)
        dist = log[(unit, "dist")].astype(float)
        dsoc = log[(unit, "dsoc")].astype(float)
        rex = log[(unit, "rex")].astype(bool) if (unit, "rex") in log.columns else None

        for rental, dti in rentals.groupby(rentals).groups.items():
            events.append(
                {
                    "pid": f"{unit}_{rental}",
                    "unit": unit,
                    "status": STATUS_SUCCESS,
                    "time_req": dti.min(),  # a log does not record when the request was placed
                    "time_dep": dti.min(),
                    "time_return": dti.max() + step,
                    "time_available": dti.max() + step,  # nor the recharging after the return
                    "dsoc": dsoc.loc[dti.min()],
                    "energy": consumption.loc[dti].sum() * (step / pd.Timedelta(hours=1)),
                    "dist": dist.loc[dti].sum(),
                    "rex": bool(rex.loc[dti].any()) if rex is not None else False,
                }
            )

    return normalize(pd.DataFrame(events, columns=COLUMNS))
