from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from revoletion.time import TimeFrame


class PeakPowerPeriodStart(str, Enum):
    CALENDAR = "CALENDAR"  # peak power periods start at calendar boundaries (e.g. month, year)
    SIMULATION = "SIMULATION"  # peak power periods start at simulation start


@dataclass(frozen=True)
class PeakPowerInterval:
    period_str: str  # pandas frequency string to convert timestamps to periods; used for PeakPowerPeriodStart.CALENDAR
    duration_offset: (
        pd.DateOffset
    )  # offset to add to the start of a period to get the end of the period; used for PeakPowerPeriodStart.SIMULATION
    periods_per_year: int | float | None


class PeakPowerPeriodFreq(Enum):
    DAY = PeakPowerInterval("D", pd.DateOffset(days=1), 365)
    WEEK = PeakPowerInterval("W-MON", pd.DateOffset(weeks=1), 7 / 365)
    MONTH = PeakPowerInterval("M", pd.DateOffset(months=1), 12)
    QUARTER = PeakPowerInterval("Q", pd.DateOffset(months=3), 4)
    YEAR = PeakPowerInterval("Y", pd.DateOffset(years=1), 1)
    SIM = PeakPowerInterval("SIM", pd.DateOffset(0), None)


def get_peak_periods(
    timeframe: TimeFrame,
    peak_period: PeakPowerPeriodFreq,
    peak_period_start: PeakPowerPeriodStart,
    peak_power_init: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # for PeakPowerPeriodFreq.SIM: set whole simulation period as one peak power period
    if peak_period == PeakPowerPeriodFreq.SIM:
        return (
            pd.DataFrame(index=["sim"], data={"fraction": 1.0, "peak_power": peak_power_init}),
            pd.DataFrame(index=timeframe.dti, data={"sim": 1.0}),
        )

    if peak_period_start == PeakPowerPeriodStart.CALENDAR:
        # convert dti to periods
        periods = pd.Series(
            index=timeframe.dti,
            # to_period() drops timezone with warning -> remove timezone before converting to period and add it back later
            data=timeframe.dti.tz_localize(None).to_period(peak_period.value.period_str),
            name="periods",
        )

        # group by periods and count timesteps in each period
        agg = periods.groupby(periods, sort=False).size().to_frame("count")

        # get start and end time of each period
        agg["start"] = agg.index.start_time.tz_localize(timeframe.timezone)
        agg["end"] = agg.index.end_time.tz_localize(timeframe.timezone).ceil(timeframe.timestep.td)

        # apply labels to the periods
        if peak_period == PeakPowerPeriodFreq.WEEK:
            # pandas periods do not use ISO weeks -> manually create ISO week labels
            iso = agg["start"].dt.isocalendar()
            agg["label"] = iso["year"].astype(str) + "-cw" + iso["week"].astype(str).str.zfill(2)
        else:
            # for all other period frequencies: convert period name to string
            agg["label"] = agg.index.astype(str)

    else:
        # build edges (few iterations → cheap)
        edges = [timeframe.dti[0]]
        while edges[-1] <= timeframe.dti[-1]:
            edges.append(edges[-1] + peak_period.value.duration_offset)

        edges = pd.to_datetime(edges)

        periods = pd.Series(
            index=timeframe.dti, data=np.searchsorted(edges, timeframe.dti, side="right"), name="periods"
        )

        # aggregate
        agg = periods.groupby(periods, sort=False).size().to_frame("count")

        # assign start/end from edges
        agg["start"] = pd.to_datetime(edges[:-1])
        agg["end"] = pd.to_datetime(edges[1:])

        agg["label"] = f"sim_{peak_period.name.lower()}_" + agg.index.astype(str).str.zfill(len(str(agg.index.max())))

    # calculate the fraction of timesteps per period included in the simulation timeframe
    agg["fraction"] = (agg["count"] * timeframe.timestep.td) / (agg["end"] - agg["start"])

    # set initial peak power for each period
    agg["peak_power"] = float(peak_power_init)

    # create activation matrix -> one column per period (bool values to save memory) and rename periods by label
    activation = pd.get_dummies(periods).rename(columns=agg["label"].to_dict())

    # set label as index instead of period object, but also keep column for easier access to labels
    agg = agg.set_index("label", drop=False)

    agg = agg[["fraction", "peak_power", "start", "end", "label"]]

    return agg, activation
