import importlib.resources as pkg_resources
from pathlib import Path

import pandas as pd

import revoletion


def convert_slp(
    path_in: Path,
    path_out: Path,
):
    """
    Convert BDEW SLP profiles from human-readable format to long format for more efficient processing.
    Before: Index = Time of the day; Columns = MultiIndex (Profile, Period, Type of Day)
    After: Index = MultiIndex (Profile, Period, Type of Day, Time of the day); Column = Power
    """

    order_months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Summer",
        "Winter",
        "Transition",
    ]

    order_profiles = [
        "H0",
        "G0",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "L0",
        "L1",
        "L2",
        "L3",
        "H25",
        "G25",
        "L25",
        "S25",
        "P25",
    ]

    order_days = ["Workday", "Saturday", "Sunday"]

    profile_map = {m: i for i, m in enumerate(order_profiles)}
    month_map = {m: i for i, m in enumerate(order_months)}
    day_map = {m: i for i, m in enumerate(order_days)}

    def key_func(x):
        if x.isin(profile_map).all():
            return x.map(profile_map)
        if x.isin(month_map).all():
            return x.map(month_map)
        if x.isin(day_map).all():
            return x.map(day_map)
        return x  # level 3: times (00:00 etc.)

    # Read BDEW SLP profiles
    slp = (
        pd.read_csv(
            path_in,
            skiprows=[0],
            header=[0, 1, 2],
            index_col=0,
        )
        .stack(level=[0, 1, 2], future_stack=True)  # "future_stack=True" to avoid FutureWarning
        .rename("power")
        .reorder_levels([1, 2, 3, 0])
        .sort_index(level=[0, 1, 2, 3], key=key_func)
    )

    slp.index.names = ["profile", "period", "day", "time"]

    slp.to_csv(path_out)

    return


if __name__ == "__main__":
    data_dir: Path = pkg_resources.files(revoletion.data)
    file_in = "slp_bdew_readable.csv"
    file_out = "slp_bdew.csv"
    convert_slp(path_in=data_dir / file_in, path_out=data_dir / file_out)
