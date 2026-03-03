#!/usr/bin/env python3


def map_timeframes(df, name):
    cs_map = {"vehicles": map_timeframes_vehicles, "brs": map_timeframes_batteries}  # fleet: mapping_function
    return cs_map[name](df)


def map_timeframes_vehicles(df):
    condition = df.index.weekday > 4
    df.loc[condition, "timeframe"] = "weekend"
    df.loc[~condition, "timeframe"] = "weekday"
    return df["timeframe"]


def map_timeframes_batteries(df):
    df["timeframe"] = "day"
    return df["timeframe"]
