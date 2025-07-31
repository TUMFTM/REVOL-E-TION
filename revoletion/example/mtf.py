#!/usr/bin/env python3

def map_timeframes(df, name):
    cs_map = {'ld': map_timeframes_vehicles, 'pue': map_timeframes_batteries}
    return cs_map[name](df)


def map_timeframes_vehicles(df):

    condition = df.index.weekday > 4

    df.loc[condition, 'timeframe'] = 'weekend'
    df.loc[condition, 'demand_mean'] = 5
    df.loc[condition, 'demand_std'] = 2

    df.loc[~condition, 'timeframe'] = 'weekday'
    df.loc[~condition, 'demand_mean'] = 12
    df.loc[~condition, 'demand_std'] = 4

    return df['timeframe'], df['demand_mean'], df['demand_std']


def map_timeframes_batteries(df):

    df.loc[:, 'timeframe'] = 'day'
    df.loc[:, 'demand_mean'] = 5
    df.loc[:, 'demand_std'] = 2

    return df['timeframe'], df['demand_mean'], df['demand_std']
