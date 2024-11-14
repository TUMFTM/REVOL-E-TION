#!/usr/bin/env python3

def map_timeframes(df, name, scenario):
    cs_map = {'bev': map_timeframes_bev, 'brs': map_timeframes_brs}
    return cs_map[name](df, scenario)


def map_timeframes_bev(df, scenario):

    condition = df.index.weekday > 4

    df.loc[condition, 'timeframe'] = 'A'
    df.loc[condition, 'demand_mean'] = 5
    df.loc[condition, 'demand_std'] = 2

    df.loc[~condition, 'timeframe'] = 'A'
    df.loc[~condition, 'demand_mean'] = 5
    df.loc[~condition, 'demand_std'] = 2

    return df['timeframe'], df['demand_mean'], df['demand_std']


def map_timeframes_brs(df, scenario):

    condition = df.index.weekday > 4

    df.loc[condition, 'timeframe'] = 'A'
    df.loc[condition, 'demand_mean'] = 5
    df.loc[condition, 'demand_std'] = 2

    df.loc[~condition, 'timeframe'] = 'A'
    df.loc[~condition, 'demand_mean'] = 5
    df.loc[~condition, 'demand_std'] = 2

    return df['timeframe'], df['demand_mean'], df['demand_std']
