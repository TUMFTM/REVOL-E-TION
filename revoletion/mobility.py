#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import scipy as sp

import revoletion.utils as utils


def lognormal_params(mean: float,
                     stdev: float) -> tuple:
    """
    calculate lognormal parameters mu and sigma from mean and standard deviation
    """
    mu = np.log(mean ** 2 / np.sqrt((mean ** 2) + (stdev ** 2)))
    sig = np.sqrt(np.log(1 + (stdev ** 2) / (mean ** 2)))
    return mu, sig


class SubFleetDemand:
    """
    abstract class
    """
    def __init__(self,
                 scenario: 'simulation.Scenario',
                 subfleet: 'blocks.SubFleet'):

        self.scenario = scenario
        self.subfleet = subfleet

        self.usecases = None  # remains unfilled if demand is read from file
        self.demand = pd.DataFrame()  # main DataFrame for demand

        self.mapper_timeframe = utils.import_module_from_path(
            module_name=self.subfleet.filename_mapper,
            file_path=os.path.join(self.scenario.run.paths['input'],
                                   f'{self.subfleet.filename_mapper}.py'))

        self.rng = np.random.default_rng()  # random number generator

    def read_usecase_file(self):
        """
        read a usecase definition csv file and perform necessary normalization for each timeframe.
        """

        usecase_path = os.path.join(self.scenario.run.paths['input'],
                                    utils.set_extension(self.subfleet.filename))
        self.usecases = pd.read_csv(usecase_path,
                                    header=[0,1],
                                    index_col=0)
        for timeframe in self.usecases.columns.levels[0]:
            self.usecases.loc[:, (timeframe, 'rel_prob_norm')] = (self.usecases.loc[:, (timeframe, 'rel_prob')] /
                                                                  self.usecases.loc[:, (timeframe, 'rel_prob')].sum())
            self.usecases.loc[:, (timeframe, 'sum_dep_magn')] = (self.usecases.loc[:, (timeframe, 'dep1_magnitude')] +
                                                                 self.usecases.loc[:, (timeframe, 'dep2_magnitude')])

            # catch cases where the sum of both departure magnitudes is not one
            self.usecases.loc[:, (timeframe, 'dep1_magnitude')] = (self.usecases.loc[:, (timeframe, 'dep1_magnitude')] /
                                                                   self.usecases.loc[:, (timeframe, 'sum_dep_magn')])
            self.usecases.loc[:, (timeframe, 'dep2_magnitude')] = (self.usecases.loc[:, (timeframe, 'dep2_magnitude')] /
                                                                   self.usecases.loc[:, (timeframe, 'sum_dep_magn')])

            self.usecases.drop(columns=[(timeframe, 'sum_dep_magn')], inplace=True)

    def sample(self):
        """
        generate demand dataframe from usecases & timeframes including all pre-dispatch information
        """

        # region sample daily total demand from timeframe mapper and lognormal distribution
        daily_total = pd.DataFrame(index=pd.to_datetime(np.unique(self.scenario.dti_sim_extd.date)))
        daily_total['timeframe'], daily_total['demand_mean'], daily_total['demand_std'] = \
            self.mapper_timeframe.map_timeframes(daily_total, self.subfleet.name, self.scenario)
        daily_total['mu'], daily_total['sigma'] = lognormal_params(daily_total['demand_mean'],
                                                                   daily_total['demand_std'])
        daily_total['demand'] = daily_total.apply(
            lambda row: np.round(self.rng.lognormal(row['mu'], row['sigma'])).astype(int),
            axis=1)
        # endregion

        # region get request dates
        self.demand['date'] = pd.to_datetime(np.repeat(daily_total.index, daily_total['demand']))
        self.demand['year'] = self.demand['date'].dt.year
        self.demand['month'] = self.demand['date'].dt.month
        self.demand['day'] = self.demand['date'].dt.day
        self.demand['timeframe'] = daily_total.loc[self.demand['date'], 'timeframe'].values

        def sample_usecases(group):
            return pd.Series(np.random.choice(self.usecases.index.values,
                                              size=len(group),
                                              replace=True,
                                              p=self.usecases.loc[:, (group.name, 'rel_prob_norm')]),
                             index=group.index)

        self.demand['usecase'] = None
        self.demand['usecase'] = self.demand.groupby('timeframe')['usecase'].transform(sample_usecases)
        # endregion

        # region sample request times of day from usecase distribution
        def sample_time_usecase(group):
            usecase = group.name[0]
            timeframe = group.name[1]

            # always sample finer than timestep to avoid rounding errors
            time_vals = np.arange(start=0, stop=24, step=self.scenario.timestep_hours / 100)

            mag1 = self.usecases.loc[usecase, (timeframe, 'dep1_magnitude')]
            mean1 = np.median([self.usecases.loc[usecase, (timeframe, 'dep1_time_mean')], 0, 24])
            std1 = np.max([self.usecases.loc[usecase, (timeframe, 'dep1_time_std')], 1e-8])
            cdf1_vals = sp.stats.norm.cdf(time_vals, mean1, std1)

            mag2 = self.usecases.loc[usecase, (timeframe, 'dep2_magnitude')]
            mean2 = np.median([self.usecases.loc[usecase, (timeframe, 'dep2_time_mean')], 0, 24])
            std2 = np.max([self.usecases.loc[usecase, (timeframe, 'dep2_time_std')], 1e-8])
            cdf2_vals = sp.stats.norm.cdf(time_vals, mean2, std2)

            cdf_vals = mag1 * cdf1_vals + mag2 * cdf2_vals
            # Generate n uniform random numbers between 0 and 1
            uniform_samples = np.random.rand(len(group))
            # Interpolate to find the samples
            time_samples = np.interp(uniform_samples, cdf_vals, time_vals)
            # round to timestep
            time_samples = np.round(time_samples / self.scenario.timestep_hours) * self.scenario.timestep_hours
            return pd.DataFrame(data=time_samples, index=group.index)

        self.demand['hour'] = (self.demand.groupby(['usecase', 'timeframe'])
                               .apply(sample_time_usecase).reset_index(level=[0, 1], drop=True).sort_index())
        self.demand['time_req'] = pd.to_datetime(self.demand[['year', 'month', 'day', 'hour']])
        self.demand.drop(['date', 'year', 'month', 'day', 'hour'], inplace=True, axis=1)
        self.demand['time_req'] = self.demand['time_req'].dt.tz_localize(self.scenario.timezone,
                                                                         ambiguous='NaT',  # fall
                                                                         nonexistent='shift_forward')  # spring
        self.demand.dropna(axis='index', subset=['time_req'], inplace=True)
        # endregion

        self.sample_energy_demand()  # specific to type of subfleet units (battery or vehicle)

        # region sample idle time
        def sample_idle_usecase(group):
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_idle_mean = self.usecases.loc[usecase, (timeframe, 'idle_mean')]
            uc_idle_stdev = self.usecases.loc[usecase, (timeframe, 'idle_std')]
            p1, p2 = lognormal_params(uc_idle_mean, uc_idle_stdev)
            idle = pd.to_timedelta(self.rng.lognormal(p1, p2, len(group)), unit='hour')
            return pd.Series(idle, index=group.index)

        self.demand['dtime_idle'] = None
        self.demand['dtime_idle'] = (self.demand
                                     .groupby(['usecase', 'timeframe'])['dtime_idle']
                                     .transform(sample_idle_usecase))
        # endregion

        # region get patience
        def get_patience_usecase(group):
            """
            groupby function
            get patience for one usecase and timeframe from usecase file
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            patience = pd.to_timedelta(self.usecases.loc[usecase, (timeframe, 'patience')],
                                       unit='hour')
            return pd.DataFrame({'patience_primary': [patience] * len(group)}, index=group.index)

        self.demand['dtime_patience'] = (self.demand
                                         .groupby(['usecase','timeframe'])
                                         .apply(get_patience_usecase)
                                         .reset_index(level=[0, 1], drop=True)
                                         .sort_index())
        # endregion

        # region save results
        if self.scenario.run.save_results_dispatch:
            demand_path = os.path.join(
                self.scenario.run.paths['output'],
                f'{self.scenario.run.runtimestamp}_'
                f'{self.scenario.run.name}_'
                f'{self.scenario.name}_'
                f'{self.subfleet.name}_'
                f'demand.csv')
            self.demand.to_csv(demand_path)
        # endregion




class BatteryDemand(SubFleetDemand):

    def sample_energy_demand(self):
        """
        Sample energy requirement for each request
        """

        def sample_energy_usecase(group):
            """
            groupby function
            sample energy requirements for one usecase and timeframe.
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_energy_mean = self.usecases.loc[usecase, (timeframe, 'energy_mean')]
            uc_energy_stdev = self.usecases.loc[usecase, (timeframe, 'energy_std')]
            p1, p2 = lognormal_params(uc_energy_mean, uc_energy_stdev)
            dist = self.rng.lognormal(p1, p2, len(group))
            return pd.Series(dist, index=group.index)

        self.demand['energy_req'] = None
        self.demand['energy_req'] = (self.demand
                                     .groupby(['usecase', 'timeframe'])['energy_req']
                                     .transform(sample_energy_usecase))
        def calc_time_active_usecase(group):
            """
            groupby function
            calculate active time for one usecase and timeframe
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            power = self.usecases.loc[usecase, (timeframe, 'power_avg')]
            dtime_active = pd.to_timedelta(group['energy_req'] / power, unit='hour')
            return pd.Series(dtime_active, index=group.index)

        self.demand['dtime_active'] = (self.demand
                                       .groupby(['usecase', 'timeframe'])
                                       .apply(calc_time_active_usecase)
                                       .reset_index(level=[0, 1], drop=True)
                                       .sort_index())


class VehicleDemand(SubFleetDemand):

    def sample_energy_demand(self):
        """
        Sample distances for each request within a VehicleFleet
        """

        def sample_distance_usecase(group):
            """
            groupby function
            sample distances for one usecase and timeframe from lognormal distribution
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_dist_mean = self.usecases.loc[usecase, (timeframe, 'dist_mean')]
            uc_dist_stdev = self.usecases.loc[usecase, (timeframe, 'dist_std')]
            p1, p2 = lognormal_params(uc_dist_mean, uc_dist_stdev)
            dist = self.rng.lognormal(p1, p2, len(group))
            return pd.Series(dist, index=group.index)

        self.demand['distance'] = np.nan
        self.demand['distance'] = (self.demand.groupby(
            ['usecase', 'timeframe'])['distance'].transform(sample_distance_usecase))

        def get_consumption_speed_usecase(group):
            """
            groupby function
            get consumption and speed values for one usecase and timeframe from the usecase file
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            consumption = self.usecases.loc[usecase, (timeframe, 'consumption')]
            speed_avg = self.usecases.loc[usecase, (timeframe, 'speed_avg')]
            return pd.DataFrame(data={'consumption': [consumption] * len(group),
                                      'speed_avg': [speed_avg] * len(group)},
                                index=group.index)

        self.demand[['consumption', 'speed_avg']] = (self.demand.groupby(['usecase', 'timeframe'])
                                                     .apply(get_consumption_speed_usecase)
                                                     .reset_index(level=[0, 1], drop=True)
                                                     .sort_index())

        self.demand['dtime_active'] = pd.to_timedelta(self.demand['distance'] / self.demand['speed_avg'],
                                                      unit='hour')

        self.demand['energy_req'] = self.demand['distance'] * self.demand['consumption']
