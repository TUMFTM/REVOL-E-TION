#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import scipy as sp

import revoletion.utils as utils


class CommodityDemand:
    def __init__(self, scenario, commodity_system):
        """
        CommodityDemand objects are only to be initialized by inheriting subclasses.
        """
        self.scenario = scenario
        self.commodity_system = commodity_system
        self.requests = pd.DataFrame()  # main DataFrame for demand

        self.path_mapper = os.path.join(self.scenario.run.path_input_data,
                                        'TimeframeMapper',
                                        f'{self.scenario.filename_mapper}.py')
        self.mapper_timeframe = utils.import_module_from_path(module_name=self.scenario.filename_mapper,
                                                              file_path=self.path_mapper)

        self.rng = np.random.default_rng()  # random number generator

    def get_patience(self):
        self.requests['dtime_patience'] = (self.requests
                                           .groupby(['usecase','timeframe'])
                                           .apply(self.get_patience_usecase)
                                           .reset_index(level=[0, 1], drop=True)
                                           .sort_index())

    def get_patience_usecase(self, group):
        usecase = group.name[0]
        timeframe = group.name[1]
        patience = pd.to_timedelta(self.commodity_system.usecases.loc[usecase, (timeframe, 'patience')],
                                   unit='hour')
        return pd.DataFrame({'patience_primary': [patience] * len(group)}, index=group.index)

    def sample_demand_total_day(self, row):
        """
        Sample total demand for one day. Method is to be used in a groupby call.
        """
        return np.round(self.rng.lognormal(row['mu'], row['sigma'])).astype(int)

    def sample_idle(self):
        """
        Sample idle times from lognormal distribution for each request.
        """
        self.requests['dtime_idle'] = None
        self.requests['dtime_idle'] = (self.requests
                                       .groupby(['usecase', 'timeframe'])['dtime_idle']
                                       .transform(self.sample_idle_usecase))

    def sample_idle_usecase(self, group):
        """
        Sample idle times for one usecase and timeframe. Method is to be used in a groupby call.
        """
        usecase = group.name[0]
        timeframe = group.name[1]
        uc_idle_mean = self.commodity_system.usecases.loc[usecase, (timeframe, 'idle_mean')]
        uc_idle_stdev = self.commodity_system.usecases.loc[usecase, (timeframe, 'idle_std')]
        p1, p2 = utils.lognormal_params(uc_idle_mean, uc_idle_stdev)
        idle = pd.to_timedelta(self.rng.lognormal(p1, p2, len(group)), unit='hour')
        return pd.Series(idle, index=group.index)

    def sample_requests(self):
        """
        Sample total daily demand within simulation time and sample request time based on usececase and timeframe
        """
        # sample daily total demand numbers
        daily_total = pd.DataFrame(index=pd.to_datetime(np.unique(self.scenario.dti_sim_extd.date)))
        daily_total['timeframe'], daily_total['demand_mean'], daily_total['demand_std'] = \
            self.mapper_timeframe.map_timeframes(daily_total, self.commodity_system.name)
        daily_total['mu'], daily_total['sigma'] = utils.lognormal_params(daily_total['demand_mean'],
                                                                         daily_total['demand_std'])
        daily_total['demand'] = daily_total.apply(lambda row: self.sample_demand_total_day(row), axis=1)

        # sample request data (excluding time)
        self.requests['date'] = pd.to_datetime(np.repeat(daily_total.index, daily_total['demand']))
        self.requests['year'] = self.requests['date'].dt.year
        self.requests['month'] = self.requests['date'].dt.month
        self.requests['day'] = self.requests['date'].dt.day
        self.requests['timeframe'] = daily_total.loc[self.requests['date'], 'timeframe'].values
        self.requests['usecase'] = None
        self.requests['usecase'] = self.requests.groupby('timeframe')['usecase'].transform(self.sample_usecases)

        # Sample request times
        self.requests['hour'] = (self.requests.groupby(['usecase', 'timeframe'])
                                 .apply(self.sample_req_times).reset_index(level=[0,1],drop=True).sort_index())
        self.requests['time_req'] = pd.to_datetime(self.requests[['year', 'month', 'day', 'hour']])
        self.requests.drop(['date', 'year', 'month', 'day', 'hour'], inplace=True, axis=1)
        self.requests['time_req'] = self.requests['time_req'].dt.tz_localize(self.scenario.timezone,
                                                                             ambiguous='NaT',  # fall
                                                                             nonexistent='shift_forward')  # spring
        self.requests.dropna(axis='index', subset=['time_req'], inplace=True)

    def sample_req_times(self,group):
        usecase = group.name[0]
        timeframe = group.name[1]

        # always sample finer than timestep to avoid rounding errors
        time_vals = np.arange(start=0, stop=24, step=self.scenario.timestep_hours / 100)

        mag1 = self.commodity_system.usecases.loc[usecase, (timeframe, 'dep1_magnitude')]
        mean1 = np.median([self.commodity_system.usecases.loc[usecase, (timeframe, 'dep1_time_mean')], 0, 24])
        std1 = np.max([self.commodity_system.usecases.loc[usecase, (timeframe, 'dep1_time_std')], 1e-8])
        cdf1_vals = sp.stats.norm.cdf(time_vals, mean1, std1)

        mag2 = self.commodity_system.usecases.loc[usecase, (timeframe, 'dep2_magnitude')]
        mean2 = np.median([self.commodity_system.usecases.loc[usecase, (timeframe, 'dep2_time_mean')], 0, 24])
        std2 = np.max([self.commodity_system.usecases.loc[usecase, (timeframe, 'dep2_time_std')], 1e-8])
        cdf2_vals = sp.stats.norm.cdf(time_vals, mean2, std2)

        cdf_vals = mag1 * cdf1_vals + mag2 * cdf2_vals
        # Generate n uniform random numbers between 0 and 1
        uniform_samples = np.random.rand(len(group))
        # Interpolate to find the samples
        time_samples = np.interp(uniform_samples, cdf_vals, time_vals)
        # round to timestep
        time_samples = np.round(time_samples / self.scenario.timestep_hours) * self.scenario.timestep_hours
        return pd.DataFrame(data=time_samples, index=group.index)

    def sample_usecases(self,group):
        return pd.Series(np.random.choice(self.commodity_system.usecases.index.values,
                                          size=len(group),
                                          replace=True,
                                          p=self.commodity_system.usecases.loc[:, (group.name, 'rel_prob_norm')]),
                         index=group.index)

    def save_data(self):
        """
        Save the request data to a csv file.
        """
        requests_path = os.path.join(
            self.scenario.run.path_result_dir,
            f'{self.scenario.run.runtimestamp}_'
            f'{self.scenario.run.scenario_file_name}_'
            f'{self.scenario.name}_'
            f'{self.commodity_system.name}_'
            f'demand.csv')
        self.requests.to_csv(requests_path)


class BatteryCommodityDemand(CommodityDemand):

    def __init__(self, scenario, commodity_system):
        super().__init__(scenario, commodity_system)

    def calc_time_active(self):
        """
        Calculate active time for each request
        """
        self.requests['dtime_active'] = (self.requests
                                          .groupby(['usecase', 'timeframe'])
                                          .apply(self.calc_time_active_usecase)
                                          .reset_index(level=[0, 1], drop=True)
                                          .sort_index())

    def calc_time_active_usecase(self, group):
        """
        Calculate active time for one usecase and timeframe. Method is to be used in a groupby call.
        """
        usecase = group.name[0]
        timeframe = group.name[1]
        power = self.commodity_system.usecases.loc[usecase, (timeframe, 'power_avg')]
        dtime_active = pd.to_timedelta(group['energy_req'] / power, unit='hour')
        return pd.Series(dtime_active, index=group.index)

    def sample(self):
        self.sample_requests()
        self.sample_energy()
        self.sample_idle()
        self.calc_time_active()
        self.get_patience()

        if self.scenario.run.save_des_results:
            self.save_data()

        return self.requests

    def sample_energy(self):
        """
        Sample energy requirements for each request
        """
        self.requests['energy_req'] = None
        self.requests['energy_req'] = (self.requests
                                       .groupby(['usecase', 'timeframe'])['energy_req']
                                       .transform(self.sample_energy_usecase))

    def sample_energy_usecase(self, group):
        """
        Sample energy requirements for one usecase and timeframe. Method is to be used in a groupby call.
        """
        usecase = group.name[0]
        timeframe = group.name[1]
        uc_energy_mean = self.commodity_system.usecases.loc[usecase, (timeframe, 'energy_mean')]
        uc_energy_stdev = self.commodity_system.usecases.loc[usecase, (timeframe, 'energy_std')]
        p1, p2 = utils.lognormal_params(uc_energy_mean, uc_energy_stdev)
        dist = self.rng.lognormal(p1, p2, len(group))
        return pd.Series(dist, index=group.index)


class VehicleCommodityDemand(CommodityDemand):

    def __init__(self, scenario, commodity_system):
        super().__init__(scenario, commodity_system)

    def calc_time_energy(self):
        """
        Calculate active time and energy requirements for each request
        """
        self.requests['dtime_active'] = pd.to_timedelta(self.requests['distance'] / self.requests['speed_avg'],
                                                         unit='hour')
        self.requests['energy_req'] = self.requests['distance'] * self.requests['consumption']

    def get_consumption_speed(self):
        """
        Get consumption and speed values for each request from the usecase file. These depend on the
        usecase and timeframe.
        """
        self.requests[['consumption', 'speed_avg']] = (self.requests.groupby(['usecase', 'timeframe'])
                                                        .apply(self.get_consumption_speed_usecase)
                                                        .reset_index(level=[0, 1], drop=True)
                                                        .sort_index())

    def get_consumption_speed_usecase(self,group):
        """
        Get consumption and speed values for one usecase and timeframe. Method is to be used in a groupby call.
        """
        usecase = group.name[0]
        timeframe = group.name[1]
        consumption = self.commodity_system.usecases.loc[usecase, (timeframe, 'consumption')]
        speed_avg = self.commodity_system.usecases.loc[usecase, (timeframe, 'speed_avg')]
        return pd.DataFrame(data={'consumption': [consumption] * len(group),
                                  'speed_avg': [speed_avg] * len(group)},
                            index=group.index)

    def sample(self):
        self.sample_requests()
        self.sample_distance()
        self.get_consumption_speed()
        self.sample_idle()
        self.calc_time_energy()
        self.get_patience()

        if self.scenario.run.save_des_results:
            self.save_data()

        return self.requests

    def sample_distance(self):
        """
        Sample distances for each request within a VehicleCommoditySystem
        """
        self.requests['distance'] = np.nan  # distance column is only used for VehicleRentalSystems
        self.requests['distance'] = (self.requests.groupby(
            ['usecase', 'timeframe'])['distance'].transform(self.sample_distance_usecase))

    def sample_distance_usecase(self, group):
        """
        Sample distances for one usecase and timeframe. Method is to be used in a groupby call.
        """
        usecase = group.name[0]
        timeframe = group.name[1]
        uc_dist_mean = self.commodity_system.usecases.loc[usecase, (timeframe, 'dist_mean')]
        uc_dist_stdev = self.commodity_system.usecases.loc[usecase, (timeframe, 'dist_std')]
        p1, p2 = utils.lognormal_params(uc_dist_mean, uc_dist_stdev)
        dist = self.rng.lognormal(p1, p2, len(group))
        return pd.Series(dist, index=group.index)
