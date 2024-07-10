#!/usr/bin/env python3
"""
commodity_des.py

--- Description ---
This file contains the Discrete Event Simulation to generate behavioral data of mobile commodity systems in the
energy system model framework. For more info, see readme.

--- Created by ---
Hannes Henglein

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Imports
###############################################################################

import os
import math
import simpy
import numpy as np
import pandas as pd
import pytz

import blocks

###############################################################################
# Simpy MultiStore Subclass Definitions
###############################################################################


class MultiStoreGet(simpy.resources.base.Get):
    def __init__(self, store, num_resources=1):
        if num_resources <= 0:
            raise ValueError(f'Number of resources taken from MultiStore ({num_resources} given)'
                             f' must be larger than or equal to zero.')
        self.amount = num_resources
        """The number of resources to be taken out of the store."""

        super(MultiStoreGet, self).__init__(store)


class MultiStorePut(simpy.resources.base.Put):
    def __init__(self, store, items):
        self.items = items
        super(MultiStorePut, self).__init__(store)


class MultiStore(simpy.resources.base.BaseResource):
    def __init__(self, env, capacity=float('inf')):
        if capacity <= 0:
            raise ValueError('"capacity" must be > 0.')

        super(MultiStore, self).__init__(env, capacity)

        self.items = []

    put = simpy.core.BoundClass(MultiStorePut)
    get = simpy.core.BoundClass(MultiStoreGet)

    def _do_put(self, event):
        if len(self.items) + len(event.items) <= self._capacity:
            self.items.extend(event.items)
            event.succeed()

    def _do_get(self, event):
        if self.items and event.amount <= len(self.items):
            elements = self.items[(len(self.items) - event.amount - 0):]
            self.items = self.items[:(len(self.items) - event.amount - 0)]
            event.succeed(elements)


###############################################################################
# Main DES classes
###############################################################################

class RentalSystem:

    def __init__(self, cs, sc):

        self.name = cs.name
        self.cs = cs  # making parent CommoditySystem callable through RentalSystem
        self.sc = sc  # making scenario callable through RentalSystem
        # all other values are accessible through the commodity system self.cs

        self.rng = np.random.default_rng()

        # buffer time is added onto minimum recharge time to ensure dispatch feasibility and give room for energy mgmt
        self.time_buffer = pd.Timedelta(hours=0)

        # draw total demand for every simulated ay from lognormal distribution
        self.demand_daily = pd.DataFrame(index=np.unique(self.sc.dti_sim.date))
        p1, p2 = lognormal_params(self.cs.daily_mean, self.cs.daily_stdev)
        self.demand_daily['num_total'] = np.round(self.rng.lognormal(p1, p2, self.demand_daily.shape[0])).astype(int)

        self.processes = pd.DataFrame(columns=['time_req',
                                               'step_req',
                                               'energy_req_total',
                                               'rex_request',
                                               'time_dep',
                                               'step_dep',
                                               'time_active',
                                               'time_idle',
                                               'time_rental',
                                               'steps_rental',
                                               'time_return',
                                               'step_return',
                                               'time_recharge_primary',
                                               'time_recharge_secondary',
                                               'time_blocked_primary',
                                               'time_blocked_secondary',
                                               'steps_blocked_primary',
                                               'steps_blocked_secondary',
                                               'time_reavail_primary',
                                               'time_reavail_secondary',
                                               'step_reavail_primary',
                                               'step_reavail_secondary',
                                               'energy_avail',
                                               'energy_req_pc_primary',
                                               'energy_req_pc_secondary',
                                               'primary_commodity',
                                               'secondary_commodity',
                                               'process_obj',
                                               'status'],
                                      dtype='object')

        self.processes['rex_request'] = False  # default value

        self.generate_demand(sc)  # child function, see subclasses

        # common calculations for both types of RentalSystem
        self.processes['time_rental'] = self.processes['time_active'] + self.processes['time_idle']
        self.processes['steps_rental'] = dt2steps(self.processes['time_rental'], sc)
        self.processes['time_blocked_primary'] = self.processes['time_recharge_primary'] + self.time_buffer
        self.processes['steps_blocked_primary'] = dt2steps(self.processes['time_blocked_primary'], sc)

        self.processes['time_blocked_secondary'] = self.processes[ 'time_recharge_secondary'] + self.time_buffer
        self.processes['steps_blocked_secondary'] = dt2steps(self.processes['time_blocked_secondary'], sc)

        # create the actual Simpy store and populate it with commodities
        self.store = MultiStore(sc.env_des, capacity=cs.num)
        for commodity in cs.commodities.values():
            self.store.put([commodity.name])

    def assign_datetime_request(self, n_processes, sc):
        self.processes['date'] = np.repeat(self.demand_daily.index, self.demand_daily['num_total'])
        self.processes['year'] = self.processes['date'].apply(get_year)
        self.processes['month'] = self.processes['date'].apply(get_month)
        self.processes['day'] = self.processes['date'].apply(get_day)
        self.processes['hour'] = (np.round(self.draw_departure_samples(n_processes) / self.sc.timestep_hours) *
                                  self.sc.timestep_hours)  # round to nearest timestep

        self.processes['time_req'] = pd.to_datetime(self.processes[['year', 'month', 'day', 'hour']])

        # Vectorized handling of time conversion to avoid slow for loop
        try:
            self.processes['time_req'] = self.processes['time_req'].dt.tz_localize(sc.timezone)

        # If DST causes a sampled time to be invalid, loop over all processes (much slower)
        # and advance the problematic one
        except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError):  # todo still fails to catch NonExistentTimeError
            for idx, process in self.processes.iterrows():
                try:
                    self.processes.iloc[idx, 'time_req'] =\
                        pd.to_datetime(self.processes.iloc[idx,['year', 'month', 'day', 'hour']])
                    self.processes.iloc[idx, 'time_req'] =\
                        self.processes.iloc[idx, 'time_req'].tz_localize(sc.timezone)
                except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError):
                    self.processes.iloc[idx, 'time_req'] =\
                         pd.to_datetime(self.processes.iloc[idx, ['year', 'month', 'day', 'hour']] +\
                         pd.Timedelta(hours=1))
                    self.processes.iloc[idx, 'time_req'] =\
                        self.processes.iloc[idx, 'time_req'].tz_localize(sc.timezone)

        self.processes['step_req'] = dt2steps(self.processes['time_req'], sc)
        self.processes.drop(['date', 'year', 'month', 'day', 'hour'], inplace=True, axis=1)

    def calc_performance_metrics(self):

        self.use_rate = dict()
        steps_total = self.data.shape[0]
        # make an individual row for each used commodity in a process
        processes_exploded = self.processes.explode('primary_commodity')

        # calculate percentage of DES (not sim, the latter is shorter) time
        # occupied by active, idle, recharge and buffer times
        for commodity in list(self.cs.commodities.keys()):
            processes = processes_exploded.loc[processes_exploded['primary_commodity'] == commodity, :]
            steps_blocked = processes['steps_blocked_primary'].sum() + processes['steps_rental'].sum()
            self.use_rate[commodity] = steps_blocked / steps_total
        self.cs.use_rate = np.mean(list(self.use_rate.values()))

        # calculate overall percentage of failed trips
        n_success = self.processes.loc[self.processes['status'] == 'success', 'status'].shape[0]
        n_total = self.processes.shape[0]
        self.fail_rate = self.cs.fail_rate = 1 - (n_success / n_total)
        pass

    def convert_process_log(self):
        """
        This function converts the process based log from DES execution into a time based log for each commodity
        as required by the energy system model as an input
        """

        commodities = list(self.cs.commodities.keys())
        column_names = []
        for commodity in commodities:
            column_names.extend([(commodity,'atbase'), (commodity,'minsoc'), (commodity,'consumption'),
                                 (commodity,'atac'), (commodity,'atdc')])
            if isinstance(self, VehicleRentalSystem):
                column_names.extend([(commodity,'tour_dist')])
        column_index = pd.MultiIndex.from_tuples(column_names, names=['time', 'time'])

        # Initialize dataframe for time based log
        self.data = pd.DataFrame(0, index=self.sc.dti_des, columns=column_index)
        self.data.loc[:, (slice(None), 'atbase')] = True
        self.data.loc[:, (slice(None), 'atac')] = False
        self.data.loc[:, (slice(None), 'atdc')] = False

        for process in [row for _, row in self.processes.iterrows() if row['status'] == 'success']:
            for commodity in process['primary_commodity']:
                # Set Availability at base for charging
                self.data.loc[process['time_dep']:(process['time_return'] - self.sc.timestep_td),
                (commodity, 'atbase')] = False

                # set consumption power as constant while rented out
                self.data.loc[process['time_dep']:(process['time_return'] - self.sc.timestep_td),
                (commodity, 'consumption')] = (process['energy_req_pc_primary'] /
                                               (process['steps_rental'] * self.sc.timestep_hours))

                # Set minimum SOC at departure makes sure that only vehicles with at least that SOC are rented out
                self.data.loc[process['time_dep'], (commodity, 'minsoc')] = self.cs.soc_dep

                if isinstance(self, VehicleRentalSystem):
                    # set distance in first timestep of rental (for distance based revenue calculation)
                    self.data.loc[process['time_dep'], (commodity, 'tour_dist')] = process['distance']

        self.cs.data = self.data

    def save_data(self, path, sc):
        """
        This function saves the converted log dataframe as a suitable input csv file for the energy system model.
        The resulting dataframe can also be handed to the energy system model directly in addition for faster
        delivery through execute_des.
        """
        processes_path = os.path.join(path, f'{sc.name}_{self.cs.name}_processes.csv')
        self.processes.to_csv(processes_path)

        log_path = os.path.join(path, f'{sc.name}_{self.cs.name}_log.csv')
        self.data.to_csv(log_path)


class VehicleRentalSystem(RentalSystem):

    def __init__(self, env: simpy.Environment, sc, cs):

        # replace the rex system name read in from scenario json with the actual CommoditySystem object
        if cs.rex_cs:
            if cs.rex_cs not in sc.blocks.keys():
                message = (f'Selected range extender system \"{cs.rex_cs}\" for VehicleCommoditySystem'
                           f' \"{cs.name}\" in scenario \"{sc.name}\" does not exist')
                sc.exception = message
                raise ValueError(message)
            elif not isinstance(sc.blocks[cs.rex_cs], blocks.BatteryCommoditySystem):
                message = (f'Selected range extender system \"{cs.rex_cs}\" for VehicleCommoditySystem'
                           f' \"{cs.name}\" in scenario \"{sc.name}\" is not a BatteryCommoditySystem')
                sc.exception = message
                raise ValueError(message)
            elif not sc.blocks[cs.rex_cs].filename == 'run_des':
                message = (f'Selected range extender system \"{cs.rex_cs}\" for VehicleCommoditySystem'
                           f' \"{cs.name}\" in scenario \"{sc.name}\" is not set to run DES itself')
                sc.exception = message
                raise ValueError(message)
            else:  # everything is fine
                cs.rex_cs = sc.blocks[cs.rex_cs]

        super().__init__(cs, sc)

    def departure_pdf(self, x):
        """
        for a given time of day (given as a float of full hours), this function returns the probability of a trip
        request.
        :param x: time of day as float of full hours
        :return: y: probability value
        """
        y = 0.6 * (1 / (self.cs.dep_stdev1 * np.sqrt(2 * np.pi)) * np.exp(- (x - self.cs.dep_mean1) ** 2 /
                                                                          (2 * self.cs.dep_stdev1 ** 2))) + \
            0.4 * (1 / (self.cs.dep_stdev2 * np.sqrt(2 * np.pi)) * np.exp(- (x - self.cs.dep_mean2) ** 2 /
                                                                          (2 * self.cs.dep_stdev2 ** 2)))
        return y

    def departure_cdf(self):
        """
        Calculate the cumulative distribution function (CDF) based on the probability density function (PDF)
        You can integrate the PDF from -infinity to x, or use numerical integration methods like np.cumsum
        Here, we use numerical integration via np.cumsum for simplicity:
        :return:
        """
        steps_per_day = int(24 / self.sc.timestep_hours)
        x_vals = np.linspace(0, 24, num=steps_per_day)
        y_vals = [self.departure_pdf(val) for val in x_vals]
        cdf_vals = np.cumsum(y_vals)
        cdf_vals /= cdf_vals[-1]  # Normalize the CDF to [0, 1]
        return x_vals, cdf_vals

    def draw_departure_samples(self, n):
        """
        Draw n samples from the departure_pdf using the inverse transform sampling method via the cumulativ function
        :param n: number of samples
        :return:
        """
        x_vals, cdf_vals = self.departure_cdf()
        uniform_samples = np.random.rand(int(n))  # Generate n uniform random numbers between 0 and 1
        samples = np.interp(uniform_samples, cdf_vals, x_vals)  # Interpolate to find the samples
        return samples

    def generate_demand(self, sc):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        :return: None
        """
        # calculate base range on internal battery for rex calculations
        self.dsoc_base = self.cs.soc_dep - self.cs.soc_return_min
        self.energy_base = self.dsoc_base * self.cs.size_pc  # VRS always has just one primary commodity (vehicle)
        self.range_base = self.energy_base / self.cs.consumption

        # create array of processes (daily number drawn before) and draw time of departure from custom function
        n_processes = self.demand_daily['num_total'].sum(axis=0)
        self.processes['date'] = np.repeat(self.demand_daily.index, self.demand_daily['num_total'])

        self.assign_datetime_request(n_processes, sc)

        # draw requested distance and time values, calculate energy used
        p1, p2 = lognormal_params(self.cs.dist_mean, self.cs.dist_stdev)
        self.processes['distance'] = self.rng.lognormal(p1, p2, n_processes)
        self.processes['time_active'] = pd.to_timedelta((self.processes['distance'] / self.cs.speed_avg), unit='hour')
        p1, p2 = lognormal_params(self.cs.idle_mean, self.cs.idle_stdev)
        self.processes['time_idle'] = pd.to_timedelta(self.rng.lognormal(p1, p2, n_processes), unit='hour')
        self.processes['energy_req_total'] = self.processes['distance'] * self.cs.consumption

        if self.cs.rex_cs:  # system can extend range. otherwise self.rex_cs is None

            self.dsoc_base_rex = self.cs.rex_cs.soc_dep - self.cs.soc_return_min
            self.energy_base_rex_pc = self.cs.rex_cs.size_pc * self.dsoc_base_rex

            # determine number of rex needed to cover missing distance and calc available energy
            self.processes['distance_missing'] = np.maximum(0, self.processes['distance'] - self.range_base)
            self.processes['energy_missing'] = self.processes['distance_missing'] * self.cs.consumption
            self.processes['num_rex'] = np.ceil(self.processes['energy_missing'] / self.energy_base_rex_pc).astype(int)
            self.processes['rex_request'] = self.processes['num_rex'] > 0
            self.processes['energy_avail'] = self.energy_base + (self.processes['num_rex'] * self.energy_base_rex_pc)
            self.processes['energy_total'] = self.cs.size_pc + (self.processes['num_rex'] * self.cs.rex_cs.size_pc)

            # calculate different delta SOC for primary and secondary commodity due to different start SOCs (linear)
            self.processes['dsoc_primary'] = (self.dsoc_base * self.processes['energy_req_total'] /
                                              self.processes['energy_avail'])
            self.processes['dsoc_secondary'] = (self.dsoc_base_rex * self.processes['energy_req_total'] /
                                                self.processes['energy_avail'])

            self.processes['energy_req_pc_primary'] = self.processes['dsoc_primary'] * self.cs.size_pc
            self.processes['energy_req_pc_secondary'] = self.processes['dsoc_secondary'] * self.cs.rex_cs.size_pc

            self.processes['time_recharge_primary'] = pd.to_timedelta(self.processes['energy_req_pc_primary'] /
                                                                     (self.cs.pwr_chg * self.cs.eff_chg),
                                                                      unit='hour')
            self.processes['time_recharge_secondary'] = pd.to_timedelta(self.processes['energy_req_pc_secondary'] /
                                                                       (self.cs.rex_cs.pwr_chg *
                                                                        self.cs.rex_cs.eff_chg),
                                                                        unit='hour')

        else:  # no rex defined
            self.processes['num_rex'] = 0
            self.processes['energy_avail'] = self.energy_base
            self.processes['energy_total'] = self.cs.size_pc

            # cap energy request at available energy (equivalent to external charging)
            self.processes['energy_req_total'] = np.minimum(self.processes['energy_req_total'],
                                                            self.processes['energy_avail'])
            self.processes['dsoc_primary'] = (self.dsoc_base * self.processes['energy_req_total'] /
                                              self.processes['energy_avail'])

            self.processes['energy_req_pc_primary'] = self.processes['dsoc_primary'] * self.cs.size_pc
            self.processes['time_recharge_primary'] = pd.to_timedelta(self.processes['energy_req_pc_primary'] /
                                                                      (self.cs.pwr_chg * self.cs.eff_chg),
                                                                      unit='hour')

    def transfer_rex_processes(self, sc):
        """
        This function takes processes requiring REX from the VehicleRentalSystem and adds them to the target
        BatteryRentalSystem's processes dataframe as these don't originate from the latter's demand pregeneration
        and are not logged there yet.
        """
        mask = (self.processes['status'] == 'success') & (self.processes['rex_request'])
        rex_processes = self.processes.loc[mask, :].copy()

        # convert values for target BatteryRentalSystem
        rex_processes['usecase_idx'] = -1
        rex_processes['usecase_name'] = f'rex_{self.cs.name}'
        rex_processes['num_resources'] = rex_processes['num_rex']
        rex_processes['soc_return'] = self.cs.rex_cs.soc_dep - rex_processes['dsoc_secondary']

        # set secondary recharge & reavailability times as primary values for BatteryRentalSystem
        rex_processes['energy_req_pc_primary'] = rex_processes['energy_req_pc_secondary']
        rex_processes['time_recharge_primary'] = rex_processes['time_recharge_secondary']
        rex_processes['time_blocked_primary'] = rex_processes['time_recharge_primary'] + self.time_buffer
        rex_processes['steps_blocked_primary'] = dt2steps(rex_processes['time_blocked_primary'], sc)
        rex_processes['time_reavail_primary'] = rex_processes['time_reavail_secondary']
        rex_processes['step_reavail_primary'] = rex_processes['step_reavail_secondary']

        # swap primary and secondary commodities as target system has other promary commodity type
        rex_processes['primary_commodity'] = rex_processes['secondary_commodity']
        rex_processes['dsoc_primary'] = rex_processes['dsoc_secondary']

        # drop all secondary columns
        rex_processes.drop([col for col in rex_processes.columns if 'secondary' in col], axis=1, inplace=True)

        # add rex processes to end of target BatteryRentalSystem's processes dataframe and create new sorted index
        self.cs.rex_cs.rs.processes = pd.concat([self.cs.rex_cs.rs.processes, rex_processes], join='inner')
        self.cs.rex_cs.rs.processes.sort_values(by='time_req', inplace=True, ignore_index=True)


class BatteryRentalSystem(RentalSystem):

    def __init__(self, env: simpy.Environment, sc, cs):

        self.usecase_file_name = cs.filename_usecases
        self.usecase_file_path = os.path.join(os.getcwd(), 'input', 'brs', f'{self.usecase_file_name}.csv')
        self.usecases = pd.read_csv(self.usecase_file_path, header=0)

        super().__init__(cs, sc)

        self.cs.rex_cs = None  # needs to be set for later check

    def draw_departure_sample(self, row):
        sample = -1  # kicking off the while loop
        while sample > 24 or sample < 0:
            sample = np.random.normal(self.usecases.loc[row['usecase_idx'], 'dep_mean'],
                                      self.usecases.loc[row['usecase_idx'], 'dep_stdev'])
        return sample

    def draw_departure_samples(self, n):
        list = self.processes.apply(self.draw_departure_sample, axis=1)
        return list

    def generate_demand(self, sc):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        :return: None
        """

        n_processes = self.demand_daily['num_total'].sum(axis=0)
        rel_prob_norm = self.usecases['rel_prob'] / self.usecases['rel_prob'].sum(axis=0)

        if not self.usecases.empty:
            self.processes['usecase_idx'] = np.random.choice(self.usecases.index.values,
                                                             n_processes,
                                                             replace=True,
                                                             p=rel_prob_norm).astype(int)
            self.processes['usecase_name'] = self.processes.apply(lambda row: self.usecases.loc[row['usecase_idx'], 'name'],
                                                                  axis=1)
            self.assign_datetime_request(n_processes, sc)

            p1, p2 = lognormal_params(self.cs.soc_return_mean, self.cs.soc_return_stdev)
            self.processes['soc_return'] = np.clip(np.random.lognormal(p1, p2, n_processes),
                                                           self.cs.soc_return_min,
                                                           self.cs.soc_dep)
            self.processes['num_resources'] = self.processes.apply(
                lambda row: self.usecases.loc[row['usecase_idx'], 'num_bat'], axis=1).astype(int)
            self.processes['energy_avail'] = self.processes['num_resources'] * self.cs.soc_dep * self.cs.size_pc

            self.processes['time_active'] = pd.to_timedelta(
                self.processes.apply(lambda row:
                                     (self.cs.soc_dep - row['soc_return']) * row['energy_avail'] /
                                     self.usecases.loc[row['usecase_idx'], 'power'],
                                     axis=1),
                unit='hour')

            p1, p2 = lognormal_params(self.cs.idle_mean, self.cs.idle_stdev)
            self.processes['time_idle'] = pd.to_timedelta(np.random.lognormal(p1, p2, n_processes), unit='hour')

            self.processes['energy_req_pc_primary'] = (self.cs.soc_dep- self.processes['soc_return']) * self.cs.size_pc
            self.processes['energy_req_primary'] = (self.processes['energy_req_pc_primary'] *
                                                    self.processes['num_resources'])

            self.processes['time_recharge_primary'] = pd.to_timedelta(self.processes['energy_req_pc_primary'] /
                                                                      (self.cs.pwr_chg * self.cs.eff_chg), unit='hour')


class RentalProcess:

    def __init__(self, idx, data, rs, sc):

        self.data = data
        self.rs = rs
        self.env = sc.env_des
        self.id = idx

        # initiate the simpy process function (define_process is not executed here, but only when the env. is run)
        self.env.process(self.define_process())

    def define_process(self):

        # wait for request time
        yield self.env.timeout(self.data['step_req'])

        # request primary resource(s) from (Multi)Store
        num_req = self.data['num_resources'] if 'num_resources' in self.data else 1
        with self.rs.store.get(num_req) as self.primary_request:
                self.primary_result = yield self.primary_request | self.env.timeout(self.rs.cs.patience)

        # Dummy values to make result checks easily possible for all possible constellations
        self.secondary_request = False
        self.secondary_result = [False]

        # request secondary resources from other MultiStore
        if isinstance(self.rs, VehicleRentalSystem):
            if self.data['rex_request'] == True:  # BatteryCommoditySystems have nan as value -> explicit True check
                with self.rs.cs.rex_cs.rs.store.get(self.data['num_rex']) as self.secondary_request:
                    self.secondary_result = yield self.secondary_request | self.env.timeout(self.rs.cs.patience_rex)

        if (self.primary_request in self.primary_result) and (self.secondary_request in self.secondary_result):

            self.rs.processes.loc[self.id, 'step_dep'] = self.env.now

            # cover the usage & idle time
            yield self.env.timeout(self.data['steps_rental'])
            self.rs.processes.loc[self.id, 'step_return'] = self.env.now

            # cover the recharge time including safety buffer
            if self.data['rex_request'] == True:  # BatteryCommoditySystems have nan as value -> explicit True check
                if self.data['steps_blocked_primary'] > self.data['steps_blocked_secondary']:
                    yield self.env.timeout(self.data['steps_blocked_secondary'])
                    self.rs.processes.loc[self.id, 'step_reavail_secondary'] = self.env.now
                    yield self.env.timeout(self.data['steps_blocked_primary'] - self.data['steps_blocked_secondary'])
                    self.rs.processes.loc[self.id, 'step_reavail_primary'] = self.env.now
                elif self.data['steps_blocked_primary'] < self.data['steps_blocked_secondary']:
                    yield self.env.timeout(self.data['steps_blocked_primary'])
                    self.rs.processes.loc[self.id, 'step_reavail_primary'] = self.env.now
                    yield self.env.timeout(self.data['steps_blocked_secondary'] - self.data['steps_blocked_primary'])
                    self.rs.processes.loc[self.id, 'step_reavail_secondary'] = self.env.now
                else:  # both blocked times are equal
                    yield self.env.timeout(self.data['steps_blocked_primary'])
                    self.rs.processes.loc[self.id, 'step_reavail_primary'] = self.env.now
                    self.rs.processes.loc[self.id, 'step_reavail_secondary'] = self.env.now
            else:  # no rex request
                yield self.env.timeout(self.data['steps_blocked_primary'])
                self.rs.processes.loc[self.id, 'step_reavail_primary'] = self.env.now

            self.rs.processes.loc[self.id, 'status'] = 'success'
            self.rs.processes.at[self.id, 'primary_commodity'] = self.primary_request.value
            if self.secondary_request:
                self.rs.processes.at[self.id, 'secondary_commodity'] = self.secondary_request.value

            self.rs.store.put(self.primary_result[self.primary_request])
            if self.secondary_request:
                self.rs.cs.rex_cs.rs.store.put(self.secondary_result[self.secondary_request])

        else:  # either or both (primary/secondary) request(s) unsuccessful

            # log type of failure
            if (self.primary_request not in self.primary_result) and (self.secondary_request not in self.secondary_result):
                self.rs.processes.loc[self.id, 'status'] = 'failure - both'
            elif (self.primary_request not in self.primary_result):
                self.rs.processes.loc[self.id, 'status'] = 'failure - primary'
            elif (self.secondary_request not in self.secondary_result):
                self.rs.processes.loc[self.id, 'status'] = 'failure - secondary'

            # make sure resources are put back, see weblink:
            # stackoverflow.com/questions/75371166/simpy-items-in-a-store-
            # disappear-while-modelling-a-carfleet-with-a-simpystore-a
            primary_resource = yield self.primary_request
            self.rs.store.put(primary_resource)
            if self.secondary_request:
                secondary_resource = yield self.secondary_request
                self.rs.cs.rex_cs.rs.store.put(secondary_resource)

###############################################################################
# global functions
###############################################################################


def dt2steps(series, sc):
    out = None  # default value
    if pd.api.types.is_datetime64_any_dtype(series):
        # ensure that the result is at least 1, as 0 would leave no time for any action in real life
        out = np.maximum(1, np.ceil((series - sc.starttime) / sc.timestep_td).astype(int))
    elif pd.api.types.is_timedelta64_dtype(series):
        out = np.maximum(1, np.ceil(series / sc.timestep_td).astype(int))
    return out


def steps2dt(series, sc, absolute=False):
    out = pd.to_timedelta(series * sc.timestep_hours, unit='hour')
    if absolute:
        out += sc.starttime
    return out


def execute_des(sc, save=False, path=None):

    # define a DES environment
    sc.env_des = simpy.Environment()

    # extend datetimeindex to simulate on by some steps to cover any shifts & predictions necessary
    sc.dti_des = sc.dti_sim.union(
        pd.date_range(start=sc.dti_sim[-1] + sc.dti_sim.freq,
                      periods=200,
                      freq=sc.dti_sim.freq))

    # create rental systems (including stochastic pregeneration of individual rental processes)
    sc.rental_systems = dict()
    for commodity_system in [sys for sys in sc.commodity_systems.values() if sys.filename == 'run_des']:
        if isinstance(commodity_system, blocks.VehicleCommoditySystem):
            commodity_system.rs = VehicleRentalSystem(sc.env_des, sc, commodity_system)
        elif isinstance(commodity_system, blocks.BatteryCommoditySystem):
            commodity_system.rs = BatteryRentalSystem(sc.env_des, sc, commodity_system)
        sc.rental_systems[commodity_system.name] = commodity_system.rs

    # generate individual RentalProcess instances for every pregenerated process
    for rs in sc.rental_systems.values():
        for idx, row in rs.processes.iterrows():
            # VehicleRentalSystem RentalProcesses can init additional processes in BatteryRentalSystems at runtime
            process = RentalProcess(idx, row, rs, sc)
            rs.processes.loc[idx, 'process_obj'] = process

    # actually run the discrete event simulation
    sc.env_des.run()

    # reconvert time steps to actual times
    for rs in sc.rental_systems.values():
        rs.processes['time_dep'] = steps2dt(rs.processes['step_dep'], sc, absolute=True)
        rs.processes['time_return'] = steps2dt(rs.processes['step_return'], sc, absolute=True)
        rs.processes['time_reavail_primary'] = steps2dt(rs.processes['step_reavail_primary'], sc, absolute=True)
        rs.processes['time_reavail_secondary'] = steps2dt(rs.processes['step_reavail_secondary'], sc, absolute=True)

    # add additional rex processes from VehicleRentalSystems with rex to BatteryRentalSystems to complete process dataframe
    for rs in [rs for rs in sc.rental_systems.values() if (rs.cs.rex_cs is not None)]:
        rs.transfer_rex_processes(sc)

    # reframe logging results to resource-based view instead of process based (and save)
    for rs in sc.rental_systems.values():
        rs.convert_process_log()
        rs.calc_performance_metrics()
        if save:
            rs.save_data(path, sc)


def lognormal_params(mean, stdev):
    if mean == 0 and stdev == 0:
        return 0, 0  # todo these parameters will not result in a dominant value of 0, but 1
    else:
        mu = np.log(mean ** 2 / math.sqrt((mean ** 2) + (stdev ** 2)))
        sig = math.sqrt(np.log(1 + (stdev ** 2) / (mean ** 2)))
        return mu, sig


def get_day(element):
    return element.day


def get_month(element):
    return element.month


def get_year(element):
    return element.year


# Execution for example file generation
if __name__ == '__main__':
    import simulation as sim
    rn = sim.SimulationRun()
    sc = sim.Scenario('des',rn)
    for rs in sc.rental_systems.values():
        folderpath = os.path.join(os.getcwd(), 'input', rs.cs.name, f'{rs.cs.name}_example.csv')
        rs.save_data(folderpath, sc)
        print(f'{folderpath} created')
