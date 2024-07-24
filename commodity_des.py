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

import os
import math
import simpy
import numpy as np
import pandas as pd
import pytz
import scipy as sp

import blocks
import mapper_timeframe as mtf


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


class RentalSystem:

    def __init__(self, cs, sc):

        self.cs = cs
        self.sc = sc
        self.name = self.cs.name

        self.rng = np.random.default_rng()

        # calculate usable energy to expect
        self.dsoc_usable_high = self.cs.soc_target_high - self.cs.soc_return
        self.dsoc_usable_low = self.cs.soc_target_low - self.cs.soc_return
        self.energy_usable_pc_high = self.dsoc_usable_high * self.cs.size_pc
        self.energy_usable_pc_low = self.dsoc_usable_low * self.cs.size_pc

        self.n_processes = self.processes = self.demand_daily = self.store = None
        self.use_rate = self.fail_rate = None

    def create_store(self):
        self.store = MultiStore(self.sc.env_des, capacity=self.cs.num)
        for commodity in self.cs.commodities.values():
            self.store.put([commodity.name])

    def calc_performance_metrics(self):

        self.use_rate = dict()
        steps_total = self.data.shape[0]
        # make an individual row for each used commodity in a process
        processes_exploded = self.processes.explode('commodities_primary')

        # calculate percentage of DES (not sim, the latter is shorter) time
        # occupied by active, idle, recharge and buffer times
        for commodity in list(self.cs.commodities.keys()):
            processes = processes_exploded.loc[processes_exploded['commodities_primary'] == commodity, :]
            steps_blocked = processes['steps_charge_primary'].sum() + processes['steps_rental'].sum()
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
            column_names.extend([(commodity,'atbase'), (commodity,'dsoc'), (commodity,'consumption'),
                                 (commodity,'atac'), (commodity,'atdc')])
            if isinstance(self, VehicleRentalSystem):
                column_names.extend([(commodity,'tour_dist')])
        column_index = pd.MultiIndex.from_tuples(column_names, names=['time', 'time'])

        # Initialize dataframe for time based log
        self.data = pd.DataFrame(data=0, index=self.sc.dti_des, columns=column_index)
        for col, dtype in {(com, col): ('bool' if 'at' in col else 'float') for com, col in column_index}.items():
            self.data[col] = self.data[col].astype(dtype)
        self.data.loc[:, (slice(None), 'atbase')] = True
        self.data.loc[:, (slice(None), 'atac')] = False
        self.data.loc[:, (slice(None), 'atdc')] = False

        for process in [row for _, row in self.processes.iterrows() if row['status'] == 'success']:
            for commodity in process['commodities_primary']:
                # Set Availability at base for charging
                self.data.loc[process['time_dep']:(process['time_return'] - self.sc.timestep_td),
                (commodity, 'atbase')] = False

                # set consumption power as constant while rented out
                self.data.loc[process['time_dep']:(process['time_return'] - self.sc.timestep_td),
                (commodity, 'consumption')] = (process['energy_req_pc_primary'] /
                                               (process['steps_rental'] * self.sc.timestep_hours))

                # Set minimum SOC at departure makes sure that only vehicles with at least that SOC are rented out
                self.data.loc[process['time_dep'], (commodity, 'dsoc')] = process['dsoc_primary']

                if isinstance(self, VehicleRentalSystem):
                    # set distance in first timestep of rental (for distance based revenue calculation)
                    self.data.loc[process['time_dep'], (commodity, 'tour_dist')] = process['distance']

        self.cs.data = self.data

    def generate_processes(self):

        self.processes = pd.DataFrame(columns=['timeframe',
                                               'usecase',
                                               'time_preblock_primary', 'step_preblock_primary',
                                               'time_preblock_secondary', 'step_preblock_secondary',
                                               'time_req', 'step_req', 'dayofweek_req', 'status',
                                               'commodities_primary', 'commodities_secondary',
                                               'time_dep', 'step_dep',
                                               'dtime_active',
                                               'dtime_idle',
                                               'dtime_rental', 'steps_rental',
                                               'time_return', 'step_return',
                                               'dtime_charge_primary', 'steps_charge_primary',
                                               'dtime_charge_secondary', 'steps_charge_secondary',
                                               'time_reavail_primary', 'step_reavail_primary',
                                               'time_reavail_secondary', 'step_reavail_secondary',
                                               'energy_total_both', 'energy_usable_both', 'energy_req_both',
                                               'rex_request', 'num_primary', 'num_secondary',
                                               'dsoc_primary', 'dsoc_secondary',
                                               'energy_req_pc_primary', 'energy_req_pc_secondary'])
        self.sample_requests()
        self.sample_request_data()

        # common calculations for both types of RentalSystem
        self.processes['dtime_rental'] = self.processes['dtime_active'] + self.processes['dtime_idle']
        self.processes['steps_rental'] = dt2steps(self.processes['dtime_rental'], self.sc)

        self.processes['steps_charge_primary'] = dt2steps(self.processes['dtime_charge_primary'], self.sc)
        self.processes['steps_charge_secondary'] = dt2steps(self.processes['dtime_charge_secondary'], self.sc)

        self.block_charge_time()  # block charge time pre-rental for vehicles

        self.processes.sort_values(by='time_preblock_primary', inplace=True, ignore_index=True)

    def sample_requests(self):

        # draw total demand for every simulated day from a lognormal distribution
        self.demand_daily = pd.DataFrame(index=pd.to_datetime(np.unique(self.sc.dti_sim.date)))
        self.demand_daily['timeframe'], self.demand_daily['demand_mean'], self.demand_daily['demand_std'] = mtf.map_timeframes(self.demand_daily, self.cs.name)
        self.demand_daily['mu'], self.demand_daily['sigma'] = lognormal_params(self.demand_daily['demand_mean'], self.demand_daily['demand_std'])

        def sample_demand(row):
            return np.round(self.rng.lognormal(row['mu'], row['sigma'])).astype(int)
        self.demand_daily['demand'] = self.demand_daily.apply(lambda row: sample_demand(row), axis=1)

        self.n_processes = self.demand_daily['demand'].sum(axis=0)

        self.processes['date'] = pd.to_datetime(np.repeat(self.demand_daily.index, self.demand_daily['demand']))
        self.processes['year'] = self.processes['date'].dt.year
        self.processes['month'] = self.processes['date'].dt.month
        self.processes['day'] = self.processes['date'].dt.day
        self.processes['timeframe'] = self.demand_daily.loc[self.processes['date'], 'timeframe'].values

        def sample_usecases(group):
            return pd.Series(np.random.choice(self.cs.usecases.index.values,
                                              size=len(group),
                                              replace=True,
                                              p=self.cs.usecases.loc[:, (group.name, 'rel_prob_norm')]),
                             index=group.index)
        self.processes['usecase'] = self.processes.groupby('timeframe')['usecase'].transform(sample_usecases)

        time_vals = np.arange(start=0, stop=24, step=self.sc.timestep_hours / 100)  # always sample finer than timestep

        def sample_req_times(group):
            usecase = group.name[0]
            timeframe = group.name[1]

            mag1 = self.cs.usecases.loc[usecase, (timeframe, 'dep1_magnitude')]
            mean1 = np.median([self.cs.usecases.loc[usecase, (timeframe, 'dep1_time_mean')], 0, 24])
            std1 = np.max([self.cs.usecases.loc[usecase, (timeframe, 'dep1_time_std')], 1e-8])
            cdf1_vals = sp.stats.norm.cdf(time_vals, mean1, std1)

            mag2 = self.cs.usecases.loc[usecase, (timeframe, 'dep2_magnitude')]
            mean2 = np.median([self.cs.usecases.loc[usecase, (timeframe, 'dep2_time_mean')], 0, 24])
            std2 = np.max([self.cs.usecases.loc[usecase, (timeframe, 'dep2_time_std')], 1e-8])
            cdf2_vals = sp.stats.norm.cdf(time_vals, mean2, std2)

            cdf_vals = mag1 * cdf1_vals + mag2 * cdf2_vals
            uniform_samples = np.random.rand(len(group))  # Generate n uniform random numbers between 0 and 1
            time_samples = np.interp(uniform_samples, cdf_vals, time_vals)  # Interpolate to find the samples
            time_samples = np.round(time_samples / self.sc.timestep_hours) * self.sc.timestep_hours  # round to timestep
            return pd.DataFrame(data=time_samples, index=group.index)

        self.processes['hour'] = (self.processes.groupby(['usecase', 'timeframe'])
                                  .apply(sample_req_times).reset_index(level=[0,1],drop=True).sort_index())

        self.processes['time_req'] = pd.to_datetime(self.processes[['year', 'month', 'day', 'hour']])
        self.processes.drop(['date', 'year', 'month', 'day', 'hour'], inplace=True, axis=1)

        self.processes['time_req'] = self.processes['time_req'].dt.tz_localize(self.sc.timezone,
                                                                               ambiguous='NaT',  # fall
                                                                               nonexistent='shift_forward')  # spring
        self.processes.dropna(axis='index', subset=['time_req'], inplace=True)

        self.n_processes = self.processes.shape[0]  # update n_processes after possibly dropping NaT values

        self.processes['step_req'] = dt2steps(self.processes['time_req'], self.sc)

    def save_data(self, path):
        """
        This function saves the converted log dataframe as a suitable input csv file for the energy system model.
        The resulting dataframe can also be handed to the energy system model directly in addition for faster
        delivery through execute_des.
        """
        processes_path = os.path.join(path, f'{self.sc.name}_{self.cs.name}_processes.csv')
        self.processes.to_csv(processes_path)

        log_path = os.path.join(path, f'{self.sc.name}_{self.cs.name}_log.csv')
        self.data.to_csv(log_path)


class VehicleRentalSystem(RentalSystem):

    def __init__(self, cs, sc):

        super().__init__(cs, sc)

        # replace the rex system name read in from scenario file with the actual CommoditySystem object
        if self.cs.rex_cs:
            self.check_rex_inputs()
            cs.rex_cs = sc.blocks[cs.rex_cs]

        self.range_usable_high = self.energy_usable_pc_high / self.cs.consumption
        self.range_usable_low = self.energy_usable_pc_low / self.cs.consumption

        self.dsoc_usable_rex_high = self.cs.rex_cs.soc_target_high - self.cs.rex_cs.soc_return if self.cs.rex_cs else 0
        self.dsoc_usable_rex_low = self.cs.rex_cs.soc_target_low - self.cs.rex_cs.soc_return if self.cs.rex_cs else 0

        self.energy_usable_rex_pc_high = self.dsoc_usable_rex_high * self.cs.rex_cs.size_pc if self.cs.rex_cs else 0
        self.energy_usable_rex_pc_low = self.dsoc_usable_rex_low * self.cs.rex_cs.size_pc if self.cs.rex_cs else 0

        self.generate_processes()
        self.create_store()

    def block_charge_time(self):
        self.processes['step_preblock_primary'] = self.processes['step_req'] - self.processes['steps_charge_primary']
        self.processes['time_preblock_primary'] = steps2dt(self.processes['step_preblock_primary'],
                                                           self.sc,
                                                           absolute=True)

    def check_rex_inputs(self):
        if self.cs.rex_cs not in self.sc.blocks.keys():
            message = (f'Selected range extender system \"{self.cs.rex_cs}\" for VehicleCommoditySystem'
                       f' \"{self.cs.name}\" in scenario \"{self.sc.name}\" does not exist')
            self.sc.exception = message
            raise ValueError(message)
        elif not isinstance(self.sc.blocks[self.cs.rex_cs], blocks.BatteryCommoditySystem):
            message = (f'Selected range extender system \"{self.cs.rex_cs}\" for VehicleCommoditySystem'
                       f' \"{self.cs.name}\" in scenario \"{self.sc.name}\" is not a BatteryCommoditySystem')
            self.sc.exception = message
            raise ValueError(message)
        elif not self.sc.blocks[self.cs.rex_cs].filename == 'run_des':
            message = (f'Selected range extender system \"{self.cs.rex_cs}\" for VehicleCommoditySystem'
                       f' \"{self.cs.name}\" in scenario \"{self.sc.name}\" is not set to run DES itself')
            self.sc.exception = message
            raise ValueError(message)

    def sample_request_data(self):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        :return: None
        """
        self.processes['num_primary'] = 1  # always one vehicle per rental process
        self.processes['distance'] = np.nan  # distance column is only used for VehicleRentalSystems

        # sample distance and active time
        def draw_usecase_distance(group):
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_dist_mean = self.cs.usecases.loc[usecase, (timeframe, 'dist_mean')]
            uc_dist_stdev = self.cs.usecases.loc[usecase, (timeframe, 'dist_std')]
            p1, p2 = lognormal_params(uc_dist_mean, uc_dist_stdev)
            dist = self.rng.lognormal(p1, p2, len(group))
            return pd.Series(dist, index=group.index)
        self.processes['distance'] = self.processes.groupby(['usecase',
                                                             'timeframe'])['distance'].transform(draw_usecase_distance)

        self.processes['dtime_active'] = pd.to_timedelta((self.processes['distance'] / self.cs.speed_avg), unit='hour')
        self.processes['energy_req_both'] = self.processes['distance'] * self.cs.consumption

        # sample idle time
        def draw_usecase_idle(group):
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_idle_mean = self.cs.usecases.loc[usecase, (timeframe, 'idle_mean')]
            uc_idle_stdev = self.cs.usecases.loc[usecase, (timeframe, 'idle_std')]
            p1, p2 = lognormal_params(uc_idle_mean, uc_idle_stdev)
            idle = pd.to_timedelta(self.rng.lognormal(p1, p2, len(group)), unit='hour')
            return pd.Series(idle, index=group.index)
        self.processes['dtime_idle'] = self.processes.groupby(['usecase',
                                                               'timeframe'])['dtime_idle'].transform(draw_usecase_idle)

        if self.cs.rex_cs:  # system can extend range
            # determine number of rex needed to cover missing distance and calc available energy
            self.processes['distance_missing'] = np.maximum(0, self.processes['distance'] - self.range_usable)
            self.processes['energy_missing'] = self.processes['distance_missing'] * self.cs.consumption
            self.processes['num_secondary'] = np.ceil(self.processes['energy_missing'] /
                                                      self.energy_usable_rex_pc).astype(int)
            self.processes['rex_request'] = self.processes['num_secondary'] > 0

            self.processes['energy_usable_both'] = (self.energy_usable_pc +
                                                    (self.processes['num_secondary'] * self.energy_usable_rex_pc))
            self.processes['energy_total_both'] = (self.cs.size_pc +
                                                   (self.processes['num_secondary'] * self.cs.rex_cs.size_pc))

        else:  # no rex defined
            self.processes['num_secondary'] = 0
            self.processes['rex_request'] = False

            self.processes['energy_usable_both'] = self.energy_usable_pc
            self.processes['energy_total_both'] = self.cs.size_pc
            # for non-rex systems, dsoc_primary is clipped to max usable dSOC (equivalent to external charging)
            self.processes['energy_req_both'] = self.processes['energy_req_both'].clip(upper=self.energy_usable_pc)


        # calculate different delta SOC for primary and secondary commodity due to different start SOCs (linear)
        self.processes['dsoc_primary'] = (self.dsoc_usable * self.processes['energy_req_both'] /
                                          self.processes['energy_usable_both'])
        self.processes['dsoc_secondary'] = (self.dsoc_usable_rex * self.processes['energy_req_both'] /
                                            self.processes['energy_usable_both']) * self.processes['rex_request']

        self.processes['energy_req_pc_primary'] = self.processes['dsoc_primary'] * self.cs.size_pc
        self.processes['dtime_charge_primary'] = pd.to_timedelta(self.processes['energy_req_pc_primary'] /
                                                                 (self.cs.pwr_chg * self.cs.eff_chg),
                                                                 unit='hour')

        if self.cs.rex_cs:
            self.processes['energy_req_pc_secondary'] = self.processes['dsoc_secondary'] * self.cs.rex_cs.size_pc
            self.processes['dtime_charge_secondary'] = pd.to_timedelta(self.processes['energy_req_pc_secondary'] /
                                                                       (self.cs.rex_cs.pwr_chg *
                                                                        self.cs.rex_cs.eff_chg),
                                                                       unit='hour')
        else:
            self.processes['energy_req_pc_secondary'] = 0
            self.processes['dtime_charge_secondary'] = None


    def transfer_rex_processes(self, sc):
        """
        This function takes processes requiring REX from the VehicleRentalSystem and adds them to the target
        BatteryRentalSystem's processes dataframe as these don't originate from the latter's demand pregeneration
        and are not logged there yet.
        """
        mask = (self.processes['status'] == 'success') & (self.processes['rex_request'])
        rex_processes = self.processes.loc[mask, :].copy()

        # convert values for target BatteryRentalSystem
        rex_processes['usecase_id'] = -1
        rex_processes['usecase_name'] = f'rex_{self.cs.name}'

        def swap_primary_secondary(col_name):
            if 'primary' in col_name:
                return col_name.replace('primary', 'secondary')
            elif 'secondary' in col_name:
                return col_name.replace('secondary', 'primary')
            return col_name
        rex_processes.columns = [swap_primary_secondary(col) for col in rex_processes.columns]

        # drop all secondary columns
        #rex_processes.drop([col for col in rex_processes.columns if 'secondary' in col], axis=1, inplace=True)
        rex_processes['time_preblock_primary'] = rex_processes['time_req']
        rex_processes['step_preblock_primary'] = rex_processes['step_req']

        # add rex processes to end of target BatteryRentalSystem's processes dataframe and create new sorted index
        self.cs.rex_cs.rs.processes = pd.concat([self.cs.rex_cs.rs.processes, rex_processes], join='inner')
        self.cs.rex_cs.rs.processes.sort_values(by='time_preblock_primary', inplace=True, ignore_index=True)


class BatteryRentalSystem(RentalSystem):

    def __init__(self, cs, sc):

        super().__init__(cs, sc)

        self.cs.usecases['rel_prob_norm'] = self.cs.usecases['rel_prob'] / self.cs.usecases['rel_prob'].sum(axis=0)

        self.cs.rex_cs = None  # needs to be set for later check

        self.generate_processes()
        self.create_store()

    def block_charge_time(self):
        self.processes['step_preblock_primary'] = self.processes['step_req']
        self.processes['time_preblock_primary'] = self.processes['time_req']

    def sample_request_data(self):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        :return: None
        """
        self.processes['rex_request'] = False

        p1, p2 = lognormal_params(self.cs.soc_return_mean, self.cs.soc_return_stdev)
        self.processes['dsoc_primary'] = (self.cs.soc_dep -
                                          np.clip(np.random.lognormal(p1, p2, self.n_processes),
                                                  self.cs.soc_return_min,
                                                  self.cs.soc_dep))
        self.processes['num_primary'] = self.processes.apply(
            lambda row: self.cs.usecases.loc[row['usecase_id'], 'num_bat'], axis=1).astype(int)

        self.processes['energy_total_both'] = self.processes['num_primary'] * self.cs.size_pc
        self.processes['energy_usable_both'] = self.processes['num_primary'] * self.energy_usable_pc
        self.processes['energy_req_both'] = self.processes['dsoc_primary'] * self.processes['energy_total_both']

        self.processes['energy_req_pc_primary'] = self.processes['dsoc_primary'] * self.cs.size_pc
        self.processes['energy_req_primary'] = self.processes['energy_req_pc_primary'] * self.processes['num_primary']

        self.processes['dtime_active'] = pd.to_timedelta(self.processes.apply(
            lambda row: row['dsoc_primary'] * row['energy_usable_both'] /
                        self.cs.usecases.loc[row['usecase_id'], 'power'],
            axis=1),
            unit='hour')

        p1, p2 = lognormal_params(self.cs.idle_mean, self.cs.idle_stdev)
        self.processes['dtime_idle'] = pd.to_timedelta(np.random.lognormal(p1, p2, self.n_processes), unit='hour')

        self.processes['dtime_charge_primary'] = pd.to_timedelta(self.processes['energy_req_pc_primary'] /
                                                                 (self.cs.pwr_chg * self.cs.eff_chg), unit='hour')


class RentalProcess:

    def __init__(self, id, data, rs, sc):

        self.data = data
        self.rs = rs
        self.env = sc.env_des
        self.id = id

        self.primary_result = self.secondary_result = [False]  # defaults equal to insuccessful request
        self.primary_request = self.secondary_request = False

        # initiate the simpy process function (define_process is not executed here, but only when the env is run)
        self.env.process(self.define_process())

    def define_process(self):

        # wait until start of preblock time
        yield self.env.timeout(self.data['step_preblock_primary'])

        # request primary resource(s) from (Multi)Store
        with self.rs.store.get(self.data['num_primary']) as self.primary_request:
            self.primary_result = yield self.primary_request | self.env.timeout(self.rs.cs.patience)

        # wait until actual request time (leaving time to charge vehicle)
        if isinstance(self.rs, VehicleRentalSystem):
            yield self.env.timeout(self.data['steps_charge_primary'])

        # request secondary resources from other MultiStore
        if self.data['num_secondary'] > 0:
            with self.rs.cs.rex_cs.rs.store.get(self.data['num_secondary']) as self.secondary_request:
                self.secondary_result = yield self.secondary_request | self.env.timeout(self.rs.cs.patience_rex)

        # if request(s) successful
        if (self.primary_request in self.primary_result) and (self.secondary_request in self.secondary_result):
            self.rs.processes.loc[self.id, 'status'] = 'success'
            self.rs.processes.at[self.id, 'commodities_primary'] = self.primary_request.value
            if self.secondary_request:
                self.rs.processes.at[self.id, 'commodities_secondary'] = self.secondary_request.value
            self.rs.processes.loc[self.id, 'step_dep'] = self.env.now

            # cover the usage & idle time
            yield self.env.timeout(self.data['steps_rental'])
            self.rs.processes.loc[self.id, 'step_return'] = self.env.now

            # cover the recharge time for BatteryCommoditySystems
            if isinstance(self.rs, VehicleRentalSystem):
                self.rs.processes.loc[self.id, 'step_reavail_primary'] = self.env.now
                if self.secondary_request:
                    yield self.env.timeout(self.data['steps_charge_secondary'])
                    self.rs.processes.loc[self.id, 'step_reavail_secondary'] = self.env.now

            elif isinstance(self.rs, BatteryRentalSystem):
                yield self.env.timeout(self.data['steps_charge_primary'])
                self.rs.processes.loc[self.id, 'step_reavail_primary'] = self.env.now

            else:
                raise ValueError('Insert description here')  # todo description

            self.rs.store.put(self.primary_result[self.primary_request])
            if self.secondary_request:
                self.rs.cs.rex_cs.rs.store.put(self.secondary_result[self.secondary_request])

        else:  # either or both (primary/secondary) request(s) unsuccessful

            # log type of failure
            if ((self.primary_request not in self.primary_result)
                    and (self.secondary_request not in self.secondary_result)):
                self.rs.processes.loc[self.id, 'status'] = 'failure_both'
            elif (self.primary_request not in self.primary_result):
                self.rs.processes.loc[self.id, 'status'] = 'failure_primary'
            elif (self.secondary_request not in self.secondary_result):
                self.rs.processes.loc[self.id, 'status'] = 'failure_secondary'

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
    ref_time = sc.starttime - pd.Timedelta(days=1)

    if pd.api.types.is_datetime64_any_dtype(series):
        # ensure that the result is at least 1, as 0 would leave no time for any action in real life
        out = np.maximum(1, np.ceil((series - ref_time) / sc.timestep_td).astype(int))

    elif pd.api.types.is_timedelta64_dtype(series):
        out = np.maximum(1, np.ceil(series / sc.timestep_td).astype(int))

    return out


def steps2dt(series, sc, absolute=False):
    ref_time = sc.starttime - pd.Timedelta(days=1)
    out = pd.to_timedelta(series * sc.timestep_hours, unit='hour')
    if absolute:
        out += ref_time
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
    for cs in [cs for cs in sc.commodity_systems.values() if cs.data_source == 'des']:
        if isinstance(cs, blocks.VehicleCommoditySystem):
            commodity_system.rs = VehicleRentalSystem(cs, sc)
        elif isinstance(cs, blocks.BatteryCommoditySystem):
            commodity_system.rs = BatteryRentalSystem(cs, sc)
        sc.rental_systems[cs.name] = cs.rs

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
            rs.save_data(path)


def lognormal_params(mean, stdev):
    mu = np.log(mean ** 2 / np.sqrt((mean ** 2) + (stdev ** 2)))
    sig = np.sqrt(np.log(1 + (stdev ** 2) / (mean ** 2)))
    return mu, sig



