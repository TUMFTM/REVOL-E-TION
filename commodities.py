#!/usr/bin/env python3
"""
commodities.py

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

import datetime as dt
import logging
import os
import math
import simpy
import numpy as np
import pandas as pd

import blocks
import simulation as sim

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
        self.cs = cs  # making cs callable through RentalSystem
        self.sc = sc  # making scenario callable through RentalSystem
        # all other values are accessible through the commodity system self.cs

        self.rng = np.random.default_rng()

        # buffer time is added onto minimum recharge time to ensure dispatch feasibility and give room for energy mgmt
        self.time_buffer = pd.Timedelta(hours=2)

        # draw total demand for every day from lognormal distribution
        self.daily_demand = pd.DataFrame(index=np.unique(self.sc.sim_dti.date))
        p1, p2 = lognormal_params(self.cs.daily_mean, self.cs.daily_stdev)
        self.daily_demand['num_total'] = np.round(self.rng.lognormal(p1, p2, self.daily_demand.shape[0])).astype(int)

        self.processes = pd.DataFrame(columns=['time_req',
                                               'step_req',
                                               'time_dep',
                                               'step_dep',
                                               'time_active',
                                               'time_idle',
                                               'time_rental',
                                               'steps_rental',
                                               'time_return',
                                               'step_return',
                                               'time_recharge',
                                               'time_blocked',
                                               'steps_blocked',
                                               'time_reavail',
                                               'step_reavail',
                                               'energy_avail',
                                               'energy_req',
                                               'soc_return',
                                               'process_obj',
                                               'status'])

        self.generate_demand(sc)  # child function, see subclasses

        # common calculations for both types of RentalSystem
        self.processes['time_rental'] = self.processes['time_active'] + self.processes['time_idle']
        self.processes['steps_rental'] = dt2steps(self.processes['time_rental'], sc)
        self.processes['time_blocked'] = self.processes['time_recharge'] + self.time_buffer
        self.processes['steps_blocked'] = dt2steps(self.processes['time_blocked'], sc)

    def assign_datetime_request(self, process_num, sc):
        self.processes['date'] = np.repeat(self.daily_demand.index, self.daily_demand['num_total'])
        self.processes['year'] = self.processes['date'].apply(get_year)
        self.processes['month'] = self.processes['date'].apply(get_month)
        self.processes['day'] = self.processes['date'].apply(get_day)
        self.processes['hour'] = (np.round(self.draw_departure_samples(process_num) / self.sc.timestep_hours) *
                                  self.sc.timestep_hours)  # round to nearest timestep
        self.processes['time_req'] = pd.to_datetime(self.processes[['year', 'month', 'day', 'hour']])
        self.processes['step_req'] = dt2steps(self.processes['time_req'], sc)
        self.processes.drop(['date', 'year', 'month', 'day', 'hour'], inplace=True, axis=1)

    def convert_process_log(self):
        """
        This function converts the process based log from DES execution into a time based log as required by the
        energy system model as an input
        :return: None
        """
        # taken from Logging.convert_to_csv:
        #
        # h = 0
        # while h < global_count:
        #
        #     # filter out failed trips for the ind_array
        #     if process_log[h][3] != "failed":
        #
        #         # which column of process log contains which info
        #         departure_timestep_log = process_log[h][3]
        #         leaving_SOC_log = process_log[h][4]
        #         used_charge_log = process_log[h][5]
        #         return_timestep_log = process_log[h][6]
        #         used_Car_log = process_log[h][8]
        #
        #         if process_log[h][8] != 0:  # check if  used car log is empty
        #
        #             for k in used_Car_log:  # for every car that was used by a trip:
        #
        #                 j = (departure_timestep_log - 1)
        #                 while (departure_timestep_log - 1) <= j <= (return_timestep_log + 1):
        #
        #                     # one timestep before rental set SoC
        #                     if j == departure_timestep_log - 1:
        #                         ind_array[j][0 + k * 3] += leaving_SOC_log
        #
        #                     # timestep of rental: remove battery capacity from minigrid
        #                     if j == departure_timestep_log:
        #                         ind_array[j][1 + k * 3] += used_charge_log
        #                         ind_array[j][2 + k * 3] = 0  # .0
        #
        #                     # during rental: set availability to 0
        #                     if departure_timestep_log < j <= return_timestep_log:
        #                         ind_array[j][2 + k * 3] = 0  # .0
        #
        #                     j += 1
        #
        #     h += 1
        pass

    def save_logs(self):
        """
        This function saves the converted log dataframe as a suitable input csv file for the energy system model.
        The resulting dataframe can also be returned to the energy system model directly in addition for faster
        delivery through execute_des.
        :return: None
        """
        # todo save dataframes directly after conversion

        # Taken from Logging.save():

        # # Save process log
        # Simulation_Log = pd.DataFrame(process_log,
        #                               columns=['Day', 'usage', 'day_Count', 'departure_timestep', 'leaving_SOC',
        #                                        'used_charge', 'return_timestep', 'chargetime', 'used_'f'{name}'])
        #
        # save_filename = os.path.join(os.getcwd(), 'input', f'{name}', f'{name}_process_log.csv')
        #
        # # print(Simulation_Log.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        # Simulation_Log.to_csv(save_filename, sep=';')
        #
        # # Save oemof compatible csv file
        # ind_bev_df = pd.DataFrame(ind_array)
        #
        # for i in range(0, fleet_capacity):
        #     ind_bev_df.rename(columns={i * 3: f'{name}{i}_minsoc',
        #                                i * 3 + 1: f'{name}{i}_consumption',
        #                                i * 3 + 2: f'{name}{i}_atbase'}, inplace=True)
        #
        # save_filename = os.path.join(os.getcwd(), 'input', f'{name}', f'{name}_log.csv')
        # # print(ind_bev_df.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        # ind_bev_df.to_csv(save_filename, sep=';')
        pass


class VehicleRentalSystem(RentalSystem):

    def __init__(self, env: simpy.Environment, sc, cs):

        # replace the rex system name with the actual CommoditySystem object
        if cs.rex_cs:
            if isinstance(sc.blocks[cs.rex_cs], blocks.BatteryCommoditySystem):
                cs.rex_cs = sc.blocks[cs.rex_cs]
            else:
                logging.error(f'Selected range extender system {cs.rex_cs.name} for VehicleCommoditySystem'
                              f' {cs.name} is not a BatteryCommoditySystem')
                exit()  # todo better exit strategy not ending the entire run

        super().__init__(cs, sc)

        self.store = simpy.Store(env, capacity=self.cs.num)  # does not need to be a MultiStore
        for commodity in self.cs.commodities:
            self.store.put(commodity.name)

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
        self.base_range = (1 - self.cs.soc_min_return) * self.cs.soc_dep * self.cs.size_pc / self.cs.consumption

        # create array of processes (daily number drawn before) and draw time of departure from custom function
        process_num = self.daily_demand['num_total'].sum(axis=0)
        self.processes['date'] = np.repeat(self.daily_demand.index, self.daily_demand['num_total'])

        self.assign_datetime_request(process_num, sc)

        # draw requested distance and time values, calculate energy used
        p1, p2 = lognormal_params(self.cs.dist_mean, self.cs.dist_stdev)
        self.processes['distance'] = self.rng.lognormal(p1, p2, process_num)
        self.processes['time_active'] = pd.to_timedelta((self.processes['distance'] / self.cs.speed_avg), unit='hour')
        p1, p2 = lognormal_params(self.cs.idle_mean, self.cs.idle_stdev)
        self.processes['time_idle'] = pd.to_timedelta(self.rng.lognormal(p1, p2, process_num), unit='hour')
        self.processes['energy_req'] = self.processes['distance'] * self.cs.consumption

        if self.cs.rex_cs:  # system can extend range. otherwise self.rex_cs is None
            self.processes['distance_rex'] = np.maximum(0, self.processes['distance'] - self.base_range)
            self.processes['energy_rex'] = self.processes['distance_rex'] * self.cs.consumption
            self.processes['num_rex'] = np.ceil(self.processes['energy_rex'] / self.cs.rex_cs.size_pc)
            self.processes['energy_avail'] = ((self.cs.size_pc * self.cs.soc_dep) +
                                              (self.processes['num_rex'] * self.cs.rex_cs.size * self.cs.rex_cs.soc_dep))
            self.processes['rex_request'] = self.processes['num_rex'] > 0
        else:
            self.processes['energy_avail'] = self.cs.size_pc * self.cs.soc_dep
            self.processes['rex_request'] = False

        # set maximum energy requirement to max available energy
        self.processes['energy_req'] = np.minimum(self.processes['energy_req'], self.processes['energy_avail'])
        self.processes['dsoc_req'] = self.processes['energy_req'] / self.processes['energy_avail']
        self.processes['time_recharge'] = pd.to_timedelta(self.processes['energy_req'] /
                                                          (self.cs.chg_pwr * self.cs.chg_eff), unit='hour')

    def generate_rex_demand(self, sc):

        rex_processes = self.processes.loc[self.processes['rex_request'], :].copy()

        rex_processes['usecase_idx'] = -1
        rex_processes['usecase_name'] = self.cs.name + '_' + rex_processes.index.to_series().astype(str)
        rex_processes['energy_req'] = rex_processes['energy_rex']
        rex_processes['num_bat'] = rex_processes['num_rex']
        rex_processes['energy_req_pc'] = rex_processes['energy_req'] / rex_processes['num_bat']
        rex_processes['soc_return'] = self.cs.rex_cs.soc_dep - rex_processes['energy_req_pc'] / self.cs.rex_cs.size_pc
        rex_processes.drop('num_rex', axis=1, inplace=True)
        self.cs.rex_cs.rs.processes = pd.concat([self.cs.rex_cs.rs.processes, rex_processes],
                                                axis=0,
                                                join='inner')
        pass


class BatteryRentalSystem(RentalSystem):

    def __init__(self, env: simpy.Environment, sc, cs):

        self.usecase_file_path = os.path.join(os.getcwd(), 'input', 'brs', 'brs_usecases.json')
        self.usecases = pd.read_json(self.usecase_file_path, orient='records', lines=True)

        super().__init__(cs, sc)

        self.store = MultiStore(env, capacity=cs.num)
        for commodity in cs.commodities:
            self.store.put([commodity.name])

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


        process_num = self.daily_demand['num_total'].sum(axis=0)
        rel_prob_norm = self.usecases['rel_prob'] / self.usecases['rel_prob'].sum(axis=0)



        self.processes['usecase_idx'] = np.random.choice(self.usecases.index.values,
                                                         process_num,
                                                         replace=True,
                                                         p=rel_prob_norm).astype(int)
        self.processes['usecase_name'] = self.processes.apply(lambda row: self.usecases.loc[row['usecase_idx'], 'name'],
                                                              axis=1)
        self.assign_datetime_request(process_num, sc)

        p1, p2 = lognormal_params(self.cs.soc_return_mean, self.cs.soc_return_stdev)
        self.processes['soc_return'] = np.minimum(1, np.random.lognormal(p1, p2, process_num))
        self.processes['num_bat'] = self.processes.apply(lambda row: self.usecases.loc[row['usecase_idx'], 'num_bat'],
                                                         axis=1).astype(int)
        self.processes['energy_avail'] = self.processes['num_bat'] * self.cs.soc_dep * self.cs.size_pc

        self.processes['time_active'] = pd.to_timedelta(
            self.processes.apply(lambda row:
                                 (1 - row['soc_return']) *
                                 row['energy_avail'] /
                                 self.usecases.loc[row['usecase_idx'],
                                 'power'],
                                 axis=1),
            unit='hour')

        p1, p2 = lognormal_params(self.cs.idle_mean, self.cs.idle_stdev)
        self.processes['time_idle'] = pd.to_timedelta(np.random.lognormal(p1, p2, process_num), unit='hour')

        self.processes['energy_req_pc'] = (1 - self.processes['soc_return']) * self.cs.size_pc
        self.processes['energy_req'] = self.processes['energy_req_pc'] * self.processes['num_bat']

        self.processes['time_recharge'] = pd.to_timedelta(self.processes['energy_req_pc'] /
                                                          (self.cs.chg_pwr * self.cs.chg_eff), unit='hour')


class RentalProcess:

    def __init__(self, idx, data, rs, sc):

        self.data = data
        self.rs = rs
        self.env = sc.des_env
        self.id = idx

        self.env.process(self.define_process())

    def define_process(self):

        # wait for request time
        yield self.env.timeout(self.data['step_req'])

        # request primary resource(s) from (Multi)Store
        with self.rs.store.get() as self.primary_request:
            self.primary_result = yield self.primary_request | self.env.timeout(self.rs.cs.patience)

        # Dummy values to make result checks easily possible for all possible constellations
        self.secondary_request = False
        self.secondary_result = [False]

        # request secondary resources from other MultiStore
        if isinstance(self.rs, blocks.VehicleCommoditySystem):
            if self.data['rex_request']:  # column does not exist for BatteryCommoditySystems
                self.secondary_request = self.rs.cs.rex_cs.rs.store.get(self.data['rex_num'])
                self.secondary_result = yield self.secondary_request | self.env.timeout(self.rs.cs.rex_patience)

        if (self.primary_request in self.primary_result) and (self.secondary_request in self.secondary_result):

            self.rs.processes.loc[self.id, 'step_dep'] = self.env.now

            # cover the usage & idle time
            yield self.env.timeout(self.data['steps_rental'])
            self.rs.processes.loc[self.id, 'step_return'] = self.env.now

            # cover the recharge time incl. buffer
            yield self.env.timeout(self.data['steps_blocked'])
            self.rs.processes.loc[self.id, 'step_reavail'] = self.env.now

            self.rs.processes.loc[self.id, 'status'] = 'sucess'
            self.rs.processes.loc[self.id, 'primary_commodity'] = self.primary_request.value
            if self.secondary_request:
                self.rs.processes.loc[self.id, 'secondary_commodity'] = self.secondary_request.value

            self.rs.store.put(self.primary_result[self.primary_request])
            if self.secondary_request:
                self.rs.cs.rex_cs.rs.store.put(self.secondary_result[self.secondary_request])

        else:  # either or both (primary/secondary) request(s) unsuccessful

            self.rs.processes.loc[self.id, 'status'] = 'failure'

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
    if pd.api.types.is_datetime64_any_dtype(series):
        out = np.round((series - sc.starttime) / sc.timestep_td).astype(int)
    elif pd.api.types.is_timedelta64_dtype(series):
        out = np.round(series / sc.timestep_td).astype(int)
    return out


def steps2dt(series, sc, absolute=False):
    out = pd.to_timedelta(series * sc.timestep_hours, unit='hour')
    if absolute:
        out += sc.starttime
    return out


def execute_des(sc):

    # define a DES environment
    sc.des_env = simpy.Environment()

    # create rental systems (including stochastic pregeneration of individual rental processes)
    sc.rental_systems = dict()
    for commodity_system in [system for system in sc.commodity_systems if system.filename == 'run_des']:
        if isinstance(commodity_system, blocks.VehicleCommoditySystem):
            commodity_system.rs = VehicleRentalSystem(sc.des_env, sc, commodity_system)
        elif isinstance(commodity_system, blocks.BatteryCommoditySystem):
            commodity_system.rs = BatteryRentalSystem(sc.des_env, sc, commodity_system)
        sc.rental_systems[commodity_system.name] = commodity_system.rs

    # generate individual RentalProcess instances for every pregenerated process
    for rs in sc.rental_systems.values():
        for idx, row in rs.processes.iterrows():
            # VehicleRentalSystem RentalProcesses can init additional processes in BatteryRentalSystems at runtime
            process = RentalProcess(idx, row, rs, sc)
            rs.processes.loc[idx, 'process_obj'] = process

    # run the discrete event simulation
    sc.des_env.run()

    for rs in sc.rental_systems.values():
        rs.processes['time_dep'] = steps2dt(rs.processes['step_dep'], sc, absolute=True)
        rs.processes['time_return'] = steps2dt(rs.processes['step_return'], sc, absolute=True)
        rs.processes['time_reavail'] = steps2dt(rs.processes['step_reavail'], sc, absolute=True)

    # save logging results
    for rs in sc.rental_systems:
        rs.convert_process_log()
        # todo implement trigger on whether to even save the .csv file as it is not needed for direct coupling to the ESM
        rs.save_logs()

    resource_logs = {rs.name: rs.resource_log for rs in sc.rental_systems}
    return resource_logs


def lognormal_params(mean, stdev):
    mu = np.log(mean ** 2 / math.sqrt((mean ** 2) + (stdev ** 2)))
    sig = math.sqrt(np.log(1 + (stdev ** 2) / (mean ** 2)))
    return mu, sig


def get_day(element):
    return element.day


def get_month(element):
    return element.month


def get_year(element):
    return element.year


###############################################################################
# Execution (only if run as standalone file)
###############################################################################

if __name__ == '__main__':
    run = sim.SimulationRun()
    scenario = sim.Scenario('both', run)
    execute_des(scenario)
