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
    def __init__(self, store, amount):
        if amount <= 0:
            raise ValueError('amount(=%s) must be > 0.' % amount)
        self.amount = amount
        """The amount of matter to be taken out of the store."""

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
                                               'time_wait',
                                               'steps_wait',
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
                                               'step_reavail'
                                               'energy_avail',
                                               'energy_req',
                                               'rex_num',
                                               'soc_return',
                                               'process_obj',
                                               'commodity',
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
        self.base_range = (1 - self.cs.min_return_soc) * self.cs.dep_soc * self.cs.size_pc / self.cs.consumption

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

        if self.cs.rex_sys:  # system can extend range. otherwise self.rex_sys is None
            self.processes['rex_distance'] = np.max(0, self.processes['distance'] - self.base_range)
            self.processes['rex_energy'] = self.rex_distance * self.cs.consumption
            self.processes['rex_num'] = math.ceil(self.rex_energy / self.cs.rex_rs.size_pc)
            self.processes['energy_avail'] = ((self.cs.size_pc * self.cs.dep_soc) +
                                              (self.processes['rex_num'] * self.cs.rex_rs.size * self.cs.rex_rs.dep_soc))
            self.processes['rex_request'] = self.processes['rex_num'] > 0
        else:
            self.processes['energy_avail'] = self.cs.size_pc * self.cs.dep_soc
            self.processes['rex_request'] = False

        # set maximum energy requirement to max available energy
        self.processes['energy_req'] = np.minimum(self.processes['energy_req'], self.processes['energy_avail'])
        self.processes['dsoc_req'] = self.processes['energy_req'] / self.processes['energy_avail']
        self.processes['time_recharge'] = pd.to_timedelta(self.processes['energy_req'] /
                                                          (self.cs.chg_pwr * self.cs.chg_eff), unit='hour')



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
                                                         p=rel_prob_norm)
        self.processes['usecase_name'] = self.processes.apply(lambda row: self.usecases.loc[row['usecase_idx'], 'name'],
                                                              axis=1)
        self.assign_datetime_request(process_num, sc)

        p1, p2 = lognormal_params(self.cs.soc_return_mean, self.cs.soc_return_stdev)
        self.processes['soc_return'] = np.minimum(1, np.random.lognormal(p1, p2, process_num))
        self.processes['num_bat'] = self.processes.apply(lambda row: self.usecases.loc[row['usecase_idx'], 'num_bat'],
                                                         axis=1)
        self.processes['energy_avail'] = self.processes['num_bat'] * self.cs.dep_soc * self.cs.size_pc

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

    def __init__(self, data, rs, sc):

        self.data = data

        # wait until time of request is reached
        yield sc.des_env.timeout(data['time_req'])  # todo time_req is a datetime - is conversion needed?

        # request resource(s) from store
        with rs.store.get() as self.request:
            self.result = yield self.request | sc.des_env.timeout(rs.cs.patience)

            # request granted
            if self.request in self.result:
                rs.processes.loc[data.index, 'time_dep'] = sc.des_env.now

                if data['rex_request']:  # range extension required & available
                    self.request_rex_commodities(rs, sc.des_env)
                    if self.rex_sucess:
                        self.execute_rental()  # vehicle
                        #self.()  # rex batteries
                    else:
                        # log failure
                        pass
                else:  # range extension not required or available
                    self.execute_rental()  #vehicle only

            # vehicle request not granted, patience is triggered
            else:
                # log failure
                if self.request.triggered:
                    # make sure that resource is placed back in store
                    resource = yield self.request
                    rs.rex_system.store.put(resource)  # todo why can we not just put the car back every time (like finally)?
            rs.store.put(self.result[self.request])  # todo this would be my approach

    def execute_rental(self, rs: RentalSystem, cs: blocks.CommoditySystem, sc):

        # cover the usage & idle time
        yield sc.des_env.timeout(rs.processes['time_rental'])   # todo does this work with timedelta?
        self.processes.loc[self.data['index'], 'time_return'] = sc.des_env.now  # todo alternative time_dep + time_rental?

        # cover the recharge time incl. buffer
        yield sc.des_env.timeout(rs.processes['time_blocked'])   # todo does this work with timedelta?
        self.processes.loc[self.data['index'], 'time_reavail'] = sc.des_env.now  # todo alternative time_dep + time_rental + time_blocked?

    def request_rex_commodities(self,
                            rs: RentalSystem,
                            cs: blocks.CommoditySystem,
                            env: simpy.Environment):

        self.rex_distance = self.distance - self.range
        self.rex_energy = self.rex_distance * cs.consumption
        self.rex_num = math.ceil(self.rex_energy / rs.rex_system.cs.size)
        self.energy_avail = (cs.size * cs.departure_soc) + (self.rex_num * rs.rex_system.cs.size * rs.rex_system.cs.departure_soc)
        self.soc_return = self.energy_req / self.energy_avail

        with rs.rex_system.store.get(self.rex_num) as self.rex_request:
            self.rex_result = yield self.rex_request | env.timeout(rs.rex_system.cs.patience)
            # todo distiguish between self patience (for vehicles) and rex patience (for rex batteries)

            self.rex_waittime = env.now - self.request_time

            if self.rex_request in self.rex_result:  # request was granted
                self.rex_time_departure = env.now
                yield env.timeout(self.duration.hours)  # todo hours doesnt exist
                self.rex_time_return = env.now
                self.rex_time_recharge = dt.timedelta(hours=self.energy_req / cs.chg_pwr)
                self.rex_time_buffer = dt.timedelta(hours=2)
                yield env.timeout(self.time_recharge.hours + self.time_buffer)  # todo hours doesnt exist
                self.rex_time_reavail = env.now
                yield rs.rex_system.store.put(self.rex_result[self.rex_request])  # put batteries back
                self.rex_sucess = True
            else:  # request for rex batteries not granted  -> abort trip

                self.rex_sucess = False


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
    sc.rental_systems = []
    for commodity_system in [block for block in sc.blocks.values() if isinstance(block, blocks.CommoditySystem)]:
        # todo make a better decision which commodity systems to initiate as VehicleRentalSystems and which as BatteryRentalSystems
        if commodity_system.name == 'bev':
            rs = VehicleRentalSystem(sc.des_env, sc, commodity_system)
        else:
            rs = BatteryRentalSystem(sc.des_env, sc, commodity_system)
        sc.rental_systems.append(rs)

    # Create additional range extension (rex) processes from vehicle rental systems in battery rental systems
    # only feasible after all rental systems have been created
    for vrs in [rs for rs in sc.rental_systems if isinstance(rs, VehicleRentalSystem)]:
        pass

    # generate individual RentalProcess instances for every pregenerated process
    for rs in sc.rental_systems:
        for idx, row in rs.processes.iterrows():
            process_info = row.to_dict()
            process = RentalProcess(process_info)
            rs.processes.loc[row, 'process_obj'] = process

    # run the discrete event simulation
    sc.des_env.run()

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
    scenario = sim.Scenario('brs_only', run)
    execute_des(scenario)
