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
# RentalSystem classes
###############################################################################

class RentalSystem:

    def __init__(self, cs, sc):

        self.name = cs.name
        self.cs = cs  # making cs callable through RentalSystem
        self.sc = sc  # making scenario callable through RentalSystem
        # all other values are accessible through the commodity system self.cs

        self.rng = np.random.default_rng()

        self.daily_demand = pd.DataFrame(index=np.unique(self.sc.sim_dti.date))
        self.processes = pd.DataFrame()
        self.generate_demand()  # child function, see subclasses

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

    def assign_datetime_request(self, process_num):
        self.processes['date'] = np.repeat(self.daily_demand.index, self.daily_demand['num_total'])
        self.processes['year'] = self.processes['date'].apply(get_year)
        self.processes['month'] = self.processes['date'].apply(get_month)
        self.processes['day'] = self.processes['date'].apply(get_day)
        self.processes['hour'] = (np.round(self.draw_departure_samples(process_num) / self.sc.timestep_hours) *
                                  self.sc.timestep_hours)  # round to nearest timestep
        self.processes['dt_req'] = pd.to_datetime(self.processes[['year', 'month', 'day', 'hour']])
        self.processes.drop(['date', 'year', 'month', 'day', 'hour'], inplace=True, axis=1)

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

        # create empty columns to be filled elementwise later on
        self.processes['time_dep'] = self.processes['time_return'] = self.processes['vehicle_sucess'] = np.nan

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
        y = 0.6 * (1 / (self.cs.dep_sig1 * np.sqrt(2 * np.pi)) * np.exp(- (x - self.cs.dep_mu1) ** 2 / (2 * self.cs.dep_sig1 ** 2))) + \
            0.4 * (1 / (self.cs.dep_sig2 * np.sqrt(2 * np.pi)) * np.exp(- (x - self.cs.dep_mu2) ** 2 / (2 * self.cs.dep_sig2 ** 2)))
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

    def generate_demand(self):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        :return: None
        """
        # calculate base range on internal battery for rex calculations
        self.base_range = (1 - self.cs.min_return_soc) * self.cs.dep_soc * self.cs.size_pc / self.cs.consumption

        # buffer time is added onto minimum recharge time to ensure dispatch feasibility and give room for energy mgmt
        self.time_buffer = pd.Timedelta(hours=2)

        # draw total demand for every day from lognormal distribution
        p1, p2 = lognormal_params(self.cs.daily_mu, self.cs.daily_sig)
        self.daily_demand['num_total'] = np.round(self.rng.lognormal(p1, p2, self.daily_demand.shape[0])).astype(int)

        # create array of processes (daily number drawn before) and draw time of departure from custom function
        process_num = self.daily_demand['num_total'].sum(axis=0)
        self.processes['date'] = np.repeat(self.daily_demand.index, self.daily_demand['num_total'])
        self.assign_datetime_request(process_num)

        # draw requested distance and time values, calculate energy used
        p1, p2 = lognormal_params(self.cs.dist_mu, self.cs.dist_sig)
        self.processes['distance'] = self.rng.lognormal(p1, p2, process_num)
        self.processes['time_active'] = pd.to_timedelta((self.processes['distance'] / self.cs.speed_avg), unit='hour')
        p1, p2 = lognormal_params(self.cs.idle_mu, self.cs.idle_sig)
        self.processes['time_idle'] = pd.to_timedelta(self.rng.lognormal(p1, p2, process_num), unit='hour')
        self.processes['time_total'] = self.processes['time_active'] + self.processes['time_idle']
        self.processes['steps_total'] = np.round(self.processes['time_total'] / self.sc.timestep_td)
        self.processes['energy_req'] = self.processes['distance'] * self.cs.consumption

        if self.cs.rex_sys:  # system can extend range. otherwise self.rex_sys is None
            self.processes['rex_distance'] = np.max(0, self.processes['distance'] - self.base_range)
            self.processes['rex_energy'] = self.rex_distance * self.cs.consumption
            self.processes['rex_num'] = math.ceil(self.rex_energy / self.cs.rex_rs.size_pc)
            self.processes['energy_avail'] = ((self.cs.size_pc * self.cs.dep_soc) +
                                              (self.processes['rex_num'] * self.cs.rex_rs.size * self.cs.rex_rs.dep_soc))
            self.processes['rex_sucess'] = np.nan
        else:
            self.processes['energy_avail'] = self.cs.size_pc * self.cs.dep_soc

        # set maximum energy requirement to max available energy
        self.processes['energy_req'] = np.minimum(self.processes['energy_req'], self.processes['energy_avail'])
        self.processes['dsoc_req'] = self.processes['energy_req'] / self.processes['energy_avail']
        self.processes['time_recharge'] = pd.to_timedelta(self.processes['energy_req'] /
                                                          (self.cs.chg_pwr * self.cs.chg_eff), unit='hour')
        self.processes['time_blocked'] = self.processes['time_recharge'] + self.time_buffer

    def start_process(env, day, inter_day_count, CRS_global_count, ID, rng):
        pass

        # # logic to print correct info depending on selected simulation
        #
        # with ID.CarFleet.get() as CAR_req:
        #     CAR_results = yield CAR_req | env.timeout(ID.CRS_patience)
        #     wait_for_car = env.now - arrive
        #     if CAR_req in CAR_results:
        #         with ID.MBFleet.get(required_number_of_MB) as MB_req:
        #                     MB_results = yield MB_req | env.timeout(ID.REX_patience)
        #                     wait_for_MB = env.now - arrive
        #                     if MB_req in MB_results:
        #                         departure_timestep = env.now
        #                         yield env.timeout(time_travelled + idle_time)
        #                         return_timestep = env.now
        #
        #                         extra_charge_time = 2
        #                         CRS_charge_time = math.ceil(total_energy_consumed / ID.CRS_charge_power)
        #                         yield env.timeout(CRS_charge_time + extra_charge_time)
        #
        #                         yield ID.MBFleet.put(MB_results[MB_req])
        #                         ID.CarFleet.put(CAR_results[CAR_req])
        #
        #                         charge_timestep = env.now
        #
        #                         # use the global BRS log variables for the MB in REX usage
        #                         global global_BRS_count
        #                         global BRS_inter_day_count
        #
        #                         # Logging for the MB the CRS+REX is using
        #                         logger.BRS_process_log[global_BRS_count,] = [day, f'REX_MB_{CRS_global_count}',
        #                                                                      BRS_inter_day_count, departure_timestep,
        #                                                                      ID.CRS_leaving_soc, used_charge_per_MB,
        #                                                                      return_timestep, charge_timestep,
        #                                                                      MB_results[MB_req]]
        #
        #                         global_BRS_count += 1
        #                         BRS_inter_day_count += 1
        #
        #                         print(
        #                             'CRS + REX: On day {} at Time {} the Trip {} with the Car {} returned the Batteries ({}) '
        #                             .format(day, env.now, CRS_global_count, CAR_results[CAR_req], MB_results[MB_req]))
        #
        #                         if ID.debug:
        #                             print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
        #                                   .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
        #                             print('')
        #
        #                         # logging for the CRS processes when REX is active
        #                         logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}',
        #                                                                      inter_day_count,
        #                                                                      departure_timestep, ID.CRS_leaving_soc,
        #                                                                      used_charge_car,
        #                                                                      return_timestep, charge_timestep,
        #                                                                      CAR_results[CAR_req]]
        #
        #                     else:  # REX: patience triggered, no MB received
        #                         # ->>> the trip is aborted
        #
        #                         # bugfix rental and timeout problem:
        #                         if MB_req.triggered:
        #                             print('--- request triggered after time out ----')
        #                             MB = yield MB_req
        #                             ID.MBFleet.put(MB)
        #
        #                         # return car to the store
        #                         ID.CarFleet.put(CAR_results[CAR_req])
        #
        #                         global failed_CRS_count
        #                         failed_CRS_count += 1
        #
        #                         print('----------------ALARM----------------------------------')
        #                         print('CRS + REX: On day {} at Time {} the Trip {} failed after waiting {} for REX '
        #                               .format(day, env.now, CRS_global_count, wait_for_MB))
        #                         print('-------------------------------------------------------')
        #
        #                         if ID.debug:
        #                             print('DEBUG: After Trip {} Failed, it returned Car {}, Cars in the Store {} '
        #                                   .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
        #                             print('')
        #
        #                         # logging for the CRS processes when REX is active but req for MB failed
        #                         logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}',
        #                                                                      inter_day_count,
        #                                                                      "failed", "failed", "failed",
        #                                                                      "failed", "failed", "failed"]
        #
        #             else:  # REX: distance travelled is not greater than threshold -> no MB needed:
        #
        #                 departure_timestep = env.now
        #
        #                 print('CRS + REX: On day {} at Time {} the Trip {} got the Car {}, no REX needed, distance {} '
        #                       .format(day, env.now, CRS_global_count, CAR_results[CAR_req], distance_travelled))
        #
        #                 if ID.debug:
        #                     print('DEBUG: 'f'cars remaining: {ID.CarFleet.items}')
        #                     print(
        #                         'DEBUG: Parameters for Trip {} with car {} Distance {}, time_travelled {}, idle_time {}, range_threshold {}'
        #                         .format(CRS_global_count, CAR_results[CAR_req], distance_travelled, time_travelled,
        #                                 idle_time, range_threshold))
        #                     print('')
        #
        #                 # trip duration:, sum of travel time and idle time, valid for the Car and MB
        #                 yield env.timeout(time_travelled + idle_time)
        #
        #                 return_timestep = env.now
        #
        #                 # IF REX but MB not necessary, used charge is travel distance * consumption
        #                 used_charge_car = round(distance_travelled * ID.energy_consumption, 1)
        #
        #                 # bug fix: calculated charge time not sufficient, extra time is introduced
        #                 extra_charge_time = 2
        #
        #                 # this represents the usage time that is valid for the Car
        #                 CRS_charge_time = math.ceil(used_charge_car / ID.CRS_charge_power)
        #                 yield env.timeout(CRS_charge_time + extra_charge_time)
        #
        #                 # return car to the store
        #                 ID.CarFleet.put(CAR_results[CAR_req])
        #
        #                 print('CRS + REX: On day {} at Time {} the Trip {} returned the Car {}  '
        #                       .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))
        #
        #                 if ID.debug:
        #                     print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
        #                           .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
        #                     print('')
        #
        #                 charge_timestep = env.now
        #
        #                 # logging for the CRS processes when REX is active, but not needed
        #                 logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}',
        #                                                              inter_day_count,
        #                                                              departure_timestep, ID.CRS_leaving_soc,
        #                                                              used_charge_car,
        #                                                              return_timestep, charge_timestep,
        #                                                              CAR_results[CAR_req]]
        #
        #         #######################################_END_REX_MB_Rental_#########################################
        #
        #         if not ID.REX:
        #
        #             if distance_travelled <= range_threshold:  # check if we can cover the distance with fixed battery
        #
        #                 # departure timestep for logging
        #                 departure_timestep = env.now
        #
        #                 print('CRS: On day {} at Time {} the Trip {} got the Car {} '
        #                       .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))
        #
        #                 if ID.debug:
        #                     print('DEBUG: 'f'cars remaining: {ID.CarFleet.items}')
        #                     print(
        #                         'DEBUG: Parameters for Trip {} with car {} Distance {}, time_travelled {}, idle_time {}, range_threshold {}'
        #                         .format(CRS_global_count, CAR_results[CAR_req], distance_travelled, time_travelled,
        #                                 idle_time, range_threshold))
        #                     print('')
        #
        #                 # trip duration:, sum of travel time and idle time, valid for the Car and MB
        #                 yield env.timeout(time_travelled + idle_time)
        #
        #                 # this is the return timestep that is used by oemof simulation csv
        #                 return_timestep = env.now
        #
        #                 # IF no REX, used charge is travel distance * consumption
        #                 used_charge_car = round(distance_travelled * ID.energy_consumption, 1)
        #
        #                 # this timeout allows oemof to recharge the MBs
        #                 CRS_charge_time = math.ceil(used_charge_car / ID.CRS_charge_power)
        #                 yield env.timeout(CRS_charge_time + extra_charge_time)
        #
        #                 charge_timestep = env.now
        #
        #                 # return car to store
        #                 ID.CarFleet.put(CAR_results[CAR_req])
        #
        #                 print('CRS: On day {} at Time {} the Trip {} returned the Car {}  '
        #                       .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))
        #
        #                 if ID.debug:
        #                     print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
        #                           .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
        #                     print('')
        #
        #                 # logging for the CRS processes when REX is deactivated
        #                 logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
        #                                                              departure_timestep, ID.CRS_leaving_soc,
        #                                                              used_charge_car,
        #                                                              return_timestep, charge_timestep,
        #                                                              CAR_results[CAR_req]]
        #
        #             else:  # REX is deactivated, thus distances greater than range_threshold can't be covered
        #                 # trip is cancelled, as intrinsic car range is not sufficient
        #
        #                 # return car immediately
        #                 ID.CarFleet.put(CAR_results[CAR_req])
        #
        #                 failed_CRS_count += 1
        #
        #                 print('-----------ALARM------------------------------')
        #                 print('CRS: On day {} at Time {} the Trip {} failed because intrinsic range not sufficient'
        #                       .format(day, env.now, CRS_global_count))
        #
        #                 if ID.debug:
        #                     print('DEBUG: Trip length was {}, max intrinsic range was {} '
        #                           .format(distance_travelled, range_threshold))
        #                     print('DEBUG: After Trip {} failed, it returned Car {}, Cars in the Store {} '
        #                           .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
        #                     print('')
        #
        #                 logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
        #                                                              "failed",
        #                                                              "failed", "failed",
        #                                                              "failed", "failed",
        #                                                              "failed"]
        #
        #     else:  # patience for CAR_req triggered, trip is canceled
        #
        #         # bugfix rental and timeout problem:
        #         if CAR_req.triggered:
        #             print('--- request triggered after time out ----')
        #             car = yield CAR_req
        #             ID.CarFleet.put(car)
        #
        #         # We quit
        #         fail_timestep = env.now
        #         failed_CRS_count += 1
        #
        #         # logging for the CRS processes when REX is deactivated
        #         logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
        #                                                      "failed",
        #                                                      "failed", "failed",
        #                                                      "failed", "failed",
        #                                                      "failed"]
        #
        #         print('-----------ALARM------------------------------')
        #         print('CRS: On day {} at Time {} the Trip {} failed bacause it waited {} for a car'
        #               .format(day, env.now, CRS_global_count, wait_for_car))


class BatteryRentalSystem(RentalSystem):

    def __init__(self, env: simpy.Environment, sc, cs):

        self.usecase_file_path = os.path.join(os.getcwd(), 'input', 'brs', 'brs_usecases.json')
        self.usecases = pd.read_json(self.usecase_file_path, orient='records')

        super().__init__(cs, sc)

        self.store = MultiStore(env, capacity=cs.num)
        for commodity in cs.commodities:
            self.store.put([commodity.name])


    def draw_departure_sample(self, row):
        sample = -1  # kicking off the while loop
        while sample > 24 or sample < 0:
            sample = np.random.normal(self.usecases.loc[row['usecase_idx'], 'dep_mu'],
                                      self.usecases.loc[row['usecase_idx'], 'dep_sig'])
        return sample

    def draw_departure_samples(self, n):
        list = self.processes.apply(self.draw_departure_sample, axis=1)
        return list

    def generate_demand(self):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        :return: None
        """
        p1,p2 = lognormal_params(self.cs.daily_mu, self.cs.daily_sig)
        self.daily_demand['num_total'] = (np.round(self.rng.lognormal(p1, p2, self.daily_demand.shape[0]))).astype(int)

        process_num = self.daily_demand['num_total'].sum(axis=0)
        rel_prob_norm = self.usecases['rel_prob'] / self.usecases['rel_prob'].sum(axis=0)
        self.processes['usecase_idx'] = np.random.choice(self.usecases.index.values,
                                                         process_num,
                                                         replace=True,
                                                         p=rel_prob_norm)
        self.processes['usecase_name'] = self.processes.apply(lambda row: self.usecases.loc[row['usecase_idx'], 'name'],
                                                              axis=1)
        self.assign_datetime_request(process_num)

        p1, p2 = lognormal_params(0.12, 0.1)  # todo introduce proper brs parameters
        self.processes['soc_return'] = np.random.lognormal(p1, p2, process_num)
        self.processes['num_bat'] = self.processes.apply(lambda row: self.usecases.loc[row['usecase_idx'], 'num_bat'],
                                                         axis=1)
        self.processes['energy_avail'] = self.processes['num_bat'] * self.cs.size_pc

        self.processes['time_active'] = pd.to_timedelta(self.processes.apply(lambda row: (1 - row['soc_return']) *
                                                                                         row['energy_avail'] /
                                                                                         self.usecases.loc[row['usecase_idx'],
                                                                                         'power'],
                                                                             axis=1), unit='hour')

        p1, p2 = lognormal_params(4, 3)  # todo implement proper brs parameters
        self.processes['time_idle'] = pd.to_timedelta(np.random.lognormal(p1, p2, process_num), unit='hour')
        self.processes['time_total'] = self.processes['time_active'] + self.processes['time_idle']

        self.processes['energy_req_pc'] = (1 - self.processes['soc_return']) * self.cs.size_pc
        self.processes['energy_req'] = self.processes['energy_req_pc'] * self.processes['num_bat']

    def start_process(self):
        # from battery_process_func
        # todo adapt to new structure


        # # draw starting time of usecase from normal distribution
        # # via "yield" wait until starting point is reached
        # yield env.timeout(round(rng.normal(n["dep_mean"], n["dep_std"])))
        #
        # print('-------------------------------------------------------------------------------')
        # print('BRS: On day {} Uscase {}_{} requires: {} Batteries at time: {}'.format(day_count, n["usecase"],
        #                                                                               inter_day_count,
        #                                                                               resources_required, env.now))
        # if ID.debug:
        #     print('DEBUG: 'f'MB available: {ID.MBFleet.items}')
        #     print('')
        #
        # # arrival time of request
        # arrival = env.now
        #
        # with ID.MBFleet.get(resources_required) as BRS_req:
        #
        #     # Wait for the MBs or abort at the end of patience time
        #     BRS_results = yield BRS_req | env.timeout(ID.BRS_patience)
        #
        #     # waiting time of request
        #     wait = env.now - arrival
        #
        #     if BRS_req in BRS_results:  # if we got the necessary number of MB in time:
        #
        #         # departure timestep for logging
        #         departure_timestep = env.now
        #
        #         print('BRS: On day {} Uscase {}_{} rented the Batteries {} at time: {}'
        #               .format(day_count, n["usecase"], inter_day_count, BRS_results[BRS_req], departure_timestep))
        #
        #         if ID.debug:
        #             print('DEBUG: 'f'MB remaining: {ID.MBFleet.items}')
        #             print('')
        #
        #         # draw the return SOC from a lognormal distribution
        #         return_soc = rng.lognormal(1, 1)
        #
        #         # exclude the unlikely case that return_soc >= 100
        #         while return_soc >= 100:
        #             return_soc = rng.lognormal(1, 1)
        #
        #         # remaining charge is calculated as percentual share of return_soc and energycontent
        #         remaining_charge = round(ID.BRS_energycontent * (return_soc / 100))
        #
        #         # calculate used charge
        #         used_charge = ID.BRS_energycontent - remaining_charge
        #
        #         # usagetime is calculated based on the used energy and power demand
        #         usagetime = math.floor((n["num_batteries"] * ID.BRS_energycontent) / n["power"])
        #
        #         # this yield timeout represents the usage time
        #         yield env.timeout(usagetime)
        #
        #         # this is the return timestep that is used by oemof simulation csv
        #         return_timestep = env.now
        #         leaving_SOC = 1
        #
        #         # bug fix: calculated charge time not sufficient, extra time is introduced
        #         extra_charge_time = 2
        #
        #         # this internal timeout gives oemof a time window to recharge the MBs
        #         BRS_charging_time = math.ceil(used_charge / ID.BRS_charge_power)
        #         yield env.timeout(BRS_charging_time + extra_charge_time)
        #
        #         # after the usage return the ressources to the store.
        #         yield ID.MBFleet.put(BRS_results[BRS_req])
        #         charge_timestep = env.now
        #
        #         print('-------------------------------------------------------------------------------')
        #         print('BRS: UseCase {}_{} that rented on day {} at {} used the Batteries {} for: {} hours'
        #               .format(n["usecase"], inter_day_count, day_count, departure_timestep, BRS_results[BRS_req],
        #                       usagetime))
        #
        #         print('BRS: UseCase {}_{} that rented on day {} returned the Batteries {} at time: {}'
        #               .format(n["usecase"], inter_day_count, day_count, BRS_results[BRS_req], return_timestep))
        #
        #         if ID.debug:
        #             print('DEBUG: After usecase {} returned Car {}, Cars in the Store {} '
        #                   .format(n["usecase"], BRS_results[BRS_req], ID.MBFleet.items))
        #             print('')
        #
        #         # fill simpy process log:
        #         logger.BRS_process_log[BRS_global_count,] = [day_count, n["usecase"], inter_day_count,
        #                                                      departure_timestep,
        #                                                      leaving_SOC, used_charge,
        #                                                      return_timestep, charge_timestep, BRS_results[BRS_req]]
        #
        #     else:  # if we didn't get MB in time: we quit the rental
        #
        #         # bug fix: if we land in else but brs_req is triggered, return the mb instantly
        #         if BRS_req.triggered:
        #             print('--- request triggered after time out ----')
        #             mb = yield BRS_req
        #             ID.MBFleet.put(mb)
        #
        #         print('-----------ALARM------------------------------')
        #         print('BRS: On day {} at Time {} the UseCase {}_{}  failed bacause it waited {}'
        #               .format(day_count, env.now, n["usecase"], inter_day_count, wait))
        #
        #         # so far unused fail timestep
        #         fail_timestep = env.now
        #
        #         # failed rental statistic
        #         global failed_BRS_count
        #         failed_BRS_count += 1
        #
        #         # fill simpy process log:
        #         logger.BRS_process_log[BRS_global_count,] = [day_count, n["usecase"], inter_day_count, "failed",
        #                                                      "failed", "failed",
        #                                                      "failed", "failed", "failed"]
        pass


class RentalProcess:

    def __init__(self, data: pd.Series, rs, sc):

        self.time_request = sc.des_env.now

        with rs.store.get() as self.request:
            # Wait for commodity or abort by a timeout
            self.result = yield self.request | sc.des_env.timeout(rs.cs.patience)

            if self.request in self.result:  # request granted

                if cs.rex and self.distance > self.range:  # range extension required & available
                    self.request_rex_commodities(rs, sc.des_env)
                    if self.rex_sucess:
                        self.execute_trip(rex=True)
                    else:
                        # log failure
                        pass
                elif self.distance <= self.range:  # range extension not required
                    self.execute_trip(rex=False)
                else:  # range extension required, but not available
                    # log failure
                    # return vehicle to store
                    pass

            else:  # request not granted, patience is triggered

                if self.request.triggered:
                    # make sure that resource is placed back in store
                    resource = yield self.request
                    rs.rex_system.store.put(resource)  # todo why can we not just put the car back every time (like finally)?

                # log failure

            rs.store.put(self.result[self.request])  # todo this would be my approach

    def execute_rental(self,
                       cs: blocks.CommoditySystem,
                       env: simpy.Environment,
                       rex=False):

        if not rex:  # if rex is active, departure time has already been set as there might be further delay
            self.time_departure = env.now

        yield env.timeout(self.duration.hours)  # todo hours doesnt exist
        self.time_return = env.now
        self.time_recharge = dt.timedelta(hours=self.energy_req / cs.chg_pwr)
        self.time_buffer = dt.timedelta(hours=2)

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


def usecase_gen(env, ID, rng):
    # parts for demand generation already cut out, only processes left

    #
    #     if ID.CRS:  # if CRS is active
    #
    #
    #         print('Day: 'f'{d}'' Daily CRS Trip demand: 'f'{daily_CRS_trip_demand}')
    #
    #         for t in range(daily_CRS_trip_demand):  # for the number of usecases per day do:
    #
    #             # start the CRS process generator with one instance of the process function called "car_process_func()"
    #             env.process(car_process_func(env, day, CRS_inter_day_count, global_CRS_count, ID, rng))
    #
    #     if ID.BRS:
    #             # start the BRS process generator with one instance of the process function called "battery_process_func()"
    #             env.process(battery_process_func(env, day, BRS_inter_day_count, n["num_batteries"], n, global_BRS_count, ID, rng))
    #             global_BRS_count += 1
    #             BRS_inter_day_count += 1
    #
    #     day += 1
    #
    #     # make sure that a day has 24h
    #     # if step length is not 1h then 24 must be changed  (viertelstunde)
    #     yield env.timeout(24)
    pass


###############################################################################
# global functions
###############################################################################

def execute_des(sc):

    # define a DES environment
    sc.des_env = simpy.Environment()

    sc.rental_systems = []
    for commodity_system in [block for block in sc.blocks.values() if isinstance(block, blocks.CommoditySystem)]:
        # todo make a better decision which commodity systems to initiate as VehicleRentalSystems and which as BatteryRentalSystems
        if commodity_system.name == 'bev':
            rs = VehicleRentalSystem(sc.des_env, sc, commodity_system)
        else:
            rs = BatteryRentalSystem(sc.des_env, sc, commodity_system)
        sc.rental_systems.append(rs)
    pass

    # call the function that generates the individual rental processes
    # sc.env.process(usecase_gen(env, ID, rng))

    # start the simulation
    sc.env.run()

    # save logging results
    for rental_system in sc.rental_systems:
        rental_system.convert_process_log()
        rental_system.save_logs()


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


