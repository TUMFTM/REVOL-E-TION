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
import random
import numpy as np
import pandas as pd

import blocks
import main


###############################################################################
# Custom subclasses from simpy.resources for stores, from which multiple resources can be taken at once
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

    def __init__(self, cs: blocks.CommoditySystem):

        self.name = cs.name
        self.cs = cs  # making cs callable from other RentalSystems
        # all other values are accessible through the commodity system

        self.rng = np.random.default_rng()
        self.processes = pd.DataFrame(columns=[''])


class VehicleRentalSystem(RentalSystem):

    def __init__(self, env: simpy.Environment, sc: main.Scenario, cs: blocks.CommoditySystem):

        super().__init__(cs)

        self.mu_daily_demand = 15
        self.sig_daily_demand = 0

        self.store = simpy.Store(env, capacity=cs.num)
        for commodity in cs.commodities:
            self.store.put({commodity.name})  # todo why the curly brackets?


    def departure_pdf(self, cs: blocks.CommoditySystem, x):

        y = 0.6 * (1 / (cs.dep_sig1 * np.sqrt(2 * np.pi)) * np.exp(- (x - cs.dep_mu1) ** 2 / (2 * cs.dep_sig1 ** 2))) + \
            0.4 * (1 / (cs.dep_sig2 * np.sqrt(2 * np.pi)) * np.exp(- (x - cs.dep_mu2) ** 2 / (2 * cs.dep_sig2 ** 2)))

        return y

    def get_departure_time(self, process, env):

        # choose a start time from all hours of a day 0-23, but differently weighted with custom probability function
        # (viertelstunde)
        process.start_time = random.choices(self.dep_time_values, self.dep_time_weights)
        yield env.timeout(int(process.start_time[0]))

    def start_process(env, day, inter_day_count, CRS_global_count, ID, rng):


        # logic to print correct info depending on selected simulation
        if ID.REX:
            print('CRS + REX: On day {} at Time {} the Trip {} is born -- Total Trip Count: {}'
                  .format(day, env.now, CRS_global_count, CRS_global_count))
        else:
            print('CRS: On day {} at Time {} the Trip {} is born -- Total Trip Count: {}'
                  .format(day, env.now, CRS_global_count, CRS_global_count))



        if ID.debug:
            print('DEBUG: 'f'cars available: {ID.CarFleet.items}')
            print('DEBUG: 'f'MB available: {ID.MBFleet.items}')
            print('')

        # request a car from the CarFleet
        with ID.CarFleet.get() as CAR_req:

            # Wait for the car or abort by a timeout
            CAR_results = yield CAR_req | env.timeout(ID.CRS_patience)

            # waiting time
            wait_for_car = env.now - arrive

            if CAR_req in CAR_results:  # we got a car, now draw trip distance and duration

                # travel distance of each trip follows log-normal:
                distance_travelled = rng.lognormal(ID.mu_trip_length, ID.sigma_trip_length)

                # calculate the driving duration of a trip based on distance travelled and avg. velocity
                time_travelled = round(distance_travelled / ID.avg_velocity)

                # consider car state that is not driving and not charging -> parking/ waiting
                # idle_time in [h] based on a log-normal distribution, value rounded to int
                # Attention: if time step is changed, idle time has to be adapted (viertelstunde)
                idle_time = round(rng.lognormal(ID.mu_idle_time, ID.sigma_idle_time))

                # threshold value in [km] used to determine:
                # 1) if trip is feasible with intrinsic range 2) IF MB are necessary for a trip
                range_threshold = ((1 - ID.min_return_soc) * ID.fix_bat_size) / ID.energy_consumption

                #######################################_BEGIN_REX_MB_Rental_#########################################

                if ID.REX:  # if REX is active, trips requiring MB can now request MB, if needed

                    if distance_travelled > range_threshold:  # if MB is needed for the trip:

                        # calculate the distance that has to be covered by MB
                        distance_to_cover_with_MB = distance_travelled - range_threshold

                        # calculate total MB energy required for the trip
                        required_MB_energy = distance_to_cover_with_MB * ID.energy_consumption

                        # calculate the number of MB needed for the trip
                        # number of MBs can only be an integer, thus value is rounded to the next full int
                        required_number_of_MB = math.ceil(required_MB_energy / ID.BRS_energycontent)

                        # total energy consumed
                        total_energy_consumed = distance_travelled * ID.energy_consumption

                        # sum of fixed_battery and total MB energy available
                        total_energy_available = ID.fix_bat_size + required_number_of_MB * ID.BRS_energycontent

                        # fraction of SOC used for trip driving distance
                        used_SOC = total_energy_consumed / total_energy_available

                        # used charge of fixed battery of a car per trip
                        used_charge_car = round(used_SOC * ID.fix_bat_size, 1)

                        # used charge per battery:
                        used_charge_per_MB = used_SOC * ID.BRS_energycontent

                        # we determined number of MB necessary, now request them from the store
                        with ID.MBFleet.get(required_number_of_MB) as MB_req:

                            MB_results = yield MB_req | env.timeout(ID.REX_patience)

                            wait_for_MB = env.now - arrive

                            if MB_req in MB_results:  # we got necessary MB:

                                # departure timestep for logging
                                departure_timestep = env.now

                                print(
                                    'CRS + REX: On day {} at Time {} the Trip {} got the Car {} and the Batteries ({}), Distance {} '
                                    .format(day, env.now, CRS_global_count, CAR_results[CAR_req], MB_results[MB_req],
                                            distance_travelled))

                                if ID.debug:
                                    print(
                                        'DEBUG: Parameters for Trip {} with car {} Distance {}, time_travelled {}, idle_time {}, range_threshold {}'
                                        .format(CRS_global_count, CAR_results[CAR_req], distance_travelled,
                                                time_travelled, idle_time, range_threshold))
                                    print('DEBUG: 'f'cars remaining: {ID.CarFleet.items}')
                                    print('DEBUG: 'f'MB remaining: {ID.MBFleet.items}')
                                    print('')

                                # trip duration:, sum of travel time and idle time, valid for the Car and MB
                                # MB can't return before the car
                                yield env.timeout(time_travelled + idle_time)

                                return_timestep = env.now

                                # bug fix: calculated charge time not sufficient, extra time is introduced
                                extra_charge_time = 2

                                # this represents the usage time that is valid for the Car and the Batteries
                                CRS_charge_time = math.ceil(total_energy_consumed / ID.CRS_charge_power)
                                yield env.timeout(CRS_charge_time + extra_charge_time)

                                # after the usage return the MB = "MB_req" to the store.
                                yield ID.MBFleet.put(MB_results[MB_req])

                                # after the usage return the car = "CAR_req" to the store.
                                ID.CarFleet.put(CAR_results[CAR_req])

                                charge_timestep = env.now

                                # use the global BRS log variables for the MB in REX usage
                                global global_BRS_count
                                global BRS_inter_day_count

                                # Logging for the MB the CRS+REX is using
                                logger.BRS_process_log[global_BRS_count,] = [day, f'REX_MB_{CRS_global_count}',
                                                                             BRS_inter_day_count, departure_timestep,
                                                                             ID.CRS_leaving_soc, used_charge_per_MB,
                                                                             return_timestep, charge_timestep,
                                                                             MB_results[MB_req]]

                                global_BRS_count += 1
                                BRS_inter_day_count += 1

                                print(
                                    'CRS + REX: On day {} at Time {} the Trip {} with the Car {} returned the Batteries ({}) '
                                    .format(day, env.now, CRS_global_count, CAR_results[CAR_req], MB_results[MB_req]))

                                if ID.debug:
                                    print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
                                          .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
                                    print('')

                                # logging for the CRS processes when REX is active
                                logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}',
                                                                             inter_day_count,
                                                                             departure_timestep, ID.CRS_leaving_soc,
                                                                             used_charge_car,
                                                                             return_timestep, charge_timestep,
                                                                             CAR_results[CAR_req]]

                            else:  # REX: patience triggered, no MB received
                                # ->>> the trip is aborted

                                # bugfix rental and timeout problem:
                                if MB_req.triggered:
                                    print('--- request triggered after time out ----')
                                    MB = yield MB_req
                                    ID.MBFleet.put(MB)

                                # return car to the store
                                ID.CarFleet.put(CAR_results[CAR_req])

                                global failed_CRS_count
                                failed_CRS_count += 1

                                print('----------------ALARM----------------------------------')
                                print('CRS + REX: On day {} at Time {} the Trip {} failed after waiting {} for REX '
                                      .format(day, env.now, CRS_global_count, wait_for_MB))
                                print('-------------------------------------------------------')

                                if ID.debug:
                                    print('DEBUG: After Trip {} Failed, it returned Car {}, Cars in the Store {} '
                                          .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
                                    print('')

                                # logging for the CRS processes when REX is active but req for MB failed
                                logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}',
                                                                             inter_day_count,
                                                                             "failed", "failed", "failed",
                                                                             "failed", "failed", "failed"]

                    else:  # REX: distance travelled is not greater than threshold -> no MB needed:

                        departure_timestep = env.now

                        print('CRS + REX: On day {} at Time {} the Trip {} got the Car {}, no REX needed, distance {} '
                              .format(day, env.now, CRS_global_count, CAR_results[CAR_req], distance_travelled))

                        if ID.debug:
                            print('DEBUG: 'f'cars remaining: {ID.CarFleet.items}')
                            print(
                                'DEBUG: Parameters for Trip {} with car {} Distance {}, time_travelled {}, idle_time {}, range_threshold {}'
                                .format(CRS_global_count, CAR_results[CAR_req], distance_travelled, time_travelled,
                                        idle_time, range_threshold))
                            print('')

                        # trip duration:, sum of travel time and idle time, valid for the Car and MB
                        yield env.timeout(time_travelled + idle_time)

                        return_timestep = env.now

                        # IF REX but MB not necessary, used charge is travel distance * consumption
                        used_charge_car = round(distance_travelled * ID.energy_consumption, 1)

                        # bug fix: calculated charge time not sufficient, extra time is introduced
                        extra_charge_time = 2

                        # this represents the usage time that is valid for the Car
                        CRS_charge_time = math.ceil(used_charge_car / ID.CRS_charge_power)
                        yield env.timeout(CRS_charge_time + extra_charge_time)

                        # return car to the store
                        ID.CarFleet.put(CAR_results[CAR_req])

                        print('CRS + REX: On day {} at Time {} the Trip {} returned the Car {}  '
                              .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))

                        if ID.debug:
                            print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
                                  .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
                            print('')

                        charge_timestep = env.now

                        # logging for the CRS processes when REX is active, but not needed
                        logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}',
                                                                     inter_day_count,
                                                                     departure_timestep, ID.CRS_leaving_soc,
                                                                     used_charge_car,
                                                                     return_timestep, charge_timestep,
                                                                     CAR_results[CAR_req]]

                #######################################_END_REX_MB_Rental_#########################################

                if not ID.REX:

                    if distance_travelled <= range_threshold:  # check if we can cover the distance with fixed battery

                        # departure timestep for logging
                        departure_timestep = env.now

                        print('CRS: On day {} at Time {} the Trip {} got the Car {} '
                              .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))

                        if ID.debug:
                            print('DEBUG: 'f'cars remaining: {ID.CarFleet.items}')
                            print(
                                'DEBUG: Parameters for Trip {} with car {} Distance {}, time_travelled {}, idle_time {}, range_threshold {}'
                                .format(CRS_global_count, CAR_results[CAR_req], distance_travelled, time_travelled,
                                        idle_time, range_threshold))
                            print('')

                        # trip duration:, sum of travel time and idle time, valid for the Car and MB
                        yield env.timeout(time_travelled + idle_time)

                        # this is the return timestep that is used by oemof simulation csv
                        return_timestep = env.now

                        # IF no REX, used charge is travel distance * consumption
                        used_charge_car = round(distance_travelled * ID.energy_consumption, 1)

                        # this timeout allows oemof to recharge the MBs
                        CRS_charge_time = math.ceil(used_charge_car / ID.CRS_charge_power)
                        yield env.timeout(CRS_charge_time + extra_charge_time)

                        charge_timestep = env.now

                        # return car to store
                        ID.CarFleet.put(CAR_results[CAR_req])

                        print('CRS: On day {} at Time {} the Trip {} returned the Car {}  '
                              .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))

                        if ID.debug:
                            print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
                                  .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
                            print('')

                        # logging for the CRS processes when REX is deactivated
                        logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
                                                                     departure_timestep, ID.CRS_leaving_soc,
                                                                     used_charge_car,
                                                                     return_timestep, charge_timestep,
                                                                     CAR_results[CAR_req]]

                    else:  # REX is deactivated, thus distances greater than range_threshold can't be covered
                        # trip is cancelled, as intrinsic car range is not sufficient

                        # return car immediately
                        ID.CarFleet.put(CAR_results[CAR_req])

                        failed_CRS_count += 1

                        print('-----------ALARM------------------------------')
                        print('CRS: On day {} at Time {} the Trip {} failed because intrinsic range not sufficient'
                              .format(day, env.now, CRS_global_count))

                        if ID.debug:
                            print('DEBUG: Trip length was {}, max intrinsic range was {} '
                                  .format(distance_travelled, range_threshold))
                            print('DEBUG: After Trip {} failed, it returned Car {}, Cars in the Store {} '
                                  .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
                            print('')

                        logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
                                                                     "failed",
                                                                     "failed", "failed",
                                                                     "failed", "failed",
                                                                     "failed"]

            else:  # patience for CAR_req triggered, trip is canceled

                # bugfix rental and timeout problem:
                if CAR_req.triggered:
                    print('--- request triggered after time out ----')
                    car = yield CAR_req
                    ID.CarFleet.put(car)

                # We quit
                fail_timestep = env.now
                failed_CRS_count += 1

                # logging for the CRS processes when REX is deactivated
                logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
                                                             "failed",
                                                             "failed", "failed",
                                                             "failed", "failed",
                                                             "failed"]

                print('-----------ALARM------------------------------')
                print('CRS: On day {} at Time {} the Trip {} failed bacause it waited {} for a car'
                      .format(day, env.now, CRS_global_count, wait_for_car))


class BatteryRentalSystem(RentalSystem):

    def __init__(self, env: simpy.Environment, sc: main.Scenario, cs: blocks.CommoditySystem):

        super().__init__(cs)

        self.usecase_file_path = os.path.join(os.getcwd(), 'input', 'brs', 'brs_usecases.json')
        self.usecases = pd.read_json(self.usecase_file_path, orient='records')

        self.store = MultiStore(env, capacity=cs.num)
        for commodity in cs.commodities:
            self.store.put([commodity.name])


    def generate_demand(self, cs: blocks.CommoditySystem, sc: main.Scenario):

        #get df of all days and draw total demand of usecases
        self.demand = pd.DataFrame(index=sc.sim_dti.date.unique())
        self.demand['num_total'] = round(self.rng.lognormal(cs.departure_mu,
                                                            cs.departure_sig,
                                                            self.demand.shape[0]))
        self.demand['usecases'] = np.random.choice(self.usecases['name'],
                                                         (self.demand.shape[0],self.demand['num_total']),
                                                         replace=True,
                                                         p=self.usecases['rel_prob'])

    def start_process(self):





class RentalProcess:

    def __init__(self,
                 rs: RentalSystem,
                 cs: blocks.CommoditySystem,
                 env: simpy.Environment,
                 rex_rs: BatteryRentalSystem = None,
                 usecase = None):

        self.time_request = env.now

        self.time_idle = dt.timedelta(hours=rs.rng.lognormal(cs.idle_mu, cs.idle_sig))

        if isinstance(rs, VehicleRentalSystem):
            self.distance = rs.rng.lognormal(cs.trip_mu, cs.trip_sig)
            self.time_active = dt.timedelta(hours=self.distance / cs.speed_avg)
            self.energy_req = self.distance * cs.consumption
        else:  # battery rental

            self.time_active = rs.usecases.loc[self.]
            self.energy_req =

        self.duration =  self.time_active + self.time_idle


        # calculate range down to minimum SOC threshold
        self.range = (1 - cs.min_return_soc)(cs.size / cs.consumption)

        with rs.store.get() as self.request:
            # Wait for commodity or abort by a timeout
            self.result = yield self.request | env.timeout(cs.patience)

            if self.request in self.result:  # request granted

                if cs.rex and self.distance > self.range:  # range extension required & available
                    self.request_rex_commodities(rs, cs, env, rex_rs)
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
                    rex_rs.store.put(resource)  # todo why can we not just put the car back every time (like finally)?

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
                            env: simpy.Environment,
                            rex_rs: BatteryRentalSystem):

        self.rex_distance = self.distance - self.range
        self.rex_energy = self.rex_distance * cs.consumption
        self.rex_num = math.ceil(self.rex_energy / rex_rs.cs.size)
        self.energy_avail = (cs.size * cs.departure_soc) + (self.rex_num * rex_rs.cs.size * rex_rs.cs.departure_soc)
        self.soc_return = self.energy_req / self.energy_avail

        with rex_rs.store.get(self.rex_num) as self.rex_request:
            self.rex_result = yield self.rex_request | env.timeout(rex_rs.cs.patience)
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
                yield rex_rs.store.put(self.rex_result[self.rex_request])  # put batteries back
                self.rex_sucess = True
            else:  # request for rex batteries not granted  -> abort trip

                self.rex_sucess = False


###############################################################################
# logging and saving results class
###############################################################################

class Logging():

    def __init__(self, ID):

        # also activate BRS logging when REX is active
        if ID.BRS or ID.REX:
            # create empty array of zeros for logging all BRS rental processes
            self.BRS_process_log = np.zeros([round((ID.simulated_days * ID.mu_daily_BRS_rental_demand)*2), 9], dtype=object)

            # create empty array of zeros for oemof compatible description of BRS usage
            self.BRS_ind_array = np.zeros([ID.number_timesteps+500, (3 * ID.BRS_capacity)])

            # fill "at_charger" columns with "1"
            for mb in range(ID.BRS_capacity):
                self.BRS_ind_array[:, 2 + mb * 3] = 1

        # activate CRS logging
        if ID.CRS:
            # create empty array of zeros for logging all CRS rental processes
            self.CRS_process_log = np.zeros([round((ID.simulated_days * ID.mu_daily_CRS_trip_demand)*2), 9], dtype=object)

            # create empty array of zeros for oemof compatible description of BRS usage
            self.CRS_ind_array = np.zeros([ID.number_timesteps+500, (3 * ID.CRS_capacity)])

            # fill "at_charger" columns with "1"
            for car in range(ID.CRS_capacity):
                self.CRS_ind_array[:, 2 + car * 3] = 1

    ###############################################################################
    # process-log is transformed to oemof compatible .csv
    ###############################################################################

    def convert_to_csv(self, process_log, ind_array, global_count):

        h = 0
        while h < global_count:

            # filter out failed trips for the ind_array
            if process_log[h][3] != "failed":

                # which column of process log contains which info
                departure_timestep_log = process_log[h][3]
                leaving_SOC_log = process_log[h][4]
                used_charge_log = process_log[h][5]
                return_timestep_log = process_log[h][6]
                used_Car_log = process_log[h][8]

                if process_log[h][8] != 0:  # check if  used car log is empty

                    for k in used_Car_log:  # for every car that was used by a trip:

                        j = (departure_timestep_log - 1)
                        while (departure_timestep_log - 1) <= j <= (return_timestep_log + 1):

                            # one timestep before rental set SoC
                            if j == departure_timestep_log - 1:
                                ind_array[j][0 + k * 3] += leaving_SOC_log

                            # timestep of rental: remove battery capacity from minigrid
                            if j == departure_timestep_log:
                                ind_array[j][1 + k * 3] += used_charge_log
                                ind_array[j][2 + k * 3] = 0  # .0

                            # during rental: set availability to 0
                            if departure_timestep_log < j <= return_timestep_log:
                                ind_array[j][2 + k * 3] = 0  # .0

                            j += 1

            h += 1

    ###############################################################################
    # .csv File creation: save results to hard-drive, in current working dir.
    ###############################################################################

    def save(self, process_log, ind_array, fleet_capacity, name):

        # Save process log
        Simulation_Log = pd.DataFrame(process_log,
                                      columns=['Day', 'usage', 'day_Count', 'departure_timestep', 'leaving_SOC',
                                               'used_charge', 'return_timestep', 'chargetime', 'used_'f'{name}'])

        save_filename = os.path.join(os.getcwd(), 'input', f'{name}', f'{name}_process_log.csv')

        # print(Simulation_Log.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        Simulation_Log.to_csv(save_filename, sep=';')

        # Save oemof compatible csv file
        ind_bev_df = pd.DataFrame(ind_array)

        for i in range(0, fleet_capacity):

            ind_bev_df.rename(columns={i * 3: f'{name}{i}_minsoc',
                                       i * 3 + 1: f'{name}{i}_consumption',
                                       i * 3 + 2: f'{name}{i}_atbase'}, inplace=True)

        save_filename = os.path.join(os.getcwd(), 'input', f'{name}', f'{name}_log.csv')
        # print(ind_bev_df.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        ind_bev_df.to_csv(save_filename, sep=';')

###############################################################################
# process generator function: creates BRS rentals and or CRS trips
###############################################################################
def usecase_gen(env, ID, rng):

    # Check input.xlsx for consistency, and input errors
    if not ID.CRS and not ID.BRS:
        print('ACHTUNG: Sowohl BRS als auch CRS ausgeschaltet, Trigger in csv überprüfen')
    if ID.BRS and ID.mu_daily_BRS_rental_demand <= 0:
        print('Achtung: BRS ist aktiviert aber "mean rental demand" ist 0! (input.xlsx überprüfen)')
    if ID.CRS and ID.mu_daily_CRS_trip_demand <= 0:
        print('Achtung: CRS ist aktiviert aber "mean trip demand" ist 0! (input.xlsx überprüfen)')

    global global_BRS_count
    global global_CRS_count

    day = 0
    for d in range(ID.simulated_days):  # for the number of simulated days do:

        global BRS_inter_day_count
        global CRS_inter_day_count
        BRS_inter_day_count = 0
        CRS_inter_day_count = 0

        if ID.CRS:  # if CRS is active

            # number of daily trips is based on normal distribution
            daily_CRS_trip_demand = round(rng.normal(ID.mu_daily_CRS_trip_demand, ID.sigma_daily_CRS_trip_demand))

            # trip demand cant be negative
            while daily_CRS_trip_demand < 0:
                daily_CRS_trip_demand = round(rng.normal(ID.mu_daily_CRS_trip_demand, ID.sigma_daily_CRS_trip_demand))

            print('Day: 'f'{d}'' Daily CRS Trip demand: 'f'{daily_CRS_trip_demand}')

            for t in range(daily_CRS_trip_demand):  # for the number of usecases per day do:

                # start the CRS process generator with one instance of the process function called "car_process_func()"
                env.process(car_process_func(env, day, CRS_inter_day_count, global_CRS_count, ID, rng))
                global_CRS_count += 1
                CRS_inter_day_count += 1

        if ID.BRS:  # if BRS is active

            # number of daily rentals is based on normal distribution
            daily_BRS_rental_demand = round(rng.normal(ID.mu_daily_BRS_rental_demand, ID.sigma_daily_BRS_rental_demand))

            # rental demand cant be negative
            while daily_BRS_rental_demand < 0:
                daily_BRS_rental_demand = round(rng.normal(ID.mu_daily_BRS_rental_demand, ID.sigma_daily_BRS_rental_demand))

            print('Day: 'f'{d}'' Daily BRS Trip demand: 'f'{daily_BRS_rental_demand}')

            for t in range(daily_BRS_rental_demand):  # for the number of usecases per day day do:

                # choose a random BRS usecase from the list
                n = random.choice(ID.list_of_customers)

                # start the BRS process generator with one instance of the process function called "battery_process_func()"
                env.process(battery_process_func(env, day, BRS_inter_day_count, n["num_batteries"], n, global_BRS_count, ID, rng))
                global_BRS_count += 1
                BRS_inter_day_count += 1

        day += 1

        # make sure that a day has 24h
        # if step length is not 1h then 24 must be changed  (viertelstunde)
        yield env.timeout(24)


###############################################################################
# Battery process function = description of rental process for Batteries
###############################################################################

def battery_process_func(env, day_count, inter_day_count, resources_required, n, BRS_global_count, ID, rng):

    # draw starting time of usecase from normal distribution
    # via "yield" wait until starting point is reached
    yield env.timeout(round(rng.normal(n["dep_mean"], n["dep_std"])))

    print('-------------------------------------------------------------------------------')
    print('BRS: On day {} Uscase {}_{} requires: {} Batteries at time: {}'.format(day_count, n["usecase"], inter_day_count,
                                                                             resources_required, env.now))
    if ID.debug:
        print('DEBUG: 'f'MB available: {ID.MBFleet.items}')
        print('')

    # arrival time of request
    arrival = env.now

    with ID.MBFleet.get(resources_required) as BRS_req:

        # Wait for the MBs or abort at the end of patience time
        BRS_results = yield BRS_req | env.timeout(ID.BRS_patience)

        # waiting time of request
        wait = env.now - arrival

        if BRS_req in BRS_results:  # if we got the necessary number of MB in time:

            # departure timestep for logging
            departure_timestep = env.now

            print('BRS: On day {} Uscase {}_{} rented the Batteries {} at time: {}'
                  .format(day_count, n["usecase"], inter_day_count, BRS_results[BRS_req], departure_timestep))

            if ID.debug:
                print('DEBUG: 'f'MB remaining: {ID.MBFleet.items}')
                print('')


            # draw the return SOC from a lognormal distribution
            return_soc = rng.lognormal(1, 1)

            # exclude the unlikely case that return_soc >= 100
            while return_soc >= 100:
                return_soc = rng.lognormal(1, 1)

            # remaining charge is calculated as percentual share of return_soc and energycontent
            remaining_charge = round(ID.BRS_energycontent * (return_soc / 100))

            # calculate used charge
            used_charge = ID.BRS_energycontent - remaining_charge

            # usagetime is calculated based on the used energy and power demand
            usagetime = math.floor((n["num_batteries"] * ID.BRS_energycontent) / n["power"])

            # this yield timeout represents the usage time
            yield env.timeout(usagetime)

            # this is the return timestep that is used by oemof simulation csv
            return_timestep = env.now
            leaving_SOC = 1

            # bug fix: calculated charge time not sufficient, extra time is introduced
            extra_charge_time = 2

            # this internal timeout gives oemof a time window to recharge the MBs
            BRS_charging_time = math.ceil(used_charge/ID.BRS_charge_power)
            yield env.timeout(BRS_charging_time+ extra_charge_time)

            # after the usage return the ressources to the store.
            yield ID.MBFleet.put(BRS_results[BRS_req])
            charge_timestep = env.now

            print('-------------------------------------------------------------------------------')
            print('BRS: UseCase {}_{} that rented on day {} at {} used the Batteries {} for: {} hours'
                  .format(n["usecase"], inter_day_count, day_count,departure_timestep, BRS_results[BRS_req], usagetime))

            print('BRS: UseCase {}_{} that rented on day {} returned the Batteries {} at time: {}'
                  .format(n["usecase"], inter_day_count, day_count, BRS_results[BRS_req], return_timestep))


            if ID.debug:
                print('DEBUG: After usecase {} returned Car {}, Cars in the Store {} '
                      .format(n["usecase"],BRS_results[BRS_req], ID.MBFleet.items))
                print('')

            # fill simpy process log:
            logger.BRS_process_log[BRS_global_count,] = [day_count, n["usecase"], inter_day_count, departure_timestep,
                                                         leaving_SOC, used_charge,
                                                         return_timestep, charge_timestep, BRS_results[BRS_req]]

        else:  # if we didn't get MB in time: we quit the rental

            # bug fix: if we land in else but brs_req is triggered, return the mb instantly
            if BRS_req.triggered:
                print('--- request triggered after time out ----')
                mb = yield BRS_req
                ID.MBFleet.put(mb)

            print('-----------ALARM------------------------------')
            print('BRS: On day {} at Time {} the UseCase {}_{}  failed bacause it waited {}'
                  .format(day_count, env.now, n["usecase"], inter_day_count, wait))

            # so far unused fail timestep
            fail_timestep = env.now

            # failed rental statistic
            global failed_BRS_count
            failed_BRS_count += 1

            # fill simpy process log:
            logger.BRS_process_log[BRS_global_count,] = [day_count, n["usecase"], inter_day_count, "failed",
                                                         "failed", "failed",
                                                         "failed", "failed", "failed"]




###############################################################################
# Car process function
###############################################################################


def execute_des():

    # generate numpy random object
    rng = np.random.default_rng()

    # global count for all usecase appearances during simulation
    global global_BRS_count
    global_BRS_count = 0
    global global_CRS_count
    global_CRS_count = 0
    global failed_BRS_count
    failed_BRS_count = 0
    global failed_CRS_count
    failed_CRS_count = 0

    # define an environment where the processes live in
    env = simpy.Environment()

    # set up IDM
    ID = InputDataManager(env, 'general_settings')

    # set up logging tables (csv)
    global logger
    logger = Logging(ID)

    # call the function that generates the individual rental processes
    env.process(usecase_gen(env, ID, rng))

    # start the simulation
    env.run()

    # save logging results
    if ID.BRS or ID.REX:
        logger.convert_to_csv(logger.BRS_process_log, logger.BRS_ind_array, global_BRS_count)
        logger.save(logger.BRS_process_log, logger.BRS_ind_array, ID.BRS_capacity, 'brs')

    if ID.CRS:
        logger.convert_to_csv(logger.CRS_process_log, logger.CRS_ind_array, global_CRS_count)
        logger.save(logger.CRS_process_log, logger.CRS_ind_array, ID.CRS_capacity, 'bev')

    # print statistic on ratio of successful vs failed trips/rentals
    print('Total BRS: {}, Failed BRS: {}, Total CRS: {}, Failed CRS: {}'
          .format(global_BRS_count, failed_BRS_count, global_CRS_count, failed_CRS_count))


###############################################################################
# Execution (only if run as standalone file)
###############################################################################

if __name__ == '__main__':
    execute_des()


