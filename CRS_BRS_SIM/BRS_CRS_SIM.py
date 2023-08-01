###############################################################################
# Imports
###############################################################################
import os
import math
import simpy
import random
import numpy as np
import pandas as pd
import pylightxl as xl
from random import choices
from simpy.core import BoundClass
from simpy.resources import base
from numpy.random import default_rng

###############################################################################
# initial setup for rng and count variables
###############################################################################

# generate numpy random object
rng = default_rng()

# global count for all uscase appearances during simulation
global_BRS_count = 0
global_CRS_count = 0
failed_count = 0


###############################################################################
# custom subclasses from simpy.resources for "multiple store get"
###############################################################################
class MyStoreGet(base.Get):
    def __init__(self, store, amount):
        if amount <= 0:
            raise ValueError('amount(=%s) must be > 0.' % amount)
        self.amount = amount
        """The amount of matter to be taken out of the store."""

        super(MyStoreGet, self).__init__(store)


class MyStorePut(base.Put):
    def __init__(self, store, items):
        self.items = items
        super(MyStorePut, self).__init__(store)


class MyStore(base.BaseResource):
    def __init__(self, env, capacity=float('inf')):
        if capacity <= 0:
            raise ValueError('"capacity" must be > 0.')

        super(MyStore, self).__init__(env, capacity)

        self.items = []

    put = BoundClass(MyStorePut)
    get = BoundClass(MyStoreGet)

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
# input data handling class
###############################################################################

class InputDataManager():

    def __init__(self, env, sheet_name):

        self.cwd = os.getcwd()
        self.input_file = os.path.join(self.cwd, 'input.xlsx')
        self.input_xdb = xl.readxl(fn=self.input_file)  # Excel database of selected file
        self.scenario_names = self.input_xdb.ws_names  # Get list of sheet names, 1 sheet is 1 scenario

        # debug boolean trigger:
        self.debug = xread('debug', sheet_name, self.input_xdb)
        if xread('debug', sheet_name, self.input_xdb) == 'True':
            self.debug = True
        else:
            self.debug = False

        # simulation duration and temporal resolution
        self.simulated_days = xread('simulated_days', sheet_name, self.input_xdb)
        self.number_of_steps_per_day = xread('number_of_steps_per_day', sheet_name, self.input_xdb)

        # number of timesteps used to determine the length of the ind_array
        self.number_timesteps = self.simulated_days * self.number_of_steps_per_day

        # Placeholders for CRS
        self.CRS_capacity = None
        self.daily_CRS_trip_demand = None
        self.CRS_patience = None
        self.REX_patience = None
        self.expected_trips = None
        self.CRS_leaving_SOC = None
        self.used_charge = None
        self.name = None
        self.CarFleet = None
        self.dep_time_values = None
        self.dep_time_weights = None
        self.fix_bat_size = None
        self.avg_velocity = None
        self.energy_consumption = None
        self.min_return_soc = None
        self.mu_trip_length = None
        self.sigma_trip_length = None
        self.mu_idle_time = None
        self.sigma_idle_time = None

        # Placeholders for BRS
        self.BRS_capacity = None
        self.daily_BRS_rental_demand = None
        self.BRS_patience = None
        self.BRS_energycontent = None
        self.list_of_customers = None
        self.MBFleet = None

        # Trigger variable functionality:
        # je nach gewünschter simulation werden nachfolgend die jeweiligen setup funktionen aufgerufen

        # CRS Setup
        if xread('CRS', sheet_name, self.input_xdb) == 'True':
            self.CRS = True
            # call method to initialize all input data for CRS
            self.setup_CRS('CRS_DES')
        else:
            self.CRS = False

        # BRS Setup
        if xread('BRS', sheet_name, self.input_xdb) == 'True':
            self.BRS = True
            # call method to initialize all input data for BRS
            self.setup_BRS('BRS_DES')
        else:
            self.BRS = False

        # REX Setup
        if xread('REX', sheet_name, self.input_xdb) == 'True':
            # trigger to enable REX functionality: Cars can also rent MB
            self.REX = True

            # If REX is activated, but basic BRS is NOT simulated: initialize BRS only for REX usage
            if not self.BRS:
                self.setup_BRS('BRS_DES')
        else:
            self.REX = False

    def setup_CRS(self, sheet_name):
        # Available Cars in the Fleet
        self.CRS_capacity = xread('CRS_capacity', sheet_name, self.input_xdb)

        # Daily trip demand, is the ex ante generated number of trips per day
        # vorerst ausgelagert in __init__ von IDM damit auch für BRS gültig
        # self.daily_trip_demand = xread('daily_trip_demand', sheet_name, self.input_xdb)

        self.daily_CRS_trip_demand = xread('daily_CRS_trip_demand', sheet_name, self.input_xdb)
        # total expected trips: used to determine the car_log length
        self.expected_trips = self.simulated_days * self.daily_CRS_trip_demand

        # starting SOC of a car, required of oemof, before a trip starts
        self.CRS_leaving_SOC = xread('CRS_leaving_SOC', sheet_name, self.input_xdb)

        # used to name and identify columns in csv file
        self.name = 'bev'

        # patience of a trip to wait for a car
        self.CRS_patience = xread('CRS_patience', sheet_name, self.input_xdb)

        # patience of a trip to wait for MB for REX
        self.REX_patience = xread('REX_patience', sheet_name, self.input_xdb)

        # Store setup:
        self.CarFleet = simpy.Store(env, capacity=self.CRS_capacity)
        # fill the store with elements = Cars
        for i in range(self.CRS_capacity):
            self.CarFleet.put({i})
            i += 1

        #################################################################################
        # CRS Departure probability setup
        #################################################################################

        # Erstelle pdf-der globalen Abfahrts-Wahrscheinlichkeit der aCar Flotte
        # Zusammengesetze normalverteilung wurde normiert

        # mean and standard deviation morning departure
        mu_1, sigma_1 = xread('mu_1', sheet_name, self.input_xdb), xread('sigma_1', sheet_name, self.input_xdb)

        # mean and standard deviation evening departure
        mu_2, sigma_2 = xread('mu_2', sheet_name, self.input_xdb), xread('sigma_2', sheet_name, self.input_xdb)

        def global_departure_pdf(x):
            # Funktion aus 2 normalverteilungs pdf zusammengesetzt (morgens, abends Abfahrt)

            y = 0.6 * (1 / (sigma_1 * np.sqrt(2 * np.pi)) * np.exp(- (x - mu_1) ** 2 / (2 * sigma_1 ** 2))) + \
                0.4 * (1 / (sigma_2 * np.sqrt(2 * np.pi)) * np.exp(- (x - mu_2) ** 2 / (2 * sigma_2 ** 2)))
            return y

        # use random choices to sample the global_departure_pdf for every hour of a day
        # in dep_time_values steckt die annahme drin, dass 1h die schritteinheit ist (viertelstunde)
        self.dep_time_values = np.linspace(0, 23, num=24)  # [0,1, 2, 3, 4, 5, 6--23]
        self.dep_time_weights = global_departure_pdf(self.dep_time_values)  # [3,3,3,3,3,3]

        # energy content of the fixed battery of the car
        self.fix_bat_size = xread('fix_bat_size', sheet_name, self.input_xdb)

        # average velocity of the car used to calculate trip length
        self.avg_velocity = xread('avg_velocity', sheet_name, self.input_xdb)

        # energy consumption of the car used to calculate remaining SOC and for REX the number of MB
        self.energy_consumption = xread('energy_consumption', sheet_name, self.input_xdb)

        # min SOC of a Car when returning home, used to calculate range and number of MB
        self.min_return_soc = xread('min_return_soc', sheet_name, self.input_xdb)

        # parameters for the log-normal distribution used to generate each trip-length in [km]
        self.mu_trip_length = xread('mu_trip_length', sheet_name, self.input_xdb)
        self.sigma_trip_length = xread('sigma_trip_length', sheet_name, self.input_xdb)

        # parameters for the log-normal distribution used to generate each idle time in [h]
        self.mu_idle_time = xread('mu_idle_time', sheet_name, self.input_xdb)
        self.sigma_idle_time = xread('sigma_idle_time', sheet_name, self.input_xdb)

    def setup_BRS(self, sheet_name):
        # Available MB in the Fleet
        self.BRS_capacity = xread('BRS_capacity', sheet_name, self.input_xdb)

        # TEST individual daily demands for BRS and CRS
        self.daily_BRS_rental_demand = xread('daily_BRS_rental_demand', sheet_name, self.input_xdb)
        # patience to wait for a MB
        self.BRS_patience = xread('BRS_patience', sheet_name, self.input_xdb)

        # energycontent of one MB in Wh
        self.BRS_energycontent = xread('BRS_energycontent', sheet_name, self.input_xdb)

        self.list_of_customers = self.excel_to_dictlist('MB_Usecases')

        # Store Setup:
        self.MBFleet = MyStore(env, capacity=self.BRS_capacity)
        # fill the store with elements = MB
        for i in range(self.BRS_capacity):
            self.MBFleet.put([i])
            i += 1

    def excel_to_dictlist(self, sheetname):

        workbook = xl.readxl(fn=self.input_file)
        sheet = workbook.ws(ws=sheetname)

        headers = sheet.row(row=1)
        data = []

        for row_idx in range(2, sheet.maxrow + 1):
            row = sheet.row(row=row_idx)
            data_dict = {}
            for col_idx, header in enumerate(headers):
                data_dict[header.lower()] = row[col_idx]
            data.append(data_dict)

        return data


###############################################################################
# function for reading Excel files from mg_ev_opti
###############################################################################

def xread(param, sheet, db):
    """
    Reading parameters from external excel file
    """
    value = None
    try:
        value = db.ws(ws=sheet).keyrow(key=param, keyindex=1)[1]
    except IndexError:
        print(f'Key \"{param}\" not found in Excel worksheet - exiting')
        exit()
    return value


###############################################################################
# logging and saving results class
###############################################################################

class Logging():

    def __init__(self):

        # ich brauche BRS log auch wenn ich REX aktiviert habe, hier werden auch MB ausgeliehen!
        if ID.BRS or ID.REX:

            self.BRS_process_log = np.zeros([ID.simulated_days * ID.daily_BRS_rental_demand + 500, 9], dtype=object)

            self.BRS_ind_array = np.zeros([ID.number_timesteps + 50, (3 * ID.BRS_capacity)])
            # fill "at_charger" columns with ones for compact code
            for mb in range(ID.BRS_capacity):
                self.BRS_ind_array[:, 2 + mb * 3] = 1

        if ID.CRS:

            self.CRS_process_log = np.zeros([ID.simulated_days * ID.daily_CRS_trip_demand + 500, 9], dtype=object)

            self.CRS_ind_array = np.zeros([ID.number_timesteps + 50, (3 * ID.CRS_capacity)])
            # fill "at_charger" columns with ones for compact code
            for car in range(ID.CRS_capacity):
                self.CRS_ind_array[:, 2 + car * 3] = 1

    ###############################################################################
    # Logging part: Simulation-Log is transformed to csv that oemof can handle
    ###############################################################################

    def convert_to_csv(self, process_log, ind_array, global_count):

        h = 0
        while h < global_count:

            # filter out failed trips for the ind_array
            if process_log[h][3] != "failed":

                departure_timestep_log = process_log[h][3]
                leaving_SOC_log = process_log[h][4]
                used_charge_log = process_log[h][5]
                return_timestep_log = process_log[h][6]
                used_Car_log = process_log[h][8]

                if process_log[h][8] != 0 and process_log[h][
                    1] != "FAIL":  # check if log is empty OR trip has status "Fail"

                    for k in used_Car_log:

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
    # .csv File creation: save results to hard-drive, in the current working dir.
    ###############################################################################

    def save(self, process_log, ind_array, fleet_capacity, name):

        # Save raw car log this is then fed into the global combined DES simulation
        Simulation_Log = pd.DataFrame(process_log,
                                      columns=['Day', 'usage', 'day_Count', 'departure_timestep', 'leaving_SOC',
                                               'used_charge', 'return_timestep', 'chargetime', 'used_'f'{name}'])

        save_filename = os.path.join(os.getcwd(), f'{name}_process_log.csv')

        # print(Simulation_Log.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        Simulation_Log.to_csv(save_filename, sep=';')

        # Save indiviudal car data in one csv
        ind_bev_df = pd.DataFrame(ind_array)

        for i in range(0, fleet_capacity):
            # ind_bev_df.rename(columns={i * 3 + 0: 'min_charge_' + str(i + 1),
            #                            i * 3 + 1: 'sink_data_' + str(i + 1),
            #                            i * 3 + 2: 'at_charger_' + str(i + 1)
            #                            }, inplace=True)

            ind_bev_df.rename(columns={i * 3: f'{name}{i}_minsoc',
                                       i * 3 + 1: f'{name}{i}_consumption',
                                       i * 3 + 2: f'{name}{i}_atbase'}, inplace=True)

        save_filename = os.path.join(os.getcwd(), f'{name}_log.csv')
        # print(ind_bev_df.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        ind_bev_df.to_csv(save_filename, sep=';')

    def sort(self, process_log):
        process_log = process_log[process_log[:, 3].argsort()]
        print(process_log)


###############################################################################
# MB SIM
# generate the individual processes aka battery rentals
###############################################################################
def usecase_gen(env):
    # Check input.xlsx for consistency, and input errors
    if not ID.CRS and not ID.BRS:
        print('ACHTUNG: Sowohl BRS als auch CRS ausgeschaltet, Trigger in csv überprüfen')
    if ID.BRS and ID.daily_BRS_rental_demand <= 0:
        print('Achtung: BRS ist aktiviert aber "rental demand" ist 0! (input.xlsx überprüfen)')
    if ID.CRS and ID.daily_CRS_trip_demand <= 0:
        print('Achtung: CRS ist aktiviert aber "trip demand" ist 0! (input.xlsx überprüfen)')

    global global_BRS_count
    global global_CRS_count

    day = 0
    for d in range(ID.simulated_days):  # for the number of simulated days do:

        global BRS_inter_day_count
        global CRS_inter_day_count
        BRS_inter_day_count = 0
        CRS_inter_day_count = 0

        if ID.CRS:
            for t in range(ID.daily_CRS_trip_demand):  # for the number of usecases per day day do:

                # start the CRS process generator with one instance of the process function called "job()"
                env.process(car_process_func(env, day, CRS_inter_day_count, global_CRS_count))
                global_CRS_count += 1
                CRS_inter_day_count += 1

        if ID.BRS:
            for t in range(ID.daily_BRS_rental_demand):  # for the number of usecases per day day do:

                # choose a random BRS usecase from the list
                n = random.choice(ID.list_of_customers)

                # start the BRS process generator with one instance of the process function called "job()"
                env.process(
                    battery_process_func(env, day, BRS_inter_day_count, n["num_batteries"], n, global_BRS_count))
                global_BRS_count += 1
                BRS_inter_day_count += 1

        day += 1

        # make sure that a day has 24h
        # wenn schrittlänge nicht 1h dann muss die 24 umgeändert werden (viertelstunde)
        yield env.timeout(24)


###############################################################################
# Battery process function = rental of Batteries
###############################################################################

def battery_process_func(env, day_count, inter_day_count, resources_required, n, BRS_global_count):
    # draw starting time of usecase from normal distribution
    # via "yield" wait until starting point is reached
    yield env.timeout(round(rng.normal(n["dep_mean"], n["dep_std"])))
    print('On day {} Uscase {}_{} requires: {} Batteries at time: {}'.format(day_count, n["usecase"], inter_day_count,
                                                                             resources_required, env.now))

    arrival = env.now

    with ID.MBFleet.get(resources_required) as BRS_req:
        # Wait for the MBs or abort at the end of patience time
        BRS_results = yield BRS_req | env.timeout(ID.BRS_patience)

        wait = env.now - arrival

        if BRS_req in BRS_results:

            # departure timestep for logging
            departure_timestep = env.now

            print('On day {} Uscase {}_{} rented: {} Batteries at time: {}'
                  .format(day_count, n["usecase"], inter_day_count, resources_required, env.now))

            print(BRS_results[BRS_req])

            # draw the return SOC from a lognormal distribution
            return_soc = rng.lognormal(1, 1)
            # exclude the unlikely case that return_soc >= 100
            while return_soc >= 100:
                return_soc = rng.lognormal(1, 1)

            # remaining charge is calculated as percentual share of return_soc and energycontent
            remaining_charge = round(ID.BRS_energycontent * (return_soc / 100))

            # caluclate used charge
            used_charge = ID.BRS_energycontent - remaining_charge

            # usagetime is calculated based on the used energy
            usagetime = math.floor((n["num_batteries"] * ID.BRS_energycontent) / n["power"])

            # this yield timeout represents the usage time
            yield env.timeout(usagetime)

            # this is the return timestep that is used by oemof simulation csv
            return_timestep = env.now
            leaving_SOC = 1

            # this timeout allows oemof to recharge the MBs
            yield env.timeout(6)

            # after the usage return the ressources to the store.
            # yield resources.put(resources_used)
            yield ID.MBFleet.put(BRS_results[BRS_req])
            charge_timestep = env.now

            print('-----------------------------------------')
            print('UseCase {}_{} that rented on day {}  used: {} Batteries for: {} hours'
                  .format(n["usecase"], inter_day_count, day_count, resources_required, usagetime))

            print('UseCase {}_{} that rented on day {} returned: {} Batteries at time: {}'
                  .format(n["usecase"], inter_day_count, day_count, resources_required, env.now))
            print('-----------------------------------------')

            # fill simpy process log:
            logger.BRS_process_log[BRS_global_count,] = [day_count, n["usecase"], inter_day_count, departure_timestep,
                                                         leaving_SOC, used_charge,
                                                         return_timestep, charge_timestep, BRS_results[BRS_req]]

        else:
            # We quit
            print('-----------ALARM------------------------------')
            print('On day {} at Time {} the UseCase {}_{}  failed bacause it waited {}'
                  .format(day_count, env.now, n["usecase"], inter_day_count, wait))

            fail_timestep = env.now
            global failed_count
            failed_count += 1

            # fill simpy process log:
            logger.BRS_process_log[BRS_global_count,] = [day_count, n["usecase"], inter_day_count, "failed",
                                                         "failed", "failed",
                                                         "failed", "failed", "failed"]

            # bug fix: if we land in else but brs_req is triggered, returne the car instantly
            if BRS_req.triggered:
                print('--- request triggered after time out ----')
                mb = yield BRS_req
                ID.MBFleet.put(mb)


###############################################################################
# Car process function
###############################################################################

def car_process_func(env, day, inter_day_count, CRS_global_count):
    # choose a start time from all hours of a day 0-23, but differently weighted with custom probability function
    # (viertelstunde)
    start_time = choices(ID.dep_time_values, ID.dep_time_weights)
    yield env.timeout(int(start_time[0]))

    if ID.REX:
        print('CRS + REX: On day {} at Time {} the Trip {} needs a Car -- Total Trip Count: {}'
              .format(day, env.now, CRS_global_count, CRS_global_count))
    else:
        print('CRS: On day {} at Time {} the Trip {} needs a Car -- Total Trip Count: {}'
              .format(day, env.now, CRS_global_count, CRS_global_count))

    arrive = env.now

    if ID.debug:
        print('DEBUG: 'f'cars zur auswahl: {ID.CarFleet.items}')
        print('')

    # hier habe ich eine Anfrage für ein Auto gestellt:
    # bisher steht nur der Zeitpunkt der ANFRAGE fest, nicht wann und ob ein Auto ausgeliehen wird
    with ID.CarFleet.get() as CAR_req:

        # Wait for the car or abort by a timeout
        CAR_results = yield CAR_req | env.timeout(ID.CRS_patience)

        wait_for_car = env.now - arrive

        if CAR_req in CAR_results:  # we got a car, now draw trip distance and duration

            # if ID.debug:
            #     print('DEBUG: On day {} at Time {} the Trip {} got the Car {}'
            #           .format(day, env.now, CRS_global_count, CAR_results[CAR_req],))

            # travel distance of each trip follows log-normal:
            distance_travelled = round(rng.lognormal(ID.mu_trip_length, ID.sigma_trip_length))

            # calculate the driving duration of a trip based on distance travelled and avg. velocity
            time_travelled = round(distance_travelled / ID.avg_velocity)

            # consider car state that is not driving and not charging -> parking/ waiting
            # idle_time in [h] based on a log-normal distribution, value rounded to int
            # Attention: if time step is changed, idle time has to be adapted (viertelstunde)
            idle_time = round(rng.lognormal(ID.mu_idle_time, ID.sigma_idle_time))

            # threshold value in [km] used to determine:
            # 1) if trip is feasible with intrinsic range 2) IF MB are necessary for a trip
            range_threshold = round(((1 - ID.min_return_soc) * ID.fix_bat_size) / ID.energy_consumption)

            #######################################_BEGIN_internal_MB_Rental_#########################################

            if ID.REX:  # REX is active, trips requiring MB can now request MB, if needed

                if distance_travelled >= range_threshold:  # if MB is needed for the trip:

                    # calculate the distance that has to be covered by MB
                    distance_to_cover_with_MB = distance_travelled - range_threshold

                    # calculate total MB energy required for the trip
                    required_MB_energy = distance_to_cover_with_MB * ID.energy_consumption

                    # calculate the number of MB needed for the trip
                    # number of MBs can only be an integer, thus value is rounded to the next full int
                    required_number_of_MB = math.ceil(required_MB_energy / ID.BRS_energycontent)

                    # IF REX: REX_threshold is the max number of km possible with the fixed battery
                    used_charge_car = range_threshold * ID.energy_consumption

                    # used charge per battery:
                    used_charge_per_MB = required_MB_energy / required_number_of_MB

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

                            # this represents the usage time that is valid for the Car and the Batteries
                            charge_time = 5
                            yield env.timeout(charge_time)

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
                                                                         ID.CRS_leaving_SOC, used_charge_per_MB,
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
                                                                         departure_timestep, ID.CRS_leaving_SOC,
                                                                         used_charge_car,
                                                                         return_timestep, charge_timestep,
                                                                         CAR_results[CAR_req]]

                        else:  # REX: patience triggered, no MB received

                            # bugfix rental and timeout problem:
                            if MB_req.triggered:
                                print('--- request triggered after time out ----')
                                MB = yield MB_req
                                ID.MBFleet.put(MB)

                            # Hier kann entschieden werden, was passiert wenn ein Trip zwar ein Auto aber keine Mb erhält
                            # Möglichkeiten: 1) we quit the trip 2) trip ohne MB

                            # We quit -> the car returns itself IMMEDIATELY to the Station
                            # ->>> the trip is aborted
                            ID.CarFleet.put(CAR_results[CAR_req])

                            print('----------------ALARM----------------------------------')
                            print('CRS + REX: On day {} at Time {} the Trip {} failed after waiting {} for REX '
                                  .format(day, env.now, CRS_global_count, wait_for_MB))
                            print('-------------------------------------------------------')

                            if ID.debug:
                                print('DEBUG: After Trip {} failed, it returned Car {}, Cars in the Store {} '
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

                    # this represents the usage time that is valid for the Car
                    charge_time = 5
                    yield env.timeout(charge_time)

                    ID.CarFleet.put(CAR_results[CAR_req])

                    print('CRS + REX: On day {} at Time {} the Trip {} returned the Car {}  '
                          .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))

                    if ID.debug:
                        print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
                              .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
                        print('')

                    charge_timestep = env.now

                    # IF REX but MB not necessary, used charge is travel distance * consumption
                    used_charge_car = distance_travelled * ID.energy_consumption

                    # logging for the CRS processes when REX is active, but not needed
                    logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}',
                                                                 inter_day_count,
                                                                 departure_timestep, ID.CRS_leaving_SOC,
                                                                 used_charge_car,
                                                                 return_timestep, charge_timestep,
                                                                 CAR_results[CAR_req]]

            #######################################_END_internal_MB_Rental_#########################################

            if not ID.REX:

                if distance_travelled <= range_threshold:  # check if we can cover the distance without REX

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

                    # this timeout allows oemof to recharge the MBs
                    yield env.timeout(10)

                    charge_timestep = env.now

                    ID.CarFleet.put(CAR_results[CAR_req])

                    # IF REX but not necessary, used charge is travel distance * consumption
                    used_charge_car = distance_travelled * ID.energy_consumption

                    print('CRS: On day {} at Time {} the Trip {} returned the Car {}  '
                          .format(day, env.now, CRS_global_count, CAR_results[CAR_req]))

                    if ID.debug:
                        print('DEBUG: After Trip {} returned Car {}, Cars in the Store {} '
                              .format(CRS_global_count, CAR_results[CAR_req], ID.CarFleet.items))
                        print('')

                    # logging for the CRS processes when REX is deactivated
                    logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
                                                                 departure_timestep, ID.CRS_leaving_SOC,
                                                                 used_charge_car,
                                                                 return_timestep, charge_timestep, CAR_results[CAR_req]]

                else:  # REX is deactivated, thus distances greater than range_threshold can't be covered
                    # trip is cancelled, as intrinsic car range is not sufficient

                    # return car immediately
                    ID.CarFleet.put(CAR_results[CAR_req])

                    print('-----------ALARM------------------------------')
                    print('CRS: On day {} at Time {} the Trip {} failed because intrinsic range not sufficient'
                          .format(day, env.now, CRS_global_count))

                    if ID.debug:
                        print('DEBUG: Trip lenght was {}, max intrinsic range was {} '
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
            global failed_count
            failed_count += 1

            # logging for the CRS processes when REX is deactivated
            logger.CRS_process_log[CRS_global_count,] = [day, f'CRS_{CRS_global_count}', inter_day_count,
                                                         "failed",
                                                         "failed", "failed",
                                                         "failed", "failed",
                                                         "failed"]

            print('-----------ALARM------------------------------')
            print('CRS: On day {} at Time {} the Trip {} failed bacause it waited {} for a car'
                  .format(day, env.now, CRS_global_count, wait_for_car))


###############################################################################
# Simpy simulation setup
###############################################################################

# define an environment where the processes live in
env = simpy.Environment()

# set up IDM
ID = InputDataManager(env, 'general_settings')

# set up logging tables (csv)
logger = Logging()

# call the function that generates the individual rental processes
env.process(usecase_gen(env))

# start the simulation
env.run()

# save logging results
if ID.BRS or ID.REX:
    # logger.sort(logger.BRS_process_log)
    logger.convert_to_csv(logger.BRS_process_log, logger.BRS_ind_array, global_BRS_count)
    logger.save(logger.BRS_process_log, logger.BRS_ind_array, ID.BRS_capacity, 'brs')

if ID.CRS:
    logger.convert_to_csv(logger.CRS_process_log, logger.CRS_ind_array, global_CRS_count)
    logger.save(logger.CRS_process_log, logger.CRS_ind_array, ID.CRS_capacity, 'bev')
