###############################################################################
# imports
###############################################################################
import simpy

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('MacOSX')

import numpy as np
from numpy.random import default_rng

from scipy.integrate import quad, simps
import scipy.stats as ss
import pandas as pd
import os

###############################################################################
# generate "2 hügel pdf der Abfahrtswahrscheinlichkeit"
###############################################################################

# generate numpy random object
rng = default_rng()

# Erstelle pdf-der globalen Abfahrts-Wahrscheinlichkeit der aCar Flotte
x = np.linspace(0, 24, 10000)
mu_1, sigma_1 = 8, 2  # mean and standard deviation morning departure
mu_2, sigma_2 = 17, 2  # mean and standard deviation evening departure


def global_departure_pdf(x):
    # Funktion aus 2 normalverteilungs pdf zusammengesetzt (morgens, abends Abfahrt)
    y = 0.6 * (1 / (sigma_1 * np.sqrt(2 * np.pi)) * np.exp(- (x - mu_1) ** 2 / (2 * sigma_1 ** 2))) + \
        0.4 * (1 / (sigma_2 * np.sqrt(2 * np.pi)) * np.exp(- (x - mu_2) ** 2 / (2 * sigma_2 ** 2)))
    return y


# Define function to normalise the PDF
def normalisation(x):
    return simps(global_departure_pdf(x), x)


# Define the distribution using rv_continuous
class GlobalDeparture(ss.rv_continuous):
    def _pdf(self, x, const):
        return (1.0 / const) * global_departure_pdf(x)


# generate Instance of GlobalDeparture Class that represents the distribution
departure_pdf = GlobalDeparture(name="global_departure_distribution", a=0.0)

# Find the normalisation constant first
norm_constant = normalisation(x)

# create pdf, cdf, random samples
pdf = departure_pdf.pdf(x=x, const=norm_constant)
# cdf = departure_pdf.cdf(x=x, const=norm_constant)
# samples = round(departure_pdf.rvs(const=norm_constant, size=1000))

departure_time = round(departure_pdf.rvs(const=norm_constant))

# plt.hist(samples, 10, density=True)
# plt.plot(x, pdf, linewidth=2, color='r')
# plt.plot(x, cdf, linewidth=2, color='r')
# plt.show()

# print(departure_time)

# Trip_Distance ist log_normalverteilt

trip_distance = rng.lognormal(1, 1)

# Extra Zeit ist log_normalverteilt

trip_extra_time = rng.lognormal(1, 1)

# Durchschnittsgeschwindigkeit
v_mean = 20  # km/h

# Available Cars in the Fleet
capacity = 1

# Daily trip demand :
daily_trip_demand = 2

# simulation duration and temporal resolution
simulated_days = 2
number_of_steps_per_day = 24
number_timesteps = simulated_days * number_of_steps_per_day

# total expected trips:
expected_trips = simulated_days * daily_trip_demand

# total count for all uscase appearances during simulation
total_count = 0

leaving_SOC = 1
used_charge = 10


################################################
class CarRentalSystem(object):
    # This Class represents the Car rental system, its store and how to get and put cars
    def __init__(self, env, capacity):
        self.env = env
        self.CarFleet = simpy.Store(env, capacity=capacity)
        # fill the store with elements = Cars
        for i in range(capacity):
            self.CarFleet.put({i})
            i += 1

    # def driving(self, value):
    # yield self.env.timeout(self.delay)
    # self.CarFleet.put(value)

    def put(self, item):
        # self.env.process(self.driving(value))
        self.CarFleet.put(item)

    def get(self):
        car = self.CarFleet.get()
        return car


###############################################################################
# Creation of numpy-arrays for logging purposes
###############################################################################
class Logging:

    def __init__(self, rows, columns):
        self.rows = rows  # Rows of the logging array
        self.columns = columns  # Columns of the logging Array
        self.car_log = np.zeros([rows, columns], dtype=object)

        self.ind_array = np.zeros([number_timesteps, (3 * capacity)])
        # fill "at_charger" columns with ones for cleaner code
        for car in range(capacity):
            self.ind_array[:, 2 + car * 3] = 1

    def to_csv(self):
        ###############################################################################
        # Logging part: Simulation-Log is transformed to csv that oemof can handle
        ###############################################################################
        print('eins')
        h = 0
        while h < total_count:

            departure_timestep_log = self.car_log[h][3]
            leaving_SOC_log = self.car_log[h][4]
            used_charge_log = self.car_log[h][5]
            return_timestep_log = self.car_log[h][6]
            used_MB_log = self.car_log[h][8]

            if self.car_log[h][8] != 0:  # check if log is empty because we have reached end of sim time

                for k in used_MB_log:

                    j = (departure_timestep_log - 1)
                    while (departure_timestep_log - 1) <= j <= (return_timestep_log + 1):

                        # one timestep before rental set SoC
                        if j == departure_timestep_log - 1:
                            self.ind_array[j][0 + k * 3] += leaving_SOC_log

                        # timestep of rental: remove battery capacity from minigrid
                        if j == departure_timestep_log:
                            self.ind_array[j][1 + k * 3] += used_charge_log
                            self.ind_array[j][2 + k * 3] = 0.0

                        # during rental: set availability to 0
                        if departure_timestep_log < j <= return_timestep_log:
                            self.ind_array[j][2 + k * 3] = 0.0

                        j += 1

            h += 1

    def save(self):
        ###############################################################################
        # .csv File creation: save results to hard-drive, in the current working dir.
        ###############################################################################

        # Save aggregated car data in one csv
        Simulation_Log = pd.DataFrame(self.car_log,
                                      columns=['Day', 'Uscase', 'Day_Count', 'departure_timestep', 'leaving_SOC',
                                               'used_charge',
                                               'return_timestep', 'chargetime', 'used_MBs'])
        save_filename = os.path.join(os.getcwd(), "simulation_log_60MB_30UC_360Days.csv")
        # print(Simulation_Log.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        Simulation_Log.to_csv(save_filename, sep=';')

        # Save indiviudal car data in one csv
        ind_bev_df = pd.DataFrame(self.ind_array)

        for i in range(0, capacity):
            ind_bev_df.rename(columns={i * 3 + 0: 'min_charge_' + str(i + 1),
                                       i * 3 + 1: 'sink_data_' + str(i + 1),
                                       i * 3 + 2: 'at_charger_' + str(i + 1)
                                       }, inplace=True)

        save_filename = os.path.join(os.getcwd(), "ind_mb_data_extended_60MB_30UC_360Days.csv")
        # print(ind_bev_df.to_latex(index=False, caption='A', label='tab:', position='H', column_format='rllllllll'))
        ind_bev_df.to_csv(save_filename, sep=';')


###############################################################################
# SimPy generator function:
# generate the individual processes aka trips
###############################################################################

def trip_gen(env):
    # start the process generator with one instance of the process function called "job()"
    day = 0
    for d in range(simulated_days):
        inter_day_count = 0
        for t in range(daily_trip_demand):
            # make global count globally accessible
            global total_count

            env.process(trip(env, day, total_count, inter_day_count))
            total_count += 1
            inter_day_count += 1

        day += 1
        # make sure that a day has 24h
        yield env.timeout(24)


###############################################################################
# main process function
###############################################################################

def trip(env, day, total_count, inter_day_count):

    # yield env.timeout(round(departure_pdf.rvs(const=norm_constant)))
    yield env.timeout(2)
    print('On day {} at Time {} the Trip {} needs a Car -- Total Trip Count: {}'.format(day, env.now, inter_day_count,
                                                                                        total_count))
    arrive = env.now

    with CRS.get() as req:
        patience = 2
        # Wait for the counter or abort at the end of our tether
        results = yield req | env.timeout(patience)

        wait = env.now - arrive

        if req in results:
            # We got to the counter
            departure_timestep = env.now
            print('On day {} at Time {} the Trip {} got a Car '.format(day, env.now, inter_day_count))
            yield env.timeout(4)
            print('####################')
            print(env.now)
            # this is the return timestep that is used by oemof simulation csv
            return_timestep = env.now

            # this timeout allows oemof to recharge the MBs
            yield env.timeout(6)

            CRS.put(results[req])
            charge_timestep = env.now
            print('On day {} at Time {} the Trip {} returned a Car '.format(day, return_timestep, inter_day_count))

            logger.car_log[total_count,] = [day, results[req], total_count, departure_timestep, leaving_SOC,
                                            used_charge, return_timestep, charge_timestep, results[req]]


        else:
            # We quit
            fail_timestep = env.now
            print('On day {} at Time {} the Trip {} failed bacause it waited {}'.format(day, env.now, inter_day_count,
                                                                                        wait))
            # logger.car_log[total_count, ] = [day, "FAIL", fail_timestep, wait, "FAIL", "FAIL", "FAIL", "FAIL", "FAIL"]


###############################################################################
# Simpy simulation setup
###############################################################################

# create Logging instance
logger = Logging(expected_trips, 9)

# define an environment where the processes live in
env = simpy.Environment()

# instantiate the Car Rental System Object including the CarFleet Store with its methods get and put
CRS = CarRentalSystem(env, capacity)

# call the function that generates the individual rental processes
env.process(trip_gen(env))

# start the simulation
env.run()  # until=200)

# print(logger.car_log)

logger.to_csv()

print(logger.ind_array)

logger.save()
