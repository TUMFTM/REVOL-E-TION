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
#plt.plot(x, pdf, linewidth=2, color='r')
# plt.plot(x, cdf, linewidth=2, color='r')
#plt.show()

#print(departure_time)

# Trip_Distance ist log_normalverteilt

trip_distance = rng.lognormal(1, 1)

# Extra Zeit ist log_normalverteilt

trip_extra_time = rng.lognormal(1, 1)

# Durchschnittsgeschwindigkeit
v_mean = 20  # km/h

# Available Cars in the Fleet
available_Cars = 10

# Daily trip demand :
daily_trip_demand= 20

# Simulated Days
simulated_days= 10

# total count for all uscase appearances during simulation
total_count = 0

###############################################################################
# Creation of numpy-arrays for logging purposes
###############################################################################

#  create an array that can store abitrary objects for internal logging purposes

# TO-DO: Dynamically adjust car_log array size depending on sum of all trips
car_log = np.zeros([2000, 9], dtype=object)


###############################################################################
# SimPy generator function:
# generate the individual processes aka trips
###############################################################################

def trip_gen(env):
    # start the process generator with one instance of the process function called "job()"
    day = 0
    for d in range(simulated_days):
        inter_day_count=0
        for t in range(daily_trip_demand):

            # make global count globally accessible
            global total_count

            #print(env.now)
            env.process(trip(env,day, total_count, inter_day_count))
            total_count += 1
            inter_day_count +=1

        day += 1
        # make sure that a day has 24h
        yield env.timeout(24)



###############################################################################
# main process function
###############################################################################

def trip(env, day, total_count, inter_day_count):

    #yield env.timeout(round(departure_pdf.rvs(const=norm_constant)))
    yield env.timeout(2)
    print('On day {} at Time {} the Trip {} needs a Car -- Total Trip Count: {}' .format(day, env.now, inter_day_count, total_count))
    arrive = env.now

    with car_Fleet.get() as req:
        patience = 2
        # Wait for the counter or abort at the end of our tether
        results = yield req | env.timeout(patience)

        wait = env.now - arrive

        if req in results:
            # We got to the counter
            print('On day {} at Time {} the Trip {} got a Car '.format(day, env.now, inter_day_count))
            yield env.timeout(4)
            yield car_Fleet.put(req)
            print('On day {} at Time {} the Trip {} returned a Car '.format(day, env.now, inter_day_count))

        else:
            # We quit
            print('On day {} at Time {} the Trip {} failed Waited {}'.format(day, env.now, inter_day_count, wait))




###############################################################################
# Simpy simulation setup
###############################################################################

# define an environment where the processes live in
env = simpy.Environment()

# define the store where the aCar-Fleet is defined with "capacity" cars
car_Fleet = simpy.Store(env, capacity=10)

# fill the store with elements = Cars
i = 0
for i in range(available_Cars):
    car_Fleet.put({i})
    i += 1

# call the function that generates the individual rental processes
env.process(trip_gen(env))

# start the simulation
env.run()#until=200)
