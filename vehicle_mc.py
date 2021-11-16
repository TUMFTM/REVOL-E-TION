# -*- coding: utf-8 -*-

"""
This script generates the electricity demand caused by electric mobility, as well as the charge the cars return with.
It generates the data based on normal distribution for the departure and arrival times, as well as the driven range.
The data is stored in a csv file, ready to be used in the corresponding oemof model.

"""

###############################################################################
# Imports
###############################################################################

import os
import pandas as pd
import numpy as np
import pprint as pp

###############################################################################
# Creating the data
###############################################################################

# Input parameters
number_timesteps = 8760
steps_per_day = 24

if number_timesteps % steps_per_day != 0:
    raise ValueError('Please simulate full days')

battery_size = 30000
max_charging_power = 5000

avg_leaving_SOC = 1
leaving_SOC_std = 0

range_average = 20000
range_std = 5000

departure_average = 7
departure_std = 1

return_average = 17
return_std = 1

car_number = 10
increment = 1/car_number

# Data creation
ind_array = np.zeros([number_timesteps, 3*car_number]) # Data for individual mobility
agr_array = np.zeros([number_timesteps, 4]) # Data for aggregated mobility
dumb_array = np.zeros([number_timesteps, 1]) # Data for dumb e-mobility (charge full power when connected)

k=0
while k < car_number:
    i = 0
    while i < number_timesteps/steps_per_day:
        departure_time = np.random.normal(departure_average, departure_std)
        departure_timestep = round((steps_per_day/24) * departure_time)
        return_time = round(np.random.normal(return_average, return_std))
        return_timestep = round((steps_per_day/24) * return_time)
        leaving_SOC = np.random.normal(avg_leaving_SOC, leaving_SOC_std)
        while leaving_SOC > 1:
            leaving_SOC = np.random.normal(avg_leaving_SOC, leaving_SOC_std)
        used_charge = round(np.random.normal(range_average, range_std))
        while used_charge > battery_size * leaving_SOC or used_charge < 0:
            used_charge = round(np.random.normal(range_average, range_std))

        missing_capacity = used_charge

        j = 0
        while j < steps_per_day:
            timestep = i*steps_per_day+j
            if j < departure_timestep:
                ind_array[timestep][2 + k*3] += 1

                agr_array[timestep][0] += increment
            if j == departure_timestep-1:
                ind_array[timestep][0 + k*3] += leaving_SOC

                agr_array[timestep][1] += increment * leaving_SOC
            if j == departure_timestep:
                ind_array[timestep][1 + k*3] += used_charge

                agr_array[timestep][3] += battery_size * leaving_SOC
            if j == return_timestep:
                ind_array[timestep][2 + k*3] += 1

                agr_array[timestep][0] += increment
                agr_array[timestep][2] += (battery_size * leaving_SOC - used_charge)

                n = 0

                while missing_capacity > 0:

                    if missing_capacity > max_charging_power/(steps_per_day/24):
                        if timestep + n < number_timesteps:
                            dumb_array[timestep + n] += max_charging_power
                            missing_capacity -= max_charging_power

                    else:
                        dumb_array[timestep + n] += missing_capacity
                        missing_capacity = 0

                    n += 1

            if j > return_timestep:
                ind_array[timestep][2 + k*3] += 1

                agr_array[timestep][0] += increment

            j += 1

        i += 1

    k += 1


#Save indiviudal car data in one csv
ind_bev_df = pd.DataFrame(ind_array)

for i in range(0, car_number):
    ind_bev_df.rename(columns={i*3 + 0: 'min_charge_' + str(i+1),
                               i*3 + 1: 'sink_data_' + str(i+1),
                               i*3 + 2: 'at_charger_' + str(i+1)
                               }, inplace=True)

save_filename = os.path.join(os.getcwd(), "ind_car_data.csv")
ind_bev_df.to_csv(save_filename, sep=';')

#Save aggregated car data in one csv
agr_bev_df= pd.DataFrame(agr_array, columns=['max_charge', 'min_charge', 'source_data', 'sink_data'])
save_filename = os.path.join(os.getcwd(), "agr_car_data.csv")
agr_bev_df.to_csv(save_filename, sep=';')

#Save dumb car data in one csv
dumb_bev_df= pd.DataFrame(dumb_array, columns=['bev_demand'])
save_filename = os.path.join(os.getcwd(), "dumb_car_data.csv")
dumb_bev_df.to_csv(save_filename, sep=';')

