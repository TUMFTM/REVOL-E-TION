"""
--- Tool name ---
Photovoltaics specific power profile generator

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
September 10th, 2021

--- Contributors ---

--- Detailed Description ---
This script uses the European Union's Joint Research Centre PVGIS database to generate representative specific PV power
curves for any location and over any datetime range.

--- Input & Output ---
The script requires input data from the PVGIS "hourly radiation tool available at:
     https://re.jrc.ec.europa.eu/pvg_tools/en/#HR
Necessary inputs are:
    Database:       PVGIS-SARAH
    Start year:     2005
    End year:       2016
    Mounting Type:  Fixed
    Optimize Slope and Azimuth: enabled
    PV power:       enabled
    PV technology:  cSi
    Installed peak power: 1 kWp
    System loss:    0
    Radiation components: enabled

The output consists of a .csv file containing a representative specific PV power for the given datetime range.

--- Requirements ---
This tool requires:
    matplotlib
    numpy
    pandas
    datetime

--- File Information ---
coding:     utf-8
license:    GPLv3

"""

###############################################################################
# Imports
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from datetime import datetime, date
from scipy import integrate

###############################################################################
# Function Definitions
###############################################################################

###############################################################################
# Input
###############################################################################

use_api = False  # If true, new data is downloaded via PVGIS APIand no local file is used. Otherwise, a local file is used.

week2watch = 2

pvgis_filename = "Zatta_CI_1kWp.csv"
pvgis_filepath = os.path.join(os.getcwd(), "scenarios", "pvgis_data", pvgis_filename)
pvgis_dtype = {'time': 'str', 'P': 'float', 'Gb (i)': 'float', 'Gd (i)': 'float', 'Gr (i)': 'float', 'H_sun': 'float', 'T2m': 'float', 'WS10m': 'float', 'Int': 'float'}
pvgis_placename = "Zatta_CI"  # Placename investigated in PVGIS, format: Name_XY, XY being the country identifier
#pvgis_lat =
#pvgis_lon =

output_ts = datetime.now().strftime("%y%m%d%H%M%S")  # create output timestamp
output_filename = output_ts + pvgis_placename + "_weekly.csv"
output_filepath = os.path.join(os.getcwd(), "scenarios", "pvgen_data", output_filename)

##########################################################################
# Download data
##########################################################################

if use_api:
    print("API download feature not implemented yet!")
    #pvgis_data =
else:
    pvgis_data = pd.read_csv(pvgis_filepath, header=10, dtype=pvgis_dtype, parse_dates=['time'], skip_blank_lines=False, skipfooter=13, engine='python')

##########################################################################
# Modify data
##########################################################################

pvgis_data["time"] = pd.to_datetime(pvgis_data.time, format='%Y%m%d:%H%M')
pvgis_data["week"] = pvgis_data["time"].dt.week
start_date = pvgis_data["time"].dt.date.min()
pvgis_data["dayid"] = (pvgis_data['time'].dt.date - start_date).dt.days
pvgis_data["hour"] = pvgis_data["time"].dt.round('H').dt.hour

pvgis_data["P"] = pvgis_data["P"] / 1000

#print(pvgis_data.head(495))
#print(pvgis_data.tail(695))

##########################################################################
# Generate weekwise statistical information
##########################################################################

weekly_data = pd.DataFrame(columns=['isoweek'])
weekly_data['isoweek'] = list(range(1,54))
key_weekdata = np.zeros((53, 5))

for week in range(1, 54):#(1, 54)
    onlyweek = pvgis_data[pvgis_data['week'] == week]
    hourly_data = pd.DataFrame()

    for day in onlyweek['dayid'].unique():
        hourly_data[str(day)] = onlyweek[onlyweek.dayid == day].P.tolist()

    integrals = np.zeros((1, len(hourly_data.columns)))
    for day in range(0,len(hourly_data.columns)):
        integrals[0, day] = integrate.trapz(hourly_data.iloc[:, day])

    hourly_data = hourly_data.append(pd.DataFrame(integrals, index=[24], columns=hourly_data.columns))
    hourly_data['mean'] = hourly_data.mean(axis=1)
    hourly_data['median'] = hourly_data.drop('mean', axis=1).median(axis=1)
    hourly_data['std'] = hourly_data.drop(['mean', 'median'], axis=1).std(axis=1)

    key_weekdata[week-1, 0] = hourly_data.loc[24,'mean']
    key_weekdata[week-1, 1] = hourly_data.loc[24,'median']
    key_weekdata[week-1, 2] = hourly_data.loc[24,'std']
    key_weekdata[week-1, 3] = hourly_data.drop(['mean', 'median', 'std'], axis=1).iloc[24].max()
    key_weekdata[week-1, 4] = hourly_data.drop(['mean', 'median', 'std'], axis=1).iloc[24].min()

    if week == week2watch:
        watchweek = hourly_data

weekly_data['mean'] = key_weekdata[:, 0]
weekly_data['median'] = key_weekdata[:, 1]
weekly_data['std'] = key_weekdata[:, 2]
weekly_data['max'] = key_weekdata[:, 3]
weekly_data['min'] = key_weekdata[:, 4]

##########################################################################
# Save and plot data
##########################################################################

weekly_data.to_csv(path_or_buf=output_filepath)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(title='Photovoltaic power prediction', xlabel='ISO calendar week', ylabel='Daily normalized PV energy generation in kWh/kWp')
plt.ylim((0,8))
plt.plot(weekly_data['isoweek'], weekly_data['mean'])
plt.plot(weekly_data['isoweek'], weekly_data['median'])
plt.plot(weekly_data['isoweek'], weekly_data['mean'] + weekly_data['std'])
plt.plot(weekly_data['isoweek'], weekly_data['mean'] - weekly_data['std'])
plt.plot(weekly_data['isoweek'], weekly_data['max'])
plt.plot(weekly_data['isoweek'], weekly_data['min'])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(title='Photovoltaic power prediction', xlabel='hour of the day', ylabel='Normalized PV power generation in kW/kWp')
plt.ylim((0,1.5))
plt.show()






