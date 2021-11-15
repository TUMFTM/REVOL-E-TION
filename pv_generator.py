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
import math

from matplotlib import rcParams
from datetime import datetime, date
from scipy import integrate

###############################################################################
# Function Definitions
###############################################################################

def circ_movmean(array, windowsize):
    oh = math.floor(windowsize/2)
    longarray = np.concatenate((array[-oh:], array, array[:oh]), axis=0)
    output = [np.average(longarray[pos - oh : pos + oh + 1], weights = [1, 1.5, 2, 1.5, 1]) for pos in range(0+oh,len(longarray)-oh)]
    return output

###############################################################################
# Input
###############################################################################

use_api = False  # If true, new data is downloaded via PVGIS APIand no local file is used. Otherwise, a local file is used.

week2watch = 2
hours2watch = list(range(6, 19))
window_size = 5  # odd number required

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

weekly_data = pd.DataFrame()
key_weekdata = np.zeros((53, 7))

for week in range(1, 54):
    dayids = list(pvgis_data[pvgis_data['week'] == week]['dayid'].unique())
    hourly_data = pvgis_data[pvgis_data['dayid'].isin(dayids)].pivot(index='hour', columns='dayid', values='P')

    integrals = []
    [integrals.append(integrate.trapz(hourly_data[dayid])) for dayid in dayids]
    integrals = pd.DataFrame([integrals], columns=hourly_data.columns, index=['int'])
    hourly_data = pd.concat([hourly_data, integrals])

    hourly_data['mean'] = hourly_data.mean(axis=1)
    hourly_data['median'] = hourly_data.drop('mean', axis=1).median(axis=1)
    hourly_data['std'] = hourly_data.drop(['mean', 'median'], axis=1).std(axis=1)

    key_weekdata[week - 1, 0] = week
    key_weekdata[week - 1, 1] = len(hourly_data.columns)
    key_weekdata[week - 1, 2] = hourly_data.loc['int', 'mean']
    key_weekdata[week - 1, 3] = hourly_data.loc['int', 'median']
    key_weekdata[week - 1, 4] = hourly_data.loc['int', 'std']
    key_weekdata[week - 1, 5] = hourly_data.drop(['mean', 'median', 'std'], axis=1).loc['int'].max()
    key_weekdata[week - 1, 6] = hourly_data.drop(['mean', 'median', 'std'], axis=1).loc['int'].min()


    if week == week2watch:
        watchweek = hourly_data

weekly_data = pd.DataFrame(key_weekdata, columns=['isoweek', 'days', 'mean', 'median', 'std', 'max', 'min'])
weekly_data['mean_sm'] = circ_movmean(weekly_data['mean'], window_size)
weekly_data['median_sm'] = circ_movmean(weekly_data['median'], window_size)
weekly_data['std_sm'] = circ_movmean(weekly_data['std'], window_size)
weekly_data['max_sm'] = circ_movmean(weekly_data['max'], window_size)
weekly_data['min_sm'] = circ_movmean(weekly_data['min'], window_size)

print('Weekly data')
print(weekly_data)
print('\n hourly data of examined week')
print(watchweek.head())

##########################################################################
# Save and plot data
##########################################################################

weekly_data.to_csv(path_or_buf=output_filepath)

p300=(0,101/255,189/255,1)
orng=(227/255,114/255,34/255,1)
p540=(0,51/255,89/255,1)
grn=(162/255,173/255,0/255,1)
blk=(0,0,0,1)
gry=(0.5,0.5,0.5,1)

plt.xlabel('ISO calendar week', fontsize=20)
plt.xticks(fontsize=20)
plt.xlim((0, 54))
plt.ylabel('Daily normalized PV energy in h', fontsize=20, loc='top')
plt.yticks(fontsize=20)
plt.ylim((0, 8.5))
plt.scatter(weekly_data['isoweek'], weekly_data['mean'], color=p540, marker='.')
plt.scatter(weekly_data['isoweek'], weekly_data['median'], color=p300, marker='+')
plt.scatter(weekly_data['isoweek'], weekly_data['mean'] + weekly_data['std'], color=grn, marker='.')
plt.scatter(weekly_data['isoweek'], weekly_data['mean'] - weekly_data['std'], color=grn, marker='.')
plt.scatter(weekly_data['isoweek'], weekly_data['max'], color=orng, marker='.')
plt.scatter(weekly_data['isoweek'], weekly_data['min'], color=orng, marker='.')
plt.plot(weekly_data['isoweek'], weekly_data['mean_sm'], color=p540, linewidth=2, label='Mean')
plt.plot(weekly_data['isoweek'], weekly_data['median_sm'], color=p300, linestyle='dashed', linewidth=2, label='Median')
plt.plot(weekly_data['isoweek'], weekly_data['mean_sm'] + weekly_data['std'], color=grn, linewidth=2, label='Std. dev.')
plt.plot(weekly_data['isoweek'], weekly_data['mean_sm'] - weekly_data['std'], color=grn, linewidth=2)
plt.plot(weekly_data['isoweek'], weekly_data['max_sm'], color=orng, linewidth=2, label='Max/Min')
plt.plot(weekly_data['isoweek'], weekly_data['min_sm'], color=orng, linewidth=2)
plt.grid(b=True, axis='y', which='major')
plt.legend(fontsize=16, loc='upper center',ncol=2)
plt.tight_layout()
rcParams.update({'figure.autolayout': True})
plt.show()

watchweek.drop(['mean', 'median', 'std'], axis=1).iloc[0:23].plot.line(x=None, y=None, color='yellow', legend=False)
watchweek['mean'].iloc[0:23].plot.line(x=None, y=None, color='blue', legend=False)
watchweek['median'].iloc[0:23].plot.line(x=None, y=None, color='cyan', legend=False)
watchweek[''].iloc[0:23].plot.line(x=None, y=None, color='yellow', legend=False)
plt.title('Photovoltaic power model')
plt.xlabel('hour of the day')
plt.ylabel('Normalized PV power generation in kW/kWp')
plt.xlim((0, 24))
plt.ylim((0, 1))
plt.show()

fig, ax=plt.subplots(2, 6, sharex=True, figsize=(25,15))
plt.xlim(left=0)
plt.title('PDFs for week ' + str(week2watch))
m=0
for i in range(2):
    for j in range(6):
        hour = hours2watch[m]
        kfactor = watchweek.drop(['mean', 'median', 'std'], axis=1).iloc[hour] / watchweek.drop(['mean', 'median', 'std'], axis=1).iloc[hour].max()
        kfactor.hist(bins=10, ax=ax[i,j])
        ax[i, j].set_title('hour' + str(hour))
        m+=1
plt.show()






