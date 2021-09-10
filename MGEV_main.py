"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
September 2nd, 2021

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021

--- Detailed Description ---
This script models an energy system representing a minigrid for rural electrification in sub-saharan Africa and its
interaction with electric vehicles. It transforms the energy system graph into a (mixed-integer) linear program and
transfers it to a solver. The results are saved and the most important aspects visualized.

--- Input & Output ---
The script requires input data in the code block "Input". Additionally, several .csv-files for timeseries data are
required.

--- Requirements ---
This tool requires oemof. Install by "pip install 'oemof.solph>=0.4,<0.5'"
All input data files need to be located in the same directory as this file

--- File Information ---
coding:     utf-8
license:    GPLv3

"""

###############################################################################
# Imports
###############################################################################

from oemof.tools import logger, economics
from oemof import solph
from oemof.solph import processing, views

from datetime import datetime as dt

import matplotlib.pyplot as plt
import importlib
import logging
import os
import pandas as pd
import numpy as np
import pprint as pp
import datetime

###############################################################################
# Function Definitions
###############################################################################

def adjust_capex(ce, me, ls, wacc):
    """
    This function adjusts a component's capex (ce) to include discounted present cost for maintenance (pme)
    """
    pme = 0                 # initialize present maintenance cost variable
    for i in range(0,ls):
        pme += present_cost(me, ls, wacc)
    adj_ce = ce + pme
    return adj_ce

def present_cost(cost, ts, wacc):
    """
    This function calculates the present cost of an actual cost in the future (ts years ahead)
    """
    pc = cost / ((1 + wacc) ** (ts + 1))
    return pc

###############################################################################
# Input
###############################################################################

# Simulation options
sim_name = "MGEV_main"      # name of scenario
sim_solver = "cbc"          # solver selection. Options: "cbc", "gplk", "gurobi"
sim_debug = False           # activates mathematical model saving
sim_ts = dt.now().strftime("%y%m%d%H%M%S")  # create simulation timestamp
sim_tsname = sim_ts + "_" + sim_name
sim_resultpath = os.path.join(os.getcwd(), "results")

# Start logging
logger.define_logging(logfile= sim_ts + sim_name + ".log")
logging.info("Get input data")

# Simulation timeframe
dti_start = "1/1/2021"      # start date of the simulation
dti_freq = 'H'              # time step length ('H'=hourly, '15M'=15 minutes)
dti_num = 24                # total number of equidistant time steps
dti = pd.date_range(dti_start, periods=dti_num, freq=dti_freq)  # create daterange object containing all timesteps

# Project data
wacc = 0.07                 # unitless weighted average cost of capital for the project
epsi = 1e-6                 # minimum variable cost in k$/kWh for transformers to incentivize minimum flow
proj_ls = 25                # project duration in years

# External data file
data_name = "New_data_24.csv"  # input data file containing timeseries for normalized power of RE and electricity demand
data_filepath = os.path.join(os.getcwd(), data_name)
data = pd.read_csv(data_filepath, sep=";")

# AC-DC bus transformer component data
trafo_eff = 0.95            # unitless bidirectional conversion efficiency

# Wind component data
wind_enable = True
wind_ce = 1.355             # capital expenses of the component in $/W
wind_me = 0                 # maintenance expenses of the component in $/(W*year)
wind_oe = 0                 # operational expenses of the component in $/Wh
wind_ls= 20                 # lifespan of the component in years
wind_ace = adjust_capex(wind_ce, wind_me, wind_ls, wacc)  # adjusted ce (including maintenance) of the component in $/W
wind_epc = economics.annuity(capex=wind_ace, n=proj_ls, wacc=wacc, u=wind_ls)

# Photovoltaic array component data
pv_enable = True
pv_ce = 0.8                 # capital expenses of the component in $/W
pv_me = 0                   # maintenance expenses of the component in $/(W*year)
pv_oe = 0                   # operational expenses of the component in $/Wh
pv_ls = 25                  # lifespan of the component in years
pv_ace = adjust_capex(pv_ce, pv_oe, pv_ls, wacc)  # adjusted ce (including maintenance) of the component in $/W
pv_epc = economics.annuity(capex=pv_ace, n=proj_ls, wacc=wacc, u=pv_ls)

# Diesel generator component data
gen_enable = False
gen_ce = 1.15               # capital expenses of the component in $/W
gen_me = 0                  # maintenance expenses of the component in $/(W*year)
gen_oe = 0.00036            # operational expenses of the component in $/Wh
gen_ls = 10                 # lifespan of the component in years
gen_ace = adjust_capex(gen_ce, gen_oe, gen_ls, wacc)  # adjusted ce (including maintenance) of the component in $/W
gen_epc = economics.annuity(capex=gen_ace, n=proj_ls, wacc=wacc, u=gen_ls)

# Stationary storage system component data
ess_enable = True
ess_ce = 0.8                # capital expenses of the component in $/Wh
ess_me = 0                  # maintenance expenses of the component in $/(Wh*year)
ess_oe = 0                  # operational expenses of the component in $/Wh
ess_ls = 10                 # lifespan of the component in years
ess_chg_eff = 0.95          # charging efficiency
ess_dis_eff = 0.85          # discharge efficiency
ess_chg_crate = 0.5         # maximum charging C-rate in 1/h
ess_dis_crate = 0.5         # maximum discharging C-rate in 1/h
ess_init_soc = 0.5          # initial state of charge
ess_sd = 0                  # self-discharge rate of the component in ???
ess_ace = adjust_capex(ess_ce, ess_me, ess_ls, wacc)  # adjusted ce (including maintenance) of the component in $/Wh
ess_epc = economics.annuity(capex=ess_ace, n=proj_ls, wacc=wacc, u=ess_ls)

# BEV
bev_enable = True          #
bev_agr = False             # boolean triggering simulationn of BEVs as a single set of components
bev_num = 10                # number of vehicles to be simulated
bev_chg_pwr = 5000          #
bev_dis_pwr = 5000          #
bev_charge_eff = 0.95       #
bev_discharge_eff = 0.85    #
bev_bat_size = 30000        #
bev_data_name = "ind_car_data.csv"
bev_filepath = os.path.join(os.getcwd(), bev_data_name)
bev_data = pd.read_csv(bev_filepath, sep=";")

##########################################################################
# Initialize oemof energy system instance
##########################################################################

logging.info("Initialize the energy system")
es = solph.EnergySystem(timeindex=dti)

logging.info("Create oemof objects")

##########################################################################
# Create basic two-bus structure
#             dc_bus              ac_bus
#               |                   |
#               |---dc_ac---------->|-->dem
#               |                   |
#               |<----------ac_dc---|-->exc
##########################################################################

ac_bus = solph.Bus(
    label="ac_bus")
dc_bus = solph.Bus(
    label="dc_bus")
ac_dc = solph.Transformer(
    label="ac_dc",
    inputs={ac_bus: solph.Flow(variable_costs=epsi)},  # variable cost to exclude circular flows
    outputs={dc_bus: solph.Flow()},
    conversion_factors={dc_bus: trafo_eff})
dc_ac = solph.Transformer(
    label="dc_ac",
    inputs={dc_bus: solph.Flow(variable_costs=epsi)},
    outputs={ac_bus: solph.Flow()},
    conversion_factors={ac_bus: trafo_eff})
dem = solph.Sink(
    label="dem",
    inputs={ac_bus: solph.Flow(fix=data["demand"],nominal_value=1)})
exc = solph.Sink(
    label="exc",
    inputs={ac_bus: solph.Flow()})
es.add(ac_bus, dc_bus, ac_dc, dc_ac, dem, exc)

##########################################################################
# Create wind power objects and add them to the energy system
#             ac_bus             wind_bus
#               |                   |
#               |<--------wind_ac---|<--wind_src
#               |                   |
#                                   |-->wind_exc
##########################################################################

if wind_enable:
    wind_bus = solph.Bus(
        label='wind_bus')
    wind_ac = solph.Transformer(
        label="wind_ac",
        inputs={wind_bus: solph.Flow(variable_costs=epsi)},
        outputs={ac_bus: solph.Flow()},
        conversion_factors={ac_bus: 1})
    wind_src = solph.Source(
        label="wind_src",
        outputs={wind_bus: solph.Flow(fix=data["wind_energy"],investment=solph.Investment(ep_costs=wind_epc))})
    wind_exc = solph.Sink(
        label="wind_exc",
        inputs={wind_bus: solph.Flow()})
    es.add(wind_bus, wind_ac, wind_src, wind_exc)

##########################################################################
# Create solar power objects and add them to the energy system
#             dc_bus              pv_bus
#               |                   |
#               |<----------pv_dc---|<--pv_src
#               |                   |
#                                   |-->pv_exc
##########################################################################

if pv_enable:
    pv_bus = solph.Bus(
        label='pv_bus')
    pv_dc = solph.Transformer(
        label="pv_dc",
        inputs={pv_bus: solph.Flow(variable_costs=epsi)},
        outputs={dc_bus: solph.Flow()},
        conversion_factors={dc_bus: 1})
    pv_src = solph.Source(
        label="pv_src",
        outputs={pv_bus: solph.Flow(fix=data["pv_radiation"],investment=solph.Investment(ep_costs=pv_epc))})
    pv_exc = solph.Sink(
        label="pv_exc",
        inputs={pv_bus: solph.Flow()})
    es.add(pv_bus, pv_dc, pv_src, pv_exc)

##########################################################################
# Create diesel generator object and add it to the energy system
#             ac_bus
#               |
#               |<--gen
#               |
##########################################################################

if gen_enable:
    gen_src = solph.Source(
        label='gen_src',
        outputs={ac_bus: solph.Flow(investment=solph.Investment(ep_costs=gen_epc),variable_costs=gen_oe)})
    es.add(gen_src)

##########################################################################
# Create stationary battery storage object and add it to the energy system
#             dc_bus
#               |
#               |<->ess
#               |
##########################################################################

if ess_enable:
    ess = solph.components.GenericStorage(
        label="ess",
        inputs={dc_bus: solph.Flow()},
        outputs={dc_bus: solph.Flow()},
        loss_rate=ess_sd,
        balanced=True,
        initial_storage_level=ess_init_soc,
        invest_relation_input_capacity=ess_chg_crate,
        invest_relation_output_capacity=ess_dis_crate,
        inflow_conversion_factor=ess_chg_eff,
        outflow_conversion_factor=ess_dis_eff,
        investment=solph.Investment(ep_costs=ess_epc),)
    es.add(ess)

##########################################################################
# Create EV objects and add them to the energy system
#
# Option 1: aggregated vehicles
#             ac_bus             bev_bus
#               |<---------bev_ac---|<--bev_src
#               |                   |
#               |---ac_bev--------->|<->bev_ess
#               |                   |
#                                   |-->bev_snk
#
# Option 2: individual vehicles with individual bevx (x=1,2,3,...bev_num) buses
#             ac_bus             bev_bus             bev1_bus
#               |<---------bev_ac---|<-------bev1_bev---|<->bev1_ess
#               |                   |                   |
#               |---ac_bev--------->|---bev_bev1------->|-->bev1_snk
#                                   |
#                                   |                bev2_bus
#                                   |<-------bev2_bev---|<->bev2_ess
#                                   |                   |
#                                   |---bev_bev2------->|-->bev2_snk
##########################################################################

if bev_enable:
    bev_bus = solph.Bus(
        label='bev_bus')
    ac_bev = solph.Transformer(
        label="ac_bev",
        inputs={ac_bus: solph.Flow(variable_costs=epsi)},
        outputs={bev_bus: solph.Flow()},
        conversion_factors={bev_bus: 1})
    bev_ac = solph.Transformer(
        label="bev_ac",
        inputs={bev_bus: solph.Flow(variable_costs=epsi)},
        outputs={ac_bus: solph.Flow()},
        conversion_factors={ac_bus: 1})
    es.add(bev_bus,ac_bev,bev_ac)
    if bev_agr:                                     # When vehicles are aggregated into three basic components
        bev_snk = solph.Sink(                       # Aggregated sink component modelling leaving vehicles
            label="bev_snk",
            inputs={bev_bus: solph.Flow(fix=bev_data["sink_data"],nominal_value=1)})
        bev_src = solph.Source(                     # Aggregated source component modelling arriving vehicles
            label='bev_src',
            outputs={bev_bus: solph.Flow(fix=bev_data['source_data'],nominal_value=1)})
        bev_ess = solph.components.GenericStorage(  # Aggregated storage modelling the connected vehicles' batteries
            label="bev_ess",
            inputs={bev_bus: solph.Flow()},
            outputs={bev_bus: solph.Flow()},
            nominal_storage_capacity=bev_num * bev_bat_size,  # Storage capacity is set to the maximum available,
            # adaptation to different numbers of vehicles happens with the min/max storage levels
            loss_rate=0,
            balanced=False,
            initial_storage_level=None,
            inflow_conversion_factor=1,
            outflow_conversion_factor=1,
            min_storage_level=bev_data['min_charge'],  # This models the varying storage capacity with (dis)connects
            max_storage_level=bev_data['max_charge'])  # This models the varying storage capacity with (dis)connects
        es.add(bev_snk, bev_src, bev_ess)
    else:                                           # When vehicles are modeled individually
        for i in range(0,bev_num):                  # Create individual vehicles having a bus, a storage and a sink
            bus_label = "bev" + str(i + 1) + "_bus"
            snk_label = "bev" + str(i + 1) + "_snk"
            ess_label = "bev" + str(i + 1) + "_ess"
            chg_label = "bev_bev" + str(i + 1)
            dis_label = "bev" + str(i + 1) + "_bev"
            snk_datalabel = 'sink_data_' + str(i + 1)
            chg_datalabel = 'at_charger_' + str(i + 1)
            maxsoc_datalabel = 'max_charge_' + str(i + 1)
            minsoc_datalabel = 'min_charge_' + str(i + 1)
            bevx_bus = solph.Bus(                        # bevx denominates an individual vehicle component
                label=bus_label)
            bev_bevx = solph.Transformer(
                label=chg_label,
                inputs={bev_bus: solph.Flow(nominal_value=bev_chg_pwr, max=bev_data[chg_datalabel], variable_costs=epsi)},
                outputs={bevx_bus: solph.Flow()},
                conversion_factors={bevx_bus: bev_charge_eff})
            bevx_bev = solph.Transformer(
                label=dis_label,
                inputs={bevx_bus: solph.Flow(nominal_value=bev_chg_pwr,max=0, variable_costs=0.000001)},
                outputs={bev_bus: solph.Flow()},
                conversion_factors={bev_bus: bev_discharge_eff})
            bevx_ess = solph.components.GenericStorage(
                label=ess_label,
                nominal_storage_capacity=bev_bat_size,
                inputs={bevx_bus: solph.Flow()},
                outputs={bevx_bus: solph.Flow()},
                loss_rate=0,
                balanced=False,
                initial_storage_level=None,
                inflow_conversion_factor=1,
                outflow_conversion_factor=1,
                max_storage_level=1,
                min_storage_level=bev_data[minsoc_datalabel],)#this ensures the vehicle is charged when it leaves the system
            bevx_snk = solph.Sink(
                label=snk_label,
                inputs={bevx_bus: solph.Flow(fix=bev_data[snk_datalabel], nominal_value=1)})
            es.add(bevx_bus, bevx_bev, bev_bevx, bevx_ess, bevx_snk)

##########################################################################
# Optimize the energy system and store results
##########################################################################

# Formulate the (MI)LP problem
logging.info("Create optimization model")
om = solph.Model(es)

if sim_debug:
    om.write("./lp_models/" + sim_ts + "_" + sim_name + ".lp", io_options={'symbolic_solver_labels': True})  # write
    # the lp file for debugging or other reasons

# Solve the optimization problem
logging.info("Solve the optimization problem")
om.solve(solver=sim_solver, solve_kwargs={"tee": True})  # If "tee" is True, solver messages will be displayed

# Add the results to the energy system object and dump it as an .oemof file
logging.info("Save model and result data into an energy system file")
es.results["main"] = solph.processing.results(om)
es.results["meta"] = solph.processing.meta_results(om)
es.dump(sim_resultpath,sim_tsname + ".oemof")

# Create a pandas dataframe from the results and dump it as a .csv file
es_results = processing.create_dataframe(om)
es_results.to_csv(os.path.join(sim_resultpath,sim_tsname + ".csv"), sep=';')

##########################################################################
# Display key results in text
##########################################################################

logging.info("Display key results")
results = solph.processing.results(om)

total_ce = 0
total_me = 0
total_oe = 0
total_ann = 0
total_en = 0
total_dem = results[(ac_bus, dem)]['sequences']['flow'].sum()
total_bev_chg = 0
total_bev_dis = 0

print("#####")

if wind_enable:
    wind_inv = results[(wind_src, wind_bus)]["scalars"]["invest"]
    wind_ten = results[(wind_src, wind_bus)]['sequences']['flow'].sum()
    wind_tce = wind_inv * wind_ce
    wind_tme = wind_inv * wind_me
    wind_toe = wind_ten * wind_oe
    wind_ann = wind_inv * wind_epc + wind_toe

    print("Wind Power Results:")
    print("Optimum Capacity: " + str(wind_inv) + "W")
    print("Capital Expenses: " + str(wind_tce) + "$")
    print("Maintenance Expenses: " + str(wind_tme) + "$")
    print("Operational Expenses: " + str(wind_toe) + "$")
    print("Total Energy: " + str(wind_ten) + "Wh")
    print("#####")

    total_ce += wind_tce
    total_me += wind_tme
    total_oe += wind_toe
    total_ann += wind_ann

if pv_enable:
    pv_inv = results[(pv_src, pv_bus)]["scalars"]["invest"]
    pv_ten = results[(pv_src, pv_bus)]['sequences']['flow'].sum()
    pv_tce = pv_inv * pv_ce
    pv_tme = pv_inv * pv_me
    pv_toe = pv_ten * pv_oe
    pv_ann = pv_inv * pv_epc + pv_toe

    print("Solar Power Results:")
    print("Optimum Capacity: " + str(pv_inv) + "Wp")
    print("Capital Expenses: " + str(pv_tce) + "$")
    print("Maintenance Expenses: " + str(pv_tme) + "$")
    print("Operational Expenses: " + str(pv_toe) + "$")
    print("Total Energy: " + str(pv_ten) + "Wh")
    print("#####")

    total_ce += pv_tce
    total_ann += pv_ann
    total_me += pv_tme
    total_oe += pv_toe

if gen_enable:
    gen_inv = results[(gen_src, ac_bus)]["scalars"]["invest"]
    gen_ten = results[(gen_src, ac_bus)]['sequences']['flow'].sum()
    gen_tce = gen_inv * gen_ce
    gen_tme = gen_inv * gen_me
    gen_toe = gen_ten * gen_oe
    gen_ann = gen_inv * gen_epc + gen_toe

    print("Diesel Power Results:")
    print("Optimum Capacity: " + str(gen_inv) + "W")
    print("Capital Expenses: " + str(gen_tce) + "$")
    print("Maintenance Expenses: " + str(gen_tme) + "$")
    print("Operational Expenses: " + str(gen_toe) + "$")
    print("Total Energy: " + str(gen_ten) + "Wh")
    print("#####")

    total_ce += gen_tce
    total_ann += gen_ann
    total_me += gen_tme
    total_oe += gen_toe

if ess_enable:
    ess_inv = results[(ess, None)]["scalars"]["invest"]
    ess_ten = results[(ess, dc_bus)]['sequences']['flow'].sum()  # absolute sum needed in the future!!!
    ess_tce = ess_inv * ess_ce
    ess_tme = ess_inv * ess_me
    ess_toe = ess_ten * ess_oe
    ess_ann = ess_inv * ess_epc + ess_toe

    print("Energy Storage Results:")
    print("Optimum Capacity: " + str(ess_inv) + "W")
    print("Capital Expenses: " + str(ess_tce) + "$")
    print("Maintenance Expenses: " + str(ess_tme) + "$")
    print("Operational Expenses: " + str(ess_toe) + "$")
    print("Total Energy: " + str(ess_ten) + "Wh")
    print("#####")

    total_ce += ess_tce
    total_ann += ess_ann
    total_me += ess_tme
    total_oe += ess_toe

if bev_enable:
    total_bev_chg += solph.views.node(results, 'bev_bus')['sequences'][(('ac_bus', 'bev_bus'), 'flow')].sum()
    total_bev_dis += solph.views.node(results, 'bev_bus')['sequences'][(('bev_bus', 'ac_bus'), 'flow')].sum()
    total_bev_dem = total_bev_chg - total_bev_dis
    total_dem += total_bev_dem

##########################################################################
# LCOE and NPC calculation
##########################################################################

# disc_factor_project = 0
# for i in range(0,proj_ls)
#     disc_factor_project += 1 / ((1 + wacc) ** (i+1))
#
# disc_wind_maint_project = wind_maint * disc_factor_project #Calculate the discounted maintainacne for the project lifespan
# disc_pv_maint_project = pv_maint * disc_factor_project
# disc_gen_maint_project = gen_maint * disc_factor_project
# disc_ess_maint_project = ess_maint * disc_factor_project
#
# disc_total_energy = yearly_prod_energy * disc_factor_project
# disc_used_energy = yearly_used_energy * disc_factor_project
#
# disc_fuel_cost = yearly_fuel_cost * disc_factor_project
#
# npc = (wind_epc * wind_invest + pv_epc * pv_invest + gen_epc * gen_invest + ess_epc * ess_invest + yearly_fuel_cost) \
#       * disc_factor_project
# prod_lcoe = npc/disc_total_energy
# used_lcoe = npc/disc_used_energy
#
# print('Invested capacities and energy usage:')
# print('Wind Investment: ' + str(wind_invest) + '[W] and ' + str(wind_invest * wind_capex) + '[Euro] upfront investment')
# print('PV Investment: ' + str(pv_invest) + '[W] and ' + str(pv_invest * pv_capex) + '[Euro] upfront investment')
# print('Generator Investment: ' + str(gen_invest) + '[W] and ' + str(gen_invest * gen_capex) + '[Euro] upfront investment')
# print('Storage Investment: ' + str(ess_invest) + '[Wh] and ' + str(ess_invest * ess_capex) + '[Euro] upfront investment')
#
# print('Yearly fuel cost: ' + str(yearly_fuel_cost) + 'Euro')
# print('Yearly maintenance: ' + str(maint_sum) + 'Euro')
#
# print('BEV demand: ' + str(bev_sum) + '[Wh]')
# print('V2G usage: ' + str(v2g_sum) + '[Wh]')
#
# print('Yearly produced energy: ' + str(yearly_prod_energy) + '[Wh]')
# print('Yearly used energy: ' + str(yearly_used_energy) + '[Wh]')
#
# print('Economic metrics: ')
# print('Initial Investment: ' + str(initial_invest) + '[Euro]')
# print('Annuity (Investments + Maintenance + Fuel): ' + str(annuity) + '[Euro]')
# print('LCOE (produced energy): ' + str(prod_lcoe*1000) + '[Euro/kWh]')
# print('LCOE (used energy): ' + str(used_lcoe*1000) + '[Euro/kWh]')
# print('NPC: ' + str(npc) + '[Euro]')

##########################################################################
# Save the results
##########################################################################

# file_name = os.path.basename(__file__)
# file_name = file_name[:-3]
# now = datetime.datetime.now()
#
# #Saving all the parameters
# parameters = solph.processing.parameter_as_dict(energysystem)
# parameters_df = pd.DataFrame.from_dict(parameters)
# parameters_filename = os.path.join(os.getcwd(), file_name + "_parameters_" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".csv")
# #parameters_df.to_csv(parameters_filename, sep=';')
#
# scalars_array = np.zeros([1, 15])
# scalars_array[0][0] = wind_invest
# scalars_array[0][1] = wind_invest * wind_epc
# scalars_array[0][2] = pv_invest
# scalars_array[0][3] = pv_invest * pv_epc
# scalars_array[0][4] = gen_invest
# scalars_array[0][5] = gen_invest * gen_epc
# scalars_array[0][6] = yearly_fuel_cost
# scalars_array[0][7] = ess_invest
# scalars_array[0][8] = ess_invest * ess_epc
# scalars_array[0][9] = yearly_prod_energy
# scalars_array[0][10] = bev_sum + demand_sum
# scalars_array[0][11] = prod_lcoe
# scalars_array[0][12] = used_lcoe
# scalars_array[0][13] = initial_invest
# scalars_array[0][14] = annuity
# scalars_df = pd.DataFrame(scalars_array, columns=['wind invest [W]', 'wind invest [Euro]', 'pv invest [W]',
#                                                   'pv invest [Euro]', 'gen invest [W]', 'gen invest [Euro]',
#                                                   'yearly fuel cost [Euro]', 'ess invest[W]', 'ess invest [Euro]',
#                                                   'Yearly produced energy [Wh]', 'Yearly used energy [Wh]',
#                                                   'LCOE (produced) [Euro / Wh]', 'LCOE (used) [Euro / Wh]',
#                                                   'Upfront invest [Euro]', 'Annuity [Euro / year]'])
# scalars_filename = os.path.join(os.getcwd(), file_name + "_scalars_" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".csv")
# #scalars_df.to_csv(scalars_filename, sep=';')
#
# #Saving selected sequences
# sequences_df = pd.concat([solph.views.node(results, 'wind_bus')['sequences'][(('wind', 'wind_bus'), 'flow')],
#                         solph.views.node(results, 'wind_bus')['sequences'][(('wind_bus', 'wind_ac'), 'flow')],
#                         solph.views.node(results, 'pv_bus')['sequences'][(('pv', 'pv_bus'), 'flow')],
#                         solph.views.node(results, 'pv_bus')['sequences'][(('pv_bus', 'pv_dc'), 'flow')],
#                         solph.views.node(results, 'dc_bus')['sequences'][(('storage', 'dc_bus'), 'flow')],
#                         solph.views.node(results, 'dc_bus')['sequences'][(('dc_bus', 'storage'), 'flow')],
#                         solph.views.node(results, 'generator')['sequences'][(('generator', 'ac_bus'), 'flow')],
#                         solph.views.node(results, 'ac_bus')['sequences'][(('ac_bus', 'ac_dc'), 'flow')],
#                         solph.views.node(results, 'ac_bus')['sequences'][(('ac_bus', 'ac_bigbev'), 'flow')],
#                         solph.views.node(results, 'ac_bus')['sequences'][(('dc_ac', 'ac_bus'), 'flow')]], axis=1)
# sequences_filename = os.path.join(os.getcwd(), file_name + "_sequences_" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".csv")
# #sequences_df.to_csv(sequences_filename, sep=';')

##########################################################################
# Plot the results
##########################################################################

# #plot dc-bus balance
# results_dc_bus = solph.views.node(results, "dc_bus")
# results_dc_bus["sequences"].plot(
#     kind="line", drawstyle="steps-post", title="DC bus balance"
# )
# plt.show()
#
# #plot ac-bus-balance
# results_ac_bus = solph.views.node(results, "ac_bus")
# results_ac_bus["sequences"].plot(
#     kind="line", drawstyle="steps-post", title="AC bus balance"
# )
# plt.show()
#
# #plot wind-bus-balance
# results_wind_bus = solph.views.node(results, "wind_bus")
# results_wind_bus["sequences"].plot(
#     kind="line", drawstyle="steps-post", title="Wind bus balance"
# )
# plt.show()
#
# #plot pv-bus-balance
# results_pv_bus = solph.views.node(results, "pv_bus")
# results_pv_bus["sequences"].plot(
#     kind="line", drawstyle="steps-post", title="PV bus balance"
# )
# plt.show()
#
# #plot the storage balance
# #column_name = (('storage', 'None'), 'storage_content')
# results_storage = solph.views.node(results, "storage")
# results_storage["sequences"].plot(
#     kind="line", drawstyle="steps-post", title="Battery Balance"
# )
# plt.show()
#
# #Plot the bus-balance and battery data for all BEVs
# for i in range(0, number_of_cars):
#     bus_label = 'bev_bus_' + str(i + 1)
#     results_ind_bev_bus = solph.views.node(results, bus_label)
#     results_ind_bev_bus["sequences"].plot(
#         kind="line", drawstyle="steps-post", title=bus_label + 'balance'
#     )
#
#     storage_label = 'bev_storage_' + str(i + 1)
#     results_ind_bev_bat = solph.views.node(results, storage_label)
#     results_ind_bev_bat["sequences"].plot(
#         kind="line", drawstyle="steps-post", title=storage_label + 'balance'
#     )
#     plt.show()


