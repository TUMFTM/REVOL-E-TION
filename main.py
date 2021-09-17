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
This tool requires oemof.solph. Install by "pip install oemof.solph==0.4.4"
All input data files need to be located in the same directory as this file

--- File Information ---
coding:     utf-8
license:    GPLv3

"""

###############################################################################
# Imports
###############################################################################

from oemof.tools import logger, economics
import oemof.solph as solph
import oemof.outputlib.processing as prcs
import oemof.outputlib.views as views

import logging
import os
import pandas as pd

from datetime import datetime as dt

###############################################################################
# Function Definitions
###############################################################################


def discount(value, deltat, discrate):
    """
    This function calculates the present cost of an actual cost in the future (in year deltat)
    """
    pc = value / ((1 + discrate) ** deltat)  # used to be (deltat + 1) - why?
    return pc


def acc_discount(value, ts, discrate):
    """
    This function calculates the accumulated present cost of a yearly cost in the future (from now to ls years ahead)
    """
    apc = 0
    for year in range(0, ts):
        apc += discount(value, year, discrate)
    return apc


def adj_ce(ce, me, ls, discrate):
    """
    This function adjusts a component's capex (ce) to include discounted present cost for time based maintenance (pme)
    """
    ace = ce + acc_discount(me, ls, discrate)
    return ace


###############################################################################
# Input
###############################################################################

# Simulation options
sim_name = "mg_ev_main"     # name of scenario
sim_solver = "cbc"          # solver selection. Options: "cbc", "gplk", "gurobi"
sim_dump = False            # activates oemof model and result saving
sim_debug = False           # activates mathematical model saving and extended solver output
epsi = 1e-6                 # minimum variable cost in $/Wh for transformers to incentivize minimum flow
sim_ts = dt.now().strftime("%y%m%d%H%M%S")  # create simulation timestamp
sim_tsname = sim_ts + "_" + sim_name
sim_resultpath = os.path.join(os.getcwd(), "results")

# Start logging
logger.define_logging(logfile=sim_tsname + ".log")
logging.info("Get input data")

# Simulation timeframe
dti_start = "1/1/2021"      # start date of the simulation
dti_freq = 'H'              # time step length ('H'=hourly, '15M'=15 minutes)
dti_num = 96                # total number of equidistant time steps
dti = pd.date_range(dti_start, periods=dti_num, freq=dti_freq)  # create daterange object containing all timesteps

# Project data
wacc = 0.07                 # unitless weighted average cost of capital for the project
proj_ls = 25                # project duration in years

# External data file
data_name = "New_data.csv"  # input data file containing timeseries for normalized power of RE and electricity demand
data_filepath = os.path.join(os.getcwd(), data_name)
data = pd.read_csv(data_filepath, sep=";")

# AC-DC bus transformer component data
trafo_eff = 0.95            # unitless bidirectional conversion efficiency

# Wind component data
wind_enable = False
wind_sce = 1.355            # specific capital expenses of the component in $/W
wind_sme = 0                # specific maintenance expenses of the component in $/(W*year)
wind_soe = 0                # specific operational expenses of the component in $/Wh
wind_ls = 20                 # lifespan of the component in years
wind_ace = adj_ce(wind_sce, wind_sme, wind_ls, wacc)  # adjusted ce (including maintenance) of the component in $/W
wind_epc = economics.annuity(capex=wind_ace, n=proj_ls, wacc=wacc, u=wind_ls)

# Photovoltaic array component data
pv_enable = True
pv_sce = 0.8                # specific capital expenses of the component in $/W
pv_sme = 0                  # specific maintenance expenses of the component in $/(W*year)
pv_soe = 0                  # specific operational expenses of the component in $/Wh
pv_ls = 25                  # lifespan of the component in years
pv_ace = adj_ce(pv_sce, pv_soe, pv_ls, wacc)  # adjusted ce (including maintenance) of the component in $/W
pv_epc = economics.annuity(capex=pv_ace, n=proj_ls, wacc=wacc, u=pv_ls)

# Diesel generator component data
gen_enable = True
gen_sce = 1.5             # specific capital expenses of the component in $/W (original 1.15)
gen_sme = 0                 # specific maintenance expenses of the component in $/(W*year)
gen_soe = 0.00065           # specific operational expenses of the component in $/Wh (original 0.00036)
gen_ls = 10                 # lifespan of the component in years
gen_ace = adj_ce(gen_sce, gen_soe, gen_ls, wacc)  # adjusted ce (including maintenance) of the component in $/W
gen_epc = economics.annuity(capex=gen_ace, n=proj_ls, wacc=wacc, u=gen_ls)

# Stationary storage system component data
ess_enable = True
ess_sce = 0.8               # specific capital expenses of the component in $/Wh
ess_sme = 0                 # specific maintenance expenses of the component in $/(Wh*year)
ess_soe = 0                 # specific operational expenses of the component in $/Wh
ess_ls = 10                 # lifespan of the component in years
ess_chg_eff = 0.95          # charging efficiency
ess_dis_eff = 0.85          # discharge efficiency
ess_chg_crate = 0.5         # maximum charging C-rate in 1/h
ess_dis_crate = 0.5         # maximum discharging C-rate in 1/h
ess_init_soc = 0.5          # initial state of charge
ess_sd = 0                  # self-discharge rate of the component in ???
ess_ace = adj_ce(ess_sce, ess_sme, ess_ls, wacc)  # adjusted ce (including maintenance) of the component in $/Wh
ess_epc = economics.annuity(capex=ess_ace, n=proj_ls, wacc=wacc, u=ess_ls)

# BEV
bev_enable = True          #
bev_agr = False             # boolean triggering simulation of BEVs as a single set of components
bev_num = 10                # number of vehicles to be simulated
bev_chg_pwr = 3600          #
bev_dis_pwr = 3600          #
bev_charge_eff = 0.95       #
bev_discharge_eff = 0.9     #
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

src_components = []         #create empty component list to iterate over later when displaying results

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
    label='dem',
    inputs={ac_bus: solph.Flow(actual_value=data['demand'], nominal_value=1, fixed=True)}
    )
exc = solph.Sink(
    label="exc",
    inputs={ac_bus: solph.Flow()})
es.add(ac_bus, dc_bus, ac_dc, dc_ac, dem)#, exc)

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
        outputs={wind_bus: solph.Flow(actual_value=data["wind_energy"], fixed=True, investment=solph.Investment(ep_costs=wind_epc))})
    wind_exc = solph.Sink(
        label="wind_exc",
        inputs={wind_bus: solph.Flow()})
    es.add(wind_bus, wind_ac, wind_src, wind_exc)
    src_components.append('wind')

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
        outputs={pv_bus: solph.Flow(actual_value=data["pv_radiation"], fixed=True, investment=solph.Investment(ep_costs=pv_epc))})
    pv_exc = solph.Sink(
        label="pv_exc",
        inputs={pv_bus: solph.Flow()})
    es.add(pv_bus, pv_dc, pv_src, pv_exc)
    src_components.append('pv')

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
        outputs={ac_bus: solph.Flow(investment=solph.Investment(ep_costs=gen_epc), variable_costs=gen_soe)})
    es.add(gen_src)
    src_components.append('gen')

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
    es.add(bev_bus, ac_bev, bev_ac)
    if bev_agr:                                     # When vehicles are aggregated into three basic components
        bev_snk = solph.Sink(                       # Aggregated sink component modelling leaving vehicles
            label="bev_snk",
            inputs={bev_bus: solph.Flow(actual_value=bev_data["sink_data"], fixed=True, nominal_value=1)})
        bev_src = solph.Source(                     # Aggregated source component modelling arriving vehicles
            label='bev_src',
            outputs={bev_bus: solph.Flow(actual_value=bev_data['source_data'], fixed=True, nominal_value=1)})
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
        for i in range(0, bev_num):                  # Create individual vehicles having a bus, a storage and a sink
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
                inputs={bevx_bus: solph.Flow(nominal_value=bev_chg_pwr, max=0, variable_costs=0.000001)},
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
                min_storage_level=bev_data[minsoc_datalabel],)  # this ensures the vehicle is charged when it leaves the system
            bevx_snk = solph.Sink(
                label=snk_label,
                inputs={bevx_bus: solph.Flow(actual_value=bev_data[snk_datalabel], fixed=True, nominal_value=1)})
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
om.solve(solver=sim_solver, solve_kwargs={"tee": sim_debug})

# Add the results to the energy system object and dump it as an .oemof file
if sim_dump:
    logging.info("Save model and result data into an energy system file")
    es.results["main"] = prcs.results(om)
    es.results["meta"] = prcs.meta_results(om)
    es.dump(sim_resultpath, sim_tsname + ".oemof")

# Create a pandas dataframe from the results and dump it as a .csv file
    es_results = prcs.create_dataframe(om)
    es_results.to_csv(os.path.join(sim_resultpath, sim_tsname + ".csv"), sep=';')

##########################################################################
# Display key results in text, add energies and costs
##########################################################################

logging.info("Display key results")
results = prcs.results(om)

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
    wind_inv = round(results[(wind_src, wind_bus)]["scalars"]["invest"])
    wind_ten = round(results[(wind_src, wind_bus)]['sequences']['flow'].sum())
    wind_tce = round(wind_inv * wind_sce)
    wind_tme = round(wind_inv * wind_sme)
    wind_toe = round(wind_ten * wind_soe)
    wind_ann = round(wind_inv * wind_epc + wind_toe)
    wind_tmpc = round(acc_discount(wind_tme, proj_ls, wacc))  # Total maintenance present cost

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
    pv_inv = round(results[(pv_src, pv_bus)]["scalars"]["invest"])
    pv_ten = round(results[(pv_src, pv_bus)]['sequences']['flow'].sum())
    pv_tce = round(pv_inv * pv_sce)
    pv_tme = round(pv_inv * pv_sme)
    pv_toe = round(pv_ten * pv_soe)
    pv_ann = round(pv_inv * pv_epc + pv_toe)
    pv_tmpc = round(acc_discount(pv_tme, proj_ls, wacc))  # Total maintenance present cost

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
    gen_inv = round(results[(gen_src, ac_bus)]["scalars"]["invest"])
    gen_ten = round(results[(gen_src, ac_bus)]['sequences']['flow'].sum())
    gen_tce = round(gen_inv * gen_sce)
    gen_tme = round(gen_inv * gen_sme)
    gen_toe = round(gen_ten * gen_soe)
    gen_ann = round(gen_inv * gen_epc + gen_toe)
    gen_tmpc = round(acc_discount(gen_tme, proj_ls, wacc))  # Total maintenance present cost

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
    ess_inv = round(results[(ess, None)]["scalars"]["invest"])
    ess_ten = round(results[(ess, dc_bus)]['sequences']['flow'].sum())  # absolute sum needed in the future!!!
    ess_tce = round(ess_inv * ess_sce)
    ess_tme = round(ess_inv * ess_sme)
    ess_toe = round(ess_ten * ess_soe)
    ess_ann = round(ess_inv * ess_epc + ess_toe)
    ess_tmpc = round(acc_discount(ess_tme, proj_ls, wacc))  # Total maintenance present cost

    print("Energy Storage Results:")
    print("Optimum Capacity: " + str(ess_inv) + "Wh")
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
    total_bev_chg = round(results[(ac_bev, bev_bus)]['sequences']['flow'].sum())
    total_bev_dis = round(results[(bev_bus, bev_ac)]['sequences']['flow'].sum())
    total_bev_dem = total_bev_chg - total_bev_dis
    total_dem += total_bev_dem

    print("Electric Vehicle Results:")
    print("Gross Charged Energy: " + str(total_bev_chg) + "Wh")
    print("Net Charged Energy: " + str(total_bev_dem) + "Wh")
    print("#####")

##########################################################################
# LCOE and NPC calculation
##########################################################################

total_discdem = acc_discount(total_dem, proj_ls, wacc)
total_lcoe = round(total_ann/total_dem,4)

print("Economic Results:")
print("Initial Investment: " + str(total_ce) + "USD")
print("Net Present Cost: functionality not implemented yet")
print("Annuity: " + str(total_ann) + "USD")
print("LCOE: " + str(total_lcoe) + "USD/kWh")
print("#####")

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
