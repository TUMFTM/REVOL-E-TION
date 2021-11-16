'''
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
Created September 2nd, 2021

Last update: September 19th, 2021

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021

--- Detailed Description ---
This script is the main model generator and optimizer for the toolset.
Its results are output to files and key ones printed to the terminal.
Visualization is done via different scripts (to be done)

--- Input & Output ---
The script requires input data in the code block "Input".
Additionally, several .csv-files for timeseries data are required.

--- Requirements ---
see readme

--- File Information ---
coding:     utf-8
license:    GPLv3
'''

###############################################################################
# Imports
###############################################################################

from oemof.tools import logger
import oemof.solph as solph
import oemof.solph.processing as prcs
import oemof.solph.views as views

import logging
import os
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import relativedelta

import functions as fcs
#import vehicle as veh

###############################################################################
# Input
###############################################################################

# Simulation options
sim_name = "mg_ev_main"  # name of scenario
sim_solver = "gurobi"  # solver selection. Options: "cbc", "gplk", "gurobi"
sim_dump = False  # "True" activates oemof model and result saving
sim_debug = False  # "True" activates mathematical model saving and extended solver output
sim_step = 'H'  # time step length ('H'=hourly, other lengths not tested yet!)
sim_eps = 1e-6  # minimum variable cost in $/Wh for transformers to incentivize minimum flow

# Project data
proj_start = "1/1/2015"  # Project start date (DD/MM/YYYY)
proj_sim = 365  # Simulation timeframe in days
proj_ls = 25  # Project duration in years
proj_wacc = 0.07  # unitless weighted average cost of capital for the project

# Demand data file
dem_filename = "dem_data.csv"  # input data file containing timeseries for electricity demand in W

# Transformer component data
ac_dc_eff = 0.95  # unitless conversion efficiency of ac-dc bus transformer component
dc_ac_eff = 0.95  # unitless conversion efficiency of dc-ac bus transformer component

# Wind component data
wind_enable = False
wind_filename = "wind_test.csv"  # name of the normalized wind power profile csv file in ./scenarios to evaluate
wind_sce = 1.355  # specific capital expenses of the component in $/W
wind_sme = 0  # specific maintenance expenses of the component in $/(W*year)
wind_soe = 0  # specific operational expenses of the component in $/Wh
wind_ls = 20  # lifespan of the component in years
wind_cdc = 1  # annual ratio of component cost decrease

# Photovoltaic array component data
pv_enable = True
pv_filename = "Zatta_CI_1kWp.csv"  # name of the normalized pv power profile csv file in ./scenarios to evaluate
pv_sce = 0.8  # specific capital expenses of the component in $/W
pv_sme = 0  # specific maintenance expenses of the component in $/(W*year)
pv_soe = 0  # specific operational expenses of the component in $/Wh
pv_ls = 25  # lifespan of the component in years
pv_cdc = 1  # annual ratio of cost decrease

# Diesel generator component data
gen_enable = True
gen_sce = 1.5  # specific capital expenses of the component in $/W (original 1.15)
gen_sme = 0  # specific maintenance expenses of the component in $/(W*year)
gen_soe = 0.00065  # specific operational expenses of the component in $/Wh (original 0.00036)
gen_ls = 10  # lifespan of the component in years
gen_cdc = 1  # annual ratio of component cost decrease

# Stationary storage system component data
ess_enable = True
ess_sce = 0.8  # specific capital expenses of the component in $/Wh
ess_sme = 0  # specific maintenance expenses of the component in $/(Wh*year)
ess_soe = 0  # specific operational expenses of the component in $/Wh
ess_ls = 10  # lifespan of the component in years
ess_chg_eff = 0.95  # charging efficiency
ess_dis_eff = 0.85  # discharge efficiency
ess_chg_crate = 0.5  # maximum charging C-rate in 1/h
ess_dis_crate = 0.5  # maximum discharging C-rate in 1/h
ess_init_soc = 0.5  # initial state of charge
ess_sd = 0  # self-discharge rate of the component in ???
ess_cdc = 1  # annual ratio of component cost decrease

# BEV
bev_enable = True
bev_agr = False  # boolean triggering simplified simulation of BEVs as a single set of components when true
bev_num = 10  # number of vehicles to be simulated
bev_chg_pwr = 3600  # maximum allowable charge power for each individual BEV
bev_dis_pwr = 3600  # maximum allowable discharge power for each individual BEV
bev_charge_eff = 0.95  # unitless charge efficiency
bev_discharge_eff = 0.95  # unitless discharge efficiency
bev_bat_size = 30000  # battery size of vehicles in Wh
bev_filename = "ind_car_data.csv"

##########################################################################
# Process input data
##########################################################################

sim_ts = datetime.now().strftime("%y%m%d%H%M%S")  # create simulation timestamp
sim_tsname = sim_ts + "_" + sim_name
sim_resultpath = os.path.join(os.getcwd(), "results")
logger.define_logging(logfile=sim_tsname + ".log")
logging.info('Processing inputs')

proj_start = datetime.strptime(proj_start, '%d/%m/%Y')
proj_simend = proj_start + relativedelta(days=proj_sim)
proj_end = proj_start + relativedelta(years=proj_ls)
proj_dur = (proj_end - proj_start).days
proj_simrat = proj_sim / proj_dur
proj_yrrat = proj_sim / 365.25
proj_dti = pd.date_range(start=proj_start, end=proj_simend, freq=sim_step).delete(-1)

dem_filepath = os.path.join(os.getcwd(), "scenarios", dem_filename)
dem_data = pd.read_csv(dem_filepath, sep=",", skip_blank_lines=False)

if wind_enable:
    wind_filepath = os.path.join(os.getcwd(), "scenarios", wind_filename)
    wind_data = pd.read_csv(wind_filepath, sep=",", skip_blank_lines=False)
    wind_ace = fcs.adj_ce(wind_sce, wind_sme, wind_ls,
                          proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
    wind_epc = fcs.ann_recur(wind_ace, wind_ls, proj_ls, proj_wacc, wind_cdc)

if pv_enable:
    pv_filepath = os.path.join(os.getcwd(), "scenarios", "pvgis_data", pv_filename)
    pv_data = pd.read_csv(pv_filepath, sep=",", header=10, skip_blank_lines=False, skipfooter=13, engine='python')
    pv_data['time'] = pd.to_datetime(pv_data['time'], format='%Y%m%d:%H%M')
    pv_data["P"] = pv_data["P"] / 1000
    pv_ace = fcs.adj_ce(pv_sce, pv_soe, pv_ls, proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
    pv_epc = fcs.ann_recur(pv_ace, pv_ls, proj_ls, proj_wacc, pv_cdc)

if gen_enable:
    gen_ace = fcs.adj_ce(gen_sce, gen_soe, gen_ls,
                         proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
    gen_epc = fcs.ann_recur(gen_ace, gen_ls, proj_ls, proj_wacc, gen_cdc)

if ess_enable:
    ess_ace = fcs.adj_ce(ess_sce, ess_sme, ess_ls,
                         proj_wacc)  # adjusted ce (including maintenance) of the component in $/Wh
    ess_epc = fcs.ann_recur(ess_ace, ess_ls, proj_ls, proj_wacc, ess_cdc)

if bev_enable:
    bev_filepath = os.path.join(os.getcwd(), "scenarios", bev_filename)
    bev_data = pd.read_csv(bev_filepath, sep=";")

##########################################################################
# Initialize oemof energy system instance
##########################################################################

logging.info("Initializing the energy system")
es = solph.EnergySystem(timeindex=proj_dti)

logging.info("Creating oemof objects")

src_components = []  # create empty component list to iterate over later when displaying results

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
    inputs={ac_bus: solph.Flow(variable_costs=sim_eps)},  # variable cost to exclude circular flows
    outputs={dc_bus: solph.Flow()},
    conversion_factors={dc_bus: ac_dc_eff})
dc_ac = solph.Transformer(
    label="dc_ac",
    inputs={dc_bus: solph.Flow(variable_costs=sim_eps)},
    outputs={ac_bus: solph.Flow()},
    conversion_factors={ac_bus: dc_ac_eff})
dem = solph.Sink(
    label='dem',
    inputs={ac_bus: solph.Flow(fix=dem_data['P'], nominal_value=1)}
)
exc = solph.Sink(
    label="exc",
    inputs={ac_bus: solph.Flow()})
es.add(ac_bus, dc_bus, ac_dc, dc_ac, dem)  # , exc)

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
        inputs={wind_bus: solph.Flow(variable_costs=sim_eps)},
        outputs={ac_bus: solph.Flow()},
        conversion_factors={ac_bus: 1})
    wind_src = solph.Source(
        label="wind_src",
        outputs={wind_bus: solph.Flow(actual_value=wind_data['P'], fixed=True,
                                      investment=solph.Investment(ep_costs=wind_epc))})
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
        inputs={pv_bus: solph.Flow(variable_costs=sim_eps)},
        outputs={dc_bus: solph.Flow()},
        conversion_factors={dc_bus: 1})
    pv_src = solph.Source(
        label="pv_src",
        outputs={
            pv_bus: solph.Flow(fix=pv_data["P"], investment=solph.Investment(ep_costs=pv_epc))})
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
        investment=solph.Investment(ep_costs=ess_epc), )
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
        inputs={ac_bus: solph.Flow(variable_costs=sim_eps)},
        outputs={bev_bus: solph.Flow()},
        conversion_factors={bev_bus: 1})
    bev_ac = solph.Transformer(
        label="bev_ac",
        inputs={bev_bus: solph.Flow(variable_costs=sim_eps)},
        outputs={ac_bus: solph.Flow()},
        conversion_factors={ac_bus: 1})
    es.add(bev_bus, ac_bev, bev_ac)
    if bev_agr:  # When vehicles are aggregated into three basic components
        bev_snk = solph.Sink(  # Aggregated sink component modelling leaving vehicles
            label="bev_snk",
            inputs={bev_bus: solph.Flow(actual_value=bev_data["sink_data"], fixed=True, nominal_value=1)})
        bev_src = solph.Source(  # Aggregated source component modelling arriving vehicles
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
    else:  # When vehicles are modeled individually
        for i in range(0, bev_num):  # Create individual vehicles having a bus, a storage and a sink
            bus_label = "bev" + str(i + 1) + "_bus"
            snk_label = "bev" + str(i + 1) + "_snk"
            ess_label = "bev" + str(i + 1) + "_ess"
            chg_label = "bev_bev" + str(i + 1)
            dis_label = "bev" + str(i + 1) + "_bev"
            snk_datalabel = 'sink_data_' + str(i + 1)
            chg_datalabel = 'at_charger_' + str(i + 1)
            maxsoc_datalabel = 'max_charge_' + str(i + 1)
            minsoc_datalabel = 'min_charge_' + str(i + 1)
            bevx_bus = solph.Bus(  # bevx denominates an individual vehicle component
                label=bus_label)
            bev_bevx = solph.Transformer(
                label=chg_label,
                inputs={bev_bus: solph.Flow(nominal_value=bev_chg_pwr, max=bev_data[chg_datalabel],
                                            variable_costs=sim_eps)},
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
                min_storage_level=bev_data[
                    minsoc_datalabel], )  # this ensures the vehicle is charged when it leaves the system
            bevx_snk = solph.Sink(
                label=snk_label,
                inputs={bevx_bus: solph.Flow(fix=bev_data[snk_datalabel], nominal_value=1)})
            es.add(bevx_bus, bevx_bev, bev_bevx, bevx_ess, bevx_snk)

##########################################################################
# Optimize the energy system
##########################################################################

# Formulate the (MI)LP problem
logging.info("Creating optimization model")
om = solph.Model(es)

if sim_debug:
    om.write("./lp_models/" + sim_ts + "_" + sim_name + ".lp", io_options={'symbolic_solver_labels': True})  # write
    # the lp file for debugging or other reasons

# Solve the optimization problem
logging.info("Solving the optimization problem")
om.solve(solver=sim_solver, solve_kwargs={"tee": sim_debug})

logging.info("Getting the results from the solver")
results = prcs.results(om)

##########################################################################
# Display key results in text, add energies and costs
##########################################################################

logging.info("Displaying key results")


tot = {}
tot = dict.fromkeys(['ice', 'tce', 'pce', 'yme', 'tme', 'pme', 'yoe', 'toe', 'poe', 'yen', 'ten', 'pen',
                     'yde', 'tde', 'pde', 'ann', 'npc', 'lcoe', 'eta'], 0)
tot['yde'] += results[(ac_bus, dem)]['sequences']['flow'].sum() / proj_yrrat

print("#####")

if wind_enable:
    wind_inv = results[(wind_src, wind_bus)]["scalars"]["invest"]
    wind_ice = wind_inv * wind_sce
    wind_tce = fcs.tce(wind_ice, wind_ice, wind_ls, proj_ls)
    wind_pce = fcs.pce(wind_ice, wind_ice, wind_ls, proj_ls, proj_wacc)
    wind_yen = results[(wind_src, wind_bus)]['sequences']['flow'].sum() / proj_yrrat
    wind_ten = wind_yen * proj_ls
    wind_pen = fcs.acc_discount(wind_yen, proj_ls, proj_wacc)
    wind_yme = wind_inv * wind_sme
    wind_tme = wind_yme * proj_ls
    wind_pme = fcs.acc_discount(wind_yme, proj_ls, proj_wacc)
    wind_yoe = wind_yen * wind_soe
    wind_toe = wind_ten * wind_soe
    wind_poe = fcs.acc_discount(wind_yoe, proj_ls, proj_wacc)
    wind_ann = fcs.ann_recur(wind_ice, wind_ls, proj_ls, proj_wacc, wind_cdc) \
                + fcs.ann_recur(wind_yme + wind_yoe, 1, proj_ls, proj_wacc, 1)

    print("Wind Power Results:")
    print("Optimum Capacity: " + str(round(wind_inv / 1e3)) + " kW")
    print("Initial Capital Expenses: " + str(round(wind_ice)) + " USD")
    print("Yearly Maintenance Expenses: " + str(round(wind_yme)) + " USD")
    print("Yearly Operational Expenses: " + str(round(wind_yoe)) + " USD")
    print("Yearly Produced Energy: " + str(round(wind_ten / 1e6)) + " MWh")
    print("#####")

    tot['ice'] += wind_ice
    tot['tce'] += wind_tce
    tot['pce'] += wind_pce
    tot['yme'] += wind_yme
    tot['tme'] += wind_tme
    tot['pme'] += wind_pme
    tot['yoe'] += wind_yoe
    tot['toe'] += wind_toe
    tot['poe'] += wind_poe
    tot['yen'] += wind_yen
    tot['ten'] += wind_ten
    tot['pen'] += wind_pen
    tot['ann'] += wind_ann

if pv_enable:

    pv_inv = results[(pv_src, pv_bus)]["scalars"]["invest"]
    pv_ice = pv_inv * pv_sce
    pv_tce = fcs.tce(pv_ice, pv_ice, pv_ls, proj_ls)
    pv_pce = fcs.pce(pv_ice, pv_ice, pv_ls, proj_ls, proj_wacc)
    pv_yen = results[(pv_src, pv_bus)]['sequences']['flow'].sum() / proj_yrrat
    pv_ten = pv_yen * proj_ls
    pv_pen = fcs.acc_discount(pv_yen, proj_ls, proj_wacc)
    pv_yme = pv_inv * pv_sme
    pv_tme = pv_yme * proj_ls
    pv_pme = fcs.acc_discount(pv_yme, proj_ls, proj_wacc)
    pv_yoe = pv_yen * pv_soe
    pv_toe = pv_ten * pv_soe
    pv_poe = fcs.acc_discount(pv_yoe, proj_ls, proj_wacc)
    pv_ann = fcs.ann_recur(pv_ice, pv_ls, proj_ls, proj_wacc, pv_cdc) \
                + fcs.ann_recur(pv_yme + pv_yoe, 1, proj_ls, proj_wacc, 1)

    print("Solar Power Results:")
    print("Optimum Capacity: " + str(round(pv_inv / 1e3)) + " kW (peak)")
    print("Initial Capital Expenses: " + str(round(pv_ice)) + " USD")
    print("Yearly Maintenance Expenses: " + str(round(pv_yme)) + " USD")
    print("Yearly Operational Expenses: " + str(round(pv_yoe)) + " USD")
    print("Total Present Cost: " + str(round(pv_pce + pv_pme + pv_poe)) + " USD")
    print("Combined Annuity: " + str(round(pv_ann)) + " USD")
    print("Yearly Produced Energy: " + str(round(pv_yen / 1e6)) + " MWh")
    print("#####")

    tot['ice'] += pv_ice
    tot['tce'] += pv_tce
    tot['pce'] += pv_pce
    tot['yme'] += pv_yme
    tot['tme'] += pv_tme
    tot['pme'] += pv_pme
    tot['yoe'] += pv_yoe
    tot['toe'] += pv_toe
    tot['poe'] += pv_poe
    tot['yen'] += pv_yen
    tot['ten'] += pv_ten
    tot['pen'] += pv_pen
    tot['ann'] += pv_ann

if gen_enable:

    gen_inv = results[(gen_src, ac_bus)]["scalars"]["invest"]
    gen_ice = gen_inv * gen_sce
    gen_tce = fcs.tce(gen_ice, gen_ice, gen_ls, proj_ls)
    gen_pce = fcs.pce(gen_ice, gen_ice, gen_ls, proj_ls, proj_wacc)
    gen_yen = results[(gen_src, ac_bus)]['sequences']['flow'].sum() / proj_yrrat
    gen_ten = gen_yen * proj_ls
    gen_pen = fcs.acc_discount(gen_yen, proj_ls, proj_wacc)
    gen_yme = gen_inv * gen_sme
    gen_tme = gen_yme * proj_ls
    gen_pme = fcs.acc_discount(gen_yme, proj_ls, proj_wacc)
    gen_yoe = gen_yen * gen_soe
    gen_toe = gen_ten * gen_soe
    gen_poe = fcs.acc_discount(gen_yoe, proj_ls, proj_wacc)
    gen_ann = fcs.ann_recur(gen_ice, gen_ls, proj_ls, proj_wacc, gen_cdc) \
                + fcs.ann_recur(gen_yme + gen_yoe, 1, proj_ls, proj_wacc, 1)

    print("Diesel Power Results:")
    print("Optimum Capacity: " + str(round(gen_inv / 1e3)) + " kW")
    print("Initial Capital Expenses: " + str(round(gen_ice)) + " USD")
    print("Yearly Maintenance Expenses: " + str(round(gen_yme)) + " USD")
    print("Yearly Operational Expenses: " + str(round(gen_yoe)) + " USD")
    print("Total Present Cost: " + str(round(gen_pce + gen_pme + gen_poe)) + " USD")
    print("Combined Annuity: " + str(round(gen_ann)) + " USD")
    print("Yearly Produced Energy: " + str(round(gen_yen / 1e6)) + " MWh")
    print("#####")

    tot['ice'] += gen_ice
    tot['tce'] += gen_tce
    tot['pce'] += gen_pce
    tot['yme'] += gen_yme
    tot['tme'] += gen_tme
    tot['pme'] += gen_pme
    tot['yoe'] += gen_yoe
    tot['toe'] += gen_toe
    tot['poe'] += gen_poe
    tot['yen'] += gen_yen
    tot['ten'] += gen_ten
    tot['pen'] += gen_pen
    tot['ann'] += gen_ann

if ess_enable:

    ess_inv = results[(ess, None)]["scalars"]["invest"]
    ess_ice = ess_inv * ess_sce
    ess_tce = fcs.tce(ess_ice, ess_ice, ess_ls, proj_ls)
    ess_pce = fcs.pce(ess_ice, ess_ice, ess_ls, proj_ls, proj_wacc)
    ess_yen = results[(ess, dc_bus)]['sequences']['flow'].sum() / proj_yrrat
    ess_ten = ess_yen * proj_ls
    ess_pen = fcs.acc_discount(ess_yen, proj_ls, proj_wacc)
    ess_yme = ess_inv * ess_sme
    ess_tme = ess_yme * proj_ls
    ess_pme = fcs.acc_discount(ess_yme, proj_ls, proj_wacc)
    ess_yoe = ess_yen * ess_soe
    ess_toe = ess_ten * ess_soe
    ess_poe = fcs.acc_discount(ess_yoe, proj_ls, proj_wacc)
    ess_ann = fcs.ann_recur(ess_ice, ess_ls, ess_ls, proj_wacc, ess_cdc) \
               + fcs.ann_recur(ess_yme + ess_yoe, 1, proj_ls, proj_wacc, 1)

    print("Energy Storage Results:")
    print("Optimum Capacity: " + str(round(ess_inv / 1e3)) + " kWh")
    print("Initial Capital Expenses: " + str(round(ess_ice)) + " USD")
    print("Yearly Maintenance Expenses: " + str(round(ess_yme)) + " USD")
    print("Yearly Operational Expenses: " + str(round(ess_yoe)) + " USD")
    print("Total Present Cost: " + str(round(ess_pce + ess_pme + ess_poe)) + " USD")
    print("Combined Annuity: " + str(round(ess_ann)) + " USD")
    print("Yearly Discharged Energy: " + str(round(ess_yen / 1e6)) + " MWh")
    print("#####")

    tot['ice'] += ess_ice
    tot['tce'] += ess_tce
    tot['pce'] += ess_pce
    tot['yme'] += ess_yme
    tot['tme'] += ess_tme
    tot['pme'] += ess_pme
    tot['yoe'] += ess_yoe
    tot['toe'] += ess_toe
    tot['poe'] += ess_poe
    tot['ann'] += ess_ann

if bev_enable:
    total_bev_chg = results[(ac_bev, bev_bus)]['sequences']['flow'].sum()
    total_bev_dis = results[(bev_bus, bev_ac)]['sequences']['flow'].sum()
    total_bev_dem = total_bev_chg - total_bev_dis
    tot['yde'] += total_bev_dem

    print("Electric Vehicle Results:")
    print("Gross charged energy: " + str(round(total_bev_chg / 1e6)) + " MWh")
    print("Energy fed back (V2G): " + str(round(total_bev_dis / 1e6)) + " MWh")
    print("Net charged energy: " + str(round(total_bev_dem / 1e6)) + " MWh")
    print("#####")

##########################################################################
# LCOE and NPC calculation
##########################################################################

tot['npc'] = tot['pce'] + tot['pme'] + tot['poe']
tot['lcoe'] = tot['npc'] / tot['pen']
tot['eta'] = tot['yde'] / tot['yen']

print("Economic Results:")
print("Yearly supplied energy: " + str(round(tot['yde'] / 1e6)) + " MWh")
print("Yearly generated energy: " + str(round(tot['yen'] / 1e6)) + " MWh")
print("Overall electrical efficiency: " + str(round(tot['eta'] * 100, 1)) + " %")
print("Total Initial Investment: " + str(round(tot['ice'] / 1e6, 2)) + " million USD")
print("Total yearly maintenance expenses: " + str(round(tot['yme'])) + " USD")
print("Total yearly operational expenses: " + str(round(tot['yoe'])) + " USD")
print("Total cost: " + str(round((tot['tce'] + tot['tme'] + tot['toe']) / 1e6, 2)) + " million USD")
print("Total present cost: " + str(round(tot['npc'] / 1e6, 2)) + " million USD")
print("Total annuity: " + str(round(tot['ann'] / 1e3, 1)) + " thousand USD")
print("LCOE: " + str(round(1e5 * tot['lcoe'], 2)) + " USct/kWh")
print("#####")

##########################################################################
# Save the results
##########################################################################

# Add the results to the energy system object and dump it as an .oemof file
if sim_dump:
    logging.info("Save model and result data")
    es.results["main"] = prcs.results(om)
    es.results["meta"] = prcs.meta_results(om)
    es.dump(sim_resultpath, sim_tsname + ".oemof")

    # Create pandas dataframes from the results and dump it as a .csv file
    es_results = prcs.create_dataframe(om)
    es_results.to_csv(os.path.join(sim_resultpath, sim_tsname + "_res_df.csv"), sep=';')
    parameters = prcs.parameter_as_dict(es)
    parameters = pd.DataFrame.from_dict(parameters)
    parameters.to_csv(os.path.join(sim_resultpath, sim_tsname + "_res_dict.csv"), sep=';')

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


##########################################################################
# Plot the results
##########################################################################
p300=(0/255,101/255,189/255,1)
p540=(0/255,51/255,89/255,1)
orng=(227/255,114/255,34/255,1)
grn=(162/255,173/255,0/255,1)

print(results[(gen_src, ac_bus)]['sequences']['flow'].head())

gen_flow = results[(gen_src, ac_bus)]['sequences']['flow']
pv_flow = results[(pv_dc, dc_bus)]['sequences']['flow']
storage_flow = results[(ess, dc_bus)]['sequences']['flow'].subtract(results[(dc_bus, ess)]['sequences']['flow'])
bev_flow = results[(bev_ac, ac_bus)]['sequences']['flow'].subtract(results[(ac_bus, ac_bev)]['sequences']['flow'])
dem_flow = -1 * (results[(ac_bus, dem)]['sequences']['flow'])

plt.plot(gen_flow.index.to_pydatetime(), gen_flow, label='Diesel generator',color=p300, linewidth=4)
plt.plot(pv_flow.index.to_pydatetime(), pv_flow, label='Photovoltaics',color=orng, linewidth=2)
plt.plot(storage_flow.index.to_pydatetime(), storage_flow, label='Battery storage',color=orng, linestyle='dashed', linewidth=2)
plt.plot(bev_flow.index.to_pydatetime(), bev_flow, label='BEV demand', color=grn, linewidth=2)
plt.plot(dem_flow.index.to_pydatetime(), dem_flow, label='Stationary demand',color=grn,linestyle='dashed',linewidth=2)
plt.axhline(y=0, linewidth=1, color='k')
plt.legend(fontsize=20)
plt.ylabel('Power in W', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Local Time', fontsize=20)
plt.xticks(fontsize=20)
#plt.xlim([datetime.date(2015, 4, 23), datetime.date(2015, 4, 26)])
plt.grid(b=True, axis='y', which='major')
plt.show()


