"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Department of Mobility Systems Engineering
School of Engineering and Design
Technical University of Munich
philipp.rosner@tum.de

Created:     September 2nd, 2021
Last update: February 2nd, 2022

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021
Marcel Brödel, B.Sc. - Semester Thesis in progress

--- Detailed Description ---
This script is the main model generator and optimizer for the toolset.
Its results are output to files and key ones printed to the terminal.
Visualization is done via different scripts (to be done)
For further information, see readme

--- Input & Output ---
The script requires input data in the code block "Input".
Additionally, several .csv-files for timeseries data are required.
For further information, see readme

--- Requirements ---
see readme

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Imports
###############################################################################

from oemof.tools import logger
import oemof.solph as solph
import oemof.solph.processing as prcs
from oemof.solph import views

import logging
import os
import pandas as pd
import numpy as np
# from pandas.plotting import register_matplotlib_converters
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime
from dateutil.relativedelta import relativedelta
import time

import functions as fcs
#import parameters as param

from global_optimum import *
from rolling_horizon import *
start = time.time()
print("Start")

###############################################################################
# Input
###############################################################################

# Defining settings in file "parameters.py"

##########################################################################
# Process input data
##########################################################################
sim_ts = datetime.now().strftime("%y%m%d%H%M%S")  # create simulation timestamp
sim_tsname = sim_ts + "_" + param.sim_name
sim_resultpath = os.path.join(os.getcwd(), "results")
logger.define_logging(logfile=sim_tsname + ".log")
logging.info('Processing inputs')

proj_start = datetime.strptime(param.proj_start, '%m/%d/%Y')
proj_simend = proj_start + relativedelta(days=param.proj_sim)
proj_end = proj_start + relativedelta(years=param.proj_ls)
proj_dur = (proj_end - proj_start).days
proj_simrat = param.proj_sim / proj_dur
proj_yrrat = param.proj_sim / 365.25
proj_dti = pd.date_range(start=proj_start, end=proj_simend, freq=param.sim_step).delete(-1)

dem_filepath = os.path.join(os.getcwd(), "scenarios", param.dem_filename)
dem_data = pd.read_csv(dem_filepath, sep=",", skip_blank_lines=False)
dem_data['time'] = pd.date_range(start=param.proj_start, periods=len(dem_data), freq='H')

wind_data = wind_epc = None
if param.sim_enable["wind"]:
    wind_filepath = os.path.join(os.getcwd(), "scenarios", param.wind_filename)
    wind_data = pd.read_csv(wind_filepath, sep=",", skip_blank_lines=False)
    wind_data['time'] = pd.date_range(start=param.proj_start, periods=len(wind_data), freq='H')
    wind_ace = fcs.adj_ce(param.wind_sce, param.wind_sme, param.wind_ls,
                          param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
    wind_epc = fcs.ann_recur(wind_ace, param.wind_ls, param.proj_ls, param.proj_wacc, param.wind_cdc)

pv_data = pv_epc = None
if param.sim_enable["pv"]:
    pv_filepath = os.path.join(os.getcwd(), "scenarios", "pvgis_data", param.pv_filename)
    pv_data = pd.read_csv(pv_filepath, sep=",", header=10, skip_blank_lines=False, skipfooter=13, engine='python')
    pv_data['time'] = pd.to_datetime(pv_data['time'], format='%Y%m%d:%H%M')
    pv_data["P"] = pv_data["P"] / 1e3
    pv_ace = fcs.adj_ce(param.pv_sce, param.pv_soe, param.pv_ls, param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
    pv_epc = fcs.ann_recur(pv_ace, param.pv_ls, param.proj_ls, param.proj_wacc, param.pv_cdc)

gen_epc = None
if param.sim_enable["gen"]:
    gen_ace = fcs.adj_ce(param.gen_sce, param.gen_soe, param.gen_ls,
                         param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
    gen_epc = fcs.ann_recur(gen_ace, param.gen_ls, param.proj_ls, param.proj_wacc, param.gen_cdc)

ess_epc = None
if param.sim_enable["ess"]:
    ess_ace = fcs.adj_ce(param.ess_sce, param.ess_sme, param.ess_ls,
                         param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/Wh
    ess_epc = fcs.ann_recur(ess_ace, param.ess_ls, param.proj_ls, param.proj_wacc, param.ess_cdc)

bev_data = bev_epc = None
if param.sim_enable["bev"]:
    bev_filepath = os.path.join(os.getcwd(), "scenarios", param.bev_filename)
    bev_data = pd.read_csv(bev_filepath, sep=";")
    bev_data['time'] = pd.date_range(start=param.proj_start, periods=len(bev_data), freq='H')
    bev_ace = fcs.adj_ce(param.bev_sce, param.bev_sme, param.bev_ls, param.proj_wacc)
    bev_epc = fcs.ann_recur(bev_ace, param.bev_ls, param.proj_ls, param.proj_wacc, param.bev_cdc)




##########################################################################
# Call simulation strategy
##########################################################################
if param.sim_os == 'go':
    logging.info("Initializing global optimum strategy")
    opt_counter, dem_in, wind_in, pv_in, ess_balancing, bev_in, bev_soc_proj_start, ess_soc_proj_start \
        = opt_strategy(proj_start, dem_data, wind_data, pv_data, bev_data)

if param.sim_os == 'rh':
    if param.sim_cs["wind"] or param.sim_cs["pv"] or param.sim_cs["gen"] or param.sim_cs["ess"] or param.sim_cs["bev"]:
        print('ATTENTION: Rolling horizon strategy is not valid if component sizing is active!')
        print('ATTENTION: Please disable sim_cs in parameters.py')
        exit()
    logging.info("Initializing rolling horizon strategy")
    # Call data for first RH iteration
    opt_counter, proj_start, proj_dti, ess_soc_proj_start, bev_soc_proj_start, ess_balancing, \
    gen_flow, pv_flow, ess_flow, bev_flow, dem_flow, wind_prod, pv_prod, gen_prod, ess_prod, bev_chg, bev_dis, sc_bevx, sc_ess \
        = rh_strategy_init(proj_start)



##########################################################################
# Iterate model (global optimum: 1 loop, rolling horizon: >1 loops
##########################################################################
for oc in range(opt_counter):
    logging.info('Optimization '+str(oc+1)+' of '+str(opt_counter))

    ##########################################################################
    # Update input files to predicted horizon (only for RH strategy)
    ##########################################################################
    if param.sim_os == 'rh':
        dem_in, wind_in, pv_in, bev_in = rh_strategy_dataupdate(proj_dti, dem_data, wind_data, pv_data, bev_data)

    ##########################################################################
    # Initialize oemof energy system instance
    ##########################################################################

    es = solph.EnergySystem(timeindex=proj_dti)
    src_components = []  # create empty component list to iterate over later when displaying results

    ##########################################################################
    # Create basic two-bus structure
    #             dc_bus              ac_bus
    #               |                   |
    #               |---dc_ac---------->|-->dem
    #               |                   |
    #               |<----------ac_dc---|
    ##########################################################################

    ac_bus = solph.Bus(
        label="ac_bus")
    dc_bus = solph.Bus(
        label="dc_bus")
    ac_dc = solph.Transformer(
        label="ac_dc",
        inputs={ac_bus: solph.Flow(variable_costs=param.sim_eps)},  # variable cost to exclude circular flows
        outputs={dc_bus: solph.Flow()},
        conversion_factors={dc_bus: param.ac_dc_eff})
    dc_ac = solph.Transformer(
        label="dc_ac",
        inputs={dc_bus: solph.Flow(variable_costs=param.sim_eps)},
        outputs={ac_bus: solph.Flow()},
        conversion_factors={ac_bus: param.dc_ac_eff})
    dem = solph.Sink(
        label='dem',
        inputs={ac_bus: solph.Flow(fix=dem_in['P'], nominal_value=1)}
    )
    es.add(ac_bus, dc_bus, ac_dc, dc_ac, dem)


    ##########################################################################
    # Create wind power objects and add them to the energy system
    #             ac_bus             wind_bus
    #               |                   |
    #               |<--------wind_ac---|<--wind_src
    #               |                   |
    #                                   |-->wind_exc
    ##########################################################################

    if param.sim_enable["wind"]:
        wind_bus = solph.Bus(
            label='wind_bus')
        wind_ac = solph.Transformer(
            label="wind_ac",
            inputs={wind_bus: solph.Flow(variable_costs=param.sim_eps)},
            outputs={ac_bus: solph.Flow()},
            conversion_factors={ac_bus: 1})
        if param.sim_cs["wind"]:
            wind_src = solph.Source(
                label="wind_src",
                outputs={wind_bus: solph.Flow(fix=wind_in['P'], investment=solph.Investment(ep_costs=wind_epc), variable_cost=param.wind_soe)})
        else:
            wind_src = solph.Source(
                label="wind_src",
                outputs={wind_bus: solph.Flow(fix=wind_in['P'], nominal_value=param.wind_cs, variable_cost=param.wind_soe)})
        wind_exc = solph.Sink(
            label="wind_exc",
            inputs={wind_bus: solph.Flow()})
        es.add(wind_bus, wind_ac, wind_src, wind_exc)
        src_components.append('wind')
    else:
        wind_bus = wind_src = wind_ac = None


    ##########################################################################
    # Create solar power objects and add them to the energy system
    #             dc_bus              pv_bus
    #               |                   |
    #               |<----------pv_dc---|<--pv_src
    #               |                   |
    #                                   |-->pv_exc
    ##########################################################################

    if param.sim_enable["pv"]:
        pv_bus = solph.Bus(
            label='pv_bus')
        pv_dc = solph.Transformer(
            label="pv_dc",
            inputs={pv_bus: solph.Flow(variable_costs=param.sim_eps)},
            outputs={dc_bus: solph.Flow()},
            conversion_factors={dc_bus: 1})
        if param.sim_cs["pv"]:
            pv_src = solph.Source(
                label="pv_src",
                outputs={
                    pv_bus: solph.Flow(fix=pv_in['P'], investment=solph.Investment(ep_costs=pv_epc), variable_cost=param.pv_soe)})
        else:
            pv_src = solph.Source(
                label="pv_src",
                outputs={
                    pv_bus: solph.Flow(fix=pv_in['P'], nominal_value=param.pv_cs, variable_cost=param.pv_soe)})
        pv_exc = solph.Sink(
            label="pv_exc",
            inputs={pv_bus: solph.Flow()})
        es.add(pv_bus, pv_dc, pv_src, pv_exc)
        src_components.append('pv')
    else:
        pv_dc = pv_bus = pv_src = None


    ##########################################################################
    # Create diesel generator object and add it to the energy system
    #             ac_bus
    #               |
    #               |<--gen
    #               |
    ##########################################################################

    if param.sim_enable["gen"]:
        if param.sim_cs["gen"]:
            gen_src = solph.Source(
                label='gen_src',
                outputs={ac_bus: solph.Flow(investment=solph.Investment(ep_costs=gen_epc), variable_costs=param.gen_soe)})
        else:
            gen_src = solph.Source(
                label='gen_src',
                outputs={ac_bus: solph.Flow(nominal_value=param.gen_cs, variable_costs=param.gen_soe)})
        es.add(gen_src)
        src_components.append('gen')
    else:
        gen_src = None



    ##########################################################################
    # Create stationary battery storage object and add it to the energy system
    #             dc_bus
    #               |
    #               |<->ess
    #               |
    ##########################################################################

    if param.sim_enable["ess"]:
        if param.sim_cs["ess"]:
            ess = solph.components.GenericStorage(
                label="ess",
                inputs={dc_bus: solph.Flow()},
                outputs={dc_bus: solph.Flow(variable_cost=param.ess_soe)},
                loss_rate=param.ess_sd,
                balanced=ess_balancing,
                initial_storage_level=ess_soc_proj_start,
                invest_relation_input_capacity=param.ess_chg_crate,
                invest_relation_output_capacity=param.ess_dis_crate,
                inflow_conversion_factor=param.ess_chg_eff,
                outflow_conversion_factor=param.ess_dis_eff,
                investment=solph.Investment(ep_costs=ess_epc),
            )
        else:
            ess = solph.components.GenericStorage(
                label="ess",
                inputs={dc_bus: solph.Flow()},
                outputs={dc_bus: solph.Flow(variable_cost=param.ess_soe)},
                loss_rate=param.ess_sd,
                balanced=ess_balancing,
                initial_storage_level=ess_soc_proj_start,
                invest_relation_input_capacity=param.ess_chg_crate,
                invest_relation_output_capacity=param.ess_dis_crate,
                inflow_conversion_factor=param.ess_chg_eff,
                outflow_conversion_factor=param.ess_dis_eff,
                nominal_storage_capacity=param.ess_cs,
            )
        es.add(ess)
    else:
        ess = None



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

    if param.sim_enable["bev"]:
        bev_bus = solph.Bus(
            label='bev_bus')
        ac_bev = solph.Transformer(
            label="ac_bev",
            inputs={ac_bus: solph.Flow(variable_costs=param.sim_eps)},
            outputs={bev_bus: solph.Flow()},
            conversion_factors={bev_bus: 1})
        bev_ac = solph.Transformer(
            label="bev_ac",
            inputs={bev_bus: solph.Flow(variable_costs=param.sim_eps)},
            outputs={ac_bus: solph.Flow()},
            conversion_factors={ac_bus: 1})
        es.add(bev_bus, ac_bev, bev_ac)
        if param.bev_agr:  # When vehicles are aggregated into three basic components
            bev_snk = solph.Sink(  # Aggregated sink component modelling leaving vehicles
                label="bev_snk",
                inputs={bev_bus: solph.Flow(actual_value=bev_in["sink_data"], fixed=True, nominal_value=1)})
            bev_src = solph.Source(  # Aggregated source component modelling arriving vehicles
                label='bev_src',
                outputs={bev_bus: solph.Flow(actual_value=bev_in['source_data'], fixed=True, nominal_value=1)})
            bev_ess = solph.components.GenericStorage(  # Aggregated storage modelling the connected vehicles' batteries
                label="bev_ess",
                inputs={bev_bus: solph.Flow()},
                outputs={bev_bus: solph.Flow(variable_cost=param.bev_soe)},
                nominal_storage_capacity=param.bev_num * param.bev_cs,  # Storage capacity is set to the maximum available,
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
            for i in range(0, param.bev_num):  # Create individual vehicles having a bus, a storage and a sink
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
                    inputs={bev_bus: solph.Flow(nominal_value=param.bev_chg_pwr, max=bev_in[chg_datalabel],
                                                variable_costs=param.sim_eps)},
                    outputs={bevx_bus: solph.Flow()},
                    conversion_factors={bevx_bus: param.bev_charge_eff})
                # TODO
                bevx_bev = solph.Transformer(
                    label=dis_label,
                    inputs={bevx_bus: solph.Flow(nominal_value=param.bev_dis_pwr, max=0, variable_costs=param.sim_eps)},
                    outputs={bev_bus: solph.Flow()},
                    conversion_factors={bev_bus: param.bev_discharge_eff})
                if param.sim_cs["bev"]:
                    bevx_ess = solph.components.GenericStorage(
                        label=ess_label,
                        inputs={bevx_bus: solph.Flow()},
                        outputs={bevx_bus: solph.Flow(variable_cost=param.bev_soe)},
                        loss_rate=0,
                        balanced=False,
                        initial_storage_level=bev_soc_proj_start[i],
                        inflow_conversion_factor=1,
                        outflow_conversion_factor=1,
                        max_storage_level=1,
                        min_storage_level=bev_in[minsoc_datalabel],  # this ensures the vehicle is charged when leaving
                        investment=solph.Investment(ep_costs=bev_epc),
                    )
                else:
                    bevx_ess = solph.components.GenericStorage(
                        label=ess_label,
                        inputs={bevx_bus: solph.Flow()},
                        outputs={bevx_bus: solph.Flow(variable_cost=param.bev_soe)},
                        loss_rate=0,
                        balanced=False,
                        initial_storage_level=bev_soc_proj_start[i],
                        #initial_storage_level=None,
                        inflow_conversion_factor=1,
                        outflow_conversion_factor=1,
                        max_storage_level=1,
                        min_storage_level=bev_in[minsoc_datalabel],  # this ensures the vehicle is charged when leaving
                        nominal_storage_capacity=param.bev_cs,
                    )
                bevx_snk = solph.Sink(
                    label=snk_label,
                    inputs={bevx_bus: solph.Flow(fix=bev_in[snk_datalabel], nominal_value=1)})
                es.add(bevx_bus, bevx_bev, bev_bevx, bevx_ess, bevx_snk)
    else:
        bev_ac = ac_bev = bev_bus = None




    ##########################################################################
    # Optimize the energy system
    ##########################################################################

    # Formulate the (MI)LP problem
    om = solph.Model(es)

    # Write the lp file for debugging or other reasons
    if param.sim_debug:
        om.write("./lp_models/" + sim_ts + "_" + param.sim_name + ".lp", io_options={'symbolic_solver_labels': True})

    # Solve the optimization problem
    om.solve(solver=param.sim_solver, solve_kwargs={"tee": param.sim_debug})

    # logging.info("Getting the results from the solver")
    results = prcs.results(om)


    ##########################################################################
    # Postprocess iteration data
    ##########################################################################
    if param.sim_os == 'go':
        gen_flow, pv_flow, ess_flow, bev_flow, dem_flow, sc_ess, \
        wind_prod, gen_prod, pv_prod, ess_prod, bev_chg, bev_dis = \
            opt_strategy_postprocessing(results, ac_bus, dc_bus, dem, gen_src, pv_dc, ess, bev_ac, ac_bev,
                                wind_src, wind_bus, pv_src, pv_bus, bev_bus)

    if param.sim_os == 'rh':
        proj_start, proj_dti, \
        dem_flow, pv_flow, gen_flow, ess_flow, bev_flow, \
        wind_prod, pv_prod, gen_prod, ess_prod, bev_chg, bev_dis, \
        ess_soc_proj_start, bev_soc_proj_start, sc_bevx, sc_ess = \
        rh_strategy_postprocessing(proj_start, results, dem, ess,
                                   dem_flow, pv_flow, gen_flow, ess_flow, bev_flow,
                                   wind_prod, pv_prod, gen_prod, ess_prod, bev_chg, bev_dis,
                                   ac_bus, dc_bus, wind_bus, pv_bus, bev_bus,
                                   wind_src, wind_ac, pv_src, pv_dc, gen_src, bev_ac, ac_bev,
                                   ess_soc_proj_start, bev_soc_proj_start, sc_bevx, sc_ess)



##########################################################################
# Display key results in text, add energies and costs
##########################################################################

logging.info("Displaying key results")

tot = dict()
tot = dict.fromkeys(['ice', 'tce', 'pce', 'yme', 'tme', 'pme', 'yoe', 'toe', 'poe', 'ype', 'ten', 'pen',
                     'yde', 'tde', 'pde', 'ann', 'npc', 'lcoe', 'eta'], 0)
tot['yde'] += dem_flow.sum() / proj_yrrat

print("#####")

if param.sim_enable["wind"]:
    if param.sim_cs["wind"]:
        wind_inv = results[(wind_src, wind_bus)]["scalars"]["invest"]
    else:
        wind_inv = param.wind_cs
    wind_ice = wind_inv * param.wind_sce
    wind_tce = fcs.tce(wind_ice, wind_ice, param.wind_ls, param.proj_ls)
    wind_pce = fcs.pce(wind_ice, wind_ice, param.wind_ls, param.proj_ls, param.proj_wacc)
    wind_ype = wind_prod.sum() / proj_yrrat
    wind_ten = wind_ype * param.proj_ls
    wind_pen = fcs.acc_discount(wind_ype, param.proj_ls, param.proj_wacc)
    wind_yme = wind_inv * param.wind_sme
    wind_tme = wind_yme * param.proj_ls
    wind_pme = fcs.acc_discount(wind_yme, param.proj_ls, param.proj_wacc)
    wind_yoe = wind_ype * param.wind_soe
    wind_toe = wind_ten * param.wind_soe
    wind_poe = fcs.acc_discount(wind_yoe, param.proj_ls, param.proj_wacc)
    wind_ann = fcs.ann_recur(wind_ice, param.wind_ls, param.proj_ls, param.proj_wacc, param.wind_cdc) \
               + fcs.ann_recur(wind_yme + wind_yoe, 1, param.proj_ls, param.proj_wacc, 1)

    print("Wind Power Results:")
    if param.sim_cs["wind"]:
        print("Optimum Capacity: " + str(round(wind_inv / 1e3)) + " kW")
    else:
        print("Set Capacity: " + str(param.wind_cs / 1e3) + " kW")
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
    tot['ype'] += wind_ype
    tot['ten'] += wind_ten
    tot['pen'] += wind_pen
    tot['ann'] += wind_ann

if param.sim_enable["pv"]:
    if param.sim_cs["pv"]:
        pv_inv = results[(pv_src, pv_bus)]["scalars"]["invest"]  # [W]
    else:
        pv_inv = param.pv_cs
    pv_ice = pv_inv * param.pv_sce
    pv_tce = fcs.tce(pv_ice, pv_ice, param.pv_ls, param.proj_ls)
    pv_pce = fcs.pce(pv_ice, pv_ice, param.pv_ls, param.proj_ls, param.proj_wacc)
    pv_ype = pv_prod.sum() / proj_yrrat
    pv_ten = pv_ype * param.proj_ls
    pv_pen = fcs.acc_discount(pv_ype, param.proj_ls, param.proj_wacc)
    pv_yme = pv_inv * param.pv_sme
    pv_tme = pv_yme * param.proj_ls
    pv_pme = fcs.acc_discount(pv_yme, param.proj_ls, param.proj_wacc)
    pv_yoe = pv_ype * param.pv_soe
    pv_toe = pv_ten * param.pv_soe
    pv_poe = fcs.acc_discount(pv_yoe, param.proj_ls, param.proj_wacc)
    pv_ann = fcs.ann_recur(pv_ice, param.pv_ls, param.proj_ls, param.proj_wacc, param.pv_cdc) \
             + fcs.ann_recur(pv_yme + pv_yoe, 1, param.proj_ls, param.proj_wacc, 1)

    print("Solar Power Results:")
    if param.sim_cs["pv"]:
        print("Optimum Capacity: " + str(round(pv_inv / 1e3)) + " kW (peak)")
    else:
        print("Set Capacity: " + str(param.pv_cs/ 1e3) + " kW (peak)")
    print("Initial Capital Expenses: " + str(round(pv_ice)) + " USD")
    print("Yearly Maintenance Expenses: " + str(round(pv_yme)) + " USD")
    print("Yearly Operational Expenses: " + str(round(pv_yoe)) + " USD")
    print("Total Present Cost: " + str(round(pv_pce + pv_pme + pv_poe)) + " USD")
    print("Combined Annuity: " + str(round(pv_ann)) + " USD")
    print("Yearly Produced Energy: " + str(round(pv_ype / 1e6)) + " MWh")
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
    tot['ype'] += pv_ype
    tot['ten'] += pv_ten
    tot['pen'] += pv_pen
    tot['ann'] += pv_ann

if param.sim_enable["gen"]:
    if param.sim_cs["gen"]:
        gen_inv = results[(gen_src, ac_bus)]["scalars"]["invest"]
    else:
        gen_inv = param.gen_cs
    gen_ice = gen_inv * param.gen_sce
    gen_tce = fcs.tce(gen_ice, gen_ice, param.gen_ls, param.proj_ls)
    gen_pce = fcs.pce(gen_ice, gen_ice, param.gen_ls, param.proj_ls, param.proj_wacc)
    gen_ype = gen_prod.sum() / proj_yrrat
    gen_ten = gen_ype * param.proj_ls
    gen_pen = fcs.acc_discount(gen_ype, param.proj_ls, param.proj_wacc)
    gen_yme = gen_inv * param.gen_sme
    gen_tme = gen_yme * param.proj_ls
    gen_pme = fcs.acc_discount(gen_yme, param.proj_ls, param.proj_wacc)
    gen_yoe = gen_ype * param.gen_soe
    gen_toe = gen_ten * param.gen_soe
    gen_poe = fcs.acc_discount(gen_yoe, param.proj_ls, param.proj_wacc)
    gen_ann = fcs.ann_recur(gen_ice, param.gen_ls, param.proj_ls, param.proj_wacc, param.gen_cdc) \
                + fcs.ann_recur(gen_yme + gen_yoe, 1, param.proj_ls, param.proj_wacc, 1)

    print("Diesel Power Results:")
    if param.sim_cs["gen"]:
        print("Optimum Capacity: " + str(round(gen_inv / 1e3)) + " kW")
    else:
        print("Set Capacity: " + str(param.gen_cs/ 1e3) + " kW")
    print("Initial Capital Expenses: " + str(round(gen_ice)) + " USD")
    print("Yearly Maintenance Expenses: " + str(round(gen_yme)) + " USD")
    print("Yearly Operational Expenses: " + str(round(gen_yoe)) + " USD")
    print("Total Present Cost: " + str(round(gen_pce + gen_pme + gen_poe)) + " USD")
    print("Combined Annuity: " + str(round(gen_ann)) + " USD")
    print("Yearly Produced Energy: " + str(round(gen_ype / 1e6)) + " MWh")
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
    tot['ype'] += gen_ype
    tot['ten'] += gen_ten
    tot['pen'] += gen_pen
    tot['ann'] += gen_ann

if param.sim_enable["ess"]:
    if param.sim_cs["ess"]:
        ess_inv = results[(ess, None)]["scalars"]["invest"]
    else:
        ess_inv = param.ess_cs
    ess_ice = ess_inv * param.ess_sce
    ess_tce = fcs.tce(ess_ice, ess_ice, param.ess_ls, param.proj_ls)
    ess_pce = fcs.pce(ess_ice, ess_ice, param.ess_ls, param.proj_ls, param.proj_wacc)
    ess_ype = ess_prod.sum() / proj_yrrat
    ess_ten = ess_ype * param.proj_ls
    ess_pen = fcs.acc_discount(ess_ype, param.proj_ls, param.proj_wacc)
    ess_yme = ess_inv * param.ess_sme
    ess_tme = ess_yme * param.proj_ls
    ess_pme = fcs.acc_discount(ess_yme, param.proj_ls, param.proj_wacc)
    ess_yoe = ess_ype * param.ess_soe
    ess_toe = ess_ten * param.ess_soe
    ess_poe = fcs.acc_discount(ess_yoe, param.proj_ls, param.proj_wacc)
    ess_ann = fcs.ann_recur(ess_ice, param.ess_ls, param.ess_ls, param.proj_wacc, param.ess_cdc) \
              + fcs.ann_recur(ess_yme + ess_yoe, 1, param.proj_ls, param.proj_wacc, 1)

    print("Energy Storage Results:")
    if param.sim_cs["ess"]:
        print("Optimum Capacity: " + str(round(ess_inv / 1e3)) + " kWh")
    else:
        print("Set Capacity: " + str(param.ess_cs / 1e3) + " kWh")
    print("Initial Capital Expenses: " + str(round(ess_ice)) + " USD")
    print("Yearly Maintenance Expenses: " + str(round(ess_yme)) + " USD")
    print("Yearly Operational Expenses: " + str(round(ess_yoe)) + " USD")
    print("Total Present Cost: " + str(round(ess_pce + ess_pme + ess_poe)) + " USD")
    print("Combined Annuity: " + str(round(ess_ann)) + " USD")
    print("Yearly Discharged Energy: " + str(round(ess_ype / 1e6)) + " MWh")
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

if param.sim_enable["bev"]:
    if param.sim_cs["bev"]:
        bev_inv = results[(bevx_ess, None)]["scalars"]["invest"]  # [Wh]
    else:
        bev_inv = param.bev_cs
    total_bev_chg = bev_chg.sum()
    total_bev_dis = bev_dis.sum()
    total_bev_dem = total_bev_chg - total_bev_dis
    tot['yde'] += total_bev_dem / proj_yrrat

    print("Electric Vehicle Results:")
    if param.sim_cs["bev"]:
        print("Optimum battery capacity: " + str(round(bev_inv / 1e3)) + " kWh")
    else:
        print("Set battery capacity: " + str(round(bev_inv / 1e3)) + " kWh")
    print("Gross charged energy: " + str(round(total_bev_chg / 1e6)) + " MWh")
    print("Energy fed back (V2G): " + str(round(total_bev_dis / 1e6)) + " MWh")
    print("Net charged energy: " + str(round(total_bev_dem / 1e6)) + " MWh")
    print("#####")


##########################################################################
# LCOE and NPC calculation
##########################################################################
tot['pde'] = fcs.acc_discount(tot['yde'], param.proj_ls, param.proj_wacc)
tot['npc'] = tot['pce'] + tot['pme'] + tot['poe']
tot['lcoe'] = tot['npc'] / tot['pde'] 
tot['eta'] = tot['yde'] / tot['ype']


print("Economic Results:")
print("Yearly supplied energy: " + str(round(tot['yde'] / 1e6,4)) + " MWh")
print("Yearly generated energy: " + str(round(tot['ype'] / 1e6,4)) + " MWh")
print("Overall electrical efficiency: " + str(round(tot['eta'] * 100, 4)) + " %")
print("Total Initial Investment: " + str(round(tot['ice'] / 1e6, 4)) + " million USD")
print("Total yearly maintenance expenses: " + str(round(tot['yme'],4)) + " USD")
print("Total yearly operational expenses: " + str(round(tot['yoe'],4)) + " USD")
print("Total cost: " + str(round((tot['tce'] + tot['tme'] + tot['toe']) / 1e6, 4)) + " million USD")
print("Total present cost: " + str(round(tot['npc'] / 1e6, 4)) + " million USD")
print("Total annuity: " + str(round(tot['ann'] / 1e3, 4)) + " thousand USD")
print("LCOE: " + str(1e5 * tot['lcoe']) + " USct/kWh")
print("#####")



##########################################################################
# Save the results
##########################################################################

# Add the results to the energy system object and dump it as an .oemof file
if param.sim_dump:
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

end = time.time()

print(f"Runtime of the program is {end - start}")

##########################################################################
# Plot the results
##########################################################################
p300 = 'rgb(0,101,189)'
p540 = 'rgb(0,51,89)'
orng = 'rgb(227,114,34)'
grn = 'rgb(162,173,0)'
# TODO: Überschrift je nach Strategy
fig = go.Figure()
if param.sim_enable["gen"]:
    fig.add_trace(
    go.Scatter(x=gen_flow.index.to_pydatetime(), y=gen_flow, mode='lines', name='Diesel generator', line_color=p300,
               line_width=4))
if param.sim_enable["pv"]:
    fig.add_trace(
    go.Scatter(x=pv_flow.index.to_pydatetime(), y=pv_flow, mode='lines', name='Photovoltaics', line_color=orng,
               line_width=2))
if param.sim_enable["ess"]:
    fig.add_trace(go.Scatter(x=ess_flow.index.to_pydatetime(), y=ess_flow, mode='lines', name='Battery storage',
                         line_color=orng, line_width=2, line_dash='dash'))
if param.sim_enable["bev"]:
    fig.add_trace(go.Scatter(x=bev_flow.index.to_pydatetime(), y=bev_flow, mode='lines', name='BEV demand', line_color=grn,
                         line_width=2))
fig.add_trace(
    go.Scatter(x=dem_flow.index.to_pydatetime(), y=-dem_flow, mode='lines', name='Stationary demand', line_color=grn,
               line_width=2, line_dash='dash'))
fig.update_layout(xaxis=dict(title='Local Time', showgrid=True, linecolor='rgb(204, 204, 204)',
                             gridcolor='rgb(204, 204, 204)',),
                             #range=[datetime.strptime(param.proj_start, '%d/%m/%Y'),(datetime.strptime(param.proj_start, '%d/%m/%Y')+relativedelta(days=3))]),
                  yaxis=dict(title='Power in W', showgrid=True, linecolor='rgb(204, 204, 204)',
                             gridcolor='rgb(204, 204, 204)',
                             range=[-620000,620000]),
                  plot_bgcolor='white')
if param.sim_os == 'go':
    fig.update_layout(title='Power Flow Results - Global Optimum')
if param.sim_os == 'rh':
    fig.update_layout(title='Power Flow Results - Rolling Horizon Strategy (PH: '+str(param.os_ph)+'h, CH: '+str(param.os_ch)+'h)')
fig.show()



##########################################################################
# Monitor ESS SOC
##########################################################################
#print(sc_ess)
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=sc_ess.index.to_pydatetime(), y=sc_ess/ param.ess_cs * 100, mode='lines', name='SOC ESS', line_color=p300,
#             line_width=4))
# fig.update_layout(title='Global Optimum - State of Charge ESS', plot_bgcolor='white',
#                 xaxis=dict(title='Local Time', showgrid=True, linecolor='rgb(204, 204, 204)',
#                         gridcolor='rgb(204, 204, 204)',
#                         range=[datetime.strptime(param.proj_start, '%m/%d/%Y'),(datetime.strptime(param.proj_start, '%m/%d/%Y')+relativedelta(days=3))]),
#                 yaxis=dict(title='SOC in %', showgrid=True, linecolor='rgb(204, 204, 204)',
#                         gridcolor='rgb(204, 204, 204)',
#                         ))
#
# fig.show()

#
# ##########################################################################
# # Monitor BEVs SOC
# ##########################################################################
# # sc_bevx = pd.DataFrame()
# # for i in range(param.bev_num):
# #     column_name = (("bev"+str(i+1)+"_ess", 'None'), 'storage_content')
# #     sc_bevx["bev_"+str(i+1)] = views.node(results, "bev"+str(i+1)+"_ess")['sequences'][column_name]
# #
# #
# # row = [1, 1, 1]
# # col = [1, 2, 3]
# # titles = ['BEV 1','BEV 2','BEV 3','BEV 4']
# # fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
# # for i in range(len(row)):
# #     fig.add_trace(go.Scatter(x=sc_bevx["bev_"+str(i+1)].index, y=sc_bevx["bev_"+str(i+1)] / param.bev_cs * 100,
# #                              line=dict(color=p300), showlegend=False), row[i], col[i])
# # fig.update_layout(plot_bgcolor='white', title_text="Global Optimum - State of Charge BEVs")
# # fig.update_yaxes(title='SOC in %', showgrid=True, linecolor='rgb(204, 204, 204)', gridcolor='rgb(204, 204, 204)',
# #                  range=[0,100])
# # fig.update_xaxes(title='Local Time', showgrid=True, linecolor='rgb(204, 204, 204)', gridcolor='rgb(204, 204, 204)',
# #                  range=[datetime.strptime(param.proj_start, '%d/%m/%Y'),(datetime.strptime(param.proj_start, '%d/%m/%Y')+relativedelta(days=3))])
# # fig.show()
#
