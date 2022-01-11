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

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021
Marcel Brödel, B.Sc. - Semester Thesis in progress
Elhussein Ismail, B.Sc. - Master Thesis in progress

--- Detailed Description ---
#TODO

"""

###############################################################################
# Imports
###############################################################################
import oemof.solph as solph
import oemof.solph.processing as prcs
from oemof.solph import views

import logging
import parameters as param
import pandas as pd

from dateutil.relativedelta import relativedelta


##########################################################################
# Create functions for "rolling horizon" strategy
##########################################################################

# Set initial simulation settings
def rh_strategy_init(proj_start):

    os_ph_steps = {'H': 1, 'T': 60}[param.sim_step] * param.os_ph  # number of timesteps for predicted horizon
    os_ch_steps = {'H': 1, 'T': 60}[param.sim_step] * param.os_ch  # number of timesteps for control horizon

    os_range_steps = {'H':24, 'T':24*60}[param.sim_step] * param.proj_sim   # number of timesteps in simulated date range
    iterations = int(os_range_steps / os_ch_steps)           # number of timeslices for simulated date range (= counter end)


    ##########################################################################
    # Variable initialization for first iteration
    ##########################################################################
    proj_simend = proj_start + relativedelta(hours=param.os_ph)
    proj_dti = pd.date_range(start=param.proj_start, end=proj_simend, freq=param.sim_step).delete(-1)
    ess_soc_proj_start = param.ess_init_soc
    bev_soc_proj_start = [None] * param.bev_num

    #TODO: check ess_balancing
    ess_balancing = False

    # Initialization of power flow results
    gen_flow = pv_flow = ess_flow = bev_flow = dem_flow = pd.Series([])
    wind_prod = pv_prod = gen_prod = ess_prod = bev_chg = bev_dis = pd.Series([])
    sc_ess = pd.Series([])
    sc_bevx = {}
    for i in range(param.bev_num):
        sc_bevx['bev_' + str(i + 1)] = pd.Series([])

    return iterations, proj_start, proj_dti, ess_soc_proj_start, bev_soc_proj_start, ess_balancing, \
           gen_flow, pv_flow, ess_flow, bev_flow, dem_flow, \
           wind_prod, pv_prod, gen_prod, ess_prod, bev_chg, bev_dis, sc_bevx, sc_ess



# Update input files to predicted horizon
def rh_strategy_dataupdate(proj_dti, dem_data, wind_data, pv_data, bev_data):

    dem_in = dem_data.loc[(dem_data.time >= proj_dti[0]) & (dem_data.time <= (proj_dti[-1]+relativedelta(hours=1)))]  # Dem data starts at proj_start
    dem_in.index = list(range(0, len(dem_in)))  # reset PV data index

    wind_in = pv_in = bev_in = None

    if param.sim_enable["wind"]:
        wind_in = wind_data.loc[(wind_data.time >= proj_dti[0] & (wind_data.time <= (proj_dti[-1]+relativedelta(hours=1))))]  # Wind data starts at proj_start
        wind_in.index = list(range(0, len(wind_in)))  # reset wind data index

    if param.sim_enable["pv"]:
        pv_in = pv_data.loc[(pv_data.time >= proj_dti[0]) & (pv_data.time <= (proj_dti[-1]+relativedelta(hours=1)))]  # PV data starts at proj_start
        pv_in.index = list(range(0, len(pv_in)))  # reset PV data index

    if param.sim_enable["bev"]:
        bev_in = bev_data.loc[(bev_data.time >= proj_dti[0]) & (bev_data.time <= (proj_dti[-1]+relativedelta(hours=1)))]  # BEV data starting at proj_start
        bev_in.index = list(range(0, len(bev_in)))  # reset BEV data index

    return dem_in, wind_in, pv_in, bev_in




# Initialize simulation data for next iteration loop
def rh_strategy_postprocessing(proj_start, results, dem, ess,
                               dem_flow, pv_flow, gen_flow, ess_flow, bev_flow,
                               wind_prod, pv_prod, gen_prod, ess_prod, bev_chg, bev_dis,
                               ac_bus, dc_bus, wind_bus, pv_bus, bev_bus,
                               wind_src, pv_src, pv_dc, gen_src, bev_ac, ac_bev,
                               ess_soc_proj_start, bev_soc_proj_start, sc_bevx, sc_ess):

    concat_start = proj_start                                                   # time mask start
    concat_end = proj_start + relativedelta(hours=param.os_ch - 1)              # time mask end


    ##########################################################################
    # Collect results and concatenate CH timeframes
    ##########################################################################
    # Demand
    dem_flow = pd.concat([dem_flow, results[(ac_bus, dem)]['sequences']['flow'][concat_start:concat_end]])

    # Wind
    if param.sim_enable["wind"]:
        wind_prod = pd.concat([wind_prod, results[(wind_src, wind_bus)]['sequences']['flow'][concat_start:concat_end]])

    # PV
    if param.sim_enable["pv"]:
        pv_flow = pd.concat([pv_flow, results[(pv_dc, dc_bus)]['sequences']['flow'][concat_start:concat_end]])
        pv_prod = pd.concat([pv_prod, results[(pv_src, pv_bus)]['sequences']['flow'][concat_start:concat_end]])

    # Generator
    if param.sim_enable["gen"]:
        gen_flow = pd.concat([gen_flow, results[(gen_src, ac_bus)]['sequences']['flow'][concat_start:concat_end]])
        gen_prod = pd.concat([gen_prod, results[(gen_src, ac_bus)]['sequences']['flow'][concat_start:concat_end]])

    # ESS
    if param.sim_enable["ess"]:
        ess_flow = pd.concat([ess_flow, (results[(ess, dc_bus)]['sequences']['flow'][concat_start:concat_end].
                                         subtract(results[(dc_bus, ess)]['sequences']['flow'][concat_start:concat_end]))])
        ess_prod = pd.concat([ess_prod, results[(ess, dc_bus)]['sequences']['flow'][concat_start:concat_end]])

    # BEV
    if param.sim_enable["bev"]:
        bev_flow = pd.concat([bev_flow, results[(bev_ac, ac_bus)]['sequences']['flow'][concat_start:concat_end].
                                        subtract(results[(ac_bus, ac_bev)]['sequences']['flow'][concat_start:concat_end])])
        bev_chg = pd.concat([bev_chg, results[(ac_bev, bev_bus)]['sequences']['flow'][concat_start:concat_end]])
        bev_dis = pd.concat([bev_dis, results[(bev_bus, bev_ac)]['sequences']['flow'][concat_start:concat_end]])


    ##########################################################################
    # Initialize start SOCs
    ##########################################################################
    # ESS
    if param.sim_enable['ess']:
        column_name = (('ess', 'None'), 'storage_content')
        SC_ph = views.node(results, 'ess')['sequences'][column_name]                                                    # storage capacity during predict horizon
        ess_soc_proj_start = SC_ph[concat_end] / param.ess_cs                                                           # SOC at end of control horizon
        sc_ess = pd.concat([sc_ess, SC_ph[concat_start:concat_end]])                                                    # storage capacity during control horizon
    else:
        ess_soc_proj_start = sc_ess = None
        #TODO: Korrekte Übergabe?

    # BEVs
    if param.sim_enable['bev']:
        for i in range(param.bev_num):
            column_name = (("bev" + str(i + 1) + "_ess", 'None'), 'storage_content')
            SC_ph = views.node(results, "bev" + str(i + 1) + "_ess")['sequences'][column_name]                          # storage capacity during predict horizon
            bev_soc_proj_start[i] = SC_ph[concat_end] / param.bev_cs                                                    # SOC at end of control horizon
            sc_bevx['bev_' + str(i + 1)] = pd.concat([sc_bevx['bev_' + str(i + 1)], SC_ph[concat_start:concat_end]])    # storage capacity during control horizon
    else:
        bev_soc_proj_start = sc_bevx = None


    ##########################################################################
    # Initialize new time horizon
    ##########################################################################
    proj_start = proj_start + relativedelta(hours=param.os_ch)
    proj_simend = proj_start + relativedelta(hours=param.os_ph)
    proj_dti = pd.date_range(start=proj_start, end=proj_simend, freq=param.sim_step).delete(-1)


    return proj_start, proj_dti, \
           dem_flow, pv_flow, gen_flow, ess_flow, bev_flow, \
           wind_prod, pv_prod, gen_prod, ess_prod, bev_chg, bev_dis, \
           ess_soc_proj_start, bev_soc_proj_start, sc_bevx, sc_ess






















