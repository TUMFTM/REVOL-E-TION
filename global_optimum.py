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

import logging
import parameters as param
from oemof.solph import views


##########################################################################
# Create functions for "global optimum" strategy
##########################################################################


def opt_strategy(proj_start, dem_data, wind_data, pv_data, bev_data):
    iterations = 1
    dem_in = dem_data
    wind_in = pv_in = bev_in = None

    if param.sim_enable["wind"]:
        wind_in = wind_data.loc[(wind_data.time >= proj_start)]  # Wind data starts at proj_start
        wind_in.index = list(range(0, len(wind_in)))  # reset wind data index

    if param.sim_enable["pv"]:
        pv_in = pv_data.loc[(pv_data.time >= proj_start)]  # PV data starts at proj_start
        pv_in.index = list(range(0, len(pv_in)))  # reset PV data index

    ess_balancing = False
    ess_soc_proj_start = param.ess_init_soc

    bev_soc_proj_start = [None] * param.bev_num
    if param.sim_enable["bev"]:
        bev_in = bev_data.loc[(bev_data.time >= proj_start)]  # BEV data starting at proj_start
        bev_in.index = list(range(0, len(bev_in)))  # reset BEV data index

    return iterations, dem_in, wind_in, pv_in, ess_balancing, bev_in, bev_soc_proj_start, ess_soc_proj_start



def opt_strategy_postprocessing(results, ac_bus, dc_bus, dem, gen_src, pv_dc, ess, bev_ac, ac_bev,
                                wind_src, wind_bus, pv_src, pv_bus, bev_bus):

    dem_flow = results[(ac_bus, dem)]['sequences']['flow']

    wind_flow = gen_flow = pv_flow = ess_flow = bev_flow = sc_ess = None
    wind_prod = gen_prod = pv_prod = ess_prod = bev_chg = bev_dis = None

    if param.sim_enable["wind"]:
        wind_prod = results[(wind_src, wind_bus)]['sequences']['flow']

    if param.sim_enable["gen"]:
        gen_flow = results[(gen_src, ac_bus)]['sequences']['flow']
        gen_prod = results[(gen_src, ac_bus)]['sequences']['flow']

    if param.sim_enable["pv"]:
        pv_flow = results[(pv_dc, dc_bus)]['sequences']['flow']
        pv_prod = results[(pv_src, pv_bus)]['sequences']['flow']

    if param.sim_enable["ess"]:
        ess_flow = results[(ess, dc_bus)]['sequences']['flow'].subtract(results[(dc_bus, ess)]['sequences']['flow'])
        column_name = (('ess', 'None'), 'storage_content')
        sc_ess = views.node(results, 'ess')['sequences'][column_name]  # storage capacity during predict horizon
        ess_prod = results[(ess, dc_bus)]['sequences']['flow']

    if param.sim_enable["bev"]:
        bev_flow = results[(bev_ac, ac_bus)]['sequences']['flow'].subtract(results[(ac_bus, ac_bev)]['sequences']['flow'])
        bev_chg = results[(ac_bev, bev_bus)]['sequences']['flow']
        bev_dis = results[(bev_bus, bev_ac)]['sequences']['flow']


    return gen_flow, pv_flow, ess_flow, bev_flow, dem_flow, sc_ess, wind_prod, gen_prod, pv_prod, ess_prod, bev_chg, bev_dis

