'''
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
February 3rd, 2022

--- Last Update ---
February 3rd, 2022

--- Contributors ---
Marcel Brödel, B.Sc. - Semester Thesis in progress

--- Detailed Description ---
This script defines various functions used by main.py for orderly getting results from the different operating strats

--- Input & Output ---
see individual functions

--- Requirements ---
none

--- File Information ---
coding:     utf-8
license:    GPLv3
'''

###############################################################################
# Imports
###############################################################################

from oemof.solph import views

import logging
import time

import parameters as param


###############################################################################
# Function definitions
###############################################################################

def acc_energy(sim, prj, dem, wind, pv, gen, ess, bev, cres):
    """

    """
    dem['sim_del_ene'] += dem['flow'].sum()
    dem['yrl_del_ene'] += cres['sim_del_ene'] / sim['yrrat']
    dem['prj_del_ene'] += cres['yrl_del_ene'] * prj['ydur']
    cres['sim_del_ene'] += dem['sim_del_ene']
    cres['yrl_del_ene'] += dem['yrl_del_ene']
    cres['prj_del_ene'] += dem['prj_del_ene']

    if param.sim_enable['wind']:
        wind['sim_pro_ene'] = wind['flow'].sum()
        wind['yrl_del_ene'] = wind['sim_pro_ene'] / sim['yrrat']
        wind_ten = wind_ype * param.proj_ls
        wind_pen = eco.acc_discount(wind_ype, param.proj_ls, param.proj_wacc)
        tot['ype'] += wind_ype
        tot['ten'] += wind_ten
        tot['pen'] += wind_pen

    pv_ype = pv_prod.sum() / proj_yrrat
    pv_ten = pv_ype * param.proj_ls
    pv_pen = eco.acc_discount(pv_ype, param.proj_ls, param.proj_wacc)
    tot['ype'] += pv_ype
    tot['ten'] += pv_ten
    tot['pen'] += pv_pen

    gen_ype = gen_prod.sum() / proj_yrrat
    gen_ten = gen_ype * param.proj_ls
    gen_pen = eco.acc_discount(gen_ype, param.proj_ls, param.proj_wacc)
    tot['ype'] += gen_ype
    tot['ten'] += gen_ten
    tot['pen'] += gen_pen

    ess_ype = ess_prod.sum() / proj_yrrat
    ess_ten = ess_ype * param.proj_ls
    ess_pen = eco.acc_discount(ess_ype, param.proj_ls, param.proj_wacc)

    total_bev_chg = bev_chg.sum()
    total_bev_dis = bev_dis.sum()
    total_bev_dem = total_bev_chg - total_bev_dis
    tot['yde'] += total_bev_dem / proj_yrrat

    tot['pde'] = eco.acc_discount(tot['yde'], param.proj_ls, param.proj_wacc)

    tot['eta'] = tot['yde'] / tot['ype']
    return cres


def append_outdfs(sim, dem, wind, pv, gen, ess, bev, results):
    """
    Get result data slice for current CH from results and save in result dataframes for later analysis
    """

    dem_flow_ch = results[(sim['components']['ac_bus'],
                           sim['components']['dem_snk'])]['sequences']['flow'][sim['ch_dti']]
    dem['flow'] = dem['flow'].append(dem_flow_ch)

    if param.sim_enable["wind"]:
        wind_flow_ch = results[(sim['components']['wind_bus'],
                                sim['components']['wind_ac'])]['sequences']['flow'][sim['ch_dti']]
        wind['flow'] = wind['flow'].append(wind_flow_ch)

    if param.sim_enable["pv"]:
        pv_flow_ch = results[(sim['components']['pv_bus'],
                              sim['components']['pv_dc'])]['sequences']['flow'][sim['ch_dti']]
        pv['flow'] = pv['flow'].append(pv_flow_ch)

    if param.sim_enable["gen"]:
        gen_flow_ch = results[(sim['components']['gen_src'],
                               sim['components']['ac_bus'])]['sequences']['flow'][sim['ch_dti']]
        gen['flow'] = gen['flow'].append(gen_flow_ch)

    if param.sim_enable["ess"]:
        ess_flow_out_ch = results[(sim['components']['ess'],
                                   sim['components']['dc_bus'])]['sequences']['flow'][sim['ch_dti']]
        ess['flow_out'] = ess['flow_out'].append(ess_flow_out_ch)

        ess_flow_in_ch = results[(sim['components']['dc_bus'],
                                  sim['components']['ess'])]['sequences']['flow'][sim['ch_dti']]
        ess['flow_in'] = ess['flow_in'].append(ess_flow_in_ch)

        ess_sc_ch = views.node(results, 'ess')['sequences'][(('ess', 'None'), 'storage_content')][
            sim['ch_dti']]  # storage content
        ess_soc_ch = ess_sc_ch / param.ess_cs
        ess['soc'] = ess['soc'].append(ess_soc_ch)  # tracking state of charge
        ess['ph_init_soc'] = ess['soc'].iloc[-1]

    if param.sim_enable["bev"]:

        bev_flow_out_ch = results[(sim['components']['bev_ac'],
                                   sim['components']['ac_bus'])]['sequences']['flow'][sim['ch_dti']]
        bev['flow_out'] = bev['flow_out'].append(bev_flow_out_ch)

        bev_flow_in_ch = results[(sim['components']['ac_bus'],
                                  sim['components']['ac_bev'])]['sequences']['flow'][sim['ch_dti']]
        bev['flow_in'] = bev['flow_in'].append(bev_flow_in_ch)

        for i in range(param.bev_num):
            bevx_ess_name = "bev" + str(i + 1) + "_ess"
            bevx_name = "bev" + str(i + 1)
            bevx_sc_ch = views.node(results,
                                    bevx_ess_name)['sequences'][((bevx_ess_name,
                                                                  'None'),
                                                                 'storage_content')][sim['ch_dti']]
            bevx_soc_ch = bevx_sc_ch / param.bev_cs
            bev[bevx_name]['soc'] = bev[bevx_name]['soc'].append(bevx_soc_ch)  # TODO Why was a -1 shift in here?
            bev[bevx_name]['ph_init_soc'] = bev[bevx_name]['soc'].iloc[-1]

    return dem, wind, pv, gen, ess, bev


def end_timing(sim):
    sim['runtimeend'] = time.time()
    sim['runtime'] = round(sim['runtimeend'] - sim['runtimestart'], 1)
    logging.info('Runtime of the program was ' + str(sim['runtime']) + " seconds")
    return sim

