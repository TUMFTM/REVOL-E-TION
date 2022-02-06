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
February 4th, 2022

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
import oemof.solph.processing as prcs

import logging
import time

import economics as eco
import parameters as param


###############################################################################
# Function definitions
###############################################################################

def acc_eco_comp(sim, prj, comp, cres):
    """
    Accumulate cost results to get economic values for a single component set
    """

    # Capital Expenses
    comp['c_init_capex'] = comp['size'] * comp['lifespan']
    comp['c_prj_capex'] = eco.tce(comp['c_init_capex'],
                                  comp['c_init_capex'],
                                  param.comp['lifespan'],
                                  prj['duration'])
    comp['c_dis_capex'] = eco.pce(comp['c_init_capex'],
                                  comp['c_init_capex'],
                                  param.comp['lifespan'],
                                  prj['duration'],
                                  prj['wacc'])
    comp['c_ann_capex'] = eco.ann_recur(comp['eco_init_capex'],
                                        param.comp['lifespan'],
                                        prj['duration'],
                                        prj['wacc'],
                                        param.comp['cost_decr'])

    cres['c_init_capex'] += comp['c_init_capex']
    cres['c_prj_capex'] += comp['c_prj_capex']
    cres['c_dis_capex'] += comp['c_dis_capex']
    cres['c_ann_capex'] += comp['c_ann_capex']

    # Maintenance Expenses (time-based maintenance)
    comp['c_yrl_mntex'] = comp['size'] * param.comp['spec_mntex']
    # TODO: Correct with ACE? Does oemof consider multiyear sim?
    comp['c_prj_mntex'] = comp['c_yrl_mntex'] * prj['duration']
    comp['c_dis_mntex'] = eco.acc_discount(comp['c_yrl_mntex'],
                                             prj['duration'],
                                             prj['wacc'])
    comp['c_ann_mntex'] = eco.ann_recur(comp['eco_yrl_mntex'],
                                        1,  # expense is every year
                                        prj['duration'],
                                        prj['wacc'],
                                        1)  # expense has no cost decrease

    cres['c_yrl_mntex'] += comp['c_yrl_mntex']
    cres['c_prj_mntex'] += comp['c_prj_mntex']
    cres['c_dis_mntex'] += comp['c_dis_mntex']
    cres['c_ann_mntex'] += comp['c_ann_mntex']

    # Operational Expenses
    comp['c_sim_opex'] = comp['e_sim_pro'] * comp['spec_opex']
    comp['c_yrl_opex'] = comp['c_sim_opex'] / sim['yrrat']
    comp['c_prj_opex'] = comp['c_yrl_opex'] * prj['duration']
    comp['c_dis_opex'] = eco.acc_discount(comp['c_yrl_opex'],
                                          prj['duration'],
                                          prj['wacc'])
    comp['c_ann_opex'] = eco.ann_recur(comp['eco_yrl_opex'],
                                       1,  # expense is every year
                                       prj['duration'],
                                       prj['wacc'],
                                       1)  # expense has no cost decrease

    cres['c_sim_opex'] += comp['c_sim_opex']
    cres['c_yrl_opex'] += comp['c_yrl_opex']
    cres['c_prj_opex'] += comp['c_prj_opex']
    cres['c_dis_opex'] += comp['c_dis_opex']
    cres['c_ann_opex'] += comp['c_ann_opex']

    # Combined expenses
    comp['c_sim_totex'] = comp['c_init_capex'] + comp['c_yrl_mntex'] + comp['c_sim_opex']
    comp['c_yrl_totex'] = comp['c_yrl_capex'] + comp['c_yrl_mntex'] + comp['c_yrl_opex']
    comp['c_prj_totex'] = comp['c_prj_capex'] + comp['c_prj_mntex'] + comp['c_prj_opex']
    comp['c_dis_totex'] = comp['c_dis_capex'] + comp['c_dis_mntex'] + comp['c_dis_opex']
    comp['c_ann_totex'] = comp['c_ann_capex'] + comp['c_ann_mntex'] + comp['c_ann_opex']

    cres['c_sim_totex'] += comp['c_sim_totex']
    cres['c_yrl_totex'] += comp['c_yrl_totex']
    cres['c_prj_totex'] += comp['c_prj_totex']
    cres['c_dis_totex'] += comp['c_dis_totex']
    cres['c_ann_totex'] += comp['c_ann_totex']

    return comp, cres


def acc_eco(sim, prj, wind, pv, gen, ess, bev, cres):
    """
    Accumulate cost results to get economic values for a all enabled components
    """

    if param.sim_enable['wind']:
        wind, cres = acc_eco_comp(sim, prj, wind, cres)

    if param.sim_enable['pv']:
        pv, cres = acc_eco_comp(sim, prj, pv, cres)

    if param.sim_enable['gen']:
        gen, cres = acc_eco_comp(sim, prj, gen, cres)

    if param.sim_enable['ess']:
        ess, cres = acc_eco_comp(sim, prj, ess, cres)

    if param.sim_enable['pv']:
        bev, cres = acc_eco_comp(sim, prj, bev, cres)

    return wind, pv, gen, ess, bev, cres


def acc_energy(sim, prj, dem, wind, pv, gen, ess, bev, cres):
    """
    Accumulate power results to get energy values from all components
    """

    if param.sim_enable['dem']:
        dem, cres = acc_energy_sink(sim, prj, dem, cres)

    if param.sim_enable['wind']:
        wind, cres = acc_energy_source(sim, prj, wind, cres)

    if param.sim_enable['pv']:
        pv, cres = acc_energy_source(sim, prj, pv, cres)

    if param.sim_enable['gen']:
        gen, cres = acc_energy_source(sim, prj, gen, cres)

    if param.sim_enable['ess']:
        ess, cres = acc_energy_storage(sim, prj, ess, cres)

    if param.sim_enable['bev']:
        bev, cres = acc_energy_bev(sim, prj, bev, cres)

    cres['e_eta'] = cres['e_sim_del'] / cres['e_sim_pro']

    return cres


def acc_energy_bev(sim, prj, comp, cres):
    """
    Calculate cumulative energy results for BEVs with in-/outflows and consumption
    """

    comp['e_sim_in'] = comp['flow_in'].sum()
    comp['e_yrl_in'] = comp['e_sim_in'] / sim['yrrat']
    comp['e_prj_in'] = comp['e_yrl_in'] * prj['duration']
    comp['e_dis_in'] = eco.acc_discount(comp['e_yrl_in'], param.proj_ls, prj['wacc'])

    comp['e_sim_out'] = comp['flow_out'].sum()
    comp['e_yrl_out'] = comp['e_sim_out'] / sim['yrrat']
    comp['e_prj_out'] = comp['e_yrl_out'] * prj['duration']
    comp['e_dis_out'] = eco.acc_discount(comp['e_yrl_out'], param.proj_ls, prj['wacc'])

    comp['e_sim_bal'] = comp['e_sim_out'] - comp['e_sim_in']
    comp['e_yrl_bal'] = comp['e_sim_bal'] / sim['yrrat']
    comp['e_prj_bal'] = comp['e_yrl_bal'] * prj['duration']
    comp['e_dis_bal'] = eco.acc_discount(comp['e_yrl_bal'], param.proj_ls, prj['wacc'])

    cres['e_sim_del'] += comp['e_sim_bal']
    cres['e_yrl_del'] += comp['e_yrl_bal']
    cres['e_prj_del'] += comp['e_prj_bal']
    cres['e_dis_del'] += comp['e_dis_bal']

    return comp, cres


def acc_energy_sink(sim, prj, comp, cres):
    """
    Calculate cumulative energy results for pure sink component sets
    """

    comp['e_sim_del'] = comp['flow'].sum()
    comp['e_yrl_del'] = comp['e_sim_del'] / sim['yrrat']
    comp['e_prj_del'] = comp['e_yrl_del'] * prj['duration']
    comp['e_dis_del'] = eco.acc_discount(comp['e_yrl_del'], param.proj_ls, prj['wacc'])

    cres['e_sim_del'] += comp['e_sim_del']
    cres['e_yrl_del'] += comp['e_yrl_del']
    cres['e_prj_del'] += comp['e_prj_del']
    cres['e_dis_del'] += comp['e_dis_del']

    return comp, cres


def acc_energy_source(sim, prj,  comp, cres):
    """
    Calculate cumulative energy results for pure source component sets
    """
    comp['e_sim_pro'] = comp['flow'].sum()
    comp['e_yrl_pro'] = comp['e_sim_pro'] / sim['yrrat']
    comp['e_prj_pro'] = comp['e_yrl_pro'] * prj['duration']
    comp['e_dis_pro'] = eco.acc_discount(comp['e_yrl_pro'], param.proj_ls, prj['wacc'])

    cres['e_sim_pro'] += comp['e_sim_pro']
    cres['e_yrl_pro'] += comp['e_yrl_pro']
    cres['e_prj_pro'] += comp['e_prj_pro']
    cres['e_dis_pro'] += comp['e_dis_pro']
    return comp, cres


def acc_energy_storage(sim, prj, comp, cres):
    """
    Calculate cumulative energy results for energy storage component sets with in- and outflows
    """

    comp['e_sim_in'] = comp['flow_in'].sum()
    comp['e_yrl_in'] = comp['e_sim_in'] / sim['yrrat']
    comp['e_prj_in'] = comp['e_yrl_in'] * prj['duration']

    comp['e_sim_out'] = comp['flow_out'].sum()
    comp['e_yrl_out'] = comp['e_sim_out'] / sim['yrrat']
    comp['e_prj_out'] = comp['e_yrl_out'] * prj['duration']

    comp['e_sim_bal'] = comp['e_sim_out'] - comp['e_sim_in']
    comp['e_yrl_bal'] = comp['e_sim_bal'] / sim['yrrat']
    comp['e_prj_bal'] = comp['e_yrl_bal'] * prj['duration']

    comp['eta'] = comp['e_sim_out'] / comp['e_sim_in']

    return comp, cres


def get_results(sim, dem, wind, pv, gen, ess, bev, model):
    """
    Get result data slice for current CH from results and save in result dataframes for later analysis
    """

    results = prcs.results(model)  # Get the results from the solver

    if sim['debugmode']:
        pass
        # dump_resultfile(sim, results)

    if param.sim_enable['dem']:
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

        # ac_bev and bev_ac have an efficiency of 1, so these energies are the ones actually transmitted
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
            bev[bevx_name]['soc'] = bev[bevx_name]['soc'].append(bevx_soc_ch)  # TODO Check for completeness of series
            bev[bevx_name]['ph_init_soc'] = bev[bevx_name]['soc'].iloc[-1]

    return dem, wind, pv, gen, ess, bev


def end_timing(sim):
    sim['runtimeend'] = time.time()
    sim['runtime'] = round(sim['runtimeend'] - sim['runtimestart'], 1)
    logging.info('Runtime of the program was ' + str(sim['runtime']) + " seconds")
    return sim


def get_cs(sim, wind, pv, gen, ess, bev, results):
    """
    Get (possibly optimized) component sizes from results to handle outputs more easily
    """

    if param.sim_enable['wind']:
        if param.sim_cs['wind']:
            wind['size'] = results[(sim['components']['wind_src'], sim['components']['wind_ac'])]["scalars"]["invest"]
        else:
            wind['size'] = param.sim_cs['wind']

    if param.sim_enable['pv']:
        if param.sim_cs['pv']:
            pv['size'] = results[(sim['components']['pv_src'], sim['components']['pv_bus'])]["scalars"]["invest"]
        else:
            pv['size'] = param.sim_cs['pv']

    if param.sim_enable['gen']:
        if param.sim_cs['gen']:
            gen['size'] = results[(sim['components']['gen_src'], sim['components']['ac_bus'])]["scalars"]["invest"]
        else:
            gen['size'] = param.sim_cs['gen']

    if param.sim_enable['ess']:
        if param.sim_cs['ess']:
            ess['size'] = results[(sim['components']['ess'], None)]["scalars"]["invest"]
        else:
            ess['size'] = param.sim_cs['ess']

    if param.sim_enable['bev']:
        if param.sim_cs['bev']:
            bev['size'] = results[(sim['components']['bevx_ess'], None)]["scalars"]["invest"]
        else:
            bev['size'] = param.sim_cs['bev']

    return wind, pv, gen, ess, bev

