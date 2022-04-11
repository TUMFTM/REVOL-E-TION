"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
February 3rd, 2022

--- Last Update ---
February 8th, 2022

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
"""

from oemof.solph import views
import oemof.solph.processing as prcs

import logging
import pprint as pp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pylightxl as xl
import numpy as np
import re
import pandas as pd


import economics as eco
import colordef as col


def acc_eco(sim, prj, wind, pv, gen, ess, bev, cres):
    """
    Accumulate cost results to get economic values for a all enabled components
    """

    if sim['enable']['wind']:
        wind, cres = acc_eco_comp(sim, prj, wind, cres)

    if sim['enable']['pv']:
        pv, cres = acc_eco_comp(sim, prj, pv, cres)

    if sim['enable']['gen']:
        gen, cres = acc_eco_comp(sim, prj, gen, cres)

    if sim['enable']['ess']:
        ess, cres = acc_eco_comp(sim, prj, ess, cres)

    if sim['enable']['bev']:
        bev, cres = acc_eco_comp(sim, prj, bev, cres)

    cres['lcoe'] = cres['dis_totex'] / cres['e_dis_del']  # NPC divided by discounted energy

    return wind, pv, gen, ess, bev, cres


def acc_eco_comp(sim, prj, comp, cres):
    """
    Accumulate cost results to get economic values for a single component set
    """

    # Capital Expenses
    comp['init_capex'] = comp['size'] * comp['spec_capex']
    comp['prj_capex'] = eco.tce(comp['init_capex'],
                                comp['init_capex'],
                                comp['lifespan'],
                                prj['duration'])
    comp['dis_capex'] = eco.pce(comp['init_capex'],
                                comp['init_capex'],
                                comp['lifespan'],
                                prj['duration'],
                                prj['wacc'])
    comp['ann_capex'] = eco.ann_recur(comp['init_capex'],
                                      comp['lifespan'],
                                      prj['duration'],
                                      prj['wacc'],
                                      comp['cost_decr'])

    cres['init_capex'] += comp['init_capex']
    cres['prj_capex'] += comp['prj_capex']
    cres['dis_capex'] += comp['dis_capex']
    cres['ann_capex'] += comp['ann_capex']

    # Maintenance Expenses (time-based maintenance)
    comp['yrl_mntex'] = comp['size'] * comp['spec_mntex']
    # TODO: Correct with ACE? Does oemof consider multiyear sim?
    comp['prj_mntex'] = comp['yrl_mntex'] * prj['duration']
    comp['dis_mntex'] = eco.acc_discount(comp['yrl_mntex'],
                                         prj['duration'],
                                         prj['wacc'])
    comp['ann_mntex'] = eco.ann_recur(comp['yrl_mntex'],
                                      1,  # expense is every year
                                      prj['duration'],
                                      prj['wacc'],
                                      1)  # expense has no cost decrease

    cres['yrl_mntex'] += comp['yrl_mntex']
    cres['prj_mntex'] += comp['prj_mntex']
    cres['dis_mntex'] += comp['dis_mntex']
    cres['ann_mntex'] += comp['ann_mntex']

    # Operational Expenses
    if 'e_sim_pro' in comp.keys():
        comp['sim_opex'] = comp['e_sim_pro'] * comp['spec_opex']  # source components
    elif 'e_sim_out' in comp.keys():
        comp['sim_opex'] = comp['e_sim_out'] * comp['spec_opex']  # storage components
    else:
        raise KeyError('neither production-based nor storage-based component accessed')
    comp['yrl_opex'] = comp['sim_opex'] / sim['yrrat']
    comp['prj_opex'] = comp['yrl_opex'] * prj['duration']
    comp['dis_opex'] = eco.acc_discount(comp['yrl_opex'],
                                        prj['duration'],
                                        prj['wacc'])
    comp['ann_opex'] = eco.ann_recur(comp['yrl_opex'],
                                     1,  # expense is every year
                                     prj['duration'],
                                     prj['wacc'],
                                     1)  # expense has no cost decrease

    cres['sim_opex'] += comp['sim_opex']
    cres['yrl_opex'] += comp['yrl_opex']
    cres['prj_opex'] += comp['prj_opex']
    cres['dis_opex'] += comp['dis_opex']
    cres['ann_opex'] += comp['ann_opex']

    # Combined expenses
    comp['sim_totex'] = comp['init_capex'] + comp['yrl_mntex'] + comp['sim_opex']
    comp['prj_totex'] = comp['prj_capex'] + comp['prj_mntex'] + comp['prj_opex']
    comp['dis_totex'] = comp['dis_capex'] + comp['dis_mntex'] + comp['dis_opex']
    comp['ann_totex'] = comp['ann_capex'] + comp['ann_mntex'] + comp['ann_opex']

    cres['sim_totex'] += comp['sim_totex']
    cres['prj_totex'] += comp['prj_totex']
    cres['dis_totex'] += comp['dis_totex']
    cres['ann_totex'] += comp['ann_totex']

    return comp, cres


def acc_energy(sim, prj, dem, wind, pv, gen, ess, bev, cres):
    """
    Accumulate power results to get energy values from all components
    """

    if sim['enable']['dem']:
        dem, cres = acc_energy_sink(sim, prj, dem, cres)

    if sim['enable']['wind']:
        wind, cres = acc_energy_source(sim, prj, wind, cres)

    if sim['enable']['pv']:
        pv, cres = acc_energy_source(sim, prj, pv, cres)

    if sim['enable']['gen']:
        gen, cres = acc_energy_source(sim, prj, gen, cres)

    if sim['enable']['ess']:
        ess, cres = acc_energy_storage(sim, prj, ess, cres)

    if sim['enable']['bev']:
        bev, cres = acc_energy_bev(sim, prj, bev, cres)

    cres['e_eta'] = cres['e_sim_del'] / cres['e_sim_pro']

    return dem, wind, pv, gen, ess, bev, cres


def acc_energy_bev(sim, prj, comp, cres):
    """
    Calculate cumulative energy results for BEVs with in-/outflows and consumption
    """

    comp['e_sim_in'] = comp['flow_in'].sum()
    comp['e_yrl_in'] = comp['e_sim_in'] / sim['yrrat']
    comp['e_prj_in'] = comp['e_yrl_in'] * prj['duration']
    comp['e_dis_in'] = eco.acc_discount(comp['e_yrl_in'], prj['duration'], prj['wacc'])

    comp['e_sim_out'] = comp['flow_out'].sum()
    comp['e_yrl_out'] = comp['e_sim_out'] / sim['yrrat']
    comp['e_prj_out'] = comp['e_yrl_out'] * prj['duration']
    comp['e_dis_out'] = eco.acc_discount(comp['e_yrl_out'], prj['duration'], prj['wacc'])

    comp['e_sim_bal'] = comp['e_sim_in'] - comp['e_sim_out']
    comp['e_yrl_bal'] = comp['e_sim_bal'] / sim['yrrat']
    comp['e_prj_bal'] = comp['e_yrl_bal'] * prj['duration']
    comp['e_dis_bal'] = eco.acc_discount(comp['e_yrl_bal'], prj['duration'], prj['wacc'])

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
    comp['e_dis_del'] = eco.acc_discount(comp['e_yrl_del'], prj['duration'], prj['wacc'])

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
    comp['e_dis_pro'] = eco.acc_discount(comp['e_yrl_pro'], prj['duration'], prj['wacc'])

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

    comp['e_sim_bal'] = comp['e_sim_in'] - comp['e_sim_out']
    comp['e_yrl_bal'] = comp['e_sim_bal'] / sim['yrrat']
    comp['e_prj_bal'] = comp['e_yrl_bal'] * prj['duration']

    comp['eta'] = comp['e_sim_out'] / comp['e_sim_in']

    return comp, cres


def plot_results(sim, dem, wind, pv, gen, ess, bev, sheet, file):
    """

    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if sim['enable']['dem']:
        fig.add_trace(go.Scatter(x=dem['flow'].index.to_pydatetime(),
                                 y=-dem['flow'],
                                 mode='lines',
                                 name='Stationary demand',
                                 line=dict(color=col.tum_p301_50, width=2, dash=None)),
                      secondary_y=False)

    if sim['enable']['wind']:
        fig.add_trace(go.Scatter(x=wind['flow'].index.to_pydatetime(),
                                 y=wind['flow'],
                                 mode='lines',
                                 name='Wind power (' + str(round(wind['size'] / 1e3, 1)) + ' kW)',
                                 line=dict(color=col.tum_p300, width=2, dash='dash')),
                      secondary_y=False)

    if sim['enable']["pv"]:
        fig.add_trace(go.Scatter(x=pv['flow'].index.to_pydatetime(),
                                 y=pv['flow'],
                                 mode='lines',
                                 name='Photovoltaic power (' + str(round(pv['size'] / 1e3, 1)) + ' kWp)',
                                 line=dict(color=col.tum_p300, width=2, dash=None)),
                      secondary_y=False)

    if sim['enable']["gen"]:
        fig.add_trace(go.Scatter(x=gen['flow'].index.to_pydatetime(),
                                 y=gen['flow'],
                                 mode='lines',
                                 name='Diesel power (' + str(round(gen['size'] / 1e3, 1)) + ' kW)',
                                 line=dict(color=col.tum_black, width=2, dash=None)),
                      secondary_y=False)

    if sim['enable']["ess"]:
        fig.add_trace(go.Scatter(x=ess['flow_bal'].index.to_pydatetime(),
                                 y=ess['flow_bal'],
                                 mode='lines',
                                 name='Stationary storage (pos=ch, ' + str(round(ess['size'] / 1e3, 1)) + ' kWh)',
                                 line=dict(color=col.tum_orange, width=2, dash=None)),
                      secondary_y=False)

        fig.add_trace(go.Scatter(x=ess['soc'].index.to_pydatetime(),
                                 y=ess['soc'],
                                 mode='lines',
                                 name='Battery storage SOC',
                                 line=dict(color=col.tum_orange, width=2, dash='dash'),
                                 visible='legendonly'),
                      secondary_y=True)

    if sim['enable']["bev"]:
        fig.add_trace(go.Scatter(x=bev['flow_bal'].index.to_pydatetime(),
                                 y=bev['flow_bal'],
                                 mode='lines',
                                 name='BEV storage (pos=ch, ' + str(round(bev['size'] / 1e3, 1)) + ' kWh)',
                                 line=dict(color=col.tum_green, width=2, dash=None)),
                      secondary_y=False)

        for bevx in bev['bevx_list']:
            fig.add_trace(go.Scatter(x=bev[bevx]['soc'].index.to_pydatetime(),
                                     y=bev[bevx]['soc'],
                                     mode='lines',
                                     name=bevx + ' SOC',
                                     line=dict(color=col.tum_green, width=2, dash='dash'),
                                     visible='legendonly'),
                          secondary_y=True)

    fig.update_layout(plot_bgcolor=col.tum_white)
    fig.update_xaxes(title='Local Time',
                     showgrid=True,
                     linecolor=col.tum_grey_20,
                     gridcolor=col.tum_grey_20,)
    fig.update_yaxes(title='Power in W',
                     showgrid=True,
                     linecolor=col.tum_grey_20,
                     gridcolor=col.tum_grey_20,
                     secondary_y=False,)
    fig.update_yaxes(title='State of Charge',
                     showgrid=False,
                     secondary_y=True)

    while '/' in file:
        file = re.sub(r'^.*?/', '', file)

    if sim['op_strat'] == 'go':
        fig.update_layout(title='Global Optimum Results (' + file + '_' + sheet + ')')
    if sim['op_strat'] == 'rh':
        title = 'Rolling Horizon Results (PH: '+str(sim['rh_ph'])+'h, CH: '+str(sim['rh_ch'])+'h), (' + file + '_' + sheet + ')'
        fig.update_layout(title=title)

    fig.show()


def print_results(sim, wind, pv, gen, ess, bev, cres):

    print('#####')
    if sim['enable']['wind']:
        print('Wind power results:')
        print_results_source(sim, wind)

    if sim['enable']['pv']:
        print('PV power results:')
        print(pv)
        print_results_source(sim, pv)

    if sim['enable']['gen']:
        print('Diesel power results:')
        print_results_source(sim, gen)

    if sim['enable']['wind']:
        print('Stationary storage results:')
        print_results_storage(sim, ess)

    if sim['enable']['wind']:
        print('Electric vehicle results:')
        print_results_storage(sim, bev)

    print_results_overall(cres)


def print_results_storage(sim, comp):
    print()
    if sim['cs_opt'][comp['name']]:
        print("Optimum Capacity: " + str(round(comp['size'] / 1e3)) + " kWh")
    else:
        print("Set Capacity: " + str(round(comp['size'] / 1e3)) + " kWh")
    print()
    print("Initial Capital Expenses: " + str(round(comp['init_capex'] / 1e3, 1)) + " thousand USD")
    print("Total Capital Expenses: " + str(round(comp['prj_capex'] / 1e3, 1)) + " thousand USD")
    print("Present Capital Expenses: " + str(round(comp['dis_capex'] / 1e3, 1)) + " thousand USD")
    print("Annuity of Capital Expenses: " + str(round(comp['ann_capex'] / 1e3, 1)) + " thousand USD")
    print()
    print("Yearly Maintenance Expenses: " + str(round(comp['yrl_mntex'] / 1e3, 1)) + " thousand USD")
    print("Total Maintenance Expenses: " + str(round(comp['prj_mntex'] / 1e3, 1)) + " thousand USD")
    print("Present Maintenance Expenses: " + str(round(comp['dis_mntex'] / 1e3, 1)) + " thousand USD")
    print("Annuity of Maintenance Expenses: " + str(round(comp['ann_mntex'] / 1e3, 1)) + " thousand USD")
    print()
    print("Yearly Operational Expenses: " + str(round(comp['yrl_opex'] / 1e3, 1)) + " thousand USD")
    print("Total Operational Expenses: " + str(round(comp['prj_opex'] / 1e3, 1)) + " thousand USD")
    print("Present Operational Expenses: " + str(round(comp['dis_opex'] / 1e3, 1)) + " thousand USD")
    print("Annuity of Operational Expenses: " + str(round(comp['ann_opex'] / 1e3, 1)) + " thousand USD")
    print()
    print("Gross Yearly Charged Energy: " + str(round(comp['e_yrl_in'] / 1e6, 1)) + " MWh")
    print("Gross Yearly Discharged Energy: " + str(round(comp['e_yrl_out'] / 1e6, 1)) + " MWh")
    print("Net Yearly Charged Energy: " + str(round(comp['e_yrl_bal'] / 1e6, 1)) + " MWh")
    print("Gross Total Charged Energy: " + str(round(comp['e_prj_in'] / 1e6, 1)) + " MWh")
    print("Gross Total Discharged Energy: " + str(round(comp['e_prj_out'] / 1e6, 1)) + " MWh")
    print("Net Total Charged Energy: " + str(round(comp['e_prj_bal'] / 1e6, 1)) + " MWh")
    print("#####")


def print_results_source(sim, comp):
    print()
    if sim['cs_opt'][comp['name']]:
        print("Optimum Capacity: " + str(round(comp['size'] / 1e3)) + " kW")
    else:
        print("Set Capacity: " + str(round(comp['size'] / 1e3)) + " kWh")
    print()
    print("Initial Capital Expenses: " + str(round(comp['init_capex'] / 1e3, 1)) + " thousand USD")
    print("Total Capital Expenses: " + str(round(comp['prj_capex'] / 1e3, 1)) + " thousand USD")
    print("Present Capital Expenses: " + str(round(comp['dis_capex'] / 1e3, 1)) + " thousand USD")
    print("Annuity of Capital Expenses: " + str(round(comp['ann_capex'] / 1e3, 1)) + " thousand USD")
    print()
    print("Yearly Maintenance Expenses: " + str(round(comp['yrl_mntex'] / 1e3, 1)) + " thousand USD")
    print("Total Maintenance Expenses: " + str(round(comp['prj_mntex'] / 1e3, 1)) + " thousand USD")
    print("Present Maintenance Expenses: " + str(round(comp['dis_mntex'] / 1e3, 1)) + " thousand USD")
    print("Annuity of Maintenance Expenses: " + str(round(comp['ann_mntex'] / 1e3, 1)) + " thousand USD")
    print()
    print("Yearly Operational Expenses: " + str(round(comp['yrl_opex'] / 1e3, 1)) + " thousand USD")
    print("Total Operational Expenses: " + str(round(comp['prj_opex'] / 1e3, 1)) + " thousand USD")
    print("Present Operational Expenses: " + str(round(comp['dis_opex'] / 1e3, 1)) + " thousand USD")
    print("Annuity of Operational Expenses: " + str(round(comp['ann_opex'] / 1e3, 1)) + " thousand USD")
    print()
    print("Yearly Produced Energy: " + str(round(comp['e_yrl_pro'] / 1e6, 1)) + " MWh")
    print("Total Produced Energy: " + str(round(comp['e_prj_pro'] / 1e6, 1)) + " MWh")
    print("Present Produced Energy: " + str(round(comp['e_dis_pro'] / 1e6, 1)) + " MWh")
    print("#####")
    print()


def print_results_overall(cres):

    print("Overall Results:")
    print()
    print("Yearly produced energy: " + str(round(cres['e_yrl_pro'] / 1e6)) + " MWh")
    print("Yearly delivered energy: " + str(round(cres['e_yrl_del'] / 1e6)) + " MWh")
    print("Total produced energy: " + str(round(cres['e_prj_pro'] / 1e6)) + " MWh")
    print("Total delivered energy: " + str(round(cres['e_prj_del'] / 1e6)) + " MWh")
    print("Present produced energy: " + str(round(cres['e_dis_pro'] / 1e6)) + " MWh")
    print("Present delivered energy: " + str(round(cres['e_dis_del'] / 1e6)) + " MWh")
    print("Overall electrical efficiency: " + str(round(cres['e_eta'] * 100, 2)) + " %")
    # print("Capacity Factor: ")
    # print("Curtailed Renewable Energy:")
    print()
    print("Initial Capital Expenses: " + str(round(cres['init_capex'] / 1e6, 2)) + " million USD")
    print("Total Capital Expenses: " + str(round(cres['prj_capex'] / 1e6, 2)) + " million USD")
    print("Present Capital Expenses: " + str(round(cres['dis_capex'] / 1e6, 2)) + " million USD")
    print("Annuity of Capital Expenses: " + str(round(cres['ann_capex'] / 1e6, 2)) + " million USD")
    print()
    print("Yearly maintenance expenses: " + str(round(cres['yrl_mntex'] / 1e3)) + " thousand USD")
    print("Total maintenance expenses: " + str(round(cres['prj_mntex'] / 1e3)) + " thousand USD")
    print("Present maintenance expenses: " + str(round(cres['dis_mntex'] / 1e3)) + " thousand USD")
    print("Annuity of maintenance expenses: " + str(round(cres['ann_mntex'] / 1e3)) + " thousand USD")
    print()
    print("Yearly operational expenses: " + str(round(cres['yrl_opex'] / 1e3)) + " thousand USD")
    print("Total operational expenses: " + str(round(cres['prj_opex'] / 1e3)) + " thousand USD")
    print("Present operational expenses: " + str(round(cres['dis_opex'] / 1e3)) + " thousand USD")
    print("Annuity of operational expenses: " + str(round(cres['ann_opex'] / 1e3)) + " thousand USD")
    print()
    print("Total project cost: " + str(round(cres['prj_totex'] / 1e6, 2)) + " million USD")
    print("Total present cost: " + str(round(cres['dis_totex'] / 1e6, 2)) + " million USD")
    print("Total annuity: " + str(round(cres['ann_totex'] / 1e6, 2)) + " thousand USD")
    print("Levelized cost of electricity: " + str(round(1e5 * cres['lcoe'], 3)) + " USct/kWh")
    print("#####")


def save_results(sim, dem, wind, pv, gen, ess, bev, cres, sheet, file, r, folder):
    """
    Dump the simulation results as a file
    """

    if sim['dump']:
        logging.info("Save model and result data")

        while '/' in file:
            file = re.sub(r'^.*?/', '', file)

        if r == 0:
            # create a blank db
            db = xl.Database()
        else:
            db = xl.readxl(fn=folder + '/Results_' + file)
        filename = folder + '/Results_' + file

        # add a blank worksheet to the db
        db.add_ws(ws=sheet)

        # header of ws
        if sim['op_strat'] == 'go':
            db.ws(ws=sheet).update_index(row=1, col=1, val='Global Optimum Results (' + file + ')')
        if sim['op_strat'] == 'rh':
            db.ws(ws=sheet).update_index(row=1, col=1, val='Rolling Horizon Results (PH: ' + str(sim['rh_ph']) + 'h, CH: ' + str(sim['rh_ch']) + 'h), (' + file + ')')

        # add sim name
        db.ws(ws=sheet).update_index(row=2, col=1, val='Logfile')
        db.ws(ws=sheet).update_index(row=2, col=2, val=sim['name'])

        # add runtime
        db.ws(ws=sheet).update_index(row=3, col=1, val='Runtime')
        db.ws(ws=sheet).update_index(row=3, col=2, val=sim['runtime'])

        name = ['Accumulated cost', 'Demand', 'Wind component', 'PV component', 'Diesel component', 'ESS component', 'BEV component']
        for i, header in enumerate(name):
            db.ws(ws=sheet).update_index(row=5, col=1+i*4, val=header+' results')

        # function to add component data to the worksheet
        def add_ws(comp, col):
            keys = list(comp.keys())
            vals = list(comp.values())
            row_id = 6
            for i in range(len(vals)):
                if type(vals[i]) == np.float64 or type(vals[i]) == np.float or type(vals[i]) == np.int:
                    db.ws(ws=sheet).update_index(row=row_id, col=col, val=keys[i])
                    db.ws(ws=sheet).update_index(row=row_id, col=col+1, val=vals[i])
                    row_id += 1
            return None

        add_ws(cres,1)
        if sim['enable']['dem']:
            add_ws(dem, 5)
        if sim['enable']['wind']:
            add_ws(wind, 9)
        if sim['enable']['pv']:
            add_ws(pv, 13)
        if sim['enable']['gen']:
            add_ws(gen, 17)
        if sim['enable']['ess']:
            add_ws(ess, 21)
        if sim['enable']['bev']:
            add_ws(bev, 25)

        # write out the db
        xl.writexl(db=db, fn=filename)

    return None


def save_results_err(sim, sheet, file, r, folder):
    """
    Dump error message in result excel file if optimization did not succeed
    """

    if sim['dump']:
        logging.info("Save model and result data")

        while '/' in file:
            file = re.sub(r'^.*?/', '', file)

        if r == 0:
            # create a blank db
            db = xl.Database()
        else:
            db = xl.readxl(fn=folder + '/Results_' + file)
        filename = folder + '/Results_' + file

        # add a blank worksheet to the db
        db.add_ws(ws=sheet)

        # header of ws
        if sim['op_strat'] == 'go':
            db.ws(ws=sheet).update_index(row=1, col=1, val='Global Optimum Results (' + file + ')')
        if sim['op_strat'] == 'rh':
            db.ws(ws=sheet).update_index(row=1, col=1, val='Rolling Horizon Results (PH: ' + str(sim['rh_ph']) + 'h, CH: ' + str(sim['rh_ch']) + 'h), (' + file + ')')

        # add sim name
        db.ws(ws=sheet).update_index(row=3, col=1, val='Logfile:')
        db.ws(ws=sheet).update_index(row=3, col=2, val=sim['name'])

        # write error message
        db.ws(ws=sheet).update_index(row=5, col=1, val='ERROR - Optimization could NOT succeed for these simulation settings')

        # write out the db
        xl.writexl(db=db, fn=filename)

    return None

def get_sizes(sim, wind, pv, gen, ess, bev, results):

    if sim['enable']['wind']:
        if sim['cs_opt']['wind']:
            wind['size'] = results[(sim['components']['wind_src'], sim['components']['wind_bus'])]["scalars"]["invest"]
        else:
            wind['size'] = wind['cs']

    if sim['enable']['pv']:
        if sim['cs_opt']['pv']:
            pv['size'] = results[(sim['components']['pv_src'], sim['components']['pv_bus'])]["scalars"]["invest"]
        else:
            pv['size'] = pv['cs']

    if sim['enable']['gen']:
        if sim['cs_opt']['gen']:
            gen['size'] = results[(sim['components']['gen_src'], sim['components']['ac_bus'])]["scalars"]["invest"]
        else:
            gen['size'] = gen['cs']

    if sim['enable']['ess']:
        if sim['cs_opt']['ess']:
            ess['size'] = results[(sim['components']['ess'], None)]["scalars"]["invest"]
        else:
            ess['size'] = ess['cs']

    if sim['enable']['bev']:
        if sim['cs_opt']['bev']:
            # All bev(x) component sizes are identical
            # there is only one sim['component'] representing the last bevx as it is repeatedly overwritten
            bev['size'] = results[(sim['components']['bevx_ess'], None)]["scalars"]["invest"]
        else:
            bev['size'] = bev['cs']

    return wind, pv, gen, ess, bev


def get_results(sim, dem, wind, pv, gen, ess, bev, model, optnum):
    """
    Get result data slice for current CH from results and save in result dataframes for later analysis
    Get (possibly optimized) component sizes from results to handle outputs more easily
    """

    results = prcs.results(model)  # Get the results from the solver

    if sim['debugmode']:
        meta_results = prcs.meta_results(model)
        pp.pprint(meta_results)
        # dump_resultfile(sim, results)

    if optnum == 0:  # first iteration
        wind, pv, gen, ess, bev = get_sizes(sim, wind, pv, gen, ess, bev, results)

    if sim['enable']['dem']:
        dem_flow_ch = results[(sim['components']['ac_bus'],
                               sim['components']['dem_snk'])]['sequences']['flow'][sim['ch_dti']]
        dem['flow'] = pd.concat([dem['flow'], dem_flow_ch])

    if sim['enable']["wind"]:
        wind_flow_ch = results[(sim['components']['wind_bus'],
                                sim['components']['wind_ac'])]['sequences']['flow'][sim['ch_dti']]
        wind['flow'] = pd.concat([wind['flow'], wind_flow_ch])

    if sim['enable']["pv"]:
        pv_flow_ch = results[(sim['components']['pv_bus'],
                              sim['components']['pv_dc'])]['sequences']['flow'][sim['ch_dti']]
        pv['flow'] = pd.concat([pv['flow'], pv_flow_ch])

    if sim['enable']["gen"]:
        gen_flow_ch = results[(sim['components']['gen_src'],
                               sim['components']['ac_bus'])]['sequences']['flow'][sim['ch_dti']]
        gen['flow'] = pd.concat([gen['flow'], gen_flow_ch])

    if sim['enable']["ess"]:
        ess_flow_out_ch = results[(sim['components']['ess'],
                                   sim['components']['dc_bus'])]['sequences']['flow'][sim['ch_dti']]
        ess['flow_out'] = pd.concat([ess['flow_out'], ess_flow_out_ch])

        ess_flow_in_ch = results[(sim['components']['dc_bus'],
                                  sim['components']['ess'])]['sequences']['flow'][sim['ch_dti']]
        ess['flow_in'] = pd.concat([ess['flow_in'], ess_flow_in_ch])

        ess_flow_bal_ch = ess_flow_in_ch - ess_flow_out_ch
        ess['flow_bal'] = pd.concat([ess['flow_bal'], ess_flow_bal_ch])

        ess_sc_ch = views.node(results, 'ess')['sequences'][(('ess', 'None'), 'storage_content')][
            sim['ch_dti']].shift(periods=1, freq=sim['step'])  # shift is needed as sc/soc is at end of timestep
        ess_soc_ch = ess_sc_ch / ess['cs']

        ess['soc'] = pd.concat([ess['soc'], ess_soc_ch])  # tracking state of charge
        ess['ph_init_soc'] = ess['soc'].iloc[-1]
        if ess['ph_init_soc'] < 0:
            logging.info('ESS init SOC < 0: '+ str(ess['ph_init_soc']))
            ess['ph_init_soc'] = 0

    if sim['enable']["bev"]:
        # ac_bev and bev_ac have an efficiency of 1, so these energies are the ones actually transmitted to the BEVs
        bev_flow_out_ch = results[(sim['components']['bev_ac'],
                                   sim['components']['ac_bus'])]['sequences']['flow'][sim['ch_dti']]
        bev['flow_out'] = pd.concat([bev['flow_out'], bev_flow_out_ch])

        bev_flow_in_ch = results[(sim['components']['ac_bus'],
                                  sim['components']['ac_bev'])]['sequences']['flow'][sim['ch_dti']]
        bev['flow_in'] = pd.concat([bev['flow_in'], bev_flow_in_ch])

        bev_flow_bal_ch = bev_flow_in_ch - bev_flow_out_ch
        bev['flow_bal'] = pd.concat([bev['flow_bal'], bev_flow_bal_ch])

        for i in range(bev['num']):
            bevx_ess_name = "bev" + str(i + 1) + "_ess"
            bevx_name = "bev" + str(i + 1)
            bevx_sc_ch = views.node(results,
                                    bevx_ess_name)['sequences'][((bevx_ess_name,
                                                                  'None'),
                                                                 'storage_content')][sim['ch_dti']]
            bevx_soc_ch = bevx_sc_ch / bev['cs']
            bev[bevx_name]['soc'] = pd.concat([bev[bevx_name]['soc'], bevx_soc_ch])
            bev[bevx_name]['ph_init_soc'] = bev[bevx_name]['soc'].iloc[-1]
            if bev[bevx_name]['ph_init_soc'] < 0:
                logging.info(bevx_name + 'init SOC < 0: ' + str(bev[bevx_name]['ph_init_soc']))
                bev[bevx_name]['ph_init_soc'] = 0

    return dem, wind, pv, gen, ess, bev


def end_timing(sim):
    sim['runtimeend'] = time.time()
    sim['runtime'] = round(sim['runtimeend'] - sim['runtimestart'], 1)
    logging.info('Runtime of the program was ' + str(sim['runtime']) + " seconds")
    return sim
