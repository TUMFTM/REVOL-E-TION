"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
February 2nd, 2022

--- Last Update ---
February 6th, 2022

--- Contributors ---
Marcel Brödel, B.Sc. - Semester Thesis in progress

--- Detailed Description ---
This script defines various functions used by main.py for preprocessing of input data and during the optimization loop

--- Input & Output ---
see individual functions

--- Requirements ---
none

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

import oemof.solph as solph
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import logging
import sys
import pylightxl as xl

import economics as eco


def add_bev(sim, es, bev):
    """
    Create EV objects and add them to the energy system
    Option 1: aggregated vehicles (x denotes the flow measurement point)
    ac_bus             bev_bus
      |<-x-------bev_ac---|<--bev_src
      |                   |
      |-x-ac_bev--------->|<->bev_ess
      |                   |
                          |-->bev_snk

    Option 2: individual vehicles with individual bevx (x=1,2,3,...bev_num) buses (x denotes the flow measurement point)
    ac_bus             bev_bus             bev1_bus
      |<-x-------bev_ac---|<-------bev1_bev---|<->bev1_ess
      |                   |                   |
      |-x-ac_bev--------->|---bev_bev1------->|-->bev1_snk
                          |
                          |                bev2_bus
                          |<-------bev2_bev---|<->bev2_ess
                          |                   |
                          |---bev_bev2------->|-->bev2_snk
    """

    sim['components']['bev_bus'] = solph.Bus(
        label='bev_bus')
    sim['components']['ac_bev'] = solph.Transformer(
        label="ac_bev",
        inputs={sim['components']['ac_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['bev_bus']: solph.Flow()},
        conversion_factors={sim['components']['bev_bus']: 1})
    sim['components']['bev_ac'] = solph.Transformer(
        label="bev_ac",
        inputs={sim['components']['bev_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['ac_bus']: solph.Flow()},
        conversion_factors={sim['components']['ac_bus']: 1})
    es.add(sim['components']['bev_bus'], sim['components']['ac_bev'], sim['components']['bev_ac'])

    if bev['agr']:  # When vehicles are aggregated into three basic components
        sim['components']['bev_snk'] = solph.Sink(  # Aggregated sink component modelling leaving vehicles
            label="bev_snk",
            inputs={sim['components']['bev_bus']: solph.Flow(actual_value=bev['ph_data']['sink_data'],
                                                             fixed=True,
                                                             nominal_value=1)})
        sim['components']['bev_src'] = solph.Source(  # Aggregated source component modelling arriving vehicles
            label='bev_src',
            outputs={sim['components']['bev_bus']: solph.Flow(actual_value=bev['ph_data']['source_data'],
                                                              fixed=True,
                                                              nominal_value=1)})
        sim['components']['bev_ess'] = solph.components.GenericStorage(  # Aggregated storage models connected vehicles
            label="bev_ess",
            inputs={sim['components']['bev_bus']: solph.Flow()},
            outputs={sim['components']['bev_bus']: solph.Flow(variable_cost=bev['spec_opex'])},
            nominal_storage_capacity=bev['num'] * bev['cs'],  # Storage capacity is set to the maximum available,
            # adaptation to different numbers of vehicles happens with the min/max storage levels
            loss_rate=0,
            balanced=False,
            initial_storage_level=None,  # TODO: Check for validity!
            inflow_conversion_factor=1,
            outflow_conversion_factor=1,
            min_storage_level=bev['ph_data']['min_charge'],
            # This models the varying storage capacity with (dis)connects
            max_storage_level=bev['ph_data'][
                'max_charge'])  # This models the varying storage capacity with (dis)connects

        es.add(sim['components']['bev_snk'],
               sim['components']['bev_src'],
               sim['components']['bev_ess'])

    else:  # When vehicles are modeled individually
        for x, bevx in enumerate(bev['bevx_list']):  # Create individual vehicles having a bus, a storage and a sink
            num_bevx = x + 1
            bus_label = bevx + "_bus"
            snk_label = bevx + "_snk"
            ess_label = bevx + "_ess"
            chg_label = "bev_" + bevx
            dis_label = bevx + "_bev"
            snk_datalabel = 'sink_data_' + str(num_bevx)
            chg_datalabel = 'at_charger_' + str(num_bevx)
            # maxsoc_datalabel = 'max_charge_' + str(num_bevx)
            minsoc_datalabel = 'min_charge_' + str(num_bevx)

            sim['components']['bevx_bus'] = solph.Bus(  # bevx denominates an individual vehicle component
                label=bus_label)
            sim['components']['bev_bevx'] = solph.Transformer(
                label=chg_label,
                inputs={sim['components']['bev_bus']: solph.Flow(nominal_value=bev['chg_pwr'],
                                                                 max=bev['ph_data'][chg_datalabel],
                                                                 variable_costs=sim['eps'])},
                outputs={sim['components']['bevx_bus']: solph.Flow()},
                conversion_factors={sim['components']['bevx_bus']: bev['charge_eff']})
            sim['components']['bevx_bev'] = solph.Transformer(
                label=dis_label,
                inputs={sim['components']['bevx_bus']: solph.Flow(nominal_value=bev['dis_pwr'],
                                                                  max=bev['dis_pwr'],
                                                                  variable_costs=sim['eps'])},
                outputs={sim['components']['bev_bus']: solph.Flow()},
                conversion_factors={sim['components']['bev_bus']: bev['discharge_eff']})
            if sim['cs_opt']["bev"]:
                sim['components']['bevx_ess'] = solph.components.GenericStorage(
                    label=ess_label,
                    inputs={sim['components']['bevx_bus']: solph.Flow()},
                    outputs={sim['components']['bevx_bus']: solph.Flow(variable_cost=bev['spec_opex'])},
                    loss_rate=0,
                    balanced=False,
                    initial_storage_level=bev[bevx]['ph_init_soc'],
                    inflow_conversion_factor=1,
                    outflow_conversion_factor=1,
                    max_storage_level=1,
                    min_storage_level=bev['ph_data'][minsoc_datalabel],  # ensures the vehicle is charged when leaving
                    investment=solph.Investment(ep_costs=bev['eq_pres_cost']),
                )
            else:
                sim['components']['bevx_ess'] = solph.components.GenericStorage(
                    label=ess_label,
                    inputs={sim['components']['bevx_bus']: solph.Flow()},
                    outputs={sim['components']['bevx_bus']: solph.Flow(variable_cost=bev['spec_opex'])},
                    loss_rate=0,
                    balanced=False,
                    initial_storage_level=bev[bevx]['ph_init_soc'],
                    inflow_conversion_factor=1,
                    outflow_conversion_factor=1,
                    max_storage_level=1,
                    min_storage_level=bev['ph_data'][minsoc_datalabel],  # ensures the vehicle is charged when leaving
                    nominal_storage_capacity=bev['cs'],
                )
            sim['components']['bevx_snk'] = solph.Sink(
                label=snk_label,
                inputs={sim['components']['bevx_bus']: solph.Flow(fix=bev['ph_data'][snk_datalabel], nominal_value=1)})

            es.add(sim['components']['bevx_bus'],
                   sim['components']['bevx_bev'],
                   sim['components']['bev_bevx'],
                   sim['components']['bevx_ess'],
                   sim['components']['bevx_snk'])
    return sim, es


def add_core(sim, es):
    """
    Create basic two-bus structure
    dc_bus              ac_bus
      |                   |
      |---dc_ac---------->|
      |                   |
      |<----------ac_dc---|
   """
    sim['components']['ac_bus'] = solph.Bus(
        label="ac_bus")
    sim['components']['dc_bus'] = solph.Bus(
        label="dc_bus")
    sim['components']['ac_dc'] = solph.Transformer(
        label="ac_dc",
        inputs={sim['components']['ac_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['dc_bus']: solph.Flow()},
        conversion_factors={sim['components']['dc_bus']: xlsxread('ac_dc_eff')})
    sim['components']['dc_ac'] = solph.Transformer(
        label="dc_ac",
        inputs={sim['components']['dc_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['ac_bus']: solph.Flow()},
        conversion_factors={sim['components']['ac_bus']: xlsxread('dc_ac_eff')})

    es.add(sim['components']['ac_bus'],
           sim['components']['dc_bus'],
           sim['components']['ac_dc'],
           sim['components']['dc_ac'])

    return sim, es


def add_dem(sim, es, dem):

    """
    Add stationary demand  (x denotes the flow measurement point)
    ac_bus
      |
      |-x->dem_snk
      |
    """

    sim['components']['dem_snk'] = solph.Sink(label='dem_snk',
                                              inputs={sim['components']['ac_bus']: solph.Flow(fix=dem['ph_data']['P'],
                                                                                              nominal_value=1)})

    es.add(sim['components']['dem_snk'])

    return sim, es


def add_ess(sim, es, ess):
    """
    Create stationary battery storage object and add it to the energy system (x denotes the flow measurement point)
    dc_bus
      |
      |<-x->ess
      |
    """

    if sim['cs_opt']["ess"]:
        sim['components']['ess'] = solph.components.GenericStorage(
            label="ess",
            inputs={sim['components']['dc_bus']: solph.Flow()},
            outputs={sim['components']['dc_bus']: solph.Flow(variable_cost=ess['spec_opex'])},
            loss_rate=ess['cost_decr'],
            balanced=ess['bal'],
            initial_storage_level=ess['ph_init_soc'],
            invest_relation_input_capacity=ess['chg_crate'],
            invest_relation_output_capacity=ess['dis_crate'],
            inflow_conversion_factor=ess['chg_eff'],
            outflow_conversion_factor=ess['dis_eff'],
            investment=solph.Investment(ep_costs=ess['eq_pres_cost']),
        )
    else:
        sim['components']['ess'] = solph.components.GenericStorage(
            label="ess",
            inputs={sim['components']['dc_bus']: solph.Flow()},
            outputs={sim['components']['dc_bus']: solph.Flow(variable_cost=ess['spec_opex'])},
            loss_rate=ess['cost_decr'],
            balanced=ess['bal'],
            initial_storage_level=ess['ph_init_soc'],
            invest_relation_input_capacity=ess['chg_crate'],
            invest_relation_output_capacity=ess['dis_crate'],
            inflow_conversion_factor=ess['chg_eff'],
            outflow_conversion_factor=ess['dis_eff'],
            nominal_storage_capacity=ess['cs'],
        )
    es.add(sim['components']['ess'])
    return sim, es


def add_gen(sim, es, gen):
    """
    Create diesel generator object and add it to the energy system (x denotes the flow measurement point)
    ac_bus
      |
      |<-x-gen
      |
    """

    if sim['cs_opt']["gen"]:
        sim['components']['gen_src'] = solph.Source(
            label='gen_src',
            outputs={sim['components']['ac_bus']: solph.Flow(investment=solph.Investment(ep_costs=gen['eq_pres_cost']),
                                                             variable_costs=gen['spec_opex'])})
    else:
        sim['components']['gen_src'] = solph.Source(
            label='gen_src',
            outputs={sim['components']['ac_bus']: solph.Flow(nominal_value=gen['cs'],
                                                             variable_costs=gen['spec_opex'])})
    es.add(sim['components']['gen_src'])
    return sim, es


def add_pv(sim, es, pv):
    """
    Create solar power objects and add them to the energy system (x denotes the flow measurement point)
    dc_bus              pv_bus
      |                   |
      |<----------pv_dc-x-|<--pv_src
      |                   |
                          |-->pv_exc
    """

    sim['components']['pv_bus'] = solph.Bus(
        label='pv_bus')

    sim['components']['pv_dc'] = solph.Transformer(
        label="pv_dc",
        inputs={sim['components']['pv_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['dc_bus']: solph.Flow()},
        conversion_factors={sim['components']['pv_bus']: 1})

    if sim['cs_opt']["pv"]:
        sim['components']['pv_src'] = solph.Source(
            label="pv_src",
            outputs={
                sim['components']['pv_bus']: solph.Flow(fix=pv['ph_data']['P'],
                                                        investment=solph.Investment(ep_costs=pv['eq_pres_cost']),
                                                        variable_cost=pv['spec_opex'])})
    else:
        sim['components']['pv_src'] = solph.Source(
            label="pv_src",
            outputs={
                sim['components']['pv_bus']: solph.Flow(fix=pv['ph_data']['P'],
                                                        nominal_value=pv['cs'],
                                                        variable_cost=pv['spec_opex'])})
    sim['components']['pv_exc'] = solph.Sink(
        label="pv_exc",
        inputs={sim['components']['pv_bus']: solph.Flow()})

    es.add(sim['components']['pv_bus'],
           sim['components']['pv_dc'],
           sim['components']['pv_src'],
           sim['components']['pv_exc'])

    return sim, es


def add_wind(sim, es, wind):
    """
    Create wind power objects and add them to the energy system (x denotes the flow measurement point)
    ac_bus             wind_bus
      |                   |
      |<--------wind_ac-x-|<--wind_src
      |                   |
                          |-->wind_exc
    """

    sim['components']['wind_bus'] = solph.Bus(
        label='wind_bus')
    sim['components']['wind_ac'] = solph.Transformer(
        label='wind_ac',
        inputs={sim['components']['wind_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['ac_bus']: solph.Flow()},
        conversion_factors={sim['components']['ac_bus']: 1})
    if sim['cs_opt']['wind']:
        sim['components']['wind_src'] = solph.Source(
            label='wind_src',
            outputs={sim['components']['wind_bus']: solph.Flow(fix=wind['ph_data']['P'],
                                                               investment=solph.Investment(ep_costs=wind['eq_pres_cost']),
                                                               variable_cost=wind['spec_opex'])})
    else:
        sim['components']['wind_src'] = solph.Source(
            label="wind_src",
            outputs={sim['components']['wind_bus']: solph.Flow(fix=wind['ph_data']['P'],
                                                               nominal_value=wind['cs'],
                                                               variable_cost=wind['spec_opex'])})
    sim['components']['wind_exc'] = solph.Sink(
        label="wind_exc",
        inputs={sim['components']['wind_bus']: solph.Flow()})
    es.add(sim['components']['wind_bus'],
           sim['components']['wind_ac'],
           sim['components']['wind_src'],
           sim['components']['wind_exc'])
    return sim, es


def build_energysystemmodel(sim, dem, wind, pv, gen, ess, bev):

    logging.info('Building energy system model')

    es = solph.EnergySystem(timeindex=sim['ph_dti'])  # Initialize oemof energy system instance for current PH

    sim, es = add_core(sim, es)

    if sim['enable']['dem']:
        sim, es = add_dem(sim, es, dem)

    if sim['enable']['wind']:
        sim, es = add_wind(sim, es, wind)
    #    else:
    #        wind_bus = wind_src = wind_ac = None

    if sim['enable']["pv"]:
        sim, es = add_pv(sim, es, pv)
    #    else:
    #        pv_dc = pv_bus = pv_src = None

    if sim['enable']["gen"]:
        sim, es = add_gen(sim, es, gen)
    #    else:
    #        gen_src = None

    if sim['enable']["ess"]:
        sim, es = add_ess(sim, es, ess)
    #    else:
    #        ess = None

    if sim['enable']["bev"]:
        sim, es = add_bev(sim, es, bev)
    #    else:
    #        bev_ac = ac_bev = bev_bus = None

    om = solph.Model(es)  # Build the mathematical linear optimization model with pyomo

    dump_modelfile(sim, om)

    return sim, om


def define_bev(prj):
    """
    This function determines the electric vehicles' equivalent costs and adds them system to the energy system
    """

    bev = dict()

    bev['name'] = 'bev'

    bev['filepath'] = os.path.join(os.getcwd(), "scenarios", xlsxread('bev_filename'))
    bev['data'] = pd.read_csv(bev['filepath'], sep=";")
    bev['data']['time'] = pd.date_range(start=prj['start'], periods=len(bev['data']), freq='H')

    bev['spec_capex'] = xlsxread('bev_sce')
    bev['spec_mntex'] = xlsxread('bev_sme')
    bev['spec_opex'] = xlsxread('bev_soe')
    bev['lifespan'] = xlsxread('bev_ls')
    bev['cost_decr'] = xlsxread('bev_cdc')
    bev['cs'] = xlsxread('bev_cs')
    bev['num'] = xlsxread('bev_num')
    bev['agr'] = xlsxread('bev_agr')
    bev['chg_pwr'] = xlsxread('bev_chg_pwr')
    bev['dis_pwr'] = xlsxread('bev_dis_pwr')
    bev['charge_eff'] = xlsxread('bev_charge_eff')
    bev['discharge_eff'] = xlsxread('bev_discharge_eff')

    bev['adj_capex'] = eco.adj_ce(bev['spec_capex'],  # adjusted ce (including maintenance) of the component in $/W
                                  bev['spec_mntex'],
                                  bev['lifespan'],
                                  prj['wacc'])

    bev['eq_pres_cost'] = eco.ann_recur(bev['adj_capex'],
                                        bev['lifespan'],
                                        prj['duration'],
                                        prj['wacc'],
                                        bev['cost_decr'])

    bev['bevx_list'] = []
    for i in range(bev['num']):
        bevx_name = 'bev' + str(i + 1)
        bev['bevx_list'].append(bevx_name)
        bev[bevx_name] = dict()
        bev[bevx_name]['init_soc'] = xlsxread('bev_init_soc')  # TODO: Don't we want to define this at random?
        bev[bevx_name]['ph_init_soc'] = bev[bevx_name]['init_soc']

    return bev


def define_components(sim, prj):
    """
    This function calls the defining functions of the individual components
    """

    if sim['enable']['dem']:
        dem = define_dem(sim, prj)
    else:
        dem = None

    if sim['enable']["wind"]:
        wind = define_wind(prj)
        sim['sources'].append('wind')
    else:
        wind = None

    if sim['enable']["pv"]:
        pv = define_pv(prj)
        sim['sources'].append('pv')
    else:
        pv = None

    if sim['enable']["gen"]:
        gen = define_gen(prj)
        sim['sources'].append('gen')
    else:
        gen = None

    if sim['enable']["ess"]:
        ess = define_ess(prj)
    else:
        ess = None

    if sim['enable']["bev"]:
        bev = define_bev(prj)
    else:
        bev = None

    return sim, dem, wind, pv, gen, ess, bev


def define_dem(sim, prj):
    """
    This function reads in the stationary demand as a dataframe
    """
    dem = dict()

    dem['name'] = 'name'

    dem['filepath'] = os.path.join(os.getcwd(), "scenarios", xlsxread('dem_filename'))
    dem['data'] = pd.read_csv(dem['filepath'], sep=",", skip_blank_lines=False)
    dem['data']['time'] = pd.date_range(start=prj['start'], periods=len(dem['data']), freq=sim['step'])

    return dem


def define_ess(prj):
    """
    This function determines storage equivalent costs and adds the energy storage system to the energy system
    """
    ess = dict()

    ess['name'] = 'ess'

    ess['spec_capex'] = xlsxread('ess_sce')
    ess['spec_mntex'] = xlsxread('ess_sme')
    ess['spec_opex'] = xlsxread('ess_soe')
    ess['lifespan'] = xlsxread('ess_ls')
    ess['cost_decr'] = xlsxread('ess_sd')
    ess['cs'] = xlsxread('ess_cs')
    ess['cdc'] = xlsxread('ess_cdc')
    ess['chg_eff'] = xlsxread('ess_chg_eff')
    ess['dis_eff'] = xlsxread('ess_dis_eff')
    ess['chg_crate'] = xlsxread('ess_chg_crate')
    ess['dis_crate'] = xlsxread('ess_dis_crate')

    ess['adj_capex'] = eco.adj_ce(ess['spec_capex'],  # adjusted ce (including maintenance) of the component in $/W
                                  ess['spec_mntex'],
                                  ess['lifespan'],
                                  prj['wacc'])

    ess['eq_pres_cost'] = eco.ann_recur(ess['adj_capex'],
                                        ess['lifespan'],
                                        prj['duration'],
                                        prj['wacc'],
                                        ess['cost_decr'])

    ess['init_soc'] = xlsxread('ess_init_soc')
    ess['ph_init_soc'] = ess['init_soc']
    ess['bal'] = False  # ESS SOC at end of prediction horizon must not be forced equal to initial SOC

    return ess


def define_gen(prj):
    """
    This function determines diesel generator equivalent costs and adds the generator to the energy system
    """
    gen = dict()

    gen['name'] = 'gen'

    gen['spec_capex'] = xlsxread('gen_sce')
    gen['spec_mntex'] = xlsxread('gen_sme')
    gen['spec_opex'] = xlsxread('gen_soe')
    gen['lifespan'] = xlsxread('gen_ls')
    gen['cost_decr'] = xlsxread('gen_cdc')
    gen['cs'] = xlsxread('gen_cs')

    gen['adj_capex'] = eco.adj_ce(gen['spec_capex'],  # adjusted ce (including maintenance) of the component in $/W
                                  gen['spec_mntex'],
                                  gen['lifespan'],
                                  prj['wacc'])

    gen['eq_pres_cost'] = eco.ann_recur(gen['adj_capex'],
                                        gen['lifespan'],
                                        prj['duration'],
                                        prj['wacc'],
                                        gen['cost_decr'])

    return gen


def define_os(sim):
    """
    Initialize simulation settings and initial states (SOCs) for first optimization iteration
    """
    if sim['op_strat'] != 'rh' and sim['op_strat'] != 'go':
        logging.error("No valid operating strategy selected - stopping execution")
        sys.exit()

    if True in sim['cs_opt'].values() and sim['op_strat'] != 'go':
        logging.error('Error: Rolling horizon strategy is not feasible if component sizing is active')
        logging.error('Please disable sim_cs in parameters.py')
        sys.exit()

    if sim['op_strat'] == 'rh':
        logging.info('Rolling horizon operational strategy initiated')
        sim['rh_ph'] = xlsxread('rh_ph')
        sim['rh_ch'] = xlsxread('rh_ch')
        sim['ph_len'] = {'H': 1, 'T': 60}[sim['step']] * sim['rh_ph']  # number of timesteps for predicted horizon
        sim['ch_len'] = {'H': 1, 'T': 60}[sim['step']] * sim['rh_ch']  # number of timesteps for control horizon
        sim['ch_num'] = int(len(sim['dti']) / sim['ch_len'])  # number of CH timeslices for simulated date range
    elif sim['op_strat'] == 'go':
        logging.info('Global optimum operational strategy initiated')
        sim['ph_len'] = None  # number of timesteps for predicted horizon
        sim['ch_len'] = None  # number of timesteps for control horizon
        sim['ch_num'] = 1  # number of CH timeslices for simulated date range

    return sim


def define_result_structure(sim, prj, dem, wind, pv, gen, ess, bev):
    """
    Initialize empty dataframes for output concatenation
    """

    cres = dict.fromkeys(['e_sim_del',  # Cumulative result counting dict
                          'e_yrl_del',
                          'e_prj_del',
                          'e_dis_del',
                          'e_sim_pro',
                          'e_yrl_pro',
                          'e_prj_pro',
                          'e_dis_pro',
                          'e_eta',
                          'init_capex',
                          'prj_capex',
                          'dis_capex',
                          'ann_capex',
                          'sim_mntex',
                          'yrl_mntex',
                          'prj_mntex',
                          'dis_mntex',
                          'ann_mntex',
                          'sim_opex',
                          'yrl_opex',
                          'prj_opex',
                          'dis_opex',
                          'ann_opex',
                          'sim_totex',
                          'prj_totex',
                          'dis_totex',
                          'ann_totex'
                          ], 0)

    dem['flow'] = pd.Series()

    if sim['enable']['wind']:
        wind['flow'] = pd.Series()

    if sim['enable']['pv']:
        pv['flow'] = pd.Series()

    if sim['enable']['gen']:
        gen['flow'] = pd.Series()

    if sim['enable']['ess']:
        ess['flow_out'] = ess['flow_in'] = ess['soc'] = ess['flow_bal'] = pd.Series()
        ess['soc'] = pd.Series(data={prj['start']: ess['init_soc']})

    if sim['enable']['bev']:
        bev['flow_out'] = bev['flow_in'] = bev['flow_bal'] = pd.Series()
        for bevx in bev['bevx_list']:
            bev[bevx]['soc'] = pd.Series(data={prj['start']: bev[bevx]['init_soc']})

    return dem, wind, pv, gen, ess, bev, cres


def define_prj(sim):
    """
    This function initializes the most basic data of the project
    to be evaluated (which is longer than the simulated timespan
    """

    prj = dict()
    prj['start'] = sim['start']
    prj['duration'] = xlsxread('proj_ls')
    prj['end'] = prj['start'] + relativedelta(years=prj['duration'])
    prj['ddur'] = (prj['start'] - prj['end']).days
    prj['simrat'] = sim['proj'] / prj['ddur']

    prj['wacc'] = xlsxread('proj_wacc')

    return prj


def define_pv(prj):
    """
    This function imports PV power data as a dataframe,
    determines equivalent costs and adds PV power to the energy system
    """
    pv = dict()

    pv['name'] = 'pv'

    pv['filepath'] = os.path.join(os.getcwd(),
                                  "scenarios",
                                  "pvgis_data",
                                  xlsxread('pv_filename'))
    pv['data'] = pd.read_csv(pv['filepath'],
                             sep=",",
                             header=10,
                             skip_blank_lines=False,
                             skipfooter=13,
                             engine='python')
    pv['data']['time'] = pd.to_datetime(pv['data']['time'],
                                        format='%Y%m%d:%H%M').dt.round('H')  # TODO: workaround
    pv['data']['P'] = pv['data']['P'] / 1e3  # data is in W for a 1kWp PV array -> convert to specific power

    pv['spec_capex'] = xlsxread('pv_sce')
    pv['spec_mntex'] = xlsxread('pv_sme')
    pv['spec_opex'] = xlsxread('pv_soe')
    pv['lifespan'] = xlsxread('pv_ls')
    pv['cost_decr'] = xlsxread('pv_cdc')
    pv['cs'] = xlsxread('pv_cs')

    pv['adj_capex'] = eco.adj_ce(pv['spec_capex'],  # adjusted ce (including maintenance) of the component in $/W
                                 pv['spec_mntex'],
                                 pv['lifespan'],
                                 prj['wacc'])

    pv['eq_pres_cost'] = eco.ann_recur(pv['adj_capex'],
                                       pv['lifespan'],
                                       prj['duration'],
                                       prj['wacc'],
                                       pv['cost_decr'])

    return pv


def define_sim():
    """
    This function initializes the most basic simulation data for the timeframe to simulate (and optimize) over
    """

    var = ['sim_name', 'sim_solver', 'sim_dump', 'sim_debug', 'sim_step', 'sim_eps', 'sim_enable', ]

    sim = dict()
    sim['runtimestart'] = time.time()
    sim['runtimestamp'] = datetime.now().strftime("%y%m%d%H%M%S")  # create simulation timestamp
    sim['name'] = sim['runtimestamp'] + "_" + xlsxread('sim_name')

    sim['start'] = datetime.strptime(xlsxread('proj_start'), '%Y/%m/%d')
    sim['proj'] = xlsxread('proj_sim')
    sim['end'] = sim['start'] + relativedelta(days=sim['proj'])
    sim['step'] = xlsxread('sim_step')
    sim['dti'] = pd.date_range(start=sim['start'], end=sim['end'], freq=sim['step']).delete(-1)
    sim['yrrat'] = sim['proj'] / 365.25

    sim['debugmode'] = xlsxread('sim_debug')
    sim['datapath'] = os.path.join(os.getcwd(), "scenarios")
    sim['resultpath'] = os.path.join(os.getcwd(), "results")
    sim['modelpath'] = os.path.join(os.getcwd(), "lp_models")
    sim['logpath'] = os.path.join(os.getcwd(), "logfiles")
    sim['modelfile'] = sim['modelpath'] + ".lp"
    sim['logfile'] = os.path.join(sim['logpath'], sim['name']) + '.log'

    sim['eps'] = 1e-6  # minimum variable cost in $/Wh for transformers to incentivize minimum flow

    sim['op_strat'] = xlsxread('sim_os')
    sim['enable'] = dict(dem=xlsxread('dem_enable'), wind=xlsxread('wind_enable'), pv=xlsxread('pv_enable'),
                         gen=xlsxread('gen_enable'), ess=xlsxread('ess_enable'), bev=xlsxread('bev_enable'))
    sim['cs_opt'] = dict(wind=xlsxread('wind_enable_cs'), pv=xlsxread('pv_enable_cs'), gen=xlsxread('gen_enable_cs'),
                         ess=xlsxread('ess_enable_cs'), bev=xlsxread('bev_enable_cs'))

    sim['components'] = dict()  # empty dict as storage for individual buses, transformers, sources and sinks
    sim['sources'] = []  # create empty list of source modules to iterate over later

    sim['solver'] = xlsxread('sim_solver')
    sim['dump'] = xlsxread('sim_dump')
    sim['eps'] = xlsxread('sim_eps')

    return sim


def define_wind(prj):
    """
    This function imports wind power data as a dataframe,
    determines equivalent costs and adds wind power to the energy system
    """

    wind = dict()

    wind['name'] = 'wind'

    wind['filepath'] = os.path.join(os.getcwd(),
                                    "scenarios",
                                    xlsxread('wind_filename'))
    wind['data'] = pd.read_csv(wind['filepath'],
                               sep=",",
                               skip_blank_lines=False)
    wind['data']['time'] = pd.date_range(start=prj['start'],
                                         periods=len(wind['data']),
                                         freq='H')

    wind['spec_capex'] = xlsxread('wind_sce')
    wind['spec_mntex'] = xlsxread('wind_sme')
    wind['spec_opex'] = xlsxread('wind_soe')
    wind['lifespan'] = xlsxread('wind_ls')
    wind['cost_decr'] = xlsxread('wind_cdc')
    wind['cs'] = xlsxread('wind_cs')

    wind['adj_capex'] = eco.adj_ce(wind['spec_capex'],  # adjusted ce (including maintenance) of the component in $/W
                                   wind['spec_mntex'],
                                   wind['lifespan'],
                                   prj['wacc'])

    wind['eq_pres_cost'] = eco.ann_recur(wind['adj_capex'],
                                         wind['lifespan'],
                                         prj['duration'],
                                         prj['wacc'],
                                         wind['cost_decr'])

    return wind


def dump_modelfile(sim, model):
    """
    Dump model and result files to working directory for later usage
    """
    if sim['debugmode']:  # save the model description for solving externally
        if sim['op_strat'] == 'rh':
            logging.info('Debug mode not implemented for RH operating strategy - no model file dumped')
        else:
            model.write(sim['modelfile'], io_options={'symbolic_solver_labels': True})

    return None


def select_data(sim, dem, wind, pv, bev):
    '''
    Update input data file slices to next prediction horizon
    '''

    if sim['enable']['dem']:
        dem['ph_data'] = slice_data(dem['data'], sim['ph_dti'])  # select correct data slice

    if sim['enable']["wind"]:
        wind['ph_data'] = slice_data(wind['data'], sim['ph_dti'])  # select correct data slice

    if sim['enable']["pv"]:
        pv['ph_data'] = slice_data(pv['data'], sim['ph_dti'])  # select correct data slice

    if sim['enable']["bev"]:
        bev['ph_data'] = slice_data(bev['data'], sim['ph_dti'])  # select correct data slice

    return dem, wind, pv, bev


def set_dti(sim, oc):
    """
    Update datetimeindices to simulate and optimize for next prediction horizon
    """
    if oc == 0:  # no advancement on first iteration (therefore also in go strategy)
        sim['ph_start'] = sim['start']  # set first prediction horizon start
        sim['ch_start'] = sim['ph_start']  # set first control horizon start
    else:
        sim['ph_start'] = sim['ph_start'] + relativedelta(hours=sim['rh_ch'])  # advance to next prediction horizon
        sim['ch_start'] = sim['ph_start']  # advance to next control horizon

    if sim['op_strat'] == 'rh':
        sim['ph_end'] = sim['ph_start'] + relativedelta(hours=sim['rh_ph'])
        sim['ch_end'] = sim['ch_start'] + relativedelta(hours=sim['rh_ch'])
    elif sim['op_strat'] == 'go':
        sim['ph_end'] = sim['end']
        sim['ch_end'] = sim['end']

    sim['ph_dti'] = pd.date_range(start=sim['ph_start'], end=sim['ph_end'], freq=sim['step']).delete(-1)
    sim['ch_dti'] = pd.date_range(start=sim['ch_start'], end=sim['ch_end'], freq=sim['step']).delete(-1)

    return sim


def slice_data(data, dti):
    """
    Selecting correct part of data that fits within the time slice marked out by dti
    """

    cond = data['time'].isin(dti)  # create boolean series marking indices within current PH
    sliced_data = data.loc[cond]  # select correct data slice
    sliced_data = sliced_data.reset_index(drop=True)  # reset data index to start from 0

    return sliced_data


def xlsxread(param):
    """
    Reading parameters from external excel file "settings.xlsx"
    """

    sheet = 'Tabelle1'
    file = 'settings.xlsx'
    db = xl.readxl(fn=file, ws=sheet)

    var = db.ws(ws=sheet).keyrow(key=param, keyindex=1)[1]
    return var
