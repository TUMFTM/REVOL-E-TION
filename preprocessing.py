"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
February 2nd, 2022

--- Contributors ---
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022

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


from oemof.tools import logger
import os

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging

import sys


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
        inputs={sim['components']['bev_bus']: solph.Flow(
            nominal_value={'uc': 0, 'cc': 0, 'tc': 0, 'v2v': 0, 'v2g': None}[bev['chg_lvl']],
            variable_costs=sim['eps'])},
        outputs={sim['components']['ac_bus']: solph.Flow()},
        conversion_factors={sim['components']['ac_bus']: 1})
    es.add(sim['components']['bev_bus'], sim['components']['ac_bev'], sim['components']['bev_ac'])

    if bev['agr'] and bev['chg_lvl'] != 'uc':  # When vehicles are aggregated into three basic components
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

    if bev['agr'] == 0 and bev['chg_lvl'] != 'uc':  # When vehicles are modeled individually
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
                inputs={sim['components']['bevx_bus']: solph.Flow(
                    nominal_value={'uc': 0, 'cc': 0, 'tc': 0, 'v2v': 1, 'v2g': 1}[bev['chg_lvl']] * bev['dis_pwr'],
                    max=bev['ph_data'][chg_datalabel],
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
                    balanced={'go': True, 'rh': False}[sim['op_strat']],
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

    if bev['chg_lvl'] == 'uc':  # When charging level "uncoordinated charging"
        sim['components']['bev_snk'] = solph.Sink(  # Aggregated sink component for charging vehicles
            label="bev_snk",
            inputs={sim['components']['bev_bus']: solph.Flow(fix=bev['ph_data']['uc_power'] / bev['charge_eff'],
                                                             nominal_value=1)})
        es.add(sim['components']['bev_snk'])

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
        conversion_factors={sim['components']['dc_bus']: ui.xread('ac_dc_eff', sim['sheet'], sim['settings_file'])})
    sim['components']['dc_ac'] = solph.Transformer(
        label="dc_ac",
        inputs={sim['components']['dc_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['ac_bus']: solph.Flow()},
        conversion_factors={sim['components']['ac_bus']: ui.xread('dc_ac_eff', sim['sheet'], sim['settings_file'])})

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
                                              inputs={sim['components']['ac_bus']:
                                                          solph.Flow(fix=dem['ph_data']['Power'], nominal_value=1)})

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
                                                               investment=solph.Investment(
                                                                   ep_costs=wind['eq_pres_cost']),
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

      # Initialize oemof energy system instance for current PH

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

    if sim['debugmode']:
        dump_modelfile(sim, om)

    return sim, om


def dump_modelfile(sim, model):
    """
    Dump model and result files to working directory for later usage
    """
    if sim['op_strat'] == 'rh':
        logging.info('Debug mode not implemented for RH operating strategy - no model file dumped')
    else:
        model.write(sim['modelfile'], io_options={'symbolic_solver_labels': True})

    return None

















