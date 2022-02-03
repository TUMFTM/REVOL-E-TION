'''
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
February 2nd, 2022

--- Last Update ---
February 3rd, 2022

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
'''

###############################################################################
# Imports
###############################################################################

import oemof.solph as solph
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import logging
import sys

import economics as eco
import parameters as param

###############################################################################
# Function definitions
###############################################################################

def add_bev(sim, es, bev):
    '''
    Create EV objects and add them to the energy system
    Option 1: aggregated vehicles
    ac_bus             bev_bus
      |<---------bev_ac---|<--bev_src
      |                   |
      |---ac_bev--------->|<->bev_ess
      |                   |
                          |-->bev_snk

    Option 2: individual vehicles with individual bevx (x=1,2,3,...bev_num) buses
    ac_bus             bev_bus             bev1_bus
      |<---------bev_ac---|<-------bev1_bev---|<->bev1_ess
      |                   |                   |
      |---ac_bev--------->|---bev_bev1------->|-->bev1_snk
                          |
                          |                bev2_bus
                          |<-------bev2_bev---|<->bev2_ess
                          |                   |
                          |---bev_bev2------->|-->bev2_snk
    '''

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

    if param.bev_agr:  # When vehicles are aggregated into three basic components
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
            outputs={sim['components']['bev_bus']: solph.Flow(variable_cost=param.bev_soe)},
            nominal_storage_capacity=param.bev_num * param.bev_cs,  # Storage capacity is set to the maximum available,
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
        for i in range(0, param.bev_num):  # Create individual vehicles having a bus, a storage and a sink
            bevx_label = "bev" + str(i + 1)
            bus_label = bevx_label + "_bus"
            snk_label = bevx_label + "_snk"
            ess_label = bevx_label + "_ess"
            chg_label = "bev_" + bevx_label
            dis_label = bevx_label + "_bev"
            snk_datalabel = 'sink_data_' + str(i + 1)
            chg_datalabel = 'at_charger_' + str(i + 1)
            maxsoc_datalabel = 'max_charge_' + str(i + 1)
            minsoc_datalabel = 'min_charge_' + str(i + 1)

            sim['components']['bevx_bus'] = solph.Bus(  # bevx denominates an individual vehicle component
                label=bus_label)
            sim['components']['bev_bevx'] = solph.Transformer(
                label=chg_label,
                inputs={sim['components']['bev_bus']: solph.Flow(nominal_value=param.bev_chg_pwr,
                                                                 max=bev['ph_data'][chg_datalabel],
                                                                 variable_costs=sim['eps'])},
                outputs={sim['components']['bevx_bus']: solph.Flow()},
                conversion_factors={sim['components']['bevx_bus']: param.bev_charge_eff})
            sim['components']['bevx_bev'] = solph.Transformer(
                label=dis_label,
                inputs={sim['components']['bevx_bus']: solph.Flow(nominal_value=param.bev_dis_pwr,
                                                                  max=param.bev_dis_pwr,
                                                                  variable_costs=sim['eps'])},
                outputs={sim['components']['bev_bus']: solph.Flow()},
                conversion_factors={sim['components']['bev_bus']: param.bev_discharge_eff})
            if param.sim_cs["bev"]:
                sim['components']['bevx_ess'] = solph.components.GenericStorage(
                    label=ess_label,
                    inputs={sim['components']['bevx_bus']: solph.Flow()},
                    outputs={sim['components']['bevx_bus']: solph.Flow(variable_cost=param.bev_soe)},
                    loss_rate=0,
                    balanced=False,
                    initial_storage_level=bev[bevx_label]['ph_init_soc'],
                    inflow_conversion_factor=1,
                    outflow_conversion_factor=1,
                    max_storage_level=1,
                    min_storage_level=bev['ph_data'][minsoc_datalabel],  # ensures the vehicle is charged when leaving
                    investment=solph.Investment(ep_costs=bev['epc']),
                )
            else:
                sim['components']['bevx_ess'] = solph.components.GenericStorage(
                    label=ess_label,
                    inputs={sim['components']['bevx_bus']: solph.Flow()},
                    outputs={sim['components']['bevx_bus']: solph.Flow(variable_cost=param.bev_soe)},
                    loss_rate=0,
                    balanced=False,
                    initial_storage_level=bev[bevx_label]['ph_init_soc'],
                    inflow_conversion_factor=1,
                    outflow_conversion_factor=1,
                    max_storage_level=1,
                    min_storage_level=bev['ph_data'][minsoc_datalabel],  # ensures the vehicle is charged when leaving
                    nominal_storage_capacity=param.bev_cs,
                )
            sim['components']['bevx_snk'] = solph.Sink(
                label=snk_label,
                inputs={sim['components']['bevx_bus']: solph.Flow(fix=bev['ph_data'][snk_datalabel], nominal_value=1)})

            es.add(sim['components']['bevx_bus'],
                   sim['components']['bevx_bev'],
                   sim['components']['bev_bevx'],
                   sim['components']['bevx_ess'],
                   sim['components']['bevx_snk'])
    return es


def add_components(sim, es, dem, wind, pv, gen, ess, bev):
    es = add_core(sim, es, dem)

    if param.sim_enable['wind']:
        es = add_wind(sim, es, wind)
    #    else:
    #        wind_bus = wind_src = wind_ac = None

    if param.sim_enable["pv"]:
        es = add_pv(sim, es, pv)
    #    else:
    #        pv_dc = pv_bus = pv_src = None

    if param.sim_enable["gen"]:
        es = add_gen(sim, es, gen)
    #    else:
    #        gen_src = None

    if param.sim_enable["ess"]:
        es = add_ess(sim, es, ess)
    #    else:
    #        ess = None

    if param.sim_enable["bev"]:
        es = add_bev(sim, es, bev)
    #    else:
    #        bev_ac = ac_bev = bev_bus = None

    return es


def add_core(sim, es, dem):
    '''
    Create basic two-bus structure
    dc_bus              ac_bus
      |                   |
      |---dc_ac---------->|-->dem
      |                   |
      |<----------ac_dc---|
   '''
    sim['components']['ac_bus'] = solph.Bus(
        label="ac_bus")
    sim['components']['dc_bus'] = solph.Bus(
        label="dc_bus")
    sim['components']['ac_dc'] = solph.Transformer(
        label="ac_dc",
        inputs={sim['components']['ac_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['dc_bus']: solph.Flow()},
        conversion_factors={sim['components']['dc_bus']: param.ac_dc_eff})
    sim['components']['dc_ac'] = solph.Transformer(
        label="dc_ac",
        inputs={sim['components']['dc_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['ac_bus']: solph.Flow()},
        conversion_factors={sim['components']['ac_bus']: param.dc_ac_eff})
    sim['components']['dem_snk'] = solph.Sink(
        label='dem_snk',
        inputs={sim['components']['ac_bus']: solph.Flow(fix=dem['ph_data']['P'], nominal_value=1)}
    )
    es.add(sim['components']['ac_bus'],
           sim['components']['dc_bus'],
           sim['components']['ac_dc'],
           sim['components']['dc_ac'],
           sim['components']['dem_snk'])
    return es


def add_ess(sim, es, ess):
    '''
    Create stationary battery storage object and add it to the energy system
    dc_bus
      |
      |<->ess
      |
    '''
    if param.sim_cs["ess"]:
        sim['components']['ess'] = solph.components.GenericStorage(
            label="ess",
            inputs={sim['components']['dc_bus']: solph.Flow()},
            outputs={sim['components']['dc_bus']: solph.Flow(variable_cost=param.ess_soe)},
            loss_rate=param.ess_sd,
            balanced=ess['bal'],
            initial_storage_level=ess['ph_ini_soc'],
            invest_relation_input_capacity=param.ess_chg_crate,
            invest_relation_output_capacity=param.ess_dis_crate,
            inflow_conversion_factor=param.ess_chg_eff,
            outflow_conversion_factor=param.ess_dis_eff,
            investment=solph.Investment(ep_costs=ess['epc']),
        )
    else:
        sim['components']['ess'] = solph.components.GenericStorage(
            label="ess",
            inputs={sim['components']['dc_bus']: solph.Flow()},
            outputs={sim['components']['dc_bus']: solph.Flow(variable_cost=param.ess_soe)},
            loss_rate=param.ess_sd,
            balanced=ess['bal'],
            initial_storage_level=ess['ph_init_soc'],
            invest_relation_input_capacity=param.ess_chg_crate,
            invest_relation_output_capacity=param.ess_dis_crate,
            inflow_conversion_factor=param.ess_chg_eff,
            outflow_conversion_factor=param.ess_dis_eff,
            nominal_storage_capacity=param.ess_cs,
        )
    es.add(sim['components']['ess'])
    return es


def add_gen(sim, es, gen):
    '''
    Create diesel generator object and add it to the energy system
    ac_bus
      |
      |<--gen
      |
    '''
    if param.sim_cs["gen"]:
        sim['components']['gen_src'] = solph.Source(
            label='gen_src',
            outputs={sim['components']['ac_bus']: solph.Flow(investment=solph.Investment(ep_costs=gen['epc']),
                                                             variable_costs=param.gen_soe)})
    else:
        sim['components']['gen_src'] = solph.Source(
            label='gen_src',
            outputs={sim['components']['ac_bus']: solph.Flow(nominal_value=param.gen_cs,
                                                             variable_costs=param.gen_soe)})
    es.add(sim['components']['gen_src'])
    return es


def add_pv(sim, es, pv):
    '''
    Create solar power objects and add them to the energy system
    dc_bus              pv_bus
      |                   |
      |<----------pv_dc---|<--pv_src
      |                   |
                          |-->pv_exc
    '''
    sim['components']['pv_bus'] = solph.Bus(
        label='pv_bus')

    sim['components']['pv_dc'] = solph.Transformer(
        label="pv_dc",
        inputs={sim['components']['pv_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['dc_bus']: solph.Flow()},
        conversion_factors={sim['components']['pv_bus']: 1})

    if param.sim_cs["pv"]:
        sim['components']['pv_src'] = solph.Source(
            label="pv_src",
            outputs={
                sim['components']['pv_bus']: solph.Flow(fix=pv['ph_data']['P'],
                                                        investment=solph.Investment(ep_costs=pv['epc']),
                                                        variable_cost=param.pv_soe)})
    else:
        sim['components']['pv_src'] = solph.Source(
            label="pv_src",
            outputs={
                sim['components']['pv_bus']: solph.Flow(fix=pv['ph_data']['P'],
                                                        nominal_value=param.pv_cs,
                                                        variable_cost=param.pv_soe)})
    sim['components']['pv_exc'] = solph.Sink(
        label="pv_exc",
        inputs={sim['components']['pv_bus']: solph.Flow()})

    es.add(sim['components']['pv_bus'],
           sim['components']['pv_dc'],
           sim['components']['pv_src'],
           sim['components']['pv_exc'])

    return es


def add_wind(sim, es, wind):
    '''
    Create wind power objects and add them to the energy system
    ac_bus             wind_bus
      |                   |
      |<--------wind_ac---|<--wind_src
      |                   |
                          |-->wind_exc
    '''
    sim['components']['wind_bus'] = solph.Bus(
        label='wind_bus')
    sim['components']['wind_ac'] = solph.Transformer(
        label='wind_ac',
        inputs={sim['components']['wind_bus']: solph.Flow(variable_costs=sim['eps'])},
        outputs={sim['components']['ac_bus']: solph.Flow()},
        conversion_factors={sim['components']['ac_bus']: 1})
    if param.sim_cs['wind']:
        sim['components']['wind_src'] = solph.Source(
            label='wind_src',
            outputs={sim['components']['wind_bus']: solph.Flow(fix=wind['ph_data']['P'],
                                                               investment=solph.Investment(ep_costs=wind['epc']),
                                                               variable_cost=param.wind_soe)})
    else:
        sim['components']['wind_src'] = solph.Source(
            label="wind_src",
            outputs={sim['components']['wind_bus']: solph.Flow(fix=wind['ph_data']['P'],
                                                               nominal_value=param.wind_cs,
                                                               variable_cost=param.wind_soe)})
    sim['components']['wind_exc'] = solph.Sink(
        label="wind_exc",
        inputs={sim['components']['wind_bus']: solph.Flow()})
    es.add(sim['components']['wind_bus'],
           sim['components']['wind_ac'],
           sim['components']['wind_src'],
           sim['components']['wind_exc'])
    return es


def define_bev():
    """
    This function determines the electric vehicles' equivalent costs and adds them system to the energy system
    """
    bev = dict()
    if param.sim_enable["bev"]:
        bev['filepath'] = os.path.join(os.getcwd(), "scenarios", param.bev_filename)
        bev['data'] = pd.read_csv(bev['filepath'], sep=";")
        bev['data']['time'] = pd.date_range(start=param.proj_start, periods=len(bev['data']), freq='H')
        bev['ace'] = eco.adj_ce(param.bev_sce, param.bev_sme, param.bev_ls, param.proj_wacc)
        bev['epc'] = eco.ann_recur(bev['ace'], param.bev_ls, param.proj_ls, param.proj_wacc, param.bev_cdc)
        for i in range(param.bev_num):
            bevx_name = 'bev' + str(i + 1)
            bev[bevx_name] = dict()
    else:
        bev['data'] = bev['epc'] = None
    return bev


def define_components(sim):
    '''
    This function calls the defining functions of the individual components
    '''
    dem = define_dem()

    sim, wind = define_wind(sim)
    sim, pv = define_pv(sim)
    sim, gen = define_gen(sim)

    ess = define_ess()
    bev = define_bev()
    return sim, dem, wind, pv, gen, ess, bev


def define_dem():
    """
    This function reads in the stationary demand as a dataframe
    """
    dem = dict()
    dem['filepath'] = os.path.join(os.getcwd(), "scenarios", param.dem_filename)
    dem['data'] = pd.read_csv(dem['filepath'], sep=",", skip_blank_lines=False)
    dem['data']['time'] = pd.date_range(start=param.proj_start, periods=len(dem['data']), freq=param.sim_step)
    return dem


def define_ess():
    """
    This function determines storage equivalent costs and adds the energy storage system to the energy system
    """
    ess = dict()
    if param.sim_enable["ess"]:
        ess['ace'] = eco.adj_ce(param.ess_sce, param.ess_sme, param.ess_ls,
                                param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/Wh
        ess['epc'] = eco.ann_recur(ess['ace'], param.ess_ls, param.proj_ls, param.proj_wacc, param.ess_cdc)
    else:
        ess['epc'] = None
    return ess


def define_gen(sim):
    """
    This function determines diesel generator equivalent costs and adds the generator to the energy system
    """
    gen = dict()
    if param.sim_enable["gen"]:
        gen['ace'] = eco.adj_ce(param.gen_sce, param.gen_soe, param.gen_ls,
                                param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
        gen['epc'] = eco.ann_recur(gen['ace'], param.gen_ls, param.proj_ls, param.proj_wacc, param.gen_cdc)
        sim['sources'].append('gen')
    else:
        gen['epc'] = None
    return sim, gen


def define_os(sim, dem, wind, pv, gen, ess, bev):
    """
    Initialize simulation settings and initial states (SOCs) for first optimization iteration
    """
    if param.sim_os != 'rh' and param.sim_os != 'go':
        logging.error("No valid operating strategy selected - stopping execution")
        sys.exit()

    if (True in param.sim_cs.values()):
        logging.error('Error: Rolling horizon strategy is not feasible if component sizing is active')
        logging.error('Please disable sim_cs in parameters.py')
        sys.exit()

    if param.sim_os == 'rh':
        sim['ph_len'] = {'H': 1, 'T': 60}[param.sim_step] * param.rh_ph  # number of timesteps for predicted horizon
        sim['ch_len'] = {'H': 1, 'T': 60}[param.sim_step] * param.rh_ch  # number of timesteps for control horizon
        sim['opt_counter'] = int(len(sim['dti']) / sim['ch_num'])  # number of CH timeslices for simulated date range
    elif param.sim_os == 'go':
        sim['ph_len'] = None  # number of timesteps for predicted horizon
        sim['ch_len'] = None  # number of timesteps for control horizon
        sim['opt_counter'] = 1  # number of CH timeslices for simulated date range

    if param.sim_enable['ess']:
        ess['ph_init_soc'] = param.ess_init_soc
        ess['bal'] = False  # ESS SOC at end of prediction horizon must not be forced equal to initial SOC
        ess['soc'] = pd.Series(data={sim['start']: param.ess_cs * ess['ph_init_soc']})  # TODO: Still valid?

    if param.sim_enable['bev']:
        for i in range(param.bev_num):
            bevx_name = 'bev' + str(i + 1)
            bev[bevx_name]['ph_init_soc'] = param.bev_init_soc  # TODO: Check for validity!

    return sim, dem, wind, pv, gen, ess, bev


def define_out(dem, wind, pv, gen, ess, bev):
    """
    Initialize empty dataframes for output concatenation
    """

    cres = dict.fromkeys(['ice', 'tce', 'pce', 'yme', 'tme', 'pme', 'yoe', 'toe', 'poe', 'ype', 'ten', 'pen',
                         'yde', 'tde', 'pde', 'ann', 'npc', 'lcoe', 'eta'], 0)

    dem['flow'] = pd.Series()

    if param.sim_enable['wind']:
        wind['flow'] = pd.Series()

    if param.sim_enable['pv']:
        pv['flow'] = pd.Series()

    if param.sim_enable['gen']:
        gen['flow'] = pd.Series()

    if param.sim_enable['ess']:
        ess['flow_out'] = ess['flow_in'] = ess['soc'] = pd.Series()

    if param.sim_enable['bev']:
        bev['flow_out'] = bev['flow_in'] = pd.Series()
        for i in range(param.bev_num):
            bevx_name = 'bev' + str(i + 1)
            bev[bevx_name]['soc'] = pd.Series()  # Power flow results initialization

    return dem, wind, pv, gen, ess, bev, cres


def define_prj(sim):
    """
    This function initializes the most basic data of the project to be evaluated (which is longer than the simulated timespan
    """
    prj = dict()
    prj['start'] = sim['start']
    prj['ydur'] = param.proj_ls
    prj['end'] = proj['start'] + relativedelta(years=prj['ydur'])
    prj['ddur'] = (proj['start'] - proj['end']).days
    prj['simrat'] = param.proj_sim / proj['ddur']
    return proj


def define_pv(sim):
    """
    This function imports PV power data as a dataframe, determines equivalent costs and adds PV power to the energy system
    """
    pv = dict()
    if param.sim_enable["pv"]:
        pv['filepath'] = os.path.join(os.getcwd(), "scenarios", "pvgis_data", param.pv_filename)
        pv['data'] = pd.read_csv(pv['filepath'], sep=",", header=10, skip_blank_lines=False, skipfooter=13,
                                 engine='python')
        pv['data']['time'] = pd.to_datetime(pv['data']['time'], format='%Y%m%d:%H%M').dt.round('H')  # TODO: workaround
        pv['data']['P'] = pv['data']['P'] / 1e3  # data is in W for a 1kWp PV array -> convert to specific power
        pv['ace'] = eco.adj_ce(param.pv_sce, param.pv_soe, param.pv_ls,
                               param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
        pv['epc'] = eco.ann_recur(pv['ace'], param.pv_ls, param.proj_ls, param.proj_wacc, param.pv_cdc)
        sim['sources'].append('pv')
    else:
        pv['data'] = pv['epc'] = None
    return sim, pv


def define_sim():
    """
    This function initializes the most basic simulation data for the timeframe to simulate (and optimize) over
    """
    sim = dict()
    sim['runtimestart'] = time.time()
    sim['runtimestamp'] = datetime.now().strftime("%y%m%d%H%M%S")  # create simulation timestamp
    sim['name'] = sim['runtimestamp'] + "_" + param.sim_name
    ###
    sim['start'] = datetime.strptime(param.proj_start, '%m/%d/%Y')
    sim['end'] = sim['start'] + relativedelta(days=param.proj_sim)
    sim['dti'] = pd.date_range(start=sim['start'], end=sim['end'], freq=param.sim_step).delete(-1)
    sim['yrrat'] = param.proj_sim / 365.25
    ###
    sim['datapath'] = os.path.join(os.getcwd(), "scenarios")
    sim['resultpath'] = os.path.join(os.getcwd(), "results")
    sim['modelpath'] = os.path.join(os.getcwd(), "lp_models")
    sim['logpath'] = os.path.join(os.getcwd(), "logfiles")
    sim['modelfile'] = sim['modelpath'] + ".lp"
    sim['logfile'] = os.path.join(sim['logpath'], sim['name']) + '.log'
    ###
    sim['eps'] = 1e-6  # minimum variable cost in $/Wh for transformers to incentivize minimum flow
    ###
    sim['components'] = dict()  # empty dict as storage for individual buses, transformers, sources and sinks
    sim['sources'] = []  # create empty list of source modules to iterate over later
    return sim


def define_wind(sim):
    """
    This function imports wind power data as a dataframe, determines equivalent costs and adds wind power to the energy system
    """
    wind = dict()
    if param.sim_enable["wind"]:
        wind['filepath'] = os.path.join(os.getcwd(), "scenarios", param.wind_filename)
        wind['data'] = pd.read_csv(wind['filepath'], sep=",", skip_blank_lines=False)
        wind['data']['time'] = pd.date_range(start=param.proj_start, periods=len(wind['data']), freq='H')
        wind['ace'] = eco.adj_ce(param.wind_sce, param.wind_sme, param.wind_ls,
                                 param.proj_wacc)  # adjusted ce (including maintenance) of the component in $/W
        wind['epc'] = eco.ann_recur(wind['ace'], param.wind_ls, param.proj_ls, param.proj_wacc, param.wind_cdc)
        sim['sources'].append('wind')
    else:
        wind['data'] = wind['epc'] = None
    return sim, wind


def select_data(sim, dem, wind, pv, bev):
    '''
    Update input data file slices to next prediction horizon
    '''

    dem['ph_data'] = slice_data(dem['data'], sim['ph_dti'])  # select correct data slice

    if param.sim_enable["wind"]:
        wind['ph_data'] = slice_data(wind['data'], sim['ph_dti'])  # select correct data slice
    else:
        wind['ph_data'] = None

    if param.sim_enable["pv"]:
        pv['ph_data'] = slice_data(pv['data'], sim['ph_dti'])  # select correct data slice
    else:
        pv['ph_data'] = None

    if param.sim_enable["bev"]:
        bev['ph_data'] = slice_data(bev['data'], sim['ph_dti'])  # select correct data slice
    else:
        bev['ph_data'] = None

    return dem, wind, pv, bev


def set_dti(sim, oc):
    """
    Update datetimeindices to simulate and optimize for next prediction horizon
    """
    if oc == 0:  # no advancement on first iteration (therefore also in go strategy)
        sim['ph_start'] = sim['start']  # set first prediction horizon start
        sim['ch_start'] = sim['ph_start']  # set first control horizon start
    else:
        sim['ph_start'] = sim['ph_start'] + relativedelta(hours=param.rh_ch)  # advance to next prediction horizon
        sim['ch_start'] = sim['ph_start']  # advance to next control horizon

    if param.sim_os == 'rh':
        sim['ph_end'] = sim['ph_start'] + relativedelta(hours=param.rh_ph)
        sim['ch_end'] = sim['ch_start'] + relativedelta(hours=param.rh_ch)
    elif param.sim_os == 'go':
        sim['ph_end'] = sim['end']
        sim['ch_end'] = sim['end']

    sim['ph_dti'] = pd.date_range(start=sim['ph_start'], end=sim['ph_end'], freq=param.sim_step).delete(-1)
    sim['ch_dti'] = pd.date_range(start=sim['ch_start'], end=sim['ch_end'], freq=param.sim_step).delete(-1)

    return sim


def slice_data(data, dti):
    """
    Selecting correct part of data that fits within the time slice marked out by dti
    """

    cond = data['time'].isin(dti)  # create boolean series marking indices within current PH
    sliced_data = data.loc[cond]  # select correct data slice
    sliced_data = sliced_data.reset_index(drop=True)  # reset data index to start from 0

    return sliced_data
