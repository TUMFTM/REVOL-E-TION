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

--- Detailed Description ---
This script defines the global settings for the MGEVOpti project

"""

import pylightxl as xl

###############################################################################
# Define Input Function
###############################################################################

def xlsxread(param):
    var = db.ws(ws=sheet).keyrow(key=param, keyindex=1)[1]
    return var

sheet = 'Tabelle1'
file = 'settings.xlsx'
db = xl.readxl(fn=file, ws=sheet)


###############################################################################
# Call Input Variables
###############################################################################
# Simulation options
#sim_name = xlsxread('sim_name')     # name of scenario
#sim_solver = xlsxread('sim_solver') # solver selection. Options: "cbc", "gplk", "gurobi"
#sim_dump = xlsxread('sim_dump')     # "True" activates oemof model and result saving
#sim_debug = xlsxread('sim_debug')   # "True" activates mathematical model saving and extended solver output
#sim_step = xlsxread('sim_step')     # time step length ('H'=hourly, other lengths not tested yet!)
#sim_eps = xlsxread('sim_eps')       # minimum variable cost in $/Wh for transformers to incentivize minimum flow
#sim_enable = dict(dem=xlsxread('dem_enable'), wind=xlsxread('wind_enable'), pv=xlsxread('pv_enable'), gen=xlsxread('gen_enable'), ess=xlsxread('ess_enable'), bev=xlsxread('bev_enable'))
##sim_os = xlsxread('sim_os')         # go: global optimum (no operating strategy), rh: rolling horizon strategy

# Options operation strategy "Rolling Horizon" (only needed if sim_os = True)
#rh_ph = xlsxread('os_ph')           # predicted horizon in h
#rh_ch = xlsxread('os_ch')           # overlap horizon in h

# Project data
#proj_start = xlsxread('proj_start') # Project start date (YYYY/MM/DD)
#proj_sim = xlsxread('proj_sim')     # Simulation timeframe in days
#proj_ls = xlsxread('proj_ls')       # Project duration in years
#proj_wacc = xlsxread('proj_wacc')   # unitless weighted average cost of capital for the project

# Transformer component data
#ac_dc_eff = xlsxread('ac_dc_eff')   # unitless conversion efficiency of ac-dc bus transformer component
#dc_ac_eff = xlsxread('dc_ac_eff')   # unitless conversion efficiency of dc-ac bus transformer component

# Demand data file
# dem_filename = xlsxread('dem_filename')     # input data file containing timeseries for electricity demand in W

# Wind component data
#wind_filename = xlsxread('wind_filename')   # name of the normalized wind power profile csv file in ./scenarios to evaluate
#wind_sce = xlsxread('wind_sce')     # specific capital expenses of the component in $/W
#wind_sme = xlsxread('wind_sme')     # specific maintenance expenses of the component in $/(W*year)
#wind_soe = xlsxread('wind_soe')     # specific operational expenses of the component in $/Wh
#wind_ls = xlsxread('wind_ls')       # lifespan of the component in years
#wind_cdc = xlsxread('wind_cdc')     # annual ratio of component cost decrease
#wind_cs = xlsxread('wind_cs')       # component size (peak) in kW, only valid if sim_cs["wind"]==False

# Photovoltaic array component data
#pv_filename = xlsxread('pv_filename')       # name of the normalized pv power profile csv file in ./scenarios to evaluate
#pv_sce = xlsxread('pv_sce')         # specific capital expenses of the component in $/W
#pv_sme = xlsxread('pv_sme')         # specific maintenance expenses of the component in $/(W*year)
#pv_soe = xlsxread('pv_soe')         # specific operational expenses of the component in $/Wh
#pv_ls = xlsxread('pv_ls')           # lifespan of the component in years
#pv_cdc = xlsxread('pv_cdc')         # annual ratio of cost decrease
#pv_cs = xlsxread('pv_cs')           # component size (peak) in W, only valid if sim_cs["pv"]==False

# Diesel generator component data
#gen_sce = xlsxread('gen_sce')       # specific capital expenses of the component in $/W (original 1.15)
#gen_sme = xlsxread('gen_sme')       # specific maintenance expenses of the component in $/(W*year)
#gen_soe = xlsxread('gen_soe')       # specific operational expenses of the component in $/Wh (original 0.00036)
#gen_ls = xlsxread('gen_ls')         # lifespan of the component in years
#gen_cdc = xlsxread('gen_cdc')       # annual ratio of component cost decrease
#gen_cs = xlsxread('gen_cs')         # component size in W, only valid if sim_cs["gen"]==False

# Stationary storage system component data
#ess_sce = xlsxread('ess_sce')       # specific capital expenses of the component in $/Wh
#ess_sme = xlsxread('ess_sme')       # specific maintenance expenses of the component in $/(Wh*year)
#ess_soe = xlsxread('ess_soe')       # specific operational expenses of the component in $/Wh
#ess_ls = xlsxread('ess_ls')         # lifespan of the component in years
#ess_chg_eff = xlsxread('ess_chg_eff')       # charging efficiency
#ess_dis_eff = xlsxread('ess_dis_eff')       # discharge efficiency
#ess_chg_crate = xlsxread('ess_chg_crate')   # maximum charging C-rate in 1/h
#ess_dis_crate = xlsxread('ess_dis_crate')   # maximum discharging C-rate in 1/h
#ess_init_soc = xlsxread('ess_init_soc')     # initial state of charge
#ess_sd = xlsxread('ess_sd')         # self-discharge rate of the component in ???
#ess_cdc = xlsxread('ess_cdc')       # annual ratio of component cost decrease
#ess_cs = xlsxread('ess_cs')         # component size in Wh, only valid if sim_cs["ess"]==False

# BEV
#bev_filename = xlsxread('bev_filename')
#bev_agr = xlsxread('bev_agr')       # boolean triggering simplified simulation of BEVs as a single set of components when true
#bev_num = xlsxread('bev_num')       # number of vehicles to be simulated
#bev_sce = xlsxread('bev_sce')       # specific capital expenses of the component in $/Wh
#bev_sme = xlsxread('bev_sme')       # specific maintenance expenses of the component in $/(Wh*year)
#bev_soe = xlsxread('bev_soe')       # specific operational expenses of the component in $/Wh
#bev_ls = xlsxread('bev_ls')         # lifespan of the component in years
#bev_init_soc = xlsxread('bev_init_soc')     # initial state of charge
#bev_chg_pwr = xlsxread('bev_chg_pwr')       # maximum allowable charge power for each individual BEV
#bev_dis_pwr = xlsxread('bev_dis_pwr')       # maximum allowable discharge power for each individual BEV
#bev_charge_eff = xlsxread('bev_charge_eff')         # unitless charge efficiency
#bev_discharge_eff = xlsxread('bev_discharge_eff')   # unitless discharge efficiency
#bev_cdc = xlsxread('bev_cdc')       # annual ratio of component cost decrease
#bev_cs = xlsxread('bev_cs')         # battery size of vehicles in Wh, only valid if sim_cs["bev"]==False




### BACKUP ###
#
# # Simulation options
# sim_name = 'mg_ev'  # name of scenario
# sim_solver = 'cbc'  # solver selection. Options: "cbc", "gplk", "gurobi"
# sim_dump = False  # "True" activates oemof model and result saving
# sim_debug = False  # "True" activates mathematical model saving and extended solver output
# sim_step = 'H'  # time step length ('H'=hourly, other lengths not tested yet!)
# sim_enable = dict(dem=True,
#                   wind=True,
#                   pv=True,
#                   gen=True,
#                   ess=True,
#                   bev=True)
# sim_cs = dict(wind=True,
#               pv=True,
#               gen=True,
#               ess=True,
#               bev=False)
# sim_os = 'go'  # go: global optimum (no operating strategy), rh: rolling horizon strategy
#
# # Options operation strategy "Rolling Horizon" (only needed if sim_os = True)
# rh_ph = 48  # predicted horizon in h
# rh_ch = 24  # overlap horizon in h
#
# # Project data
# proj_start = "1/15/2005"  # Project start date (MM/DD/YYYY) - Caution US date format
# proj_sim = 10  # Simulation timeframe in days
# proj_ls = 25  # Project duration in years
# proj_wacc = 0.07  # unitless weighted average cost of capital for the project
#
# # Demand data file
# dem_filename = "dem_data.csv"  # input data file containing timeseries for electricity demand in W
#
# # Transformer component data
# ac_dc_eff = 0.95  # unitless conversion efficiency of ac-dc bus transformer component
# dc_ac_eff = 0.95  # unitless conversion efficiency of dc-ac bus transformer component
#
# # Wind component data
# wind_filename = "wind_data.csv"  # name of the normalized wind power profile csv file in ./scenarios to evaluate
# wind_sce = 1.355  # specific capital expenses of the component in $/W
# wind_sme = 0  # specific maintenance expenses of the component in $/(W*year)
# wind_soe = 0  # specific operational expenses of the component in $/Wh
# wind_ls = 20  # lifespan of the component in years
# wind_cdc = 1  # annual ratio of component cost decrease
# wind_cs = 100e3  # component size (peak) in W, only valid if sim_cs["wind"]==False
#
# # Photovoltaic array component data
# pv_filename = "Zatta_CI_1kWp.csv"  # name of the normalized pv power profile csv file in ./scenarios to evaluate
# pv_sce = 0.8  # specific capital expenses of the component in $/W
# pv_sme = 0  # specific maintenance expenses of the component in $/(W*year)
# pv_soe = 0  # specific operational expenses of the component in $/Wh
# pv_ls = 25  # lifespan of the component in years
# pv_cdc = 1  # annual ratio of cost decrease
# pv_cs = 205e3  # component size (peak) in W, only valid if sim_cs["pv"]==False
#
# # Diesel generator component data
# gen_sce = 1.5  # specific capital expenses of the component in $/W (original 1.15)
# gen_sme = 0  # specific maintenance expenses of the component in $/(W*year)
# gen_soe = 0.00065  # specific operational expenses of the component in $/Wh (original 0.00036)
# gen_ls = 10  # lifespan of the component in years
# gen_cdc = 1  # annual ratio of component cost decrease
# gen_cs = 85e3  # component size in W, only valid if sim_cs["gen"]==False
#
# # Stationary storage system component data
# ess_sce = 0.8  # specific capital expenses of the component in $/Wh
# ess_sme = 0  # specific maintenance expenses of the component in $/(Wh*year)
# ess_soe = 0  # specific operational expenses of the component in $/Wh
# ess_ls = 10  # lifespan of the component in years
# ess_chg_eff = 0.95  # charging efficiency
# ess_dis_eff = 0.85  # discharge efficiency
# ess_chg_crate = 0.5  # maximum charging C-rate in 1/h
# ess_dis_crate = 0.5  # maximum discharging C-rate in 1/h
# ess_init_soc = 0.7  # initial state of charge
# ess_sd = 0  # self-discharge rate of the component in ???
# ess_cdc = 1  # annual ratio of component cost decrease
# ess_cs = 65e3  # component size in Wh, only valid if sim_cs["ess"]==False
#
# # BEV
# bev_filename = "ind_car_data.csv"
# bev_agr = False  # boolean triggering simplified simulation of BEVs as a single set of components when true
# bev_num = 10  # number of vehicles to be simulated
# bev_sce = 0.8  # specific capital expenses of the component in $/Wh
# bev_sme = 0  # specific maintenance expenses of the component in $/(Wh*year)
# bev_soe = 0  # specific operational expenses of the component in $/Wh
# bev_ls = 10  # lifespan of the component in years
# bev_init_soc = 0.5  # initial state of charge
# bev_chg_pwr = 3600  # maximum allowable charge power for each individual BEV
# bev_dis_pwr = 3600  # maximum allowable discharge power for each individual BEV
# bev_charge_eff = 0.95  # unitless charge efficiency
# bev_discharge_eff = 0.95  # unitless discharge efficiency
# bev_cdc = 1  # annual ratio of component cost decrease
# bev_cs = 30e3  # battery size of vehicles in Wh, only valid if sim_cs["bev"]==False
#
