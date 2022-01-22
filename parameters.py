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
This script defines the global settings for the MGEVOpti project

"""

###############################################################################
# Input
###############################################################################

# Simulation options
sim_name = "mg_ev_main"  # name of scenario
sim_solver = "cbc"  # solver selection. Options: "cbc", "gplk", "gurobi"
sim_dump = False  # "True" activates oemof model and result saving
sim_debug = False  # "True" activates mathematical model saving and extended solver output
sim_step = 'H'  # time step length ('H'=hourly, other lengths not tested yet!)
sim_eps = 1e-6  # minimum variable cost in $/Wh for transformers to incentivize minimum flow
sim_enable = dict(wind=False, pv=True, gen=True, ess=True, bev=True)
sim_cs = dict(wind=False, pv=False, gen=False, ess=False, bev=False)
sim_os = dict(opt=True, rh=False) # opt: global optimum (no operating strategy), rh: rolling horizon strategy

# Options operation strategy "Rolling Horizon" (only needed if sim_os = True)
os_ph = 48  # predicted horizon in h
os_ch = 24  # overlap horizon in h

# Project data
proj_start = "1/1/2015"  # Project start date (DD/MM/YYYY)
proj_sim = 10 # Simulation timeframe in days
proj_ls = 25  # Project duration in years
proj_wacc = 0.07  # unitless weighted average cost of capital for the project

# Demand data file
dem_filename = "dem_data.csv"  # input data file containing timeseries for electricity demand in W

# Transformer component data
ac_dc_eff = 0.95  # unitless conversion efficiency of ac-dc bus transformer component
dc_ac_eff = 0.95  # unitless conversion efficiency of dc-ac bus transformer component

# Wind component data
wind_filename = "wind_data.csv"  # name of the normalized wind power profile csv file in ./scenarios to evaluate
wind_sce = 1.355  # specific capital expenses of the component in $/W
wind_sme = 0  # specific maintenance expenses of the component in $/(W*year)
wind_soe = 0  # specific operational expenses of the component in $/Wh
wind_ls = 20  # lifespan of the component in years
wind_cdc = 1  # annual ratio of component cost decrease
wind_cs = 100e3  # component size (peak) in kW, only valid if sim_cs["wind"]==False

# Photovoltaic array component data
pv_filename = "Zatta_CI_1kWp.csv"  # name of the normalized pv power profile csv file in ./scenarios to evaluate
pv_sce = 0.8  # specific capital expenses of the component in $/W
pv_sme = 0  # specific maintenance expenses of the component in $/(W*year)
pv_soe = 0  # specific operational expenses of the component in $/Wh
pv_ls = 25  # lifespan of the component in years
pv_cdc = 1  # annual ratio of cost decrease
pv_cs = 850e3  # component size (peak) in W, only valid if sim_cs["pv"]==False

# Diesel generator component data
gen_sce = 1.5  # specific capital expenses of the component in $/W (original 1.15)
gen_sme = 0  # specific maintenance expenses of the component in $/(W*year)
gen_soe = 0.00065  # specific operational expenses of the component in $/Wh (original 0.00036)
gen_ls = 10  # lifespan of the component in years
gen_cdc = 1  # annual ratio of component cost decrease
gen_cs = 100e3  # component size in W, only valid if sim_cs["gen"]==False

# Stationary storage system component data
ess_sce = 0.8  # specific capital expenses of the component in $/Wh
ess_sme = 0  # specific maintenance expenses of the component in $/(Wh*year)
ess_soe = 0  # specific operational expenses of the component in $/Wh
ess_ls = 10  # lifespan of the component in years
ess_chg_eff = 0.95  # charging efficiency
ess_dis_eff = 0.85  # discharge efficiency
ess_chg_crate = 0.5  # maximum charging C-rate in 1/h
ess_dis_crate = 0.5  # maximum discharging C-rate in 1/h
ess_init_soc = 0.0  # initial state of charge
ess_sd = 0  # self-discharge rate of the component in ???
ess_cdc = 1  # annual ratio of component cost decrease
ess_cs = 1000e3  # component size in Wh, only valid if sim_cs["ess"]==False

# BEV
bev_filename = "ind_car_data.csv"
bev_agr = False  # boolean triggering simplified simulation of BEVs as a single set of components when true
bev_num = 10  # number of vehicles to be simulated
bev_sce = 0.8  # specific capital expenses of the component in $/Wh
bev_sme = 0  # specific maintenance expenses of the component in $/(Wh*year)
bev_soe = 0  # specific operational expenses of the component in $/Wh
bev_ls = 10  # lifespan of the component in years
bev_chg_pwr = 3600  # maximum allowable charge power for each individual BEV
bev_dis_pwr = 3600  # maximum allowable discharge power for each individual BEV
bev_charge_eff = 0.95  # unitless charge efficiency
bev_discharge_eff = 0.95  # unitless discharge efficiency
bev_cdc = 1  # annual ratio of component cost decrease
bev_cs = 30e3  # battery size of vehicles in Wh, only valid if sim_cs["bev"]==False

