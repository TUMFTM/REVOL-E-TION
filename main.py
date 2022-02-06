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

Created:     September 2nd, 2021
Last update: February 4th, 2022

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021
Marcel Brödel, B.Sc. - Semester Thesis in progress

--- Detailed Description ---
This script is the main model generator and optimizer for the toolset.
Its results are output to files and key ones printed to the terminal.
Visualization is done via different scripts (to be done)
For further information, see readme

--- Input & Output ---
The script requires input data in the code block "Input".
Additionally, several .csv-files for timeseries data are required.
For further information, see readme

--- Requirements ---
see readme

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Imports
###############################################################################

from oemof.tools import logger
import oemof.solph as solph
import oemof.solph.processing as prcs
# from oemof.solph import views

import logging
# import os
# import pandas as pd
# import numpy as np
# from pandas.plotting import register_matplotlib_converters
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# import time
# import sys

import preprocessing as pre
import postprocessing as post
# import economics as eco
import parameters as param
# import load_following as lf  # Preparation for future setup

##########################################################################
# Preprocessing
##########################################################################
sim = pre.define_sim()  # Initialize basic simulation data

logger.define_logging(logfile=sim['logfile'])
logging.info('Processing inputs')

prj = pre.define_prj(sim)  # Initialize project data for later economic extrapolation on project lifespan
sim, dem, wind, pv, gen, ess, bev = pre.define_components(sim, prj)  # Initialize component data
sim = pre.define_os(sim, dem, wind, pv, gen, ess, bev)  # Initialize operational strategy
dem, wind, pv, gen, ess, bev, cres = pre.define_result_structure(dem, wind, pv, gen, ess, bev)

##########################################################################
# Optimization Loop
##########################################################################

for ch in range(sim['opt_counter']):  # Iterate over number of prediction horizons (1 for GO, more for RH)
    logging.info('Prediction Horizon ' + str(oc + 1) + ' of ' + str(sim['opt_counter']))

    sim = pre.set_dti(sim, oc)  # set datetimeindices to fit the current prediction and control horizons
    dem, wind, pv, bev = pre.select_data(sim, dem, wind, pv, bev)  # select correct input data slices for datetimeindices

    sim, om = pre.build_energysystemmodel(sim, dem, wind, pv, gen, ess, bev)

    om.solve(solver=param.sim_solver, solve_kwargs={"tee": param.sim_debug})

    dem, wind, pv, gen, ess, bev = post.get_results(sim, dem, wind, pv, gen, ess, bev, om, oc)

##########################################################################
# Postprocessing
##########################################################################

logging.info("Calculating key results")
wind, pv, gen, ess, bev = post.get_cs(sim, wind, pv, gen, ess, bev, results)  # get (optimized) component sizes
cres = post.acc_energy(sim, prj, dem, wind, pv, gen, ess, bev, cres)  # calculate cumulative energy results
# cres = post.acc_eco(sim, prj, dem, wind, pv, gen, ess, bev, cres)  # calculate cumulative economic results

logging.info("Displaying key results")
# post.display_cres(cres)  # display cumulative results

sim = post.end_timing(sim)

# post.plot_power(dem, wind, pv, gen, ess, bev)
# post.plot_energy(ess, bev)
# post.save_results(sim, dem, wind, pv, gen, ess, bev, cres)



#
#
# print("#####")

#
#     print("Wind Power Results:")
#     if param.sim_cs["wind"]:
#         print("Optimum Capacity: " + str(round(wind_inv / 1e3)) + " kW")
#     else:
#         print("Set Capacity: " + str(param.wind_cs / 1e3) + " kW")
#     print("Initial Capital Expenses: " + str(round(wind_ice)) + " USD")
#     print("Yearly Maintenance Expenses: " + str(round(wind_yme)) + " USD")
#     print("Yearly Operational Expenses: " + str(round(wind_yoe)) + " USD")
#     print("Yearly Produced Energy: " + str(round(wind_ten / 1e6)) + " MWh")
#     print("#####")
#
#     tot['ice'] += wind_ice
#     tot['tce'] += wind_tce
#     tot['pce'] += wind_pce
#     tot['yme'] += wind_yme
#     tot['tme'] += wind_tme
#     tot['pme'] += wind_pme
#     tot['yoe'] += wind_yoe
#     tot['toe'] += wind_toe
#     tot['poe'] += wind_poe
#
#     tot['ann'] += wind_ann
#
# if param.sim_enable["pv"]:
#     if param.sim_cs["pv"]:
#         pv_inv = results[(pv_src, pv_bus)]["scalars"]["invest"]  # [W]
#     else:
#         pv_inv = param.pv_cs
#     pv_ice = pv_inv * param.pv_sce
#     pv_tce = eco.tce(pv_ice, pv_ice, param.pv_ls, param.proj_ls)
#     pv_pce = eco.pce(pv_ice, pv_ice, param.pv_ls, param.proj_ls, param.proj_wacc)
#
#     pv_yme = pv_inv * param.pv_sme
#     pv_tme = pv_yme * param.proj_ls
#     pv_pme = eco.acc_discount(pv_yme, param.proj_ls, param.proj_wacc)
#     pv_yoe = pv_ype * param.pv_soe
#     pv_toe = pv_ten * param.pv_soe
#     pv_poe = eco.acc_discount(pv_yoe, param.proj_ls, param.proj_wacc)
#     pv_ann = eco.ann_recur(pv_ice, param.pv_ls, param.proj_ls, param.proj_wacc, param.pv_cdc) \
#              + eco.ann_recur(pv_yme + pv_yoe, 1, param.proj_ls, param.proj_wacc, 1)
#
#     print("Solar Power Results:")
#     if param.sim_cs["pv"]:
#         print("Optimum Capacity: " + str(round(pv_inv / 1e3)) + " kW (peak)")
#     else:
#         print("Set Capacity: " + str(param.pv_cs/ 1e3) + " kW (peak)")
#     print("Initial Capital Expenses: " + str(round(pv_ice)) + " USD")
#     print("Yearly Maintenance Expenses: " + str(round(pv_yme)) + " USD")
#     print("Yearly Operational Expenses: " + str(round(pv_yoe)) + " USD")
#     print("Total Present Cost: " + str(round(pv_pce + pv_pme + pv_poe)) + " USD")
#     print("Combined Annuity: " + str(round(pv_ann)) + " USD")
#     print("Yearly Produced Energy: " + str(round(pv_ype / 1e6)) + " MWh")
#     print("#####")
#
#     tot['ice'] += pv_ice
#     tot['tce'] += pv_tce
#     tot['pce'] += pv_pce
#     tot['yme'] += pv_yme
#     tot['tme'] += pv_tme
#     tot['pme'] += pv_pme
#     tot['yoe'] += pv_yoe
#     tot['toe'] += pv_toe
#     tot['poe'] += pv_poe
#
#     tot['ann'] += pv_ann
#
# if param.sim_enable["gen"]:
#     if param.sim_cs["gen"]:
#         gen_inv = results[(gen_src, ac_bus)]["scalars"]["invest"]
#     else:
#         gen_inv = param.gen_cs
#     gen_ice = gen_inv * param.gen_sce
#     gen_tce = eco.tce(gen_ice, gen_ice, param.gen_ls, param.proj_ls)
#     gen_pce = eco.pce(gen_ice, gen_ice, param.gen_ls, param.proj_ls, param.proj_wacc)
#     gen_yme = gen_inv * param.gen_sme
#     gen_tme = gen_yme * param.proj_ls
#     gen_pme = eco.acc_discount(gen_yme, param.proj_ls, param.proj_wacc)
#     gen_yoe = gen_ype * param.gen_soe
#     gen_toe = gen_ten * param.gen_soe
#     gen_poe = eco.acc_discount(gen_yoe, param.proj_ls, param.proj_wacc)
#     gen_ann = eco.ann_recur(gen_ice, param.gen_ls, param.proj_ls, param.proj_wacc, param.gen_cdc) \
#                 + eco.ann_recur(gen_yme + gen_yoe, 1, param.proj_ls, param.proj_wacc, 1)
#
#     print("Diesel Power Results:")
#     if param.sim_cs["gen"]:
#         print("Optimum Capacity: " + str(round(gen_inv / 1e3)) + " kW")
#     else:
#         print("Set Capacity: " + str(param.gen_cs/ 1e3) + " kW")
#     print("Initial Capital Expenses: " + str(round(gen_ice)) + " USD")
#     print("Yearly Maintenance Expenses: " + str(round(gen_yme)) + " USD")
#     print("Yearly Operational Expenses: " + str(round(gen_yoe)) + " USD")
#     print("Total Present Cost: " + str(round(gen_pce + gen_pme + gen_poe)) + " USD")
#     print("Combined Annuity: " + str(round(gen_ann)) + " USD")
#     print("Yearly Produced Energy: " + str(round(gen_ype / 1e6)) + " MWh")
#     print("#####")
#
#     tot['ice'] += gen_ice
#     tot['tce'] += gen_tce
#     tot['pce'] += gen_pce
#     tot['yme'] += gen_yme
#     tot['tme'] += gen_tme
#     tot['pme'] += gen_pme
#     tot['yoe'] += gen_yoe
#     tot['toe'] += gen_toe
#     tot['poe'] += gen_poe

#     tot['ann'] += gen_ann
#
# if param.sim_enable["ess"]:
#     if param.sim_cs["ess"]:
#         ess_inv = results[(ess, None)]["scalars"]["invest"]
#     else:
#         ess_inv = param.ess_cs
#     ess_ice = ess_inv * param.ess_sce
#     ess_tce = eco.tce(ess_ice, ess_ice, param.ess_ls, param.proj_ls)
#     ess_pce = eco.pce(ess_ice, ess_ice, param.ess_ls, param.proj_ls, param.proj_wacc)
#     ess_yme = ess_inv * param.ess_sme
#     ess_tme = ess_yme * param.proj_ls
#     ess_pme = eco.acc_discount(ess_yme, param.proj_ls, param.proj_wacc)
#     ess_yoe = ess_ype * param.ess_soe
#     ess_toe = ess_ten * param.ess_soe
#     ess_poe = eco.acc_discount(ess_yoe, param.proj_ls, param.proj_wacc)
#     ess_ann = eco.ann_recur(ess_ice, param.ess_ls, param.ess_ls, param.proj_wacc, param.ess_cdc) \
#               + eco.ann_recur(ess_yme + ess_yoe, 1, param.proj_ls, param.proj_wacc, 1)
#
#     print("Energy Storage Results:")
#     if param.sim_cs["ess"]:
#         print("Optimum Capacity: " + str(round(ess_inv / 1e3)) + " kWh")
#     else:
#         print("Set Capacity: " + str(param.ess_cs / 1e3) + " kWh")
#     print("Initial Capital Expenses: " + str(round(ess_ice)) + " USD")
#     print("Yearly Maintenance Expenses: " + str(round(ess_yme)) + " USD")
#     print("Yearly Operational Expenses: " + str(round(ess_yoe)) + " USD")
#     print("Total Present Cost: " + str(round(ess_pce + ess_pme + ess_poe)) + " USD")
#     print("Combined Annuity: " + str(round(ess_ann)) + " USD")
#     print("Yearly Discharged Energy: " + str(round(ess_ype / 1e6)) + " MWh")
#     print("#####")
#
#     tot['ice'] += ess_ice
#     tot['tce'] += ess_tce
#     tot['pce'] += ess_pce
#     tot['yme'] += ess_yme
#     tot['tme'] += ess_tme
#     tot['pme'] += ess_pme
#     tot['yoe'] += ess_yoe
#     tot['toe'] += ess_toe
#     tot['poe'] += ess_poe
#     tot['ann'] += ess_ann
#
# if param.sim_enable["bev"]:
#     if param.sim_cs["bev"]:
#         bev_inv = results[(bevx_ess, None)]["scalars"]["invest"]  # [Wh]
#     else:
#         bev_inv = param.bev_cs

#
#
#     print("Electric Vehicle Results:")
#     if param.sim_cs["bev"]:
#         print("Optimum battery capacity: " + str(round(bev_inv / 1e3)) + " kWh")
#     else:
#         print("Set battery capacity: " + str(round(bev_inv / 1e3)) + " kWh")
#     print("Gross charged energy: " + str(round(total_bev_chg / 1e6)) + " MWh")
#     print("Energy fed back (V2G): " + str(round(total_bev_dis / 1e6)) + " MWh")
#     print("Net charged energy: " + str(round(total_bev_dem / 1e6)) + " MWh")
#     print("#####")
#
#
# ##########################################################################
# # LCOE and NPC calculation
# ##########################################################################
#
# tot['npc'] = tot['pce'] + tot['pme'] + tot['poe']
# tot['lcoe'] = tot['npc'] / tot['pde']

#
#
# print("Economic Results:")
# print("Yearly supplied energy: " + str(round(tot['yde'] / 1e6,4)) + " MWh")
# print("Yearly generated energy: " + str(round(tot['ype'] / 1e6,4)) + " MWh")
# print("Overall electrical efficiency: " + str(round(tot['eta'] * 100, 4)) + " %")
# print("Total Initial Investment: " + str(round(tot['ice'] / 1e6, 4)) + " million USD")
# print("Total yearly maintenance expenses: " + str(round(tot['yme'],4)) + " USD")
# print("Total yearly operational expenses: " + str(round(tot['yoe'],4)) + " USD")
# print("Total cost: " + str(round((tot['tce'] + tot['tme'] + tot['toe']) / 1e6, 4)) + " million USD")
# print("Total present cost: " + str(round(tot['npc'] / 1e6, 4)) + " million USD")
# print("Total annuity: " + str(round(tot['ann'] / 1e3, 4)) + " thousand USD")
# print("LCOE: " + str(1e5 * tot['lcoe']) + " USct/kWh")
# print("#####")
#
#
#
# ##########################################################################
# # Save the results
# ##########################################################################
#
# # Add the results to the energy system object and dump it as an .oemof file
# if param.sim_dump:
#     logging.info("Save model and result data")
#     es.results["main"] = prcs.results(om)
#     es.results["meta"] = prcs.meta_results(om)
#     es.dump(sim_resultpath, sim_tsname + ".oemof")
#
#     # Create pandas dataframes from the results and dump it as a .csv file
#     es_results = prcs.create_dataframe(om)
#     es_results.to_csv(os.path.join(sim_resultpath, sim_tsname + "_res_df.csv"), sep=';')
#     parameters = prcs.parameter_as_dict(es)
#     parameters = pd.DataFrame.from_dict(parameters)
#     parameters.to_csv(os.path.join(sim_resultpath, sim_tsname + "_res_dict.csv"), sep=';')
#

#
# ##########################################################################
# # Plot the results
# ##########################################################################
#
# p300 = 'rgb(0,101,189)'
# p540 = 'rgb(0,51,89)'
# orng = 'rgb(227,114,34)'
# grn = 'rgb(162,173,0)'
# # TODO: Überschrift je nach Strategy
# fig = go.Figure()
# if param.sim_enable["gen"]:
#     fig.add_trace(
#     go.Scatter(x=gen_flow.index.to_pydatetime(), y=gen_flow, mode='lines', name='Diesel generator', line_color=p300,
#                line_width=4))
# if param.sim_enable["pv"]:
#     fig.add_trace(
#     go.Scatter(x=pv_flow.index.to_pydatetime(), y=pv_flow, mode='lines', name='Photovoltaics', line_color=orng,
#                line_width=2))
# if param.sim_enable["ess"]:
#     fig.add_trace(go.Scatter(x=ess_flow.index.to_pydatetime(), y=ess_flow, mode='lines', name='Battery storage',
#                          line_color=orng, line_width=2, line_dash='dash'))
# if param.sim_enable["bev"]:
#     fig.add_trace(go.Scatter(x=bev_flow.index.to_pydatetime(), y=bev_flow, mode='lines', name='BEV demand', line_color=grn,
#                          line_width=2))
# fig.add_trace(
#     go.Scatter(x=dem_flow.index.to_pydatetime(), y=-dem_flow, mode='lines', name='Stationary demand', line_color=grn,
#                line_width=2, line_dash='dash'))
# fig.update_layout(xaxis=dict(title='Local Time', showgrid=True, linecolor='rgb(204, 204, 204)',
#                              gridcolor='rgb(204, 204, 204)',),
#                              #range=[datetime.strptime(param.proj_start, '%d/%m/%Y'),(datetime.strptime(param.proj_start, '%d/%m/%Y')+relativedelta(days=3))]),
#                   yaxis=dict(title='Power in W', showgrid=True, linecolor='rgb(204, 204, 204)',
#                              gridcolor='rgb(204, 204, 204)',
#                              range=[-620000,620000]),
#                   plot_bgcolor='white')
# if param.sim_os == 'go':
#     fig.update_layout(title='Power Flow Results - Global Optimum')
# if param.sim_os == 'rh':
#     fig.update_layout(title='Power Flow Results - Rolling Horizon Strategy (PH: '+str(param.os_ph)+'h, CH: '+str(param.os_ch)+'h)')
# fig.show()

##########################################################################
# Monitor ESS SOC
##########################################################################
#print(sc_ess)
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=sc_ess.index.to_pydatetime(), y=sc_ess/ param.ess_cs * 100, mode='lines', name='SOC ESS', line_color=p300,
#             line_width=4))
# fig.update_layout(title='Global Optimum - State of Charge ESS', plot_bgcolor='white',
#                 xaxis=dict(title='Local Time', showgrid=True, linecolor='rgb(204, 204, 204)',
#                         gridcolor='rgb(204, 204, 204)',
#                         range=[datetime.strptime(param.proj_start, '%m/%d/%Y'),(datetime.strptime(param.proj_start, '%m/%d/%Y')+relativedelta(days=3))]),
#                 yaxis=dict(title='SOC in %', showgrid=True, linecolor='rgb(204, 204, 204)',
#                         gridcolor='rgb(204, 204, 204)',
#                         ))
#
# fig.show()

#
# ##########################################################################
# # Monitor BEVs SOC
# ##########################################################################
# # sc_bevx = pd.DataFrame()
# # for i in range(param.bev_num):
# #     column_name = (("bev"+str(i+1)+"_ess", 'None'), 'storage_content')
# #     sc_bevx["bev_"+str(i+1)] = views.node(results, "bev"+str(i+1)+"_ess")['sequences'][column_name]
# #
# #
# # row = [1, 1, 1]
# # col = [1, 2, 3]
# # titles = ['BEV 1','BEV 2','BEV 3','BEV 4']
# # fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
# # for i in range(len(row)):
# #     fig.add_trace(go.Scatter(x=sc_bevx["bev_"+str(i+1)].index, y=sc_bevx["bev_"+str(i+1)] / param.bev_cs * 100,
# #                              line=dict(color=p300), showlegend=False), row[i], col[i])
# # fig.update_layout(plot_bgcolor='white', title_text="Global Optimum - State of Charge BEVs")
# # fig.update_yaxes(title='SOC in %', showgrid=True, linecolor='rgb(204, 204, 204)', gridcolor='rgb(204, 204, 204)',
# #                  range=[0,100])
# # fig.update_xaxes(title='Local Time', showgrid=True, linecolor='rgb(204, 204, 204)', gridcolor='rgb(204, 204, 204)',
# #                  range=[datetime.strptime(param.proj_start, '%d/%m/%Y'),(datetime.strptime(param.proj_start, '%d/%m/%Y')+relativedelta(days=3))])
# # fig.show()
#
