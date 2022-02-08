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
Last update: February 8th, 2022

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

import logging

import preprocessing as pre
import postprocessing as post
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
sim = pre.define_os(sim)  # Initialize operational strategy
dem, wind, pv, gen, ess, bev, cres = pre.define_result_structure(prj, dem, wind, pv, gen, ess, bev)

##########################################################################
# Optimization Loop
##########################################################################

for ph in range(sim['ch_num']):  # Iterate over number of prediction horizons (1 for GO, more for RH)
    logging.info('Prediction Horizon ' + str(ph + 1) + ' of ' + str(sim['ch_num']))

    sim = pre.set_dti(sim, ph)  # set datetimeindices to fit the current prediction and control horizons
    dem, wind, pv, bev = pre.select_data(sim, dem, wind, pv, bev)  # select correct input data slices for selected dti

    sim, om = pre.build_energysystemmodel(sim, dem, wind, pv, gen, ess, bev)

    logging.info('Solving optimization problem')
    om.solve(solver=param.sim_solver, solve_kwargs={"tee": sim['debugmode']})

    dem, wind, pv, gen, ess, bev = post.get_results(sim, dem, wind, pv, gen, ess, bev, om, ph)

##########################################################################
# Postprocessing
##########################################################################

logging.info("Calculating key results")
dem, wind, pv, gen, ess, bev, cres = post.acc_energy(sim, prj, dem, wind, pv, gen, ess, bev, cres)
wind, pv, gen, ess, bev, cres = post.acc_eco(sim, prj, wind, pv, gen, ess, bev, cres)

logging.info("Displaying key results")
post.print_results(sim, wind, pv, gen, ess, bev, cres)

sim = post.end_timing(sim)

post.plot_results(sim, dem, wind, pv, gen, ess, bev)
# post.save_results(sim, dem, wind, pv, gen, ess, bev, cres)
