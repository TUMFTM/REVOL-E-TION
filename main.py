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
Created September 2nd, 2021

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022

--- Detailed Description ---
This script is the main model generator and optimizer for the toolset.
Its results are output to files and key ones printed to the terminal.
Visualization is done via different scripts (to be done)
For further information, see readme

--- Input & Output ---
The model requires an Excel input file for scenario definition.
Additionally, several .csv-files for timeseries data are required.
For further information, see readme.

--- Requirements ---
For package requirements, see requirements.txt
For all other requirements, see readme.

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

import logging
# import multiprocessing as multi  # TODO parallelize scenario loop

import preprocessing as pre
import postprocessing as post


def simulate_scenario():
    '''

    '''

    try:
        scenario = pre.Scenario(run, scenario_name)  # Create scenario instance & read data from excel sheet.

        for horizon_index in range(scenario.horizon_num):  # Inner optimization loop over all prediction horizons
            horizon = pre.PredictionHorizon(horizon_index, scenario, run)
            horizon.model.solve(solver=run.solver, solve_kwargs={"tee": run.solver_debugmode})
            horizon.get_results(scenario, run)

        #dem, wind, pv, gen, ess, bev, cres = post.acc_energy(sim, prj, dem, wind, pv, gen, ess, bev, cres)  # TODO OO
        #wind, pv, gen, ess, bev, cres = post.acc_eco(sim, prj, wind, pv, gen, ess, bev, cres)  # TODO OO

        #post.print_results(sim, wind, pv, gen, ess, bev, cres)  # TODO OO

        #sim = post.end_timing(sim)  # TODO OO

        #post.plot_results(sim, dem, wind, pv, gen, ess, bev)  # TODO OO
        #post.save_results(sim, dem, wind, pv, gen, ess, bev, cres)  # TODO OO

    except KeyError:  # TODO proper error handling
        logging.warning(f"Scenario {scenario_name} failed or infeasible - continue on next scenario")
        #post.save_results_err(sim)
        #continue


if __name__ == '__main__':

    run = pre.SimulationRun()  # get all global information about the run

    for scenario_index, scenario_name in enumerate(run.scenario_names):
        simulate_scenario()  # TODO integrate multiprocessing
