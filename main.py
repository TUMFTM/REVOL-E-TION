#!/usr/bin/env python3
"""
main.py

--- Description ---
This script is the main executable for the oemof mg_ev toolset.
For further information, see readme.md

--- Created by ---
Philipp Rosner

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Module imports
###############################################################################

import logging
import logging.handlers
import threading
import warnings
import multiprocessing as mp
import platform

from itertools import repeat
from simulation import PredictionHorizon, Scenario, SimulationRun

warnings.filterwarnings("error")  # needed for catching UserWarning during infeasibility of scenario

###############################################################################
# Function definitions
###############################################################################


def read_mplogger_queue(queue):
    while True:
        record = queue.get()
        if platform.system() == 'Windows':
            run.logger.handle(record)  # This line causes double logger outputs on Linux
        if record is None:
            break

def simulate_scenario(name: str, run: SimulationRun, log_queue):  # needs to be a function for starpool

    if run.parallel:
        run.process = mp.current_process()
        run.queue_handler = logging.handlers.QueueHandler(log_queue)
        run.logger = logging.getLogger()
        if run.debugmode:
            run.logger.setLevel(logging.DEBUG)
        else:
            run.logger.setLevel(logging.INFO)
        run.logger.addHandler(run.queue_handler)

    scenario = Scenario(name, run)  # Create scenario instance

    for horizon_index in range(scenario.nhorizons):  # Inner optimization loop over all prediction horizons
        try:
            horizon = PredictionHorizon(horizon_index, scenario, run)
        except IndexError:
            scenario.exception = 'Input data does not cover full sim timespan'
            logging.warning(f'Input data in scenario \"{scenario.name}\" does not cover full simulation timespan'
                            f' - continuing on next scenario')
            scenario.save_results(run)
            break

        if scenario.strategy in ['go', 'rh']:
            horizon.run_optimization(scenario, run)
        else:
            logging.error(f'Scenario {scenario.name}: energy management strategy unknown')
            break  # todo better error handling

        if scenario.exception and run.save_results:
            scenario.save_results(run)
            break
        else:
            horizon.get_results(scenario, run)

        # free up memory before garbage collector can act - mostly useful in rolling horizon strategy
        del horizon

    scenario.end_timing(run)

    if not scenario.exception:
        if run.save_results or run.print_results:
            scenario.accumulate_results(run)
            if run.save_results:
                scenario.save_results(run)
            if run.print_results:
                scenario.print_results(run)

        if run.save_plots or run.show_plots:
            scenario.generate_plots(run)
            if run.save_plots:
                scenario.save_plots()
            if run.show_plots:
                scenario.show_plots()

    # make sure to clear up memory space
    del scenario

###############################################################################
# Execution code
###############################################################################


if __name__ == '__main__':

    run = SimulationRun()  # get all global information about the run

    if run.parallel:
        with mp.Manager() as manager:
            log_queue = manager.Queue()
            log_thread = threading.Thread(target=read_mplogger_queue, args=(log_queue,))
            log_thread.start()
            with mp.Pool(processes=run.process_num) as pool:
                pool.starmap(simulate_scenario, zip(run.scenario_names, repeat(run), repeat(log_queue)))
            log_queue.put(None)
            log_thread.join()
            # TODO improve error handling - scenarios that fail wait to the end and are memory hogs
    else:
        for scenario_name in run.scenario_names:
            simulate_scenario(scenario_name, run, None)  # no logger queue

    if run.save_results:
        run.join_results()

    run.end_timing()
