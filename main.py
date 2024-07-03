#!/usr/bin/env python3
"""
main.py

--- Description ---
This script is the main executable for the REVOL-E-TION toolset.
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
from logging.handlers import QueueHandler
import threading
import warnings
import multiprocessing as mp
import platform

from itertools import repeat
from simulation import PredictionHorizon, Scenario, SimulationRun

# raise UserWarnings about infeasibility as errors to catch them properly
warnings.simplefilter(action='error', category=UserWarning)

# only print FutureWarnings once (in theory)
# Set to 'ignore' to suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

###############################################################################
# Function definitions
###############################################################################


def setup_logger(name, log_queue, debugmode):
    logger = logging.getLogger(name)

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set up QueueHandler and configure logger
    queue_handler = QueueHandler(log_queue)
    logger.setLevel(logging.DEBUG if debugmode else logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)

    return logger


def read_mplogger_queue(queue):
    main_logger = logging.getLogger('main')

    for handler in main_logger.handlers[:]:
        main_logger.removeHandler(handler)

    while True:
        record = queue.get()
        if record is None:
            break
        elif platform.system() in ['Windows', 'Darwin']:  # Darwin is macOS
            main_logger.handle(record)  # This line causes double logger outputs on Linux



def simulate_scenario(name: str, run: SimulationRun, log_queue):  # needs to be a function for starpool

    logger = setup_logger(name, log_queue, run.debugmode)

    run.logger = logger
    run.process = mp.current_process()

    scenario = Scenario(name, run, logger)  # Create scenario instance

    for horizon_index in range(scenario.nhorizons):  # Inner optimization loop over all prediction horizons
        horizon = PredictionHorizon(horizon_index, scenario, run)

        if scenario.strategy in ['go', 'rh']:
            horizon.run_optimization(scenario, run)
        else:
            logging.error(f'Scenario {scenario.name}: energy management strategy unknown')
            break

        if scenario.exception and run.save_results:
            scenario.save_result_summary(run)
            break
        else:
            horizon.get_results(scenario, run)

        # free up memory before garbage collector can act - mostly useful in rolling horizon strategy
        del horizon

    scenario.end_timing(run)

    if not scenario.exception:
        if run.save_results or run.print_results:
            scenario.get_results(run)
            scenario.calc_meta_results(run)
            if run.save_results:
                scenario.save_result_summary(run)
                scenario.save_result_timeseries()
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

    # parallelization activated in settings file
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

    # sequential scenario processing selected
    else:
        for scenario_name in run.scenario_names:
            simulate_scenario(scenario_name, run, None)  # no logger queue

    if run.save_results:
        run.join_results()

    run.end_timing()
