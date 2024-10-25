#!/usr/bin/env python3

import itertools
import os
import sys
import threading
import tkinter as tk
import tkinter.filedialog
import warnings
import multiprocessing as mp

from revoletion import simulation as sim
from revoletion import logger as logger_fcs


# raise UserWarnings about infeasibility as errors to catch them properly
warnings.simplefilter(action='error', category=UserWarning)

# only print FutureWarnings once (in theory)
# Set to 'ignore' to suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def simulate_scenario(name: str, run: sim.SimulationRun, log_queue):  # needs to be a function for starpool
    logger = logger_fcs.setup_logger(name, log_queue, run)

    run.process = mp.current_process() if run.parallel else None

    scenario = sim.Scenario(name, run, logger)  # Create scenario instance

    for horizon_index in range(scenario.nhorizons):  # Inner optimization loop over all prediction horizons
        horizon = sim.PredictionHorizon(horizon_index, scenario, run)

        if scenario.strategy in ['go', 'rh']:
            horizon.run_optimization(scenario, run)
        else:
            logger.error(f'Scenario {scenario.name}: energy management strategy unknown')
            break

        if scenario.exception and run.save_results:
            scenario.save_result_summary()
            break
        else:
            horizon.get_results(scenario, run)

        # free up memory before garbage collector can act - mostly useful in rolling horizon strategy
        del horizon

    scenario.end_timing()

    if not scenario.exception:
        if run.save_results or run.print_results:
            scenario.get_results()
            scenario.calc_meta_results()
            if run.save_results:
                scenario.save_result_summary()
                scenario.save_result_timeseries()
            if run.print_results:
                scenario.print_results()

        if run.save_plots or run.show_plots:
            scenario.generate_plots()
            if run.save_plots:
                scenario.save_plots()
            if run.show_plots:
                scenario.show_plots()

    # make sure to clear up memory space
    del scenario

def read_arguments(cwd):

    if os.path.isfile(sys.argv[1]):
        scenarios_file_path = sys.argv[1]
    elif os.path.isfile(os.path.join(cwd, 'input', 'scenarios', sys.argv[1])):
        scenarios_file_path = os.path.join(cwd, 'input', 'scenarios', sys.argv[1])
    else:
        raise FileNotFoundError(f'Scenario file or path not found: {sys.argv[1]}')

    if os.path.isfile(sys.argv[2]):
        settings_file_path = sys.argv[2]
    elif os.path.isfile(os.path.join(cwd, 'input', 'settings', sys.argv[2])):
        settings_file_path = os.path.join(cwd, 'input', 'settings', sys.argv[2])
    else:
        raise FileNotFoundError(f'Settings file or pathnot found: {sys.argv[2]} not found')

    return scenarios_file_path, settings_file_path


def select_arguments(cwd):

    root = tk.Tk()
    root.withdraw()  # hide small tk-window
    root.lift()  # make sure all tk windows appear in front of other windows

    # get scenarios file
    scenarios_default_dir = os.path.join(cwd, 'input', 'scenarios')
    scenarios_file_path = tk.filedialog.askopenfilename(initialdir=scenarios_default_dir,
                                                        title="Select scenario file",
                                                        filetypes=(("CSV files", "*.csv"),
                                                                   ("All files", "*.*")))
    if not scenarios_file_path:
        raise FileNotFoundError('No scenario file selected')

    # get settings file
    settings_default_dir = os.path.join(cwd, 'input', 'settings')
    settings_file_path = tk.filedialog.askopenfilename(initialdir=settings_default_dir,
                                                       title="Select settings file",
                                                       filetypes=(("CSV files", "*.csv"),
                                                                  ("All files", "*.*")))
    if not settings_file_path:
        raise FileNotFoundError('No settings file selected')

    return scenarios_file_path, settings_file_path


def main():
    cwd = os.getcwd()
    if len(sys.argv) == 1:  # no arguments passed
        path_scenario, path_settings = select_arguments(cwd)
    elif len(sys.argv) == 3:  # two arguments passed
        path_scenario, path_settings = read_arguments(cwd)
    else:
        raise ValueError('Invalid number of arguments - please provide either none (GUI input) '
                         'or two arguments: scenarios file name or path and settings file name or path')

    run = sim.SimulationRun(path_scenario=path_scenario,
                            path_settings=path_settings,
                            execute=False)

    # parallelization activated in settings file
    if run.parallel:
        with mp.Manager() as manager:
            log_queue = manager.Queue()

            log_thread = threading.Thread(target=logger_fcs.read_mplogger_queue, args=(log_queue,))
            log_thread.start()
            with mp.Pool(processes=run.process_num) as pool:
                pool.starmap(
                    simulate_scenario,
                    zip(run.scenario_names, itertools.repeat(run), itertools.repeat(log_queue)))
            log_queue.put(None)
            log_thread.join()
    else:
        for scenario_name in run.scenario_names:
            simulate_scenario(scenario_name, run, None)

    if run.save_results:
        run.join_results()

    run.end_timing()
    # TODO improve error handling - scenarios that fail wait to the end and are memory hogs


if __name__ == '__main__':
    main()
