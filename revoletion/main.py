#!/usr/bin/env python3

import itertools
import logging
import threading
import warnings
import multiprocessing as mp

from revoletion import simulation as sim


# raise UserWarnings about infeasibility as errors to catch them properly
warnings.simplefilter(action='error', category=UserWarning)

# only print FutureWarnings once (in theory)
# Set to 'ignore' to suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def setup_logger(name, log_queue, run):
    logger = logging.getLogger(name)
    if log_queue:
        logger.setLevel(logging.DEBUG if run.debugmode else logging.INFO)
        formatter = logging.Formatter('%(message)s')

        queue_handler = logging.handlers.QueueHandler(log_queue)
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)

    else:
        logger.setLevel(run.logger.level)
        for handler in run.logger.handlers:
            logger.addHandler(handler)

    return logger


def read_mplogger_queue(queue):
    main_logger = logging.getLogger('main')

    while True:
        record = queue.get()
        if record is None:
            break
        main_logger.handle(record)


def simulate_scenario(name: str, run: sim.SimulationRun, log_queue):  # needs to be a function for starpool
    logger = setup_logger(name, log_queue, run)

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


def main():

    run = sim.SimulationRun()  # get all global information about the run

    # parallelization activated in settings file
    if run.parallel:
        with mp.Manager() as manager:
            log_queue = manager.Queue()

            log_thread = threading.Thread(target=read_mplogger_queue, args=(log_queue,))
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