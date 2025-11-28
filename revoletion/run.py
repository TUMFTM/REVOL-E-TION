#!/usr/bin/env python3

import itertools
import logging
import multiprocessing as mp
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path

import pandas as pd
from oemof import solph as solph

from . import logger as logger_fcs
from . import simulation, utils

_LOGGER = logging.getLogger(__name__)


class SimulationRun:
    def __init__(
        self,
        paths: simulation.SimulationPaths,
        settings: simulation.SimulationSettings | None = None,
    ):
        self.paths = paths
        self.settings = settings or simulation.SimulationSettings()

        self.runtime = utils.RunTime()
        self.runtime.start()

        self.name = self.paths.scenario.stem  # set name of scenario file as run name

        # region get version information
        self.version_solph = solph.__version__
        self.version_revoletion = utils.get_revoletion_python_package_version()

        self.commit_hash = utils.get_current_project_git_commit_hash()
        # endregion

        # region read, copy and check scenario data
        self.scenario_data = utils.read_scenario_from_file(self.paths.scenario)
        self.scenario_names = [name for name in self.scenario_data.columns if not name.startswith("#")]

        # region define logger structure
        self.logger = _LOGGER
        # make sure that uncaught errors (i.e. errors occurring outside simulate_scenario method) are logged to logfile
        sys.excepthook = self.handle_exception
        # endregion

        if self.paths.rerun:
            # only run scenarios which have not been optimized successfully (or were infeasible)
            self.scenario_status = pd.read_csv(
                self.paths.status,
                index_col=0,
                dtype=str,  # empty columns are interpreted as float -> avoid
            )

            dont_rerun = ["successful", "infeasible"] if self.settings.rerun_infeasible else ["successful"]
            scenarios_rerun = self.scenario_status[~self.scenario_status["status"].isin(dont_rerun)].index.to_list()
            self.scenario_names = [name for name in self.scenario_names if name in scenarios_rerun]

            if not self.scenario_names:
                raise ValueError(
                    f'Parameter "--rerun" was set to {self.paths.rerun}, but the status file contains no scenarios '
                    f"to rerun.\n"
                    f"All scenarios were {'either infeasible or ' if self.settings.rerun_infeasible else ''}"
                    f"already completed successfully.\n"
                    f"Check {(Path(self.paths.status.parent.name) / self.paths.status.name)} "
                    f"for additional information."
                )

            # delete all temporary results of files which are rerun (happens if SimulationRun terminates unexpected)
            for scenario in self.scenario_names:
                for file_name in [f"{scenario}_summary_temp.csv", f"{scenario}_results.csv"]:
                    file_path = self.paths.output / file_name
                    if file_path.is_file():
                        file_path.unlink()

            # reset status of scenarios to be run to 'queued'
            self.scenario_status.loc[self.scenario_names, ["status", "exception", "traceback"]] = [
                ["queued", pd.NA, pd.NA]
            ] * len(self.scenario_names)
            self.scenario_status.to_csv(self.paths.status, index=True)

        else:
            self.scenario_status = pd.DataFrame(
                index=self.scenario_names, data={"status": "queued", "exception": None, "traceback": None}
            ).rename_axis("scenario")
            self.copy_scenario_file()
        self.scenario_num = len(self.scenario_names)

        if self.scenario_num == 0:
            raise ValueError("No executable scenarios found in scenario file")

        self.settings.n_processes = min(self.settings.n_processes, os.cpu_count(), self.scenario_num)
        # endregion

        self.logger.info(f"{'Reading scenarios from:':<25} {self.paths.scenario}")
        self.logger.info(f"{'Reading input data from:':<25} {self.paths.input}")
        self.logger.info(f"{'Writing results to:':<25} {self.paths.output}")
        self.logger.info(
            f"Running {self.scenario_num} scenario{('s' if self.scenario_num > 1 else '')} "
            f"with {self.settings.n_processes} process{('es' if self.settings.n_processes > 1 else '')}"
        )

    def copy_scenario_file(self):
        target = self.paths.output / f"{self.name}.csv"
        try:  # with metadata
            shutil.copy2(self.paths.scenario, target)
        except PermissionError:  # can happen if metadata is not writable, e.g. on network drives
            shutil.copyfile(self.paths.scenario, target)

    def execute(self, plot: bool = True):
        if self.settings.n_processes > 1:
            with mp.Manager() as manager:
                lock = manager.Lock()

                status_queue = manager.Queue()
                status_thread = threading.Thread(target=self.read_status_queue, args=(status_queue,))
                status_thread.start()

                log_queue = manager.Queue()
                log_thread = threading.Thread(target=logger_fcs.read_mplogger_queue, args=(log_queue,))
                log_thread.start()

                with mp.Pool(
                    processes=self.settings.n_processes,
                    initializer=_worker_init,
                    initargs=(log_queue, self.settings.debugmode),
                ) as pool:
                    pool.starmap(
                        self.execute_scenario,
                        zip(
                            self.scenario_names,
                            itertools.repeat(status_queue),
                            itertools.repeat(lock),
                            itertools.repeat(plot),
                        ),
                    )
                status_queue.put(None)
                status_thread.join()
                log_queue.put(None)
                log_thread.join()
        else:
            for scenario_name in self.scenario_names:
                self.execute_scenario(name=scenario_name, plot=plot)

        self.runtime.stop()
        self.logger.info(f"Total runtime for all scenarios: {self.runtime}")

        self.join_results()

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error(f"Exception type: {exc_type.__name__}")
        self.logger.error(f"Exception message: {str(exc_value)}")
        self.logger.error("Traceback:")
        self.logger.error("".join(traceback.format_tb(exc_traceback)))

        self.logger.error(msg="Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def join_results(self):
        filenames = [
            file
            for file in self.paths.output.iterdir()
            if file.name.endswith("_summary_temp.pkl")
            and self.scenario_status.loc[file.name.removesuffix("_summary_temp.pkl"), "status"] == "successful"
        ]

        scenario_frames = [pd.read_pickle(file) for file in filenames]

        if scenario_frames:  # empty scenario_frames, if all scenarios fail during initialization
            joined_results = pd.concat(scenario_frames, axis=1)

            if self.paths.rerun and self.paths.summary_pkl.is_file():  # only happens for infeasible scenarios
                results_summary_prev = pd.read_pickle(self.paths.summary_pkl)
                joined_results = pd.concat([results_summary_prev, joined_results], axis=1)

            # apply same order of scenarios as in scenario input file
            joined_results = joined_results[[c for c in self.scenario_data.columns if c in joined_results.columns]]

            # get results of run
            results_run = pd.concat(
                [
                    pd.Series(
                        {
                            key: value
                            for key, value in self.__dict__.items()
                            if isinstance(value, (int, float, bool, str))
                        }
                    ),
                    self.runtime.result_summary,
                ]
            )
            # apply MultiIndex
            results_run.index = pd.MultiIndex.from_tuples(
                tuples=[("run", key) for key in results_run.index], names=["block", "key"]
            )

            # convert to DataFrame and repeat for all scenarios
            results_run = pd.DataFrame([results_run] * len(joined_results.columns)).T
            results_run.columns = joined_results.columns

            joined_results = pd.concat(
                [
                    results_run,
                    joined_results,
                ]
            )

            joined_results.to_csv(self.paths.summary_csv, index=True)
            joined_results.to_pickle(self.paths.summary_pkl)
            self.logger.info("Result summary file created")

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in filenames:
            file.unlink()

    def read_status_queue(self, queue):
        while True:
            status_msg = queue.get()
            if status_msg is None:  # Exit signal
                break
            self.update_scenario_status(status_msg)

    def execute_scenario(self, name: str, status_queue: mp.Queue = None, lock: mp.Lock = None, plot: bool = True):
        # this method is necessary as running Scenario() directly from the starmap fails as Scenario object contains
        # objects which cannot be pickled.

        max_scenario_name_len = max([len(scenario_name) for scenario_name in self.scenario_names])
        scenario_logger = logger_fcs.ContextLoggerAdapter(
            self.logger, {"context_str": f"{name:<{max_scenario_name_len}}"}
        )
        try:
            scenario = simulation.Scenario(
                paths=self.paths,
                settings=self.settings,
                name=name,
                parameters=self.scenario_data[name],
                logger=scenario_logger,
                lock=lock,
                status_update=self.trigger_scenario_status_update,
                status_queue=status_queue,
            )
            scenario.execute()
            if plot:
                scenario.generate_and_save_plot()
            scenario.save_result_summary()
        except Exception as e:
            self.trigger_scenario_status_update(
                queue=status_queue,
                status_msg={
                    "scenario": name,
                    "status": "failed",
                    "exception": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

            self.logger.error(
                msg=f"{str(e)} - continue on next scenario",  # todo is not written to log or stream
                exc_info=True,
            )

    def trigger_scenario_status_update(self, queue, status_msg):
        if queue is not None:
            queue.put(status_msg)
        else:
            self.update_scenario_status(status_msg)

    def update_scenario_status(self, status_msg):
        for col in [key for key, value in status_msg.items() if key != "scenario" and value is not None]:
            self.scenario_status.loc[status_msg["scenario"], col] = status_msg[col]
        self.scenario_status.to_csv(self.paths.status, index=True)


def _worker_init(log_queue: mp.Queue, debugmode: bool) -> None:
    """
    Initialize a worker process and configure logging.

    :param log_queue: The queue to which log messages will be sent.
    :param debugmode: Configure the log level in the worker process according to the level of the parent.
    """
    logger_fcs.configure_process_logger_parallel(log_queue, debugmode)
