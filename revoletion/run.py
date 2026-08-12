#!/usr/bin/env python3

import enum
import itertools
import logging
import multiprocessing as mp
import multiprocessing.queues as mpq
import multiprocessing.synchronize as mps
import os
import shutil
import sys
import threading
import time as pytime
import traceback
import typing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from oemof import solph as solph
from tqdm import tqdm

from . import location, simulation, time, utils
from . import logger as logger_fcs
from . import scenario as scn

_LOGGER = logging.getLogger(__name__)


class _ScenarioStatus(enum.Enum):
    QUEUED = "queued"
    STARTED = "started"
    INITIALIZED = "initialized"
    COMPLETED_HORIZON = "completed_horizon"
    FAILED = "failed"
    INFEASIBLE = "infeasible"
    SUCCESSFUL = "successful"


# Statuses that mark a scenario as done (in one way or another). Reaching any of these
# advances the scenario progress bar shown in large-scale mode.
_TERMINAL_STATUSES = frozenset({_ScenarioStatus.SUCCESSFUL, _ScenarioStatus.FAILED, _ScenarioStatus.INFEASIBLE})


@dataclass
class _ScenarioStatusMessage:
    """
    Container for information sent by worker processes to the parent `SimulationRun`.
    """

    scenario_name: str
    """The name of the scenario that is currently being processed by the worker."""

    status: _ScenarioStatus
    """Indicate the current status of the worker process."""

    extras: dict[str, typing.Any] | None = None
    """The worker process can include additional information, like an error message or a traceback."""

    worker_name: str | None = None
    """Name of the process handling the scenario (used to route per-process horizon progress bars)."""

    n_horizons_total: int | None = None
    """Total number of prediction horizons of the scenario, sent once the scenario is initialized."""


class _StatusUpdateCallback(typing.Protocol):
    """Type signature for the callback that is passed to each worker process so it can communicate its status to the parent `SimulationRun`."""

    def __call__(self, status_msg: _ScenarioStatusMessage, queue: mpq.Queue | None = None) -> None: ...


class SimulationRun:
    def __init__(
        self,
        paths: scn.SimulationPaths,
        settings: simulation.SimulationSettings | None = None,
    ):
        self.paths = paths
        self.settings = settings or simulation.SimulationSettings()

        self.run_timer = time.RunTimer()

        # Progress bars, only used in large-scale mode (created in `execute`).
        # `progress_bar` tracks finished scenarios; `horizon_bars` shows completed horizons per
        # worker slot (one per process; a single bar in single-process mode).
        self.progress_bar: tqdm | None = None
        self.horizon_bars: list[tqdm] = []
        self._worker_bar_map: dict[str, tqdm] = {}

        self.name = self.paths.scenario.stem  # set name of scenario file as run name

        self.version_solph = solph.__version__
        self.version_revoletion = utils.get_revoletion_python_package_version()

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

        logger_fcs.LogFormatter.format_scenarioname.max_width = max(len(name) for name in self.scenario_names)

        self.scenario_num = len(self.scenario_names)

        if self.scenario_num == 0:
            raise ValueError("No executable scenarios found in scenario file")

        self.settings.n_processes = min(
            self.scenario_num,
            self.settings.n_processes,
            *(x for x in [os.cpu_count()] if x is not None),  # filter out undetermined CPU count
        )
        # endregion

        self.logger.info(f"{'Reading scenarios from:':<25} {self.paths.scenario}")
        self.logger.info(f"{'Reading input data from:':<25} {self.paths.input}")
        self.logger.info(f"{'Writing results to:':<25} {self.paths.output}")
        self.logger.info(
            f"Running {self.scenario_num} scenario{('s' if self.scenario_num > 1 else '')} "
            f"with {self.settings.n_processes} process{('es' if self.settings.n_processes > 1 else '')}"
        )

    def __getstate__(self):
        # `execute_scenario` is dispatched to worker processes via `pool.starmap`, which pickles `self`.
        # The tqdm progress bars hold unpicklable file handles and only live in the parent process,
        # so they are dropped from the state sent to workers.
        state = self.__dict__.copy()
        state["progress_bar"] = None
        state["horizon_bars"] = []
        state["_worker_bar_map"] = {}
        return state

    def copy_scenario_file(self):
        target = self.paths.output / f"{self.name}{self.paths.scenario.suffix}"
        try:  # with metadata
            shutil.copy2(self.paths.scenario, target)
        except PermissionError:  # can happen if metadata is not writable, e.g. on network drives
            shutil.copyfile(self.paths.scenario, target)

    def execute(self):
        if self.settings.largescalemode:
            self.progress_bar = tqdm(
                total=self.scenario_num,
                desc="Scenarios finished",
                unit="scenario",
                position=0,
                dynamic_ncols=True,  # re-layout on terminal resize
                smoothing=0,  # rate/ETA from the average over all finished scenarios instead of the last one
                file=sys.stdout,
            )
            # one horizon bar per worker slot below the overall bar (a single bar in single-process mode)
            self.horizon_bars = [
                tqdm(
                    total=None,
                    desc="idle",
                    unit="horizon",
                    position=slot + 1,
                    leave=False,
                    dynamic_ncols=True,  # re-layout on terminal resize
                    file=sys.stdout,
                )
                for slot in range(self.settings.n_processes)
            ]

        try:
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
                                itertools.repeat(self.settings.largescalemode),
                            ),
                        )
                    status_queue.put(None)
                    status_thread.join()
                    log_queue.put(None)
                    log_thread.join()
            else:
                for scenario_name in self.scenario_names:
                    self.execute_scenario(name=scenario_name, largescalemode=self.settings.largescalemode)

            self.run_timer.stop()
            self.logger.info(f"Total runtime for all scenarios: {self.run_timer}")

            self.join_cashflows()
            self.join_results()
        finally:
            for bar in self.horizon_bars:
                bar.close()
            if self.progress_bar is not None:
                self.progress_bar.close()

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error(f"Exception type: {exc_type.__name__}")
        self.logger.error(f"Exception message: {str(exc_value)}")
        self.logger.error("Traceback:")
        self.logger.error("".join(traceback.format_tb(exc_traceback)))

        self.logger.error(msg="Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def join_cashflows(self):
        suffix = "_cashflow_temp.pkl"
        filenames = [
            file
            for file in self.paths.output.iterdir()
            if file.name.endswith(suffix)
            and self.scenario_status.loc[file.name.removesuffix(suffix), "status"] == "successful"
        ]

        scenario_frames = [pd.read_pickle(file) for file in filenames]

        if scenario_frames:  # empty scenario_frames, if all scenarios fail during initialization
            joined_df = pd.concat(scenario_frames, axis=0)

            if self.paths.rerun and self.paths.cashflow_pkl.is_file():
                # as in join_results: a rerun only covers part of the run, so the
                # originally successful scenarios have to be carried over from the
                # previous file - writing only the rerun ones would drop their cashflows
                cashflows_prev = pd.read_pickle(self.paths.cashflow_pkl)
                cashflows_prev = cashflows_prev.drop(
                    index=joined_df.index.get_level_values("scenario").unique(),
                    level="scenario",
                    errors="ignore",
                )
                joined_df = pd.concat([cashflows_prev, joined_df], axis=0)

            joined_df.to_csv(self.paths.cashflow_csv, index=True)
            joined_df.to_pickle(self.paths.cashflow_pkl)
            self.logger.info("Result cashflow file created")

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in filenames:
            file.unlink()

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
            results_run_prev = None

            if self.paths.rerun and self.paths.summary_pkl.is_file():  # only happens for infeasible scenarios
                results_summary_prev = pd.read_pickle(self.paths.summary_pkl)
                # the previous summary carries a run block of its own. Keep it for the
                # scenarios it describes - those were not simulated again - but take it
                # out of the results, so the block prepended below remains the only one
                # and every (block, key) row of the merged summary stays unique
                results_run_prev = results_summary_prev.loc[["run"]].drop(
                    columns=joined_results.columns, errors="ignore"
                )
                results_summary_prev = results_summary_prev.drop(index="run", level="block")
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
                    self.run_timer.result_summary,
                ]
            )
            # apply MultiIndex
            results_run.index = pd.MultiIndex.from_tuples(
                tuples=[("run", key) for key in results_run.index], names=["block", "key"]
            )

            # convert to DataFrame and repeat for all scenarios
            results_run = pd.DataFrame([results_run] * len(joined_results.columns)).T
            results_run.columns = joined_results.columns

            if results_run_prev is not None:
                # scenarios carried over from the previous summary keep the run they
                # were simulated in; only the rerun ones get this run's block
                results_run[results_run_prev.columns] = results_run_prev

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

    def execute_scenario(
        self,
        name: str,
        status_queue: mpq.Queue | None = None,
        lock: mps.Lock | None = None,
        largescalemode: bool = False,
    ):
        # this method is necessary as running Scenario() directly from the starmap fails as Scenario object contains
        # objects which cannot be pickled.

        scenario_logger = logger_fcs.ContextLoggerAdapter(self.logger, {"scenarioname": name})
        try:
            worker = ScenarioWorker(
                paths=self.paths,
                settings=self.settings,
                name=name,
                parameters=self.scenario_data[name],
                logger=scenario_logger,
                lock=lock,
                status_update=self.trigger_scenario_status_update,
                status_queue=status_queue,
            )
            worker.execute()
        except Exception as e:
            self.trigger_scenario_status_update(
                status_msg=_ScenarioStatusMessage(
                    scenario_name=name,
                    status=_ScenarioStatus.FAILED,
                    extras={"exception": str(e), "traceback": traceback.format_exc()},
                    worker_name=mp.current_process().name,
                ),
                queue=status_queue,
            )

            self.logger.error(
                msg=f"{str(e)} - continue on next scenario",
                exc_info=True,
            )

    def read_status_queue(self, queue: mpq.Queue):
        while True:
            status_msg = queue.get()
            if status_msg is None:  # Exit signal
                break
            self.update_scenario_status(status_msg)

    def trigger_scenario_status_update(
        self, status_msg: _ScenarioStatusMessage, queue: mpq.Queue | None = None
    ) -> None:
        if queue is not None:
            queue.put(status_msg)
        else:
            self.update_scenario_status(status_msg)

    def update_scenario_status(self, status_msg: _ScenarioStatusMessage):
        self.scenario_status.loc[status_msg.scenario_name, "status"] = status_msg.status.value

        # If the worker provided extra information, this information is directly dumped into the CSV.
        if status_msg.extras is not None:
            for key, value in status_msg.extras.items():
                if value is None:
                    continue
                self.scenario_status.loc[status_msg.scenario_name, key] = value

        self.scenario_status.to_csv(self.paths.status, index=True)

        # update the large-scale mode progress bars
        if self.progress_bar is not None:
            self._update_horizon_bar(status_msg)
            # advance the overall scenario bar once a scenario reaches a terminal state
            if status_msg.status in _TERMINAL_STATUSES:
                self.progress_bar.update(1)

    def _get_horizon_bar(self, worker_name: str) -> tqdm:
        """Return the horizon bar assigned to `worker_name`, assigning a free slot on first sight."""
        bar = self._worker_bar_map.get(worker_name)
        if bar is None:
            assigned = set(self._worker_bar_map.values())
            bar = next(b for b in self.horizon_bars if b not in assigned)
            self._worker_bar_map[worker_name] = bar
        return bar

    def _update_horizon_bar(self, status_msg: _ScenarioStatusMessage) -> None:
        """Drive the per-worker horizon progress bar according to the reported scenario status."""
        if not self.horizon_bars or status_msg.worker_name is None:
            return

        bar = self._get_horizon_bar(status_msg.worker_name)
        status = status_msg.status

        if status == _ScenarioStatus.STARTED:
            # total is not known until the scenario is initialized: show "initializing" instead of a counter
            bar.reset()
            bar.bar_format = "{desc}initializing"
            bar.set_description(status_msg.scenario_name)
        elif status == _ScenarioStatus.INITIALIZED and status_msg.n_horizons_total is not None:
            bar.bar_format = None  # restore the default proportional bar layout
            bar.reset(total=status_msg.n_horizons_total)
            bar.set_description(status_msg.scenario_name)
        elif status == _ScenarioStatus.COMPLETED_HORIZON:
            bar.update(1)
        elif status in _TERMINAL_STATUSES:
            bar.reset()
            bar.bar_format = "{desc}"
            bar.set_description_str("idle")


class ScenarioWorker:
    """
    Worker to process a scenario.

    Handles the execution of a scenario for single- and multiprocess runs.
    """

    def __init__(
        self,
        paths: scn.SimulationPaths,
        settings: simulation.SimulationSettings,
        name: str,
        parameters: pd.Series,
        logger: logging.Logger,
        status_update: _StatusUpdateCallback,
        status_queue: mpq.Queue | None = None,
        lock: mps.Lock | None = None,
    ) -> None:
        self._paths = paths
        self._settings = settings
        self._name = name
        self._parameters = parameters
        self._logger = logger
        self._lock = lock
        self._status_update = status_update
        self._status_queue = status_queue

    def update_scenario_status(
        self,
        status: _ScenarioStatus,
        extras: dict[str, str] | None = None,
        n_horizons_total: int | None = None,
    ) -> None:
        status_msg = _ScenarioStatusMessage(
            scenario_name=self._name,
            status=status,
            extras=extras,
            worker_name=mp.current_process().name,
            n_horizons_total=n_horizons_total,
        )

        self._status_update(status_msg, self._status_queue)

    def execute(self) -> None:
        self.update_scenario_status(_ScenarioStatus.STARTED)

        run_timer = time.RunTimer()

        worker = mp.current_process()
        msg_parallel = (
            f" on {worker.name.ljust(18)} - Parent: {worker._parent_name}" if hasattr(worker, "_parent_name") else ""
        )
        self._logger.info(f"Scenario initialization{msg_parallel}")

        if self._lock:
            # avoid rate limiting by geocoding API
            _ = self._lock.acquire()
            pytime.sleep(2)

        loc = location.Location.create_from_lat_lon(
            latitude=self._parameters[("scenario", "latitude")],
            longitude=self._parameters[("scenario", "longitude")],
            country=self._parameters[("scenario", "country")],
            state=self._parameters[("scenario", "state")],
            logger=self._logger,
        )

        if self._lock:
            self._lock.release()

        scenario_settings = scn.ScenarioSettings(
            largescalemode=self._settings.largescalemode,
            key_solcast_api=self._settings.key_solcast_api,
        )

        try:
            scenario = scn.Scenario(
                paths=self._paths,
                settings=scenario_settings,
                name=self._name,
                parameters=self._parameters,
                location=loc,
                logger=self._logger,
            )
        except Exception as e:
            self.update_scenario_status(
                status=_ScenarioStatus.FAILED, extras={"exception": str(e), "traceback": traceback.format_exc()}
            )
            self._logger.error(
                msg=f"{str(e)} - continue on next scenario",
                exc_info=True,
            )
            return

        self._logger.info("Scenario fully initialized")
        self.update_scenario_status(status=_ScenarioStatus.INITIALIZED, n_horizons_total=scenario.nhorizons)

        try:
            for horizon_index in range(scenario.nhorizons):
                prediction_horizon = simulation.PredictionHorizon(
                    index=horizon_index, scenario=scenario, settings=self._settings, logger=scenario.logger
                )
                prediction_horizon.execute()

                self.update_scenario_status(status=_ScenarioStatus.COMPLETED_HORIZON)
            self.update_scenario_status(status=_ScenarioStatus.SUCCESSFUL)
        except Exception as e:
            # Scenario has failed -> store scenario name to dataframe containing failed scenarios
            status = (
                _ScenarioStatus.INFEASIBLE if isinstance(e, simulation.OptimizationError) else _ScenarioStatus.FAILED
            )
            self.update_scenario_status(
                status=status,
                extras={"exception": str(e), "traceback": traceback.format_exc()},
            )

            self._logger.error(
                msg=f"{str(e)} - continue on next scenario", exc_info=(not isinstance(e, simulation.OptimizationError))
            )
        finally:
            scenario.process_results()

        run_timer.stop()
        self._logger.info(f"Scenario finished - runtime {run_timer}")

        if not self._settings.largescalemode:
            scenario.generate_and_save_plot()

        scenario.save_result_cashflow()
        scenario.save_result_summary([run_timer.result_summary])


def _worker_init(log_queue: mp.Queue, debugmode: bool) -> None:
    """
    Initialize a worker process and configure logging.

    :param log_queue: The queue to which log messages will be sent.
    :param debugmode: Configure the log level in the worker process according to the level of the parent.
    """
    logger_fcs.configure_process_logger_parallel(log_queue, debugmode)
