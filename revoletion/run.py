#!/usr/bin/env python3

import importlib.metadata
import itertools
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd
from oemof import solph as solph

from . import logger as logger_fcs
from . import utils
from . import simulation


class SimulationRun:

    def __init__(self,
                 paths: simulation.SimulationPaths,
                 settings: simulation.SimulationSettings = None,
                 ):

        self.paths = paths
        self.settings = settings if settings is not None else simulation.SimulationSettings()

        self.runtime_start = time.perf_counter()
        self.runtime_end = self.runtime_len = None

        self.name = self.paths.scenario.stem

        if not self.settings.rerun:
            self.runtimestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
        else:
            # get timestamp from rerun directory name (for both absolute and relative (to settings output dir) paths)
            self.runtimestamp = '_'.join(Path(self.settings.rerun).name.split('_')[:2])

        self.paths.basename = Path(f'{self.runtimestamp}_{self.name}')

        # region get version information
        self.version_solph = solph.__version__
        self.version_revoletion = importlib.metadata.version('revoletion')

        try:  # todo additionally get commit hash of revoletion if possible
            self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()[0:6]
        except subprocess.CalledProcessError:
            self.commit_hash = 'unknown'
        # endregion

        # region read, copy and check scenario data
        if not self.paths.scenario.is_file():
            raise FileNotFoundError(f'Scenario file {self.paths.scenario} does not exist')
        if self.paths.scenario.suffix == '.csv':
            self.scenario_data = pd.read_csv(self.paths.scenario,
                                             index_col=[0, 1],
                                             keep_default_na=False)
        elif self.paths.scenario.suffix == '.pkl':
            self.scenario_data = pd.read_pickle(self.paths.scenario)
            if not isinstance(self.scenario_data, pd.DataFrame):
                raise ValueError(f'Scenario file {self.paths.scenario} must be a DataFrame')
        else:
            raise ValueError(f'Scenario file {self.paths.scenario} must be a CSV or PKL file')
        self.scenario_data = self.scenario_data.sort_index(sort_remaining=True).map(utils.infer_dtype)
        self.scenario_names = [name for name in self.scenario_data.columns if not name.startswith('#')]

        # region define logger structure
        self.logger = logger_fcs.get_root_logger(paths=self.paths,
                                                 settings=self.settings,
                                                 len_scn_max=max([len(el)
                                                                  for el
                                                                  in list(self.scenario_names) + ["root"]]))

        # make sure that uncaught errors (i.e. errors occurring outside simulate_scenario method) are logged to logfile
        sys.excepthook = self.handle_exception
        # endregion

        if self.settings.rerun:
            # only run scenarios which have not been optimized successfully (or were infeasible)
            self.scenario_status = pd.read_csv(self.paths.status,
                                               index_col=0,
                                               dtype=str,  # empty columns are interpreted as float -> avoid
                                               )

            dont_rerun = ['successful', 'infeasible'] if self.settings.rerun_infeasible else ['successful']
            scenarios_rerun = self.scenario_status[~self.scenario_status['status'].isin(dont_rerun)].index.to_list()
            self.scenario_names = [name for name in self.scenario_names if name in scenarios_rerun]

            if not self.scenario_names:
                raise ValueError(
                    f'Parameter "--rerun" was set to {self.settings.rerun}, but the status file contains no scenarios '
                    f'to rerun.\n'
                    f'All scenarios were {"either infeasible or " if self.settings.rerun_infeasible else ""}'
                    f'already completed successfully.\n'
                    f'Check {(Path(self.paths.status.parent.name) / self.paths.status.name)} '
                    f'for additional information.')

            # delete all temporary results of files which are rerun (happens if SimulationRun terminates unexpected)
            for scenario in self.scenario_names:
                for file_name in [f'{scenario}_summary_temp.csv',
                                  f'{scenario}_results.csv']:
                    file_path = self.paths.output / file_name
                    if file_path.is_file():
                        file_path.unlink()

            # reset status of scenarios to be run to 'queued'
            self.scenario_status.loc[self.scenario_names, ['status', 'exception', 'traceback']] = (
                    [['queued', pd.NA, pd.NA]] * len(self.scenario_names))
            self.scenario_status.to_csv(self.paths.status, index=True)

        else:
            self.scenario_status = pd.DataFrame(index=self.scenario_names,
                                                data={'status': 'queued',
                                                      'exception': None,
                                                      'traceback': None}).rename_axis('scenario')
            self.copy_scenario_file()
        self.scenario_num = len(self.scenario_names)

        if self.scenario_num == 0:
            raise ValueError('No executable scenarios found in scenario file')

        self.settings.n_processes = min(self.settings.n_processes, os.cpu_count(), self.scenario_num)
        # endregion

        self.logger.info(f'Reading scenarios from:\t{self.paths.scenario}')
        self.logger.info(f'Reading input data from:\t{self.paths.input}')
        self.logger.info(f'Writing results to:\t\t{self.paths.output}')

        # plural extensions
        pe1 = 's' if self.scenario_num > 1 else ''
        pe2 = 'es' if self.settings.n_processes > 1 else ''
        self.logger.info(f'Running {self.scenario_num} scenario{pe1}'
                         f' with {self.settings.n_processes} process{pe2}')

        self.execute()

    def copy_scenario_file(self):
        target = self.paths.output / f'{self.name}.csv'
        try:  # with metadata
            shutil.copy2(self.paths.scenario, target)
        except PermissionError:  # can happen if metadata is not writable, e.g. on network drives
            shutil.copyfile(self.paths.scenario, target)

    def execute(self):
        if self.settings.n_processes > 1:
            with mp.Manager() as manager:
                lock = manager.Lock()

                status_queue = manager.Queue()
                status_thread = threading.Thread(target=self.read_status_queue, args=(status_queue,))
                status_thread.start()

                log_queue = manager.Queue()
                log_thread = threading.Thread(target=logger_fcs.read_mplogger_queue, args=(log_queue,))
                log_thread.start()

                with mp.Pool(processes=self.settings.n_processes) as pool:
                    pool.starmap(self.execute_scenario,
                                 zip(self.scenario_names,
                                     itertools.repeat(log_queue),
                                     itertools.repeat(status_queue),
                                     itertools.repeat(lock)))
                status_queue.put(None)
                status_thread.join()
                log_queue.put(None)
                log_thread.join()
        else:
            for scenario_name in self.scenario_names:
                self.execute_scenario(name=scenario_name)

        # region end runtime
        self.runtime_end = time.perf_counter()
        self.runtime_len = self.runtime_end - self.runtime_start
        self.logger.info(f'Total runtime for all scenarios: {self.runtime_len:.1f} s')
        # endregion

        self.join_results()

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error(f'Exception type: {exc_type.__name__}')
        self.logger.error(f'Exception message: {str(exc_value)}')
        self.logger.error('Traceback:')
        self.logger.error(''.join(traceback.format_tb(exc_traceback)))

        self.logger.error(msg='Uncaught exception',
                          exc_info=(exc_type, exc_value, exc_traceback))

    def join_results(self):

        filenames = [file
                     for file in self.paths.output.iterdir()
                     if file.name.endswith('_summary_temp.pkl')
                     and self.scenario_status.loc[file.name.removesuffix('_summary_temp.pkl'), 'status'] == 'successful'
                     ]

        scenario_frames = [pd.read_pickle(file) for file in filenames]

        if scenario_frames:  # empty scenario_frames, if all scenarios fail during initialization
            joined_results = pd.concat(scenario_frames, axis=1)
            joined_results.loc[('run', 'runtime_end'), :] = self.runtime_end
            joined_results.loc[('run', 'runtime_len'), :] = self.runtime_len

            if self.settings.rerun and self.paths.summary_pkl.is_file():  # only happens for infeasible scenarios
                results_summary_prev = pd.read_pickle(self.paths.summary_pkl)
                joined_results = pd.concat([results_summary_prev, joined_results], axis=1)

            # apply same order of scenarios as in scenario input file
            joined_results = joined_results[[c for c in self.scenario_data.columns if c in joined_results.columns]]

            # get results of run
            results_run = pd.Series({key: value for key, value in self.__dict__.items()
                                     if isinstance(value, (int, float, bool, str))})
            # apply MultiIndex
            results_run.index = pd.MultiIndex.from_tuples(tuples=[('run', key) for key in results_run.index],
                                                          names=['block', 'key'])

            # convert to DataFrame and repeat for all scenarios
            results_run = pd.DataFrame([results_run] * len(joined_results.columns)).T
            results_run.columns = joined_results.columns

            joined_results = pd.concat([results_run,
                                        joined_results,
                                        ])

            joined_results.to_csv(self.paths.summary_csv, index=True)
            joined_results.to_pickle(self.paths.summary_pkl)
            self.logger.info('Result summary file created')

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in filenames:
            file.unlink()

    def read_status_queue(self, queue):
        while True:
            status_msg = queue.get()
            if status_msg is None:  # Exit signal
                break
            self.update_scenario_status(status_msg)

    def execute_scenario(self,
                         name: str,
                         log_queue: mp.Queue = None,
                         status_queue: mp.Queue = None,
                         lock: mp.Lock = None):

        # this method is necessary as running Scenario() directly from the starmap fails as Scenario object contains
        # objects which cannot be pickled.
        try:
            simulation.Scenario(paths=self.paths,
                                settings=self.settings,
                                run_execution=True,
                                name=name,
                                parameters=self.scenario_data[name],
                                log_queue=log_queue,
                                lock=lock,
                                status_update=self.trigger_scenario_status_update,
                                status_queue=status_queue)
        except Exception as e:
            self.trigger_scenario_status_update(queue=status_queue,
                                                status_msg={'scenario': name,
                                                            'status': 'failed',
                                                            'exception': str(e),
                                                            'traceback': traceback.format_exc()})

            self.logger.error(msg=f'{str(e)} - continue on next scenario', # todo is not written to log or stream
                              exc_info=True)

    def trigger_scenario_status_update(self, queue, status_msg):
        if queue is not None:
            queue.put(status_msg)
        else:
            self.update_scenario_status(status_msg)

    def update_scenario_status(self, status_msg):
        for col in [key for key, value in status_msg.items() if key != 'scenario' and value is not None]:
            self.scenario_status.loc[status_msg['scenario'], col] = status_msg[col]
        self.scenario_status.to_csv(self.paths.status,
                                    index=True)
