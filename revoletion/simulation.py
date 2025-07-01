#!/usr/bin/env python3
import webbrowser

from dataclasses import dataclass, field
import geopy
import holidays
import importlib.metadata
from importlib.resources import files
import itertools
import logging
import logging.handlers
import math
import os
from pathlib import Path
import numpy as np
import plotly.subplots
import pprint
import psutil
import pytz
import re
import shutil
import simpy
import subprocess
import sys
import threading
import time

import simpy
import timezonefinder
import traceback
import warnings

import multiprocessing as mp
import numpy_financial as npf
import oemof.solph as solph
import pandas as pd
import pyomo.environ as po

from revoletion import blocks
from revoletion import constraints
from revoletion import dispatch
from revoletion import economics as eco
from revoletion import logger as logger_fcs
from revoletion import scheduler
from revoletion import utils


class OptimizationError(Exception):
    pass


@dataclass
class SimulationPaths:
    scenarios: Path
    input: Path = None
    output: Path = None
    basename: Path = None

    _revoletion: Path = field(default_factory=lambda: files(__package__),
                              init=False,  # cannot be set manually
                              )
    _cwd: Path = field(default_factory=Path.cwd,
                       init=False,  # cannot be set manually
                       )

    def __post_init__(self):
        if self.basename is None:
            self.basename = Path(pd.Timestamp.now().strftime('%y%m%d_%H%M%S'))
        if self.input is None:
            self.input = self.scenarios.parent
        if self.output is None:
            self.output = self._cwd / 'results'
        self.output = self.output / self.basename

        # ensure all paths are absolute
        self.scenarios = self.scenarios.resolve()
        self.input = self.input.resolve()
        self.output = self.output.resolve()
        self._revoletion = self._revoletion.resolve()
        self._cwd = self._cwd.resolve()

        # ensure that all paths exist
        if not self.scenarios.is_file():
            raise FileNotFoundError(f'Scenario file not found: {self.scenarios}')
        if not self.input.is_dir():
            raise NotADirectoryError(f'Input directory path not interpretable: {self.input}')
        if not self.output.is_dir():
            self.output.mkdir(parents=True)  # create parents if missing -> relevant for default "results"

    def create_result_path(self,
                           suffix: str) -> Path:
        return self.output / f'{self.basename}_{suffix}'

    @property
    def cwd(self) -> Path:
        return self._cwd

    @property
    def revoletion(self) -> Path:
        return self._revoletion

    @property
    def data_persist(self) -> Path:
        return self.revoletion / 'data'

    @property
    def summary_csv(self) -> Path:
        return self.create_result_path(suffix='summary.csv')

    @property
    def summary_pkl(self) -> Path:
        return self.create_result_path(suffix='summary.pkl')

    @property
    def status(self) -> Path:
        return self.create_result_path(suffix='status.csv')

    @property
    def dump(self) -> Path:
        return self.create_result_path(suffix='model.lp')

    @property
    def log(self) -> Path:
        return self.create_result_path(suffix='log.log')


@dataclass
class SimulationSettings:
    solver: str
    n_processes: int
    largescalemode: bool
    debugmode: bool
    rerun: bool | Path
    rerun_infeasible: bool
    key_solcast_api: str


class SimulationRun:

    def __init__(self,
                 path_scenarios: Path,
                 path_input: Path = None,
                 path_output: Path = None,
                 solver: str = 'gurobi',
                 n_processes: int = 1,
                 largescalemode: bool = False,
                 debugmode: bool = False,
                 rerun: bool | Path = False,
                 rerun_infeasible: bool = True,
                 key_solcast_api: str = None):

        self.settings = SimulationSettings(solver=solver,
                                           n_processes=n_processes,
                                           largescalemode=largescalemode,
                                           debugmode=debugmode,
                                           rerun=rerun,
                                           rerun_infeasible=rerun_infeasible,
                                           key_solcast_api=key_solcast_api,  # ToDo: find more elegant solution
                                           )
        del solver, n_processes, largescalemode, debugmode, rerun, rerun_infeasible, key_solcast_api

        # region start runtime
        self.runtime_start = time.perf_counter()
        if not self.settings.rerun:
            self.runtimestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
        else:
            # get timestamp from rerun directory name (for both absolute and relative (to settings output dir) paths)
            self.runtimestamp = '_'.join(Path(self.settings.rerun).name.split('_')[:2])
        self.runtime_end = self.runtime_len = None
        # endregion

        self.name = path_scenarios.stem

        self.paths = SimulationPaths(scenarios=path_scenarios,
                                     input=path_input,
                                     output=path_output,
                                     basename=Path(f'{self.runtimestamp}_{self.name}'),
                                     )

        del path_scenarios, path_input, path_output

        # region get version information
        self.version_solph = solph.__version__
        self.version_revoletion = importlib.metadata.version('revoletion')

        try:  # todo additionally get commit hash of revoletion if possible
            self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()[0:6]
        except subprocess.CalledProcessError:
            self.commit_hash = 'unknown'
        # endregion

        # region read, copy and check scenario data
        self.scenario_data = pd.read_csv(self.paths.scenarios,
                                         index_col=[0, 1],
                                         keep_default_na=False)
        self.scenario_data = self.scenario_data.sort_index(sort_remaining=True).map(utils.infer_dtype)
        self.scenario_names = [name for name in self.scenario_data.columns if not name.startswith('#')]

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

        # region define logger structure
        self.logger = logger_fcs.get_root_logger(paths=self.paths,
                                                 settings=self.settings)

        self.logger.info(f'Reading scenarios from:\t{self.paths.scenarios}')
        self.logger.info(f'Reading input data from:\t{self.paths.input}')
        self.logger.info(f'Writing results to:\t\t{self.paths.output}')

        # plural extensions
        pe1 = 's' if self.scenario_num > 1 else ''
        pe2 = 'es' if self.settings.n_processes > 1 else ''
        self.logger.info(f'Running {self.scenario_num} scenario{pe1}'
                         f' with {self.settings.n_processes} process{pe2}')

        # make sure that uncaught errors (i.e. errors occurring outside simulate_scenario method) are logged to logfile
        sys.excepthook = self.handle_exception
        # endregion

        self.execute()

    def copy_scenario_file(self):
        target = self.paths.output / f'{self.name}.csv'
        try:  # with metadata
            shutil.copy2(self.paths.scenarios, target)
        except PermissionError:  # can happen if metadata is not writable, e.g. on network drives
            shutil.copyfile(self.paths.scenarios, target)

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
            Scenario(name=name,
                     run=self,
                     parameters=self.scenario_data[name],
                     paths=self.paths,
                     settings=self.settings,
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


class Scenario:

    def __init__(self,
                 name: str,
                 parameters: pd.Series | str,
                 run: SimulationRun,
                 paths: SimulationPaths = None,
                 settings: SimulationSettings = None,
                 log_queue: mp.Queue = None,
                 lock: mp.Lock = None,
                 status_update: SimulationRun.trigger_scenario_status_update = None,
                 status_queue: mp.Queue = None):

        self.name = name
        self.run = run

        if isinstance(parameters, pd.Series):
            self.parameters = parameters
        # check whether file exists
        elif isinstance(parameters, str) and Path(parameters).is_file():
            if parameters.endswith('.csv'):
                self.parameters = pd.read_csv(parameters,
                                              index_col=[0, 1],
                                              keep_default_na=False)
                self.parameters = self.parameters.sort_index(sort_remaining=True).map(utils.infer_dtype)
            elif parameters.endswith('.pkl'):
                self.parameters = pd.read_pickle(parameters)
        else:
            raise ValueError('Parameters must be a pandas Series or filename of a CSV or PKL file')

        self.paths = paths
        self.settings = settings

        if False:  # ToDo: activate for standalone execution of scenario
            self.logger = logger_fcs.get_root_logger(paths=self.paths,
                                                     settings=self.settings,
                                                     )

        elif log_queue is not None:
            self.logger = logger_fcs.get_process_logger_parallel(name=self.name,
                                                                 settings=self.settings,
                                                                 log_queue=log_queue,
                                                                 )
        else:
            self.logger = logger_fcs.get_process_logger_sequential(name=self.name,
                                                                   settings=self.settings,
                                                                   )

        self.status_update = status_update
        self.status_queue = status_queue

        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            # Force warnings in custom formatting and ignore warnings about infeasible or unbounded optimizations
            if not 'Optimization ended with status warning and termination condition' in str(message):
                self.logger.warning(f'{category.__name__}: {message} (in {filename}, line {lineno})')

        warnings.showwarning = custom_warning_handler

        self.update_scenario_status(status_msg={'status': 'started'})

        # General Information --------------------------------

        # integration levels at which power consumption is determined a priori
        self.apriori_lvls = ['uc', 'fcfs', 'equal', 'soc']

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        self.worker = mp.current_process()

        msg_parallel = (f' on {self.worker.name.ljust(18)} - Parent: {self.worker._parent_name}'
                        if hasattr(self.worker, '_parent_name') else '')
        self.logger.info(f'Scenario initialized{msg_parallel}')

        for key, value in self.parameters.loc['scenario', :].items():
            setattr(self, key, value)  # this sets all the parameters defined in the csv file

        if not isinstance(self.blocks, dict):
            raise ValueError(f'Scenario parameter "blocks" has to be defined in a dictionary format '
                             '("{\'name1\':\'classname1\',\'name2\':\'classname2\'}") - '
                             f'check for missing or additional single or double quotes')

        if not self.blocks:
            raise ValueError(f'Scenario parameter "blocks" is empty - Definition of at least one block is required')

        self.currency = self.currency.upper()  # all other parameters are .lower()-ed

        self.tzfinder = timezonefinder.TimezoneFinder()
        self.timezone = pytz.timezone(self.tzfinder.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        geolocator = geopy.geocoders.Nominatim(user_agent=f'location_finder')
        self.country = 'DE'  # set default country
        self.state = 'BY'  # set default state
        try:
            if lock is None:  # sequential
                location = geolocator.reverse((self.latitude, self.longitude), language="en", exactly_one=True)
            else:  # parallel
                with lock:
                    time.sleep(2)  # max 1 request per second --> wait for 2 seconds to make sure to not avoid the limit
                    location = geolocator.reverse((self.latitude, self.longitude), language="en", exactly_one=True)
            if location:
                self.country, self.state = location.raw['address']['ISO3166-2-lvl4'].split('-')
        except geopy.exc.GeocoderUnavailable:
            self.logger.warning(f'Connection to Geocoder failed. '
                                f'Using default country ({self.country}) and state ({self.state}).')

        # convert to datetime and calculate time(delta) values
        # simulation and project timeframe start simultaneously
        # simulation vs. extended simulation: for rh strategy and truncate_ph = False, the extended simulation timeframe
        # is longer than the simulation timeframe defined by the example parameter duration. Otherwise, they are the same.
        # ToDo: check for format not only len of string
        self.starttime = self.starttime if len(self.starttime) > 10 else self.starttime + ' 00:00'
        self.starttime = pd.to_datetime(self.starttime, format='%d.%m.%Y %H:%M').floor(self.timestep).tz_localize(self.timezone)

        # sim_duration and sim_endtime are defined
        if self.sim_duration is not None and self.sim_endtime is not None:
            raise ValueError('Both parameters "sim_duration" and "sim_endtime" are defined. '
                             'Please define only one of these parameters.')
        # sim_duration is defined, sim_endtime is not
        elif self.sim_duration is not None:
            self.sim_duration = (pd.Timedelta(days=self.sim_duration) if isinstance(self.sim_duration, (float, int))
                                 else pd.Timedelta(self.sim_duration)).floor(self.timestep)
            self.sim_endtime = self.starttime + self.sim_duration
        # sim_endtime is defined, sim_duration is not
        elif self.sim_endtime is not None:
            # ToDo: check for format not only len of string
            # ToDo: use function for starttime and endtime conversion
            self.sim_endtime = self.sim_endtime if len(self.sim_endtime) > 10 else self.sim_endtime + ' 00:00'
            self.sim_endtime = (pd.to_datetime(self.sim_endtime, format='%d.%m.%Y %H:%M')
                            .floor(self.timestep)
                            .tz_localize(self.timezone)
                            )
            self.sim_duration = self.sim_endtime - self.starttime

        self.sim_extd_duration = self.sim_duration
        self.sim_extd_endtime = self.sim_endtime
        self.prj_duration_yrs = self.prj_duration
        self.prj_endtime = self.starttime + pd.DateOffset(years=self.prj_duration)
        self.prj_duration = self.prj_endtime - self.starttime  # takes leap years into account

        # generate variables for calculations
        self.timestep_td = pd.Timedelta(self.timestep)
        self.timestep_hours = self.timestep_td.total_seconds() / 3600
        self.sim_yr_rat = self.sim_duration / pd.Timedelta(days=365)  # no leap years
        self.sim_prj_rat = self.sim_duration / self.prj_duration

        if self.strategy == 'rh':
            self.len_ph = pd.Timedelta(hours=self.len_ph).floor(self.timestep_td)
            self.len_ch = pd.Timedelta(hours=self.len_ch).floor(self.timestep_td)
            self.nhorizons = math.ceil(self.sim_duration / self.len_ch)  # number of timeslices to run
            if not self.truncate_ph:
                # if PH is not truncated, the end of the last PH may be later than the end of the evaluation period
                self.sim_extd_duration = self.len_ch * (self.nhorizons - 1) + self.len_ph
                self.sim_extd_endtime = self.starttime + self.sim_extd_duration
        elif self.strategy in ['go']:
            self.len_ph = self.sim_duration
            self.len_ch = self.sim_duration
            self.nhorizons = 1
        else:
            raise ValueError(f'Optimization strategy "{self.strategy}" unknown')

        if self.len_ph == self.timestep_td:
            raise ValueError('Single timestep optimization not possible. Adjust simulation duration, timestep or '
                             'prediction horizon length / truncate_ph (for RH only)')

        # generate a datetimeindex for the energy system model to run on
        self.dti_eval = pd.date_range(start=self.starttime, end=self.sim_endtime, freq=self.timestep, inclusive='left')
        self.dti_eval_extd = utils.extend_dti(dti=self.dti_eval, freq=self.timestep_td)
        # extended index covers PHs that are not truncated after simulation end time
        self.dti_sim = pd.date_range(start=self.starttime, end=self.sim_extd_endtime, freq=self.timestep,
                                     inclusive='left')
        self.dti_sim_extd = utils.extend_dti(dti=self.dti_sim, freq=self.timestep_td)

        # get holidays during simulation timeframe
        years = range(min(self.dti_sim).year, max(self.dti_sim).year + 1)
        try:
            self.holiday_dates = sorted(
                getattr(holidays, self.country)(years=years,
                                                state=self.state))
        except:  # not for all countries the states are available (e.g. France)
            try:
                self.holiday_dates = sorted(
                    getattr(holidays, self.country)(years=years))
                self.logger.warning(f'Holidays for state {self.state} not available. '
                                    f'Country-wide holidays for {self.country} are used instead.')
            except AttributeError:  # not all countries worldwide are available
                self.holiday_dates = []
                self.logger.warning(f'Holidays for country {self.country} not available. '
                                    f'No public holidays are considered in this scenario.')

        # region set air temperature
        temp_air = pd.DataFrame(index=self.dti_sim,
                                columns=['temp_air'],
                                dtype=float)

        if isinstance(self.temp_air, (float, int)):
            temp_air['temp_air'] = self.temp_air
            self.temp_air = temp_air

        elif (isinstance(self.temp_air, str)
              and self.blocks.get(self.temp_air, '') == 'PVSource'):
            # PVSource checks for temp_scn in parameters and writes temperature to this variable
            self.parameters.loc[(self.temp_air, 'temp_scn')] = True
            self.temp_air = temp_air

        elif (isinstance(self.temp_air, str)
              and (self.paths.input / utils.set_extension(filename=self.temp_air,
                                                          default_extension='.csv')).is_file()
        ):
            self.temp_air = utils.read_timeseries_csv(path_input_file=(self.paths.input /
                                                                       utils.set_extension(filename=self.temp_air,
                                                                                           default_extension='.csv')),
                                                      block=self,  # only uses block.name -> scenario works, too
                                                      scenario=self)
        else:
            self.logger.warning(f'Specified argument for scenario parameter temp_air ({self.temp_air}) not found - '
                                f'Using default of 25 °C')
            temp_air['temp_air'] = 25
            self.temp_air = temp_air
        # endregion

        # region initialize result variables
        self.periods_prj = np.arange(0, self.prj_duration_yrs)
        self.periods_prj_extd = np.arange(0, self.prj_duration_yrs + 1)  # add. year for salvage values
        self.discount_factors = pd.DataFrame(index=self.periods_prj_extd,
                                             columns=['beginning', 'mid', 'end'],
                                             data={occ: eco.discount(future_value=1,
                                                                     periods=self.periods_prj_extd + 1,
                                                                     discount_rate=self.wacc,
                                                                     occurs_at=occ)
                                                   for occ in ['beginning', 'mid', 'end']},
                                             dtype='float64')

        self.aggregator = eco.EconomicAggregator(name='scenario', block=None, scenario=self)

        self.block_registry = dict()

        # Define priorities of blocks to ensure correct initialization order
        priority_default = 2
        priority_blocks = {'SystemCore': 0,  # always first -> ac and dc bus required for all other ElectricBlocks
                           'PVSource': 1,  # holds temperature and wind data -> required by StorageBlock and WindSource
                           }

        for name, class_name in sorted({'core': 'SystemCore', **self.blocks}.items(),
                                       key=lambda item: priority_blocks.get(item[1], priority_default),):
            class_obj = getattr(blocks, class_name, None)
            if class_obj is not None and isinstance(class_obj, type):
                class_obj(name, self)
            else:
                raise ValueError(f'Class "{class_name}" not found in blocks.py file - '
                                 f'Check for typos or add class.')

        if self.invest_max is not None and self.invest_max < self.aggregator.capex['preexisting']:
            raise ValueError(f'Initial investment costs of {self.aggregator.capex["preexisting"]:.2f} {self.currency} '
                             f'exceed maximum investment limit of {self.invest_max} {self.currency}')

        self.objective_opt = None  # unused for rh strategy
        self.cashflows = pd.DataFrame()
        self.energies = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples=[('renewable', 'act'),
                                                                             ('sources', 'pro'),
                                                                             ('sinks', 'del')],
                                                                     names=['block', 'key']),
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,
                                     dtype=float)

        self.figure = None
        self.plot_traces = {'powers': [],
                            'states': []}

        self.result_messages = []
        self.result_summary = []
        self.result_timeseries = []

        self.e_eta = None
        self.renewable_share = None
        self.lcoe_total = self.lcoe_wocs = None
        self.npv = self.irr = self.mirr = None
        # endregion

        # region preexecution
        self.dispatcher = dispatch.SiteDispatcher(scenario=self)

        for block in self.block_registry.get('TopLevelBlock', {}).values():
            block.pre_scenario()

        self.scheduler = None
        if self.block_registry.get('SubFleetScheduling', {}):
            self.scheduler = scheduler.AprioriPowerScheduler(scenario=self)
        # endregion

        self.logger.debug(f'Scenario initialization completed')

        self.update_scenario_status(status_msg={'status': 'fully initialized'})

        # todo adapt to new fleet structure
        # # check example parameter configuration of rulebased charging for validity
        # if fleet_unlim := [fleet for fleet in self.block_registry.get('Fleet', {}).values() if
        #                 (fleet.mode_scheduling in self.apriori_lvls)
        #                 and fleet.mode_scheduling != 'uc'
        #                 and not fleet.power_lim_static]:
        #     if [block for block in self.blocks.values() if getattr(block, 'invest', False)]:
        #         raise ValueError(f'Rulebased charging except for uncoordinated charging (uc) '
        #                          f'without static load management (lm_static) is not compatible'
        #                          f' with size optimization')
        #     if [block for block in self.blocks.values() if isinstance(block, blocks.StationaryBattery)]:
        #         raise ValueError(f'Rulebased charging except for uncoordinated charging (uc) '
        #                          f'without static load management (lm_static) is not implemented for systems with '
        #                          f'stationary energy storage')
        #     if len(set([cs.mode_scheduling for cs in cs_unlim])) > 1:
        #         raise ValueError(f'All rulebased CommoditySystems with dynamic load management '
        #                          f'have to follow the same strategy. Different strategies are not possible')
        #     if cs_unlim[0].mode_scheduling == 'equal' and len(set([cs.bus_connected for cs in cs_unlim])) > 1:
        #         raise ValueError(f'If strategy "equal" is chosen for CommoditySystems with'
        #                          f' dynamic load management, all CommoditySystems with dynamic load management have to'
        #                          f' be connected to the same bus')

        # region execute scenario
        try:
            for horizon_index in range(self.nhorizons):  # Inner optimization loop over all prediction horizons
                PredictionHorizon(index=horizon_index,
                                  scenario=self)

                self.update_scenario_status(status_msg={'status': f'completed horizon '
                                                                  f'{horizon_index + 1} out of '
                                                                  f'{self.nhorizons}'})

            self.update_scenario_status(status_msg={'status': 'successful'})

        except Exception as e:
            # Scenario has failed -> store scenario name to dataframe containing failed scenarios
            status = 'infeasible' if isinstance(e, OptimizationError) else 'failed'
            self.update_scenario_status(status_msg={'status': status,
                                                    'exception': str(e),
                                                    'traceback': traceback.format_exc()})

            self.logger.error(msg=f'{str(e)} - continue on next scenario',
                              exc_info=(not isinstance(e, OptimizationError)))

            self.end_timing()  # ToDo: does timing end here? Should that better be called at the end of result writing?

        finally:  # save results up to exception - valuable in RH strategy

            for block in self.block_registry.get('TopLevelBlock', {}).values():
                block.post_scenario()
            self.aggregator.post_scenario()

            self.calc_meta_results()

            self.save_result_summary()

            if not self.settings.largescalemode:
                self.result_timeseries = pd.concat(self.result_timeseries, axis=1)
                self.result_timeseries.to_csv(self.paths.create_result_path(suffix=f'{self.name}_results_ts.csv'))
                for msg in self.result_messages:
                    self.logger.info(msg)
                self.generate_plots()
                self.figure.write_html(self.paths.create_result_path(suffix=f'{self.name}.html'))
                try:
                    self.figure.show(renderer='browser')
                except webbrowser.Error:  # webbrowser is not available on most remote machines
                    pass

        logging.shutdown()
        # endregion

    def update_scenario_status(self,
                               status_msg: dict):

        if self.status_update is not None:
            status_msg.update(scenario=self.name)
            self.status_update(queue=self.status_queue,
                               status_msg=status_msg)

    def calc_meta_results(self):

        # pandas creates a RuntimeWarning at division by 0 -> try/except does not work
        if self.energies.loc[('sources', 'pro'), 'sim'] == 0:
            self.logger.warning(f'Core efficiency calculation: division by zero')
        else:
            self.e_eta = self.energies.loc[('sinks', 'del'), 'sim'] / self.energies.loc[('sources', 'pro'), 'sim']

        if self.energies.loc[('sources', 'pro'), 'sim'] == 0:
            self.logger.warning(f'Renewable share calculation: division by zero')
        else:
            self.renewable_share = (self.energies.loc[('renewable', 'act'), 'sim'] /
                                    self.energies.loc[('sources', 'pro'), 'sim'])

        if self.energies.loc[('sinks', 'del'), 'sim'] == 0:
            self.logger.warning(f'LCOE calculation: division by zero')
        else:
            self.lcoe_total = self.aggregator.totex['dis'] / self.energies.loc[('sinks', 'del'), 'dis']
            self.lcoe_wocs = ((self.aggregator.totex['dis'] -
                               # ToDo: check whether calculation of totex['dis'] of fleets is correct
                               sum([fleet.aggregator.totex['dis'] for fleet in self.block_registry.get('Fleet', {}).values()])) /
                              self.energies.loc[('sinks', 'del'), 'dis'])

        self.npc = self.aggregator.totex['dis']
        self.npv = self.aggregator.value['dis']
        # ToDo: implement self.cashflows
        self.irr = npf.irr(self.cashflows.sum(axis=1).to_numpy())
        self.mirr = npf.mirr(self.cashflows.sum(axis=1).to_numpy(), self.wacc, self.wacc)

        # print basic results
        self.logger.info(f'NPC {f"{self.npc:,.2f}" if pd.notna(self.npc) else "-"} {self.currency} | '
                         f'NPV {f"{self.npv:,.2f}" if pd.notna(self.npv) else "-"} {self.currency} | '
                         f'LCOE {f"{self.lcoe_wocs * 1e5:,.2f}" if pd.notna(self.lcoe_wocs) else "-"} {self.currency}-ct/kWh | '
                         f'mIRR {f"{self.mirr * 100:,.2f}" if pd.notna(self.mirr) else "-"} %')

    def end_timing(self):
        self.runtime_end = time.perf_counter()
        self.runtime_len = round(self.runtime_end - self.runtime_start, 2)
        self.logger.info(f'Scenario finished - runtime {self.runtime_len} s')

    def generate_plots(self):

        self.figure = plotly.subplots.make_subplots(specs=[[{'secondary_y': True}]])

        self.figure.add_traces(self.plot_traces['powers'],
                               secondary_ys=[False] * len(self.plot_traces['powers']))

        self.figure.add_traces(self.plot_traces['states'],
                               secondary_ys=[True] * len(self.plot_traces['states']))

        self.figure.update_layout(plot_bgcolor='white')
        self.figure.update_xaxes(title='Local Time',
                                 showgrid=True,
                                 linecolor='gray',
                                 gridcolor='gray')
        self.figure.update_yaxes(title='Power in W',
                                 showgrid=True,
                                 linecolor='gray',
                                 gridcolor='gray',
                                 secondary_y=False, )
        self.figure.update_yaxes(title='State of Charge',
                                 showgrid=False,
                                 secondary_y=True)

        if self.strategy == 'go':
            self.figure.update_layout(title=f'Global Optimum Results - '
                                            f'{self.paths.basename} - '
                                            f'Scenario: {self.name}')
        if self.strategy == 'rh':
            self.figure.update_layout(title=f'Rolling Horizon Results - '
                                            f'{self.paths.basename} - '
                                            f'Scenario: {self.name} - '
                                            f'PH: {self.len_ph}h - '
                                            f'CH: {self.len_ch}h')

    def save_result_summary(self):
        """
        Saves all int, float and str attributes of run, scenario (incl. technoeconomic KPIs) and all blocks to the
        results dataframe
        :return: none
        """

        # get results of run
        results_run = pd.Series({key: value for key, value in self.run.__dict__.items()
                                 if isinstance(value, (int, float, bool, str))})
        # apply MultiIndex
        results_run.index = pd.MultiIndex.from_tuples(tuples=[('run', key) for key in results_run.index],
                                                      names=['block', 'key'])

        # get results of scenario
        results_scenario = pd.concat([
            # get attributes of type int, float, bool and str for scenario.result_summary
            pd.Series({key: value for key, value in self.__dict__.items()
                       if isinstance(value, (int, float, bool, str))}),
            # get dict of blocks with class names
            pd.Series(index=['blocks'], data=str({key: value.classname for key, value in self.block_registry.get('TopLevelBlock', {}).items()})),
            # get energies dataframes results for scenario.result_summary
            utils.create_results_from_dataframe(df=self.energies, name_prefix='energy'),
            # get economic results for scenario.result_summary
            self.aggregator.write_result_summary()])

        # apply MultiIndex
        results_scenario.index = pd.MultiIndex.from_tuples(tuples=[('scenario', key) for key in results_scenario.index],
                                                           names=['block', 'key'])

        # write results from run and scenario to result_summary
        self.result_summary = pd.concat([results_run, results_scenario, *self.result_summary])

        # convert result_summary to DataFrame and save to temporary file
        pd.DataFrame(self.result_summary, columns=[self.name]).to_pickle(
            self.paths.output / f'{self.name}_summary_temp.pkl'
        )


class PredictionHorizon:

    def __init__(self, index, scenario):

        self.index = index
        self.scenario = scenario

        self.results = None

        # region time and data generation and slicing
        self.starttime = self.scenario.starttime + (index * self.scenario.len_ch)  # calc both start times
        self.ch_endtime = self.starttime + self.scenario.len_ch
        self.ph_endtime = self.starttime + self.scenario.len_ph
        self.timestep = self.scenario.timestep

        self.constraints = constraints.CustomConstraints(scenario=self.scenario)

        # Display logger message if PH exceeds simulation end time and has to be truncated
        if self.ph_endtime > self.scenario.sim_endtime and self.scenario.truncate_ph:
            self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - ' +
                                      f'Prediction Horizon truncated to simulation end time')

        # Truncate PH and CH to simulation or eval end time
        self.ph_endtime = min(self.ph_endtime, self.scenario.sim_extd_endtime)
        self.ch_endtime = min(self.ch_endtime, self.scenario.sim_endtime)

        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - ' +
                                  f'Start: {self.starttime} - ' +
                                  f'CH end: {self.ch_endtime} - ' +
                                  f'PH end: {self.ph_endtime}')

        # Create datetimeindex for ph and ch; neglect last timestep as this is the first timestep of the next ph / ch
        self.dti_ph = pd.date_range(start=self.starttime, end=self.ph_endtime, freq=self.scenario.timestep, inclusive='left')
        self.dti_ph_extd = utils.extend_dti(dti=self.dti_ph, freq=self.scenario.timestep_td)
        self.dti_ch = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=self.scenario.timestep, inclusive='left')
        self.dti_ch_extd = utils.extend_dti(dti=self.dti_ch, freq=self.scenario.timestep_td)

        # if apriori power scheduling is necessary, calculate power schedules:
        if self.scenario.scheduler:
            self.scenario.logger.debug(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                       f'Calculating power schedules for commodities with rulebased charging strategies')
            self.scenario.scheduler.calc_ph_schedule(self)
        # endregion

        # region build energy system model
        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Building oemof model')

        self.es = solph.EnergySystem(timeindex=self.dti_ph,
                                     infer_last_interval=True)  # initialize energy system model instance

        for block in self.scenario.block_registry.get('TopLevelBlock', {}).values():
            block.pre_horizon(self)

        self.scenario.logger.debug(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                   f'Model build completed')
        # endregion

        # region build optimization problem
        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Building optimization problem from oemof model')

        self.model = solph.Model(self.es, debug=self.scenario.settings.debugmode)
        self.constraints.apply_constraints(model=self.model)

        if self.scenario.settings.debugmode and self.index == 1:
            self.model.write(self.scenario.path.dump, io_options={'symbolic_solver_labels': True})
        # endregion

        # region solve optimization problem
        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Model built, starting optimization')
        results = self.model.solve(solver=self.scenario.settings.solver,
                                   solve_kwargs={'tee': self.scenario.settings.debugmode})

        if (results.solver.status == po.SolverStatus.ok) and \
                (results.solver.termination_condition == po.TerminationCondition.optimal):
            self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                      f'Optimization completed, getting results')
            if self.scenario.nhorizons == 1:  # Don't store objective for multiple horizons in scenario (most RH scenarios)
                self.scenario.objective_opt = self.model.objective()
        elif results.solver.termination_condition == po.TerminationCondition.infeasible:
            raise OptimizationError(
                f'Horizon {self.index + 1} of {self.scenario.nhorizons} - Scenario failed: Infeasible')
        elif results.solver.termination_condition == po.TerminationCondition.unbounded:
            raise OptimizationError(
                f'Horizon {self.index + 1} of {self.scenario.nhorizons} - Scenario failed: Unbounded')
        elif results.solver.termination_condition == po.TerminationCondition.infeasibleOrUnbounded:
            raise OptimizationError(
                f'Horizon {self.index + 1} of {self.scenario.nhorizons} - Scenario failed: Infeasible or Unbounded '
                f'(To solve this error try to set investment limits for blocks or for the scenario)')
        else:
            raise Exception(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                            f'Optimization terminated with unknown status: {results.solver.termination_condition}')
        # endregion

        # region get results
        # Get result data slice for current CH from results and save in result dataframes for later analysis
        # Get (possibly optimized) component sizes from results to handle outputs more easily
        self.results = solph.processing.results(self.model)  # Get the results of the solved horizon from the solver

        self.scenario.logger.debug(pprint.pformat(solph.processing.meta_results(self.model)))

        # free up RAM
        del self.model

        for block in self.scenario.block_registry.get('TopLevelBlock', {}).values():
            block.post_horizon(self)
        # endregion
