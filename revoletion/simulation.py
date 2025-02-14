#!/usr/bin/env python3

import geopy
import graphviz
import holidays
import importlib.metadata
import itertools
import logging
import logging.handlers
import math
import os
import pathlib
import numpy as np
import plotly.subplots
import pprint
import psutil
import pytz
import re
import shutil
import subprocess
import sys
import threading
import time
import timezonefinder
import traceback
import warnings

import multiprocessing as mp
import numpy_financial as npf
import oemof.solph as solph
import pandas as pd
import pyomo.environ as po

from revoletion import blocks
from revoletion import checker
from revoletion import constraints
from revoletion import dispatch
from revoletion import economics as eco
from revoletion import logger as logger_fcs
from revoletion import scheduler
from revoletion import utils


class OptimizationError(Exception):
    pass


class OptimizationSuccessfulFilter(logging.Filter):
    def filter(self, record):
        # Filter out log messages from the root logger
        return not (record.name == 'root' and record.msg == 'Optimization successful...')


class SimulationRun:

    def __init__(self,
                 path_scenarios,
                 path_settings,
                 rerun=False,
                 rerun_infeasible=True,
                 execute=False):

        self.paths = {'revoletion': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'scenarios': path_scenarios,
                      'settings': path_settings}
        self.rerun = rerun
        self.rerun_infeasible = rerun_infeasible

        # region start runtime
        self.runtime_start = time.perf_counter()
        if not self.rerun:
            self.runtimestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
        else:
            # get timestamp from rerun directory name (for both absolute and relative (to settings output dir) paths)
            self.runtimestamp = '_'.join(os.path.basename(os.path.normpath(self.rerun)).split('_')[0:2])
        self.runtime_end = self.runtime_len = None
        # endregion

        # region get version information
        self.version_solph = solph.__version__
        self.version_revoletion = importlib.metadata.version('revoletion')

        try:  # todo additionally get commit hash of revoletion if possible
            self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()[0:6]
        except subprocess.CalledProcessError:
            self.commit_hash = 'unknown'
        # endregion

        self.name = pathlib.Path(self.paths['scenarios']).stem
        self.input_checker = checker.InputChecker(self)

        # region read and check settings
        self.settings = pd.read_csv(self.paths['settings'],
                                    index_col=[0])
        self.settings = self.settings.map(utils.infer_dtype)
        for key, value in self.settings['value'].items():
            if key.startswith('path_'):
                self.paths[''.join(key.split('_')[1:])] = value
            else:
                setattr(self, key, value)
        self.input_checker.check_settings()
        # endregion

        # region define paths
        if self.paths['input'] == 'example':
            self.paths['input'] = os.path.join(self.paths['revoletion'], 'example')
        elif self.paths['input'] == 'scenario_file_dir':
            self.paths['input'] = os.path.dirname(self.paths['scenarios'])
        elif os.path.isdir(self.paths['input']):
            pass  # no modification of path necessary
        else:
            raise NotADirectoryError(f'Input directory not found: {self.paths["input"]}')

        if self.paths['output'] in ['package', 'example']:
            self.paths['output'] = os.path.join(self.paths['revoletion'],
                                                'results',
                                                f'{self.runtimestamp}_{self.name}')
        elif os.path.isdir(self.path_output_data):
            self.paths['output'] = os.path.join(self.paths['output'],
                                                f'{self.runtimestamp}_{self.name}')  # no modification of path necessary
        else:
            raise NotADirectoryError(f'Output directory not found: {self.paths["output"]}')

        if not os.path.isdir(self.paths['output']):
            os.mkdir(self.paths['output'])

        self.paths['data_persist'] = os.path.join(self.paths['revoletion'], 'data')

        basename = f'{self.runtimestamp}_{self.name}'

        self.paths['summary_csv'] = os.path.join(self.paths['output'], f'{basename}_summary.csv')
        self.paths['summary_pkl'] = os.path.join(self.paths['output'], f'{basename}_summary.pkl')
        self.paths['status'] = os.path.join(self.paths['output'], f'{basename}_status.csv')
        self.paths['dump'] = os.path.join(self.paths['output'], f'{basename}_model.lp')
        self.paths['log'] = os.path.join(self.paths['output'], f'{basename}.log')
        # endregion

        # region read, copy and check scenario data
        self.scenario_data = pd.read_csv(self.paths['scenarios'],
                                         index_col=[0, 1],
                                         keep_default_na=False)
        self.scenario_data = self.scenario_data.sort_index(sort_remaining=True).map(utils.infer_dtype)
        self.scenario_names = [name for name in self.scenario_data.columns if not name.startswith('#')]

        if self.rerun:
            # only run scenarios which have not been optimized successfully (or were infeasible)
            self.scenario_status = pd.read_csv(os.path.join(self.paths['output'],
                                                            f'{self.runtimestamp}_'
                                                            f'{self.name}_'
                                                            f'scenarios_status.csv'),
                                               index_col=0)

            dont_rerun = ['successful', 'infeasible'] if self.rerun_infeasible else ['successful']
            scenarios_rerun = self.scenario_status[~self.scenario_status['status'].isin(dont_rerun)].index.to_list()
            self.scenario_names = [name for name in self.scenario_names if name in scenarios_rerun]

            # delete all temporary results of files which are rerun (happens if SimulationRun terminates unexpected)
            for scenario in self.scenario_names:
                for file in [f'{scenario}_summary_temp.csv',
                             f'{scenario}_results.csv',
                             f'{scenario}_graph.pdf',
                             f'{scenario}_graph.html']:
                    if os.path.isfile(os.path.join(self.paths['output'], file)):
                        os.remove(os.path.join(self.paths['output'], file))

            # reset status of scenarios to be run to 'queued'
            self.scenario_status.loc[self.scenario_names, ['status', 'exception', 'traceback']] = (
                    [['queued', pd.NA, pd.NA]] * len(self.scenario_names))
            self.scenario_status.to_csv(self.path_result_status_file, index=True)

        else:
            self.scenario_status = pd.DataFrame(index=self.scenario_names,
                                                data={'status': 'queued',
                                                      'exception': None,
                                                      'traceback': None}).rename_axis('scenario')
            self.copy_scenario_file()
        self.scenario_num = len(self.scenario_names)

        if self.scenario_num == 0:
            raise ValueError('No executable scenarios found in scenario file')
        # endregion

        # region calculate number of threads to use
        self.process = None
        if self.max_process_num == 'max':
            self.max_process_num = os.cpu_count()
        elif self.max_process_num == 'physical':
            self.max_process_num = psutil.cpu_count(logical=False)
        else:
            self.max_process_num = int(self.max_process_num)
        self.process_num = min(self.scenario_num, os.cpu_count(), self.max_process_num)

        if (len(self.scenario_names) <= 1 or self.process_num == 1) and self.parallel:
            # logger not defined yet, use print as logger definition needs to be done after process_num is defined
            print('Single scenario or process: Parallel mode not possible - switching to sequential mode')
            self.parallel = False
        # endregion

        # region define logger structure
        self.logger = logging.getLogger()
        log_formatter = logging.Formatter(f'%(levelname)-{len("WARNING")}s  '
                                          f'%(name)-{max([len(el) for el in list(self.scenario_names) + ["root"]])}s  '
                                          f'%(message)s')
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setFormatter(log_formatter)
        log_file_handler = logging.FileHandler(os.environ.get('LOGFILE', self.paths['log']))
        log_file_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_stream_handler)
        self.logger.addHandler(log_file_handler)

        # Adding the custom filter to prevent root logger messages
        log_stream_handler.addFilter(OptimizationSuccessfulFilter())
        log_file_handler.addFilter(OptimizationSuccessfulFilter())

        if self.parallel:
            log_stream_handler.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)
        else:
            if self.debugmode:
                self.logger.setLevel(os.environ.get('LOGLEVEL', 'DEBUG'))
                log_stream_handler.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))
                log_stream_handler.setLevel(logging.INFO)

        # plural extensions
        pe1 = 's' if self.scenario_num > 1 else ''
        pe2 = 'es' if self.process_num > 1 else ''

        mode = f'parallel mode with {self.process_num} process{pe2}' if self.parallel else 'sequential mode'
        self.logger.info(f'Global settings read - running {self.scenario_num} scenario{pe1} in {mode}')

        # make sure that uncaught errors (i.e. errors occurring outside simulate_scenario method) are logged to logfile
        sys.excepthook = self.handle_exception
        # endregion

        # integration levels at which power consumption is determined a priori
        self.apriori_lvls = ['uc', 'fcfs', 'equal', 'soc']

        self.generate_plots = True if self.show_plots or self.save_plots else False

        if execute:
            self.execute()

    def copy_scenario_file(self):
        target = os.path.join(self.paths['output'], f'{self.name}.csv')
        try:  # with metadata
            shutil.copy2(self.paths['scenarios'], target)
        except PermissionError:  # can happen if metadata is not writable, e.g. on network drives
            shutil.copyfile(self.paths['scenarios'], target)

    def execute(self):
        # parallelization activated in settings file
        if self.parallel:
            with mp.Manager() as manager:
                lock = manager.Lock()

                status_queue = manager.Queue()
                status_thread = threading.Thread(target=self.read_status_queue, args=(status_queue,))
                status_thread.start()

                log_queue = manager.Queue()
                log_thread = threading.Thread(target=logger_fcs.read_mplogger_queue, args=(log_queue,))
                log_thread.start()

                with mp.Pool(processes=self.process_num) as pool:
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
                self.execute_scenario(scenario_name)

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

        self.logger.error('Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback))

    def join_results(self):

        files = [filename for filename in os.listdir(self.paths['output']) if filename.endswith('_summary_temp.pkl')]

        scenario_frames = []

        for file in files:
            # only add results of successful scenarios to summary
            if self.scenario_status.loc[file.removesuffix('_summary_temp.pkl'), 'status'] != 'successful':
                continue
            file_path = os.path.join(self.paths['output'], file)
            file_results = pd.read_pickle(file_path)
            scenario_frames.append(file_results)

        if len(scenario_frames) > 0:  # empty scenario_frames, if all scenarios fail during initialization
            joined_results = pd.concat(scenario_frames, axis=1)
            joined_results.loc[('run', 'runtime_end'), :] = self.runtime_end
            joined_results.loc[('run', 'runtime_len'), :] = self.runtime_len
            if self.rerun and os.path.isfile(os.path.join(self.path_result_summary_file_pkl)):
                results_summary_prev = pd.read_pickle(os.path.join(self.path_result_summary_file_pkl))
                joined_results = pd.concat([results_summary_prev, joined_results], axis=1)
            # apply same order of scenarios as in scenario input file
            joined_results = joined_results[[col for col in self.scenario_data.columns if col in joined_results.columns]]
            joined_results.to_csv(self.paths['summary_csv'], index=True)
            joined_results.to_pickle(self.paths['summary_pkl'])
            self.logger.info('Result summary file created')

        # deletion loop at the end to avoid premature execution of results in case of error
        for file in files:
            file_path = os.path.join(self.paths['output'], file)
            os.remove(file_path)

    def read_status_queue(self, queue):
        while True:
            status_msg = queue.get()
            if status_msg is None:  # Exit signal
                break
            self.update_scenario_status(status_msg)

    def execute_scenario(self,
                         name: str,
                         log_queue=None,
                         status_queue=None,
                         lock=None):
        # this method is necessary as running Scenario() directly from the starmap fails as Scenario object contains
        # objects which cannot be pickled.
        try:
            Scenario(name=name,
                     run=self,
                     log_queue=log_queue,
                     lock=lock,
                     status_queue=status_queue)
        except:
            self.logger.info(f'Error occurred in scenario {name}')

    def trigger_scenario_status_update(self, queue, status_msg):
        if queue is not None:
            queue.put(status_msg)
        else:
            self.update_scenario_status(status_msg)

    def update_scenario_status(self, status_msg):
        for col in [key for key, value in status_msg.items() if key != 'scenario' and value is not None]:
            self.scenario_status.loc[status_msg['scenario'], col] = status_msg[col]
        self.scenario_status.to_csv(self.paths['status'],
                                    index=True)


class Scenario:

    def __init__(self, name, run, log_queue, lock, status_queue=None):
        self.name = name
        self.run = run
        self.logger = logger_fcs.setup_logger(name, log_queue, self.run)
        self.logger.propagate = False
        self.status_queue = status_queue

        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            # Force warnings in custom formatting and ignore warnings about infeasible or unbounded optimizations
            if not 'Optimization ended with status warning and termination condition' in str(message):
                self.logger.warning(f'{category.__name__}: {message} (in {filename}, line {lineno})')

        warnings.showwarning = custom_warning_handler

        self.run.trigger_scenario_status_update(queue=self.status_queue,
                                                status_msg={'scenario': self.name,
                                                            'status': 'started'})

        # General Information --------------------------------

        self.runtime_start = time.perf_counter()
        self.runtime_end = None  # placeholder
        self.runtime_len = None  # placeholder

        self.worker = mp.current_process()

        msg_parallel = (f' on {self.worker.name.ljust(18)} - Parent: {self.worker._parent_name}'
                        if hasattr(self.worker, '_parent_name') else '')
        self.logger.info(f'Scenario initialized{msg_parallel}')

        self.parameters = self.run.scenario_data[self.name]
        for key, value in self.parameters.loc['scenario', :].items():
            setattr(self, key, value)  # this sets all the parameters defined in the csv file

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

        self.sim_duration = (pd.Timedelta(days=self.sim_duration) if isinstance(self.sim_duration, (float, int))
                             else pd.Timedelta(self.sim_duration)).floor(self.timestep)
        self.sim_extd_duration = self.sim_duration
        self.sim_endtime = self.starttime + self.sim_duration
        self.sim_extd_endtime = self.sim_endtime
        self.prj_duration_yrs = self.prj_duration
        self.prj_endtime = self.starttime + pd.DateOffset(years=self.prj_duration)
        self.prj_duration = self.prj_endtime - self.starttime  # takes leap years into account

        if self.strategy == 'rh':
            self.len_ph = pd.Timedelta(hours=self.len_ph)
            self.len_ch = pd.Timedelta(hours=self.len_ch)
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
            self.exception = f'Optimization strategy "{self.strategy}" unknown'
            raise ValueError(self.exception)

        # generate a datetimeindex for the energy system model to run on
        self.dti_sim = pd.date_range(start=self.starttime, end=self.sim_endtime, freq=self.timestep, inclusive='left')
        # extended index covers PHs that are not truncated after simulation end time
        self.dti_sim_extd = pd.date_range(start=self.starttime, end=self.sim_extd_endtime, freq=self.timestep,
                                          inclusive='left')

        # generate variables for calculations
        self.timestep_td = pd.Timedelta(self.dti_sim_extd.freq)
        self.timestep_hours = self.timestep_td.total_seconds() / 3600
        self.sim_yr_rat = self.sim_duration / pd.Timedelta(days=365)  # no leap years
        self.sim_prj_rat = self.sim_duration / self.prj_duration

        # get holidays during simulation timeframe
        years = range(min(self.dti_sim_extd).year, max(self.dti_sim_extd).year + 1)
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

        # prepare for system graph saving later on
        self.path_system_graph_file = os.path.join(
            self.run.paths['output'],
            f'{self.run.runtimestamp}_{self.run.name}_{self.name}_system_graph.pdf')

        # prepare for dispatch plot saving later on
        self.plot_file_path = os.path.join(self.run.paths['output'], f'{run.runtimestamp}_'
                                                                     f'{run.name}_'
                                                                     f'{self.name}.html')

        # prepare for cumulative result saving later on
        self.result_summary = pd.Series(index=pd.MultiIndex.from_tuples(tuples=[], names=['block', 'key']))
        self.path_result_summary_tempfile = os.path.join(self.run.paths['output'],
                                                         f'{self.name}_summary_temp.pkl')

        self.result_timeseries = pd.DataFrame(index=utils.extend_dti(self.dti_sim_extd),
                                              columns=pd.MultiIndex.from_tuples(tuples=[],
                                                                                names=['block', 'timeseries']))
        self.path_result_ts_file = os.path.join(
            self.run.paths['output'],
            f'{self.run.runtimestamp}_{self.run.name}_{self.name}_results_ts.csv')

        self.exception = None  # placeholder for possible infeasibility

        # Energy System Blocks --------------------------------
        # initialize variable to store initial investment costs given in scenario definition
        self.discount_factors = pd.DataFrame(index=range(self.prj_duration_yrs),
                                             columns=['beginning', 'mid', 'end'],
                                             data={occ: eco.discount(future_value=1,
                                                                     periods=np.arange(1, self.prj_duration_yrs + 1),
                                                                     discount_rate=self.wacc,
                                                                     occurs_at=occ)
                                                   for occ in ['beginning', 'mid', 'end']},
                                             dtype='float64')
        self.aggregator = eco.EconomicAggregator(name='scenario', block=None, scenario=self)

        # add SystemCore to blocks ensuring SystemCore is the first component to be built
        self.blocks = {**{'core': 'SystemCore'}, **self.blocks}

        # create all block objects defined in the scenario DataFrame under "scenario/blocks" as a dict
        self.storage_blocks = {}
        self.fleets = {}
        self.renewable_sources = {}
        self.blocks = self.create_block_objects()

        if self.invest_max is not None and self.invest_max < self.capex_init_existing:
            raise ValueError(f'Initial investment costs of {self.capex_init_existing:.2f} {self.currency} '
                             f'exceed maximum investment limit of {self.invest_max} {self.currency}')

        # Execute commodity system discrete event simulation
        # can only be started after all blocks have been initialized, as the different systems depend on each other.
        # if any([fleet.data_source in ['demand', 'usecases'] for fleet in self.fleets.values()]):
        #     dispatch.execute_des(self, self.run)  # todo implement des across fleets

        # todo adapt to new fleet structure
        # # check example parameter configuration of rulebased charging for validity
        # if fleet_unlim := [fleet for fleet in self.fleets.values() if
        #                 (fleet.mode_scheduling in self.run.apriori_lvls)
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

        self.scheduler = None
        # todo adapt to new fleet structure
        # if any([fleet for fleet in self.fleets.values() if fleet.mode_scheduling in self.run.apriori_lvls]):
        #     self.scheduler = scheduler.AprioriPowerScheduler(scenario=self)

        # Result variables --------------------------------
        self.figure = None  # placeholder for plotting

        self.cashflows = pd.DataFrame()

        self.objective_opt = None  # placeholder for objective optimised by the optimizer. Not used for Rolling Horizon

        self.print_results_msgs = []
        self.plot_traces = {'powers': [],
                            'states': []}

        # Result variables - Energy
        self.energies = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples=[('renewable', 'act'),],
                                                                     names=['block', 'key']),
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,
                                     dtype=float)

        self.e_eta = None
        self.renewable_share = None

        self.lcoe_total = self.lcoe_wocs = None
        self.npv = self.irr = self.mirr = None

        self.logger.debug(f'Scenario initialization completed')

        self.run.trigger_scenario_status_update(queue=self.status_queue,
                                                status_msg={'scenario': self.name,
                                                            'status': 'fully initialized'})

        # region execute scenario
        for block in self.blocks.values():
            block.pre_scenario()

        try:
            for horizon_index in range(self.nhorizons):  # Inner optimization loop over all prediction horizons
                PredictionHorizon(index=horizon_index,
                                  scenario=self)

                self.run.trigger_scenario_status_update(queue=self.status_queue,
                                                        status_msg={'scenario': self.name,
                                                                    'status': f'completed horizon '
                                                                              f'{horizon_index + 1} out of '
                                                                              f'{self.nhorizons}'})
            self.run.trigger_scenario_status_update(queue=self.status_queue,
                                                    status_msg={'scenario': self.name,
                                                                'status': 'successful'})

        except Exception as e:
            # Scenario has failed -> store scenario name to dataframe containing failed scenarios
            status = 'infeasible' if isinstance(e, OptimizationError) else 'failed'
            self.run.trigger_scenario_status_update(queue=self.status_queue,
                                                    status_msg={'scenario': self.name,
                                                                'status': status,
                                                                'exception': str(e),
                                                                'traceback': traceback.format_exc()})

            # show error message and traceback in console; suppress traceback if problem was infeasible or unbounded
            self.logger.error(msg=f'{str(e)} - continue on next scenario',
                              exc_info=(not isinstance(e, OptimizationError)))

            self.exception = str(e)
            self.end_timing()  # ToDo: does timing end here? Should that better be called at the end of result writing?

        finally:
            try:
                for block in self.blocks.values():
                    block.post_scenario()
                self.aggregator.post_scenario()

                self.calc_meta_results()
                self.save_result_summary()

                if self.run.save_results_timeseries:
                    self.result_timeseries.to_csv(self.path_result_ts_file)

                if self.run.print_results:
                    for msg in self.print_results_msgs:
                        self.logger.info(msg)

                if self.run.generate_plots:
                    self.generate_plots()
                    if self.run.save_plots:
                        self.figure.write_html(self.plot_file_path)
                    if self.run.show_plots:
                        self.figure.show(renderer='browser')

            except Exception as e:
                self.logger.error(e, exc_info=True)

        logging.shutdown()
        # endregion

    def calc_meta_results(self):

        energy_pro = self.energies.loc[(self.energies['sim'] > 0) &
                                       (self.energies.index.get_level_values(1) == 'total'), :].sum()

        energy_del = abs(self.energies.loc[(self.energies['sim'] < 0) &
                                           (self.energies.index.get_level_values(1) == 'total'), :].sum())

        # pandas creates a RuntimeWarning at division by 0 -> try/except does not work
        if energy_pro['sim'] == 0:
            self.logger.warning(f'Core efficiency calculation: division by zero')
        else:
            self.e_eta = energy_del['sim'] / energy_pro['sim']

        if energy_pro['sim'] == 0:
            self.logger.warning(f'Renewable share calculation: division by zero')
        else:
            self.renewable_share = self.energies.loc[('renewable', 'act'), 'sim'] / energy_pro['sim']

        if energy_del['dis'] == 0:
            self.logger.warning(f'LCOE calculation: division by zero')
        else:
            self.lcoe_total = self.aggregator.totex['dis'] / energy_del['dis']
            self.lcoe_wocs = ((self.aggregator.totex['dis'] -
                               # ToDo: check whether calculation of totex['dis'] of fleets is correct
                               sum([fleet.aggregator.totex['dis'] for fleet in self.fleets.values()])) /
                              energy_del['dis'])

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

    def create_block_objects(self):
        class_dict = self.blocks
        objects = {}
        for name, class_name in class_dict.items():
            class_obj = getattr(blocks, class_name, None)
            if class_obj is not None and isinstance(class_obj, type):
                objects[name] = class_obj(name, self)
            else:
                raise ValueError(f'Class "{class_name}" not found in blocks.py file - '
                                 f'Check for typos or add class.')
        return objects

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
                                            f'{self.run.name} - '
                                            f'Scenario: {self.name}')
        if self.strategy == 'rh':
            self.figure.update_layout(title=f'Rolling Horizon Results - '
                                            f'{self.run.name} - '
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
            pd.Series(index=['blocks'], data=str({key: value.classname for key, value in self.blocks.items()})),
            # get energies dataframes results for scenario.result_summary
            utils.get_dataframe_results(df=self.energies, name_prefix='energy'),
            # get economic results for scenario.result_summary
            self.aggregator.write_result_summary()])

        # apply MultiIndex
        results_scenario.index = pd.MultiIndex.from_tuples(tuples=[('scenario', key) for key in results_scenario.index],
                                                           names=['block', 'key'])

        # write results from run and scenario to result_summary
        self.result_summary = pd.concat([results_run, results_scenario, self.result_summary])

        # convert result_summary to DataFrame and save to temporary file
        pd.DataFrame(self.result_summary, columns=[self.name]).to_pickle(self.path_result_summary_tempfile)


class PredictionHorizon:

    def __init__(self, index, scenario):

        self.index = index
        self.scenario = scenario

        self.results = None
        self.meta_results = None

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
        self.dti_ch = pd.date_range(start=self.starttime, end=self.ch_endtime, freq=self.scenario.timestep, inclusive='left')

        # if apriori power scheduling is necessary, calculate power schedules:
        if self.scenario.scheduler:
            self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                      f'Calculating power schedules for commodities with rulebased charging strategies')
            self.scenario.scheduler.calc_ph_schedule(self)
        # endregion

        # region build energy system model
        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Building oemof model')

        self.es = solph.EnergySystem(timeindex=self.dti_ph,
                                     infer_last_interval=True)  # initialize energy system model instance

        for block in self.scenario.blocks.values():
            block.pre_horizon(self)

        self.scenario.logger.debug(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                   f'Model build completed')
        # endregion

        # region draw graph of energy model
        if self.index == 0 and self.scenario.run.save_system_graphs:  # first horizon - create graph of energy system
            # Initialize the graph with the filepath without extension
            dot = graphviz.Digraph(filename=os.path.splitext(self.scenario.path_system_graph_file)[0])

            # Define drawing styles for certain components
            dot.node('Bus', shape='rectangle', fontsize='10', color='red')
            dot.node('Sink', shape='trapezium', fontsize='10')
            dot.node('Source', shape='invtrapezium', fontsize='10')
            dot.node('Storage', shape='rectangle', style='dashed', fontsize='10', color='green')

            busses = []
            # draw a node for each of the network's components.
            for nd in self.es.nodes:
                if isinstance(nd, solph.Bus):
                    dot.node(nd.label,
                             shape='rectangle',
                             fontsize='10',
                             fixedsize='shape',
                             width='2.4',
                             height='0.6',
                             color='red')
                    # keep the bus reference for drawing edges later
                    busses.append(nd)
                elif isinstance(nd, solph.components.Sink):
                    dot.node(nd.label, shape='trapezium', fontsize='10')
                elif isinstance(nd, solph.components.Source):
                    dot.node(nd.label, shape='invtrapezium', fontsize='10')
                elif isinstance(nd, solph.components.Converter):
                    dot.node(nd.label, shape='rectangle', fontsize='10')
                elif isinstance(nd, solph.components.GenericStorage):
                    dot.node(nd.label, shape='rectangle', style='dashed', fontsize='10', color='green')
                else:
                    self.scenario.logger.debug(f'System Node {nd.label} - Type {type(nd)} not recognized')

            # draw the edges between the nodes based on each bus inputs/outputs
            for bus in busses:
                for component in bus.inputs:
                    # draw an arrow from the component to the bus
                    dot.edge(component.label, bus.label)
                for component in bus.outputs:
                    # draw an arrow from the bus to the component
                    dot.edge(bus.label, component.label)

            try:
                dot.render(cleanup=True)
            except Exception as e:  # inhibiting failing renderer from stopping model execution
                self.scenario.logger.warning(f'System graph rendering failed - Traceback: {e}')
        # endregion

        # region build optimization problem
        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Building optimization problem from oemof model')

        self.model = solph.Model(self.es, debug=self.scenario.run.debugmode)
        self.constraints.apply_constraints(model=self.model)

        if self.scenario.run.dump_model and self.scenario.strategy != 'rh':
            self.model.write(self.scenario.run.path_dump_file, io_options={'symbolic_solver_labels': True})
        # endregion

        # region solve optimization problem
        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - '
                                  f'Model built, starting optimization')
        results = self.model.solve(solver=self.scenario.run.solver, solve_kwargs={'tee': self.scenario.run.debugmode})

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

        if self.scenario.run.debugmode:
            self.meta_results = solph.processing.meta_results(self.model)
            pprint.pprint(self.meta_results)

        # free up RAM
        del self.model

        for block in self.scenario.blocks.values():
            block.post_horizon(self)
        # endregion
