#!/usr/bin/env python3

from dataclasses import dataclass, field
from functools import cached_property
import geopy
import holidays
import importlib.resources
import logging
import math
from pathlib import Path
import numpy as np
import plotly.subplots
import pprint
import pytz
import time
from typing import List

import timezonefinder
import traceback
import warnings
import webbrowser

import multiprocessing as mp
import numpy_financial as npf
import oemof.solph as solph
import pandas as pd
import plotly.graph_objects as go
import pyomo.environ as po

from . import blocks
from . import constraints
from . import dispatch
from . import economics as eco
from . import logger as logger_fcs
from . import scheduler
from . import utils

import revoletion.data


class OptimizationError(Exception):
    pass


@dataclass
class Location:
    latitude: float
    longitude: float
    _lock: mp.Lock = field(repr=False)
    _logger: logging.Logger = field(repr=False)

    timezone: pytz.BaseTzInfo = field(init=False,
                                      default_factory=lambda: pytz.timezone('Europe/Berlin'))
    country: str = field(init=False,
                         default='DE')
    state: str = field(init=False,
                       default='BY')

    def __post_init__(self):

        tzfinder = timezonefinder.TimezoneFinder()
        self.timezone = pytz.timezone(tzfinder.certain_timezone_at(lat=self.latitude, lng=self.longitude))

        geolocator = geopy.geocoders.Nominatim(user_agent=f'location_finder')
        try:
            location = None
            if self._lock is None:  # sequential
                location = geolocator.reverse(query=(self.latitude, self.longitude),
                                              language="en",
                                              exactly_one=True)
            else:  # parallel
                with self._lock:
                    time.sleep(2)  # max 1 request per second --> wait for 2 seconds to make sure to not avoid the limit
                    location = geolocator.reverse(query=(self.latitude, self.longitude),
                                                  language="en",
                                                  exactly_one=True)
            if location:
                self.country, self.state = location.raw['address']['ISO3166-2-lvl4'].split('-')
        except geopy.exc.GeocoderUnavailable:
            self._logger.warning(f'Connection to Geocoder failed. '
                                 f'Using default country ({self.country}) and state ({self.state}).')


@dataclass
class TimeSettings:
    start: pd.Timestamp
    _timestep: pd.Timedelta
    end: pd.Timestamp = field(default=None)
    duration: pd.Timedelta = field(default=None)

    def __post_init__(self):
        if (self.end is None and self.duration is None) or (self.end is not None and self.duration is not None):
            raise ValueError('Exactly one of the parameters "end" or "duration" must be provided.')

        elif self.duration is None:
            self.duration = (self.end - self.start).floor(self._timestep)
        elif self.end is None:
            self.duration = self.duration.floor(self._timestep)
        # always recalculate end to ensure consistency
        self.end = self.start + self.duration

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start,
                             end=self.end,
                             freq=self._timestep,
                             inclusive='left')

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start,
                             end=self.end,
                             freq=self._timestep,
                             inclusive='both')


@dataclass
class SimulationTimes:
    _scenario: 'Scenario'

    sim: TimeSettings = field(init=False)
    eval: TimeSettings = field(init=False)
    prj: TimeSettings = field(init=False)

    def __post_init__(self):
        starttime = self._scenario.starttime  # ToDo: reformat starttime
        starttime = starttime if len(starttime) > 10 else starttime + ' 00:00'
        starttime = pd.to_datetime(starttime, format='%d.%m.%Y %H:%M').floor(self._scenario.timestep).tz_localize(
            self._scenario.location.timezone)

        timestep = utils.convert2timedelta(self._scenario.timestep,
                                           unit='minute')

        self.sim = TimeSettings(start=starttime,
                                _timestep=timestep,
                                end=(pd.to_datetime(self._scenario.sim_endtime, format='%d.%m.%Y %H:%M')
                                     .floor(timestep)
                                     .tz_localize(self._scenario.location.timezone)
                                     if self._scenario.sim_endtime is not None else None),
                                duration=utils.convert2timedelta(value=self._scenario.sim_duration,
                                                                       unit='day'))
        self.eval = TimeSettings(start=starttime,
                                 _timestep=timestep,
                                 end=(pd.to_datetime(self._scenario.sim_endtime, format='%d.%m.%Y %H:%M')
                                      .floor(timestep)
                                      .tz_localize(self._scenario.location.timezone)
                                      if self._scenario.sim_endtime is not None else None),
                                 duration=utils.convert2timedelta(value=self._scenario.sim_duration,
                                                                        unit='day'))
        self.prj = TimeSettings(start=starttime,
                                _timestep=timestep,
                                end=starttime + pd.DateOffset(years=self._scenario.prj_duration))

        for attr in ['starttime', 'sim_endtime', 'sim_duration', 'prj_duration']:
            if hasattr(self._scenario, attr):
                delattr(self._scenario, attr)
        delattr(self, '_scenario')


@dataclass
class Timestep:
    str: str
    hours: float = field(init=False,
                         default=None)

    td: pd.Timedelta = field(init=False,
                             default=None)

    def __post_init__(self):
        self.td = pd.Timedelta(self.str)
        self.hours = self.td.total_seconds() / 3600


@dataclass
class SimulationPaths:
    """
    Contains all paths relevant for the simulation run.
    scenario: Path to the scenario file
    input: Path to the input data directory (default: same directory as scenario file)
    output: Path to the output directory (default: current working directory/results)
    rerun: Path to the rerun directory (can also contain the string "latest")

    data_persist: Path to the persistent data directory within the revoletion package
    summary_csv: Path to the summary CSV file
    summary_pkl: Path to the summary pickle file
    status: Path to the status csv file
    dump: Path to the pyomo model
    log: Path to the log file
    """

    scenario: Path | str
    input: Path | str = None
    output: Path | str = None
    rerun: Path | str = None

    def __post_init__(self):
        self.scenario = Path(self.scenario)
        if self.input is None:
            self.input = self.scenario.parent
        else:
            self.input = Path(self.input)
        if self.output is None:
            self.output = Path.cwd() / 'results'
        else:
            self.output = Path(self.output)
        if self.rerun is None:
            self.output = self.output / Path(f'{pd.Timestamp.now().strftime("%y%m%d_%H%M%S")}_{self.scenario.stem}')
        elif self.rerun == 'latest':
            # get all directories in the output directory already sorted alphabetically
            directories = [d for d in sorted(self.output.iterdir()) if d.is_dir()]

            # return the last directory (if any)
            if directories:
                self.output = directories[-1]
            else:
                raise NotADirectoryError(f'No previous runs available in specified output directory {self.output}')
        else:
            self.rerun = Path(self.rerun)
            if self.rerun.is_absolute():
                self.output = self.rerun
            else:
                self.output = self.output / self.rerun.name

        # ensure all paths are absolute
        self.scenario = self.scenario.resolve()
        self.input = self.input.resolve()
        self.output = self.output.resolve()

        # ensure that all paths exist
        if not self.scenario.is_file():
            raise FileNotFoundError(f'Scenario file not found: {self.scenario}')
        if not self.input.is_dir():
            raise NotADirectoryError(f'Input directory path not interpretable: {self.input}')
        if not self.rerun:
            self.output.mkdir(parents=True)  # create parents if missing -> relevant for default "results"
        else:
            if not self.output.is_dir():
                raise NotADirectoryError(f'Specified rerun directory {self.output} does not exist.')

    def create_result_path(self,
                           suffix: str) -> Path:
        return self.output / f'{self.output.name}_{suffix}'

    @property
    def data_persist(self) -> Path:
        with importlib.resources.as_file(importlib.resources.files(revoletion.data)) as data_dir:
            return data_dir

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
    solver: str = 'gurobi'
    n_processes: int = 1
    largescalemode: bool = False
    debugmode: bool = False
    rerun_infeasible: bool = True
    key_solcast_api: str = None


@dataclass
class PlotTraces:
    _plot_traces: List[go.Scatter] = field(default_factory=list,
                                           repr=False)
    _secondary_y: List[bool] = field(default_factory=list,
                                     repr=False)

    def append(self,
               plot_line: go.Scatter,
               secondary_y: bool = False) -> None:
        self._plot_traces.append(plot_line)
        self._secondary_y.append(secondary_y)

    def extend(self,
                plot_lines: List[go.Scatter],
                secondary_ys: List[bool] = None) -> None:

        if not secondary_ys:
            secondary_ys = [False] * len(plot_lines)

        self._plot_traces.extend(plot_lines)
        self._secondary_y.extend(secondary_ys)

    @property
    def plot_lines(self) -> List[go.Scatter]:
        return self._plot_traces

    @property
    def secondary_ys(self) -> List[bool]:
        return self._secondary_y


class Scenario:

    def __init__(self,
                 paths: SimulationPaths,
                 settings: SimulationSettings,
                 run_execution: bool = False,
                 name: str = None,  # will be set to the stem of the scenario filename for single scenario execution
                 parameters: pd.Series = None,
                 log_queue: mp.Queue = None,
                 lock: mp.Lock = None,
                 status_update: 'SimulationRun.trigger_scenario_status_update' = None,
                 status_queue: mp.Queue = None):

        self.paths = paths
        self.settings = settings

        if run_execution:
            if name is None:
                raise ValueError('Scenario name must be provided when run_execution is True')
            if parameters is None:
                raise ValueError('Parameters must be provided when run_execution is True')

        self.name = name
        self.parent = None  # attribute needs to exist for economic aggregation

        if not run_execution:
            self.logger = logger_fcs.get_root_logger(paths=self.paths,
                                                     settings=self.settings,
                                                     len_scn_max=len('root'),
                                                     )

            # read scenario file
            if self.paths.scenario.suffix == '.csv':
                self.parameters = pd.read_csv(self.paths.scenario,
                                              index_col=[0, 1],
                                              keep_default_na=False)
                self.parameters = self.parameters.sort_index(sort_remaining=True).map(utils.infer_dtype)
            elif self.paths.scenario.suffix == '.pkl':
                self.parameters = pd.read_pickle(self.paths.scenario)
            else:
                raise ValueError('Scenario file specified in SimulationPaths object is neither CSV nor PKL file.')

            # check if scenario file contains more than one scenario (then it has to be run via a SimulationRun)
            if len(self.parameters.columns) > 1:
                raise ValueError('More than one scenario detected. Provide a single column CSV or PKL file.')

            if self.name is None:
                self.name = self.parameters.columns[0]

            # convert DataFrame to Series
            self.parameters = self.parameters.iloc[:, 0]

        else:
            # Define logger
            if log_queue is not None:
                self.logger = logger_fcs.get_process_logger_parallel(name=self.name,
                                                                     settings=self.settings,
                                                                     log_queue=log_queue,
                                                                     )
            else:
                self.logger = logger_fcs.get_process_logger_sequential(name=self.name,
                                                                       settings=self.settings,
                                                                       )

            # Set given parameters as attribute
            if not isinstance(parameters, pd.Series):
                raise ValueError('Parameters of type pd.Series must be provided to scenario when run_execution is True')
            self.parameters = parameters

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

        self.runtime = utils.RunTime()

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

        self.location = Location(latitude=self.latitude,
                                 longitude=self.longitude,
                                 _lock=lock,
                                 _logger=self.logger)

        for param in ['latitude', 'longitude']:
            delattr(self, param)

        self.prj_duration_yrs = self.prj_duration
        self.times = SimulationTimes(_scenario=self)
        self.timestep = Timestep(self.timestep)

        # generate variables for calculations
        self.sim_yr_rat = self.times.sim.duration / pd.Timedelta(days=365)  # no leap years
        self.sim_prj_rat = self.times.sim.duration / self.times.prj.duration

        if self.strategy == 'rh':
            self.len_ph = utils.convert2timedelta(self.len_ph, unit='hour').floor(self.timestep.td)
            self.len_ch = utils.convert2timedelta(self.len_ch, unit='hour').floor(self.timestep.td)
        elif self.strategy in ['go']:
            self.len_ph = self.times.sim.duration
            self.len_ch = self.times.sim.duration
        else:
            raise ValueError(f'Optimization strategy "{self.strategy}" unknown')

        if self.len_ph == self.timestep.td:
            raise ValueError('Single timestep optimization not possible. Adjust simulation duration, timestep or '
                             'prediction horizon length / truncate_ph (for RH only)')

        self.nhorizons = math.ceil(self.times.sim.duration / self.len_ch)  # number of timeslices to run
        if not self.truncate_ph:
            # if PH is not truncated, the end of the last PH may be later than the end of the evaluation period
            self.times.sim = TimeSettings(start=self.times.sim.start,
                                          _timestep=self.timestep.td,
                                          duration=(self.len_ch * (self.nhorizons - 1) + self.len_ph))

        # get holidays during simulation timeframe
        years = range(min(self.times.eval.dti_extd).year, max(self.times.eval.dti_extd).year + 1)
        try:
            self.holiday_dates = sorted(
                getattr(holidays, self.location.country)(years=years,
                                                         state=self.location.state))
        except:  # not for all countries the states are available (e.g. France)
            try:
                self.holiday_dates = sorted(
                    getattr(holidays, self.location.country)(years=years))
                self.logger.warning(f'Holidays for state {self.location.state} not available. '
                                    f'Country-wide holidays for {self.location.country} are used instead.')
            except AttributeError:  # not all countries worldwide are available
                self.holiday_dates = []
                self.logger.warning(f'Holidays for country {self.location.country} not available. '
                                    f'No public holidays are considered in this scenario.')

        # region set air temperature
        temp_air = pd.Series(index=self.times.sim.dti,
                             dtype=float)

        if isinstance(self.temp_air, (float, int)):
            temp_air[:] = self.temp_air
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
                                                      scenario=self).iloc[:, 0]
        else:
            self.logger.warning(f'Specified argument for scenario parameter temp_air ({self.temp_air}) not found - '
                                f'Using default of 25 °C')
            temp_air[:] = 25
            self.temp_air = temp_air
        # endregion

        # region initialize result variables
        self.periods_prj = np.arange(0, self.prj_duration_yrs)
        self.periods_prj_extd = np.arange(0, self.prj_duration_yrs + 1)  # add. year for salvage values
        self.discount_factors = pd.DataFrame(index=self.periods_prj_extd,
                                             columns=['beginning', 'mid', 'end'],
                                             data={occ: eco.EcoTools.discount(future_value=1,
                                                                              periods=self.periods_prj_extd + 1,
                                                                              discount_rate=self.wacc,
                                                                              occurs_at=occ)
                                                   for occ in ['beginning', 'mid', 'end']},
                                             dtype='float64')

        self.aggregator = eco.EcoAggregator(name='scenario',
                                            scenario=self)
        self.capex_preexisting_considered = 0

        self.block_registry = dict()

        # Define priorities of blocks to ensure correct initialization order
        priority_default = 3
        priority_blocks = {'SystemCore': 0,  # always first -> ac and dc bus required for all other ElectricBlocks
                           'ThermalCore': 1,  # holds the busses to connect thermal components
                           'PVSource': 2,  # holds temperature and wind data -> required by StorageBlock and WindSource
                           }

        for name, class_name in sorted({'core': 'SystemCore',
                                        'thcore': 'ThermalCore',
                                        **self.blocks}.items(),
                                       key=lambda item: priority_blocks.get(item[1], priority_default),):
            class_obj = getattr(blocks, class_name, None)
            if class_obj is not None and isinstance(class_obj, type):
                class_obj(name, self)
            else:
                raise ValueError(f'Class "{class_name}" not found in blocks.py file - '
                                 f'Check for typos or add class.')

        if self.invest_max is not None and self.invest_max < self.capex_preexisting_considered:
            raise ValueError(f'Initial investment costs of {self.capex_preexisting_considered:.2f} {self.currency} '
                             f'exceed maximum investment limit of {self.invest_max} {self.currency}')

        self.objective_opt = None  # unused for rh strategy
        self.energies = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples=[('renewable', 'act'),
                                                                             ('sources', 'pro'),
                                                                             ('sinks', 'del')],
                                                                     names=['block', 'key']),
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,
                                     dtype=float)

        # Define object to store all traces for plotting
        self.plot_traces = PlotTraces()

        self.result_messages = []
        self.result_summary = []
        self.result_timeseries = []

        self.e_eta = None
        self.renewable_share = None
        self.lcoe_total = self.lcoe_wocs = None
        self.npc = self.npv = self.irr = self.mirr = None
        # endregion

        # region preexecution
        self.dispatch_environment = dispatch.DispatchEnvironment(scenario=self)

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

        finally:  # save results up to exception - valuable in RH strategy

            for block in self.block_registry.get('TopLevelBlock', {}).values():
                block.post_scenario()
            self.aggregator.aggregate()

            self.calc_meta_results()

            if not self.settings.largescalemode:
                self.result_timeseries = pd.concat(self.result_timeseries, axis=1)
                self.result_timeseries.to_csv(self.paths.create_result_path(suffix=f'{self.name}_results_ts.csv'))
                for msg in self.result_messages:
                    self.logger.info(msg)
                self.generate_and_save_plot()

            self.runtime.stop()
            self.logger.info(f'Scenario finished - runtime {self.runtime.duration:.2f} s')

            self.save_result_summary()

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
            self.lcoe_total = self.aggregator.totex.dis / self.energies.loc[('sinks', 'del'), 'dis']
            self.lcoe_wocs = ((self.aggregator.totex.dis -
                               # ToDo: check whether calculation of totex['dis'] of fleets is correct
                               sum([fleet.aggregator.totex.dis for fleet in self.block_registry.get('Fleet', {}).values()])) /
                              self.energies.loc[('sinks', 'del'), 'dis'])

        self.npc = self.aggregator.totex.dis
        self.npv = self.aggregator.value.dis
        self.irr = npf.irr(self.aggregator.value.cashflows)
        self.mirr = npf.mirr(self.aggregator.value.cashflows, self.wacc, self.wacc)

        # print basic results
        self.logger.info(f'NPC {f"{self.npc:,.2f}" if pd.notna(self.npc) else "-"} {self.currency} | '
                         f'NPV {f"{self.npv:,.2f}" if pd.notna(self.npv) else "-"} {self.currency} | '
                         f'LCOE {f"{self.lcoe_wocs * 1e5:,.2f}" if pd.notna(self.lcoe_wocs) else "-"} {self.currency}-ct/kWh | '
                         f'mIRR {f"{self.mirr * 100:,.2f}" if pd.notna(self.mirr) else "-"} %')

    def generate_and_save_plot(self):

        figure = plotly.subplots.make_subplots(specs=[[{'secondary_y': True}]])

        figure.add_traces(self.plot_traces.plot_lines,
                          secondary_ys=self.plot_traces.secondary_ys)

        if self.strategy == 'go':
            title = f'Global Optimum Results - {self.paths.output.name} - Scenario: {self.name}'
        elif self.strategy == 'rh':
            title = (f'Rolling Horizon Results - {self.paths.output.name} - Scenario: {self.name} - '
                     f'PH: {self.len_ph}h - CH: {self.len_ch}h')
        else:
            title = f'Results - {self.paths.output.name} - Scenario: {self.name}'

        linecolor = 'gray'
        gridcolor = 'gray'

        figure.update_layout(
            title=title,
            plot_bgcolor='white',
            xaxis=dict(
                title='Local Time',
                showgrid=True,
                linecolor=linecolor,
                gridcolor=gridcolor,
            ),
            yaxis=dict(
                title='Power in W',
                showgrid=True,
                linecolor=linecolor,
                gridcolor=gridcolor,
            ),
            yaxis2=dict(
                title='State of Charge',
                showgrid=False,
                overlaying='y',
                side='right',
                range=[0, 1],
            )
        )

        figure.write_html(self.paths.create_result_path(suffix=f'{self.name}.html'))
        try:
            figure.show(renderer='browser')
        except webbrowser.Error:  # webbrowser is not available on most remote machines
            pass

    def save_result_summary(self):
        """
        Saves all int, float and str attributes of run, scenario (incl. technoeconomic KPIs) and all blocks to the
        results dataframe
        :return: none
        """

        # get results of scenario
        results_scenario = pd.concat([
            # get attributes of type int, float, bool and str for scenario.result_summary
            pd.Series({key: value for key, value in self.__dict__.items()
                       if isinstance(value, (int, float, bool, str))}),
            # get dict of blocks with class names
            pd.Series(index=['blocks'],
                      data=str({key: value.classname
                                for key, value in self.block_registry.get('TopLevelBlock', {}).items()})),
            # get energies dataframes results for scenario.result_summary
            utils.create_results_from_dataframe(df=self.energies, name_prefix='energy'),
            # get economic results for scenario.result_summary
            self.aggregator.write_result_summary(),
            # get RunTime results
            self.runtime.result_summary
        ])

        # apply MultiIndex
        results_scenario.index = pd.MultiIndex.from_tuples(tuples=[('scenario', key) for key in results_scenario.index],
                                                           names=['block', 'key'])

        # write results from run and scenario to result_summary
        self.result_summary = pd.concat([results_scenario,
                                         *self.result_summary])

        # convert result_summary to DataFrame and save to temporary file
        pd.DataFrame(self.result_summary, columns=[self.name]).to_pickle(
            self.paths.output / f'{self.name}_summary_temp.pkl'
        )


class PredictionHorizon:

    def __init__(self, index, scenario):

        self.index = index
        self.scenario = scenario

        del index, scenario

        self.results = None

        # region time and data generation and slicing
        start = self.scenario.times.sim.start + (self.index * self.scenario.len_ch)
        self.ph = TimeSettings(start=start,
                               _timestep=self.scenario.timestep.td,
                               end=min(start + self.scenario.len_ph,
                                             self.scenario.times.sim.end),
                               )

        self.ch = TimeSettings(start=start,
                               _timestep=self.scenario.timestep.td,
                               end=min(start + self.scenario.len_ch,
                                             self.scenario.times.eval.end),
                               )
        del start

        def log_msg(msg: str):
            return f'Horizon {self.index + 1} of {self.scenario.nhorizons} - {msg}'

        self.constraints = constraints.CustomConstraints(scenario=self.scenario)

        # Display logger message if PH exceeds simulation end time and has to be truncated
        if self.ph.duration < self.scenario.len_ph:
            self.scenario.logger.info(log_msg(msg='Prediction Horizon truncated to simulation end time'))

        self.scenario.logger.info(f'Horizon {self.index + 1} of {self.scenario.nhorizons} - ' +
                                  f'Start: {self.ph.start} - ' +
                                  f'CH end: {self.ch.end} - ' +
                                  f'PH end: {self.ph.end}')

        # if apriori power scheduling is necessary, calculate power schedules:
        if self.scenario.scheduler:
            self.scenario.logger.debug(log_msg(msg='Calculating power schedules for commodities with rulebased charging strategies'))
            self.scenario.scheduler.calc_ph_schedule(self)
        # endregion

        # region build energy system model
        self.scenario.logger.info(log_msg(msg='Building oemof model'))

        self.es = solph.EnergySystem(timeindex=self.ph.dti,
                                     infer_last_interval=True)  # initialize energy system model instance

        for block in self.scenario.block_registry.get('TopLevelBlock', {}).values():
            block.pre_horizon(self)

        self.scenario.logger.debug(log_msg('Model build completed'))
        # endregion

        # region build optimization problem
        self.scenario.logger.info(log_msg(msg='Building optimization problem from oemof model'))

        self.model = solph.Model(self.es, debug=self.scenario.settings.debugmode)
        self.constraints.apply_constraints(model=self.model)

        if self.scenario.settings.debugmode and self.index == 1:
            self.model.write(self.scenario.path.dump, io_options={'symbolic_solver_labels': True})
        # endregion

        # region solve optimization problem
        self.scenario.logger.info(log_msg(msg='Model built, starting optimization'))
        results = self.model.solve(solver=self.scenario.settings.solver,
                                   solve_kwargs={'tee': self.scenario.settings.debugmode})

        if (results.solver.status == po.SolverStatus.ok) and \
                (results.solver.termination_condition == po.TerminationCondition.optimal):
            self.scenario.logger.info(log_msg(msg='Optimization completed, getting results'))
            if self.scenario.nhorizons == 1:  # Don't store objective for multiple horizons in scenario (most RH scenarios)
                self.scenario.objective_opt = self.model.objective()
        elif results.solver.termination_condition == po.TerminationCondition.infeasible:
            raise OptimizationError(log_msg(msg='Scenario failed: Infeasible'))
        elif results.solver.termination_condition == po.TerminationCondition.unbounded:
            raise OptimizationError(log_msg(msg='Scenario failed: Unbounded'))
        elif results.solver.termination_condition == po.TerminationCondition.infeasibleOrUnbounded:
            raise OptimizationError(log_msg(msg='Scenario failed: Infeasible or Unbounded (To solve this error try to '
                                                'set investment limits for blocks or for the scenario)'))
        else:
            raise Exception(log_msg(msg=f'Optimization terminated with unknown status: '
                                        f'{results.solver.termination_condition}'))
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
