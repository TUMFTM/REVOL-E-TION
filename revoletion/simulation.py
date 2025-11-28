#!/usr/bin/env python3

import importlib.resources
import logging
import math
import multiprocessing as mp
import multiprocessing.synchronize as mps
import pprint
import time
import traceback
import warnings
import webbrowser
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import geopy
import geopy.geocoders
import holidays
import numpy as np
import numpy_financial as npf
import oemof.solph as solph
import pandas as pd
import plotly.subplots
import pyomo.environ as po
import pytz
import timezonefinder
from typing_extensions import Self

import revoletion.data

from . import blocks, constraints, dispatch, scheduler, utils
from . import economics as eco
from . import logger as logger_fcs

_LOGGER = logging.getLogger(__name__)


class OptimizationError(Exception):
    def __init__(
        self, msg: str, prediction_horizon_idx: int | None = None, prediction_horizon_num: int | None = None
    ) -> None:
        """
        Create a new OptimizationError.

        :param msg: The error message.
        :param prediction_horizon_idx: Optionally provide the index of the prediction horizon that failed, to include more information in the error message.
        :param prediction_horizon_num: Optionally provide the total number of the prediction horizon, to include more information in the error message.

        :returns: The new OptimizationError.
        """
        self.prediction_horizon_idx = prediction_horizon_idx
        self.prediction_horizon_num = prediction_horizon_num

        if self.prediction_horizon_idx is not None and self.prediction_horizon_num is not None:
            msg = f"Horizon {self.prediction_horizon_idx} of {self.prediction_horizon_num} - {msg}"

        super().__init__(msg)


@dataclass
class Location:
    latitude: float
    longitude: float
    timezone: pytz.BaseTzInfo = field(default_factory=lambda: pytz.timezone("Europe/Berlin"))
    country: str = "DE"
    state: str = "BY"

    @classmethod
    def create_from_lat_lon(
        cls, latitude: float, longitude: float, logger: logging.Logger, lock: mps.Lock | None = None
    ) -> Self:
        tzfinder = timezonefinder.TimezoneFinder()
        timezone_raw = tzfinder.certain_timezone_at(lat=latitude, lng=longitude)
        if timezone_raw is None:
            raise ValueError(f"Failed to determine timezone at {latitude}/{longitude}")

        timezone = pytz.timezone(timezone_raw)

        location = cls._reverse_geocode_location(latitude, longitude, lock)

        if location is None:
            location = cls(
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
            )
            logger.warning(
                f"Connection to Geocoder failed. "
                f"Using default country ({location.country}) and state ({location.state})."
            )

            return location

        address = location.raw.get("address", {})

        if "ISO3166-2-lvl4" in address:
            country, state = address["ISO3166-2-lvl4"].split("-")
        elif "ISO3166-2-lvl3" in address:
            country, state = address["ISO3166-2-lvl3"].split("-")
        else:
            # fallback: try country_code + state name
            country = address.get("country_code", "").upper()
            state = address.get("state", "")

        return cls(latitude=latitude, longitude=longitude, timezone=timezone, country=country, state=state)

    @staticmethod
    def _reverse_geocode_location(
        latitude: float, longitude: float, lock: mps.Lock | None = None
    ) -> None | geopy.Location:
        geolocator = geopy.geocoders.Nominatim(user_agent="location_finder")
        try:
            if lock is None:  # sequential
                return geolocator.reverse(query=(latitude, longitude), language="en", exactly_one=True)
            else:  # parallel
                with lock:
                    time.sleep(2)  # max 1 request per second --> wait for 2 seconds to make sure to not avoid the limit
                    return geolocator.reverse(query=(latitude, longitude), language="en", exactly_one=True)
        except geopy.exc.GeocoderUnavailable:
            return None


@dataclass
class TimeSettings:
    start: pd.Timestamp
    end: pd.Timestamp
    duration: pd.Timedelta

    _timestep: pd.Timedelta

    @classmethod
    def create_from_start_timestamp(
        cls,
        start: pd.Timestamp,
        timestep: pd.Timedelta,
        end: pd.Timestamp | None = None,
        duration: pd.Timedelta | None = None,
    ) -> Self:
        if (end is None and duration is None) or (end is not None and duration is not None):
            raise ValueError('Exactly one of the parameters "end" or "duration" must be provided.')
        elif duration is None:
            duration = (end - start).floor(timestep)
        elif end is None:
            duration = duration.floor(timestep)
        # always recalculate end to ensure consistency
        end = start + duration

        return cls(start=start, end=end, duration=duration, _timestep=timestep)

    @cached_property
    def dti(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep, inclusive="left")

    @cached_property
    def dti_extd(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start, end=self.end, freq=self._timestep, inclusive="both")


@dataclass
class SimulationTimes:
    sim: TimeSettings
    eval: TimeSettings
    prj: TimeSettings

    @classmethod
    def create_from_plain(
        cls,
        timestep: str,
        timezone: pytz.BaseTzInfo,
        starttime: str,
        sim_endtime: str,
        sim_duration: str,
        prj_duration: str,
    ) -> Self:
        starttime_timestamp = cls._convert_time_str(starttime, timestep, timezone)
        if starttime_timestamp is None:
            raise ValueError(f"Failed to convert starttime ({starttime}) to pd.Timestamp")

        sim_endtime_timestamp = cls._convert_time_str(sim_endtime, timestep, timezone)

        timestep_timedelta = utils.convert2timedelta(timestep, unit="minute")
        if timestep_timedelta is None:
            raise ValueError(f"Failed to convert timestep ({timestep}) to pd.Timedelta")

        sim_duration_timedelta = utils.convert2timedelta(sim_duration, unit="day")

        sim = TimeSettings.create_from_start_timestamp(
            start=starttime_timestamp,
            timestep=timestep_timedelta,
            end=sim_endtime_timestamp,
            duration=sim_duration_timedelta,
        )
        eval = TimeSettings.create_from_start_timestamp(
            start=starttime_timestamp,
            timestep=timestep_timedelta,
            end=sim_endtime_timestamp,
            duration=sim_duration_timedelta,
        )
        prj = TimeSettings.create_from_start_timestamp(
            start=starttime_timestamp,
            timestep=timestep_timedelta,
            end=starttime_timestamp + pd.DateOffset(years=prj_duration),
        )

        return cls(sim=sim, eval=eval, prj=prj)

    @staticmethod
    def _convert_time_str(time_str: str | None, timestep: str, timezone: pytz.BaseTzInfo) -> pd.Timestamp | None:
        if time_str is None:
            return None

        # ToDo: reformat time
        time_str = time_str if len(time_str) > 10 else time_str + " 00:00"
        value = pd.to_datetime(time_str, format="%d.%m.%Y %H:%M").floor(timestep).tz_localize(timezone)
        return value


@dataclass
class Timestep:
    hours: float
    td: pd.Timedelta

    @classmethod
    def from_str(cls, timestep_str: str) -> Self:
        td = pd.Timedelta(timestep_str)

        return cls(td=td, hours=td.total_seconds() / 3600)


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

    scenario: Path
    input: Path
    output: Path
    rerun: Path | str | None

    @classmethod
    def from_plain_paths(
        cls,
        scenario: Path | str,
        input: Path | str | None = None,
        output: Path | str | None = None,
        rerun: Path | str | None = None,
    ):
        scenario_path = Path(scenario)

        if input is None:
            input_path = scenario_path.parent
        else:
            input_path = Path(input)

        if output is None:
            output_path = Path.cwd() / "results"
        else:
            output_path = Path(output)

        rerun_path = None
        if rerun is None:
            output_path = output_path / Path(f"{pd.Timestamp.now().strftime('%y%m%d_%H%M%S')}_{scenario_path.stem}")
        elif rerun == "latest":
            # get all directories in the output directory already sorted alphabetically
            directories = [d for d in sorted(output_path.iterdir()) if d.is_dir()]

            # return the last directory (if any)
            if directories:
                output_path = directories[-1]
            else:
                raise NotADirectoryError(f"No previous runs available in specified output directory {output_path}")
        else:
            rerun_path = Path(rerun)
            if rerun_path.is_absolute():
                output_path = rerun_path
            else:
                output_path = output_path / rerun_path.name

        # ensure all paths are absolute
        scenario_path = scenario_path.resolve()
        input_path = input_path.resolve()
        output_path = output_path.resolve()

        # ensure that all paths exist
        if not scenario_path.is_file():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input directory path not interpretable: {input_path}")
        if not rerun:
            output_path.mkdir(parents=True)  # create parents if missing -> relevant for default "results"
        else:
            if not output_path.is_dir():
                raise NotADirectoryError(f"Specified rerun directory {output_path} does not exist.")

        return cls(
            scenario=scenario_path,
            output=output_path,
            input=input_path,
            rerun=rerun_path if rerun_path is not None else rerun,
        )

    def create_result_path(self, suffix: str) -> Path:
        return self.output / f"{self.output.name}_{suffix}"

    @property
    def data_persist(self) -> Path:
        with importlib.resources.as_file(importlib.resources.files(revoletion.data)) as data_dir:
            return data_dir

    @property
    def summary_csv(self) -> Path:
        return self.create_result_path(suffix="summary.csv")

    @property
    def summary_pkl(self) -> Path:
        return self.create_result_path(suffix="summary.pkl")

    @property
    def status(self) -> Path:
        return self.create_result_path(suffix="status.csv")

    @property
    def dump(self) -> Path:
        return self.create_result_path(suffix="model.lp")

    @property
    def log(self) -> Path:
        return self.create_result_path(suffix="log.log")


@dataclass
class SimulationSettings:
    solver: str = "gurobi"
    n_processes: int = 1
    largescalemode: bool = False
    debugmode: bool = False
    rerun_infeasible: bool = True
    key_solcast_api: str = None


class Scenario:
    def __init__(
        self,
        paths: SimulationPaths,
        settings: SimulationSettings,
        name: str,  # will be set to the stem of the scenario filename for single scenario execution
        parameters: pd.Series,
        logger: logging.Logger | None = None,
        lock: mp.Lock = None,
        status_update: "SimulationRun.trigger_scenario_status_update" = None,
        status_queue: mp.Queue = None,
    ):
        self.paths = paths
        self.settings = settings

        self.name = name
        self.parent = None  # attribute needs to exist for economic aggregation

        # Set given parameters as attribute
        if not isinstance(parameters, pd.Series):
            raise ValueError("Parameters of type pd.Series must be provided to scenario")
        self.parameters = parameters

        if logger is None:
            logger = logger_fcs.ContextLoggerAdapter(_LOGGER, {"context_str": name})
        self.logger = logger

        self.status_update = status_update
        self.status_queue = status_queue

        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            # Force warnings in custom formatting and ignore warnings about infeasible or unbounded optimizations
            if "Optimization ended with status warning and termination condition" not in str(message):
                self.logger.warning(f"{category.__name__}: {message} (in {filename}, line {lineno})")

        warnings.showwarning = custom_warning_handler

        self.update_scenario_status(status_msg={"status": "started"})

        # General Information --------------------------------

        # integration levels at which power consumption is determined a priori
        self.apriori_lvls = ["uc", "fcfs", "equal", "soc"]

        self.runtime = utils.RunTime()
        self.runtime.start()

        self.worker = mp.current_process()

        msg_parallel = (
            f" on {self.worker.name.ljust(18)} - Parent: {self.worker._parent_name}"
            if hasattr(self.worker, "_parent_name")
            else ""
        )
        self.logger.info(f"Scenario initialized{msg_parallel}")

        for key, value in self.parameters.loc["scenario", :].items():
            setattr(self, key, value)  # this sets all the parameters defined in the csv file

        if not isinstance(self.blocks, dict):
            raise ValueError(
                'Scenario parameter "blocks" has to be defined in a dictionary format '
                "(\"{'name1':'classname1','name2':'classname2'}\") - "
                "check for missing or additional single or double quotes"
            )

        if not self.blocks:
            raise ValueError('Scenario parameter "blocks" is empty - Definition of at least one block is required')

        self.currency = self.currency.upper()  # all other parameters are .lower()-ed

        self.location = Location.create_from_lat_lon(
            latitude=self.latitude, longitude=self.longitude, logger=self.logger, lock=lock
        )

        self.prj_duration_yrs = self.prj_duration
        self.times = SimulationTimes.create_from_plain(
            timestep=self.timestep,
            timezone=self.location.timezone,
            starttime=self.starttime,
            sim_endtime=self.sim_endtime,
            sim_duration=self.sim_duration,
            prj_duration=self.prj_duration,
        )
        self.timestep = Timestep.from_str(self.timestep)

        for param in ["latitude", "longitude", "starttime", "sim_endtime", "sim_duration", "prj_duration"]:
            if hasattr(self, param):
                delattr(self, param)

        # generate variables for calculations
        self.sim_yr_rat = self.times.sim.duration / pd.Timedelta(days=365)  # no leap years
        self.sim_prj_rat = self.times.sim.duration / self.times.prj.duration

        if self.strategy == "rh":
            self.len_ph = utils.convert2timedelta(self.len_ph, unit="hour").floor(self.timestep.td)
            self.len_ch = utils.convert2timedelta(self.len_ch, unit="hour").floor(self.timestep.td)
        elif self.strategy in ["go"]:
            self.len_ph = self.times.sim.duration
            self.len_ch = self.times.sim.duration
        else:
            raise ValueError(f'Optimization strategy "{self.strategy}" unknown')

        if self.len_ph == self.timestep.td:
            raise ValueError(
                "Single timestep optimization not possible. Adjust simulation duration, timestep or "
                "prediction horizon length / truncate_ph (for RH only)"
            )

        self.nhorizons = math.ceil(self.times.sim.duration / self.len_ch)  # number of timeslices to run
        if not self.truncate_ph:
            # if PH is not truncated, the end of the last PH may be later than the end of the evaluation period
            self.times.sim = TimeSettings.create_from_start_timestamp(
                start=self.times.sim.start,
                timestep=self.timestep.td,
                duration=(self.len_ch * (self.nhorizons - 1) + self.len_ph),
            )

        # get holidays during simulation timeframe
        years = range(min(self.times.eval.dti_extd).year, max(self.times.eval.dti_extd).year + 1)
        try:
            self.holiday_dates = sorted(
                getattr(holidays, self.location.country)(years=years, state=self.location.state)
            )
        except:  # not for all countries the states are available (e.g. France)
            try:
                self.holiday_dates = sorted(getattr(holidays, self.location.country)(years=years))
                self.logger.warning(
                    f"Holidays for state {self.location.state} not available. "
                    f"Country-wide holidays for {self.location.country} are used instead."
                )
            except AttributeError:  # not all countries worldwide are available
                self.holiday_dates = []
                self.logger.warning(
                    f"Holidays for country {self.location.country} not available. "
                    f"No public holidays are considered in this scenario."
                )

        # region set air temperature
        temp_air = pd.Series(index=self.times.sim.dti, dtype=float)

        if isinstance(self.temp_air, (float, int)):
            temp_air[:] = self.temp_air
            self.temp_air = temp_air

        elif isinstance(self.temp_air, str) and self.blocks.get(self.temp_air, "") == "PVSource":
            # PVSource checks for temp_scn in parameters and writes temperature to this variable
            self.parameters.loc[(self.temp_air, "temp_scn")] = True
            self.temp_air = temp_air

        elif (
            isinstance(self.temp_air, str)
            and (self.paths.input / utils.set_extension(filename=self.temp_air, default_extension=".csv")).is_file()
        ):
            try:
                self.temp_air = utils.read_timeseries_csv(
                    path_input_file=(
                        self.paths.input / utils.set_extension(filename=self.temp_air, default_extension=".csv")
                    ),
                    scenario=self,
                ).iloc[:, 0]
            except IndexError as exc:
                raise IndexError(f"Failed to load air temperature timeseries data: {exc}")

        else:
            self.logger.warning(
                f"Specified argument for scenario parameter temp_air ({self.temp_air}) not found - "
                f"Using default of 25 °C"
            )
            temp_air[:] = 25
            self.temp_air = temp_air
        # endregion

        # region initialize result variables
        self.periods_prj = np.arange(0, self.prj_duration_yrs)
        self.periods_prj_extd = np.arange(0, self.prj_duration_yrs + 1)  # add. year for salvage values
        self.discount_factors = pd.DataFrame(
            index=self.periods_prj_extd,
            columns=["beginning", "mid", "end"],
            data={
                occ: eco.EcoTools.discount(
                    future_value=1, periods=self.periods_prj_extd + 1, discount_rate=self.wacc, occurs_at=occ
                )
                for occ in ["beginning", "mid", "end"]
            },
            dtype="float64",
        )

        self.aggregator = eco.EcoAggregator(name="scenario", scenario=self)
        self.capex_preexisting_considered = 0

        self.block_registry = dict()

        # Define priorities of blocks to ensure correct initialization order
        priority_default = 2
        priority_blocks = {
            "SystemCore": 0,  # always first -> ac and dc bus required for all other ElectricBlocks
            "PVSource": 1,  # holds temperature and wind data -> required by StorageBlock and WindSource
        }

        for name, class_name in sorted(
            {"core": "SystemCore", **self.blocks}.items(),
            key=lambda item: priority_blocks.get(item[1], priority_default),
        ):
            class_obj = getattr(blocks, class_name, None)
            if class_obj is not None and isinstance(class_obj, type):
                class_obj(name, self)
            else:
                raise ValueError(f'Class "{class_name}" not found in blocks.py file - Check for typos or add class.')

        if self.invest_max is not None and self.invest_max < self.capex_preexisting_considered:
            raise ValueError(
                f"Initial investment costs of {self.capex_preexisting_considered:.2f} {self.currency} "
                f"exceed maximum investment limit of {self.invest_max} {self.currency}"
            )

        self.objective_opt = None  # unused for rh strategy
        self.energies = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                tuples=[("renewable", "act"), ("sources", "pro"), ("sinks", "del")], names=["block", "key"]
            ),
            columns=["sim", "yrl", "prj", "dis"],
            data=0,
            dtype=float,
        )

        self.e_eta = None
        self.renewable_share = None
        self.lcoe_total = self.lcoe_wocs = None
        self.npc = self.npv = self.irr = self.mirr = None
        # endregion

        # region preexecution
        self.dispatch_environment = dispatch.DispatchEnvironment(scenario=self)

        for block in self.block_registry.get("TopLevelBlock", {}).values():
            block.pre_scenario()

        self.scheduler = None
        if self.block_registry.get("SubFleetScheduling", {}):
            self.scheduler = scheduler.AprioriPowerScheduler(scenario=self)
        # endregion

        self.logger.debug("Scenario initialization completed")

        self.update_scenario_status(status_msg={"status": "fully initialized"})

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

    @classmethod
    def create_from_file(cls, paths: SimulationPaths, settings: SimulationSettings) -> Self:
        """
        Create a new scenario from a scenario file.

        Args:
            paths: The paths configuration for the scenario.
            settings: The settings for the simulation.

        Returns:
            A new `Scenario` which is ready for execution.
        """
        parameters = utils.read_scenario_from_file(paths.scenario)
        # check if scenario file contains more than one scenario (then it has to be run via a SimulationRun)
        if len(parameters.columns) > 1:
            raise ValueError("More than one scenario detected. Provide a single column CSV or PKL file.")

        name = str(parameters.columns[0])

        # convert DataFrame to Series
        parameters_series = parameters.iloc[:, 0]

        scenario_logger = logger_fcs.ContextLoggerAdapter(_LOGGER, {"context_str": name})

        return cls(paths=paths, settings=settings, name=name, parameters=parameters_series, logger=scenario_logger)

    def execute(self) -> None:
        try:
            for horizon_index in range(self.nhorizons):  # Inner optimization loop over all prediction horizons
                prediction_horizon = PredictionHorizon(index=horizon_index, scenario=self, logger=self.logger)

                prediction_horizon.execute()

                self.update_scenario_status(
                    status_msg={"status": f"completed horizon {horizon_index + 1} out of {self.nhorizons}"}
                )

            self.update_scenario_status(status_msg={"status": "successful"})

        except Exception as e:
            # Scenario has failed -> store scenario name to dataframe containing failed scenarios
            status = "infeasible" if isinstance(e, OptimizationError) else "failed"
            self.update_scenario_status(
                status_msg={"status": status, "exception": str(e), "traceback": traceback.format_exc()}
            )

            self.logger.error(
                msg=f"{str(e)} - continue on next scenario", exc_info=(not isinstance(e, OptimizationError))
            )

        finally:  # save results up to exception - valuable in RH strategy
            for block in self.block_registry.get("TopLevelBlock", {}).values():
                block.post_scenario()
            self.aggregator.aggregate()

            self.calc_meta_results()

            if not self.settings.largescalemode:
                result_timeseries = blocks.TimeseriesCollectionBlockVisitor().collect_timeseries(self.block_registry)
                result_timeseries_aggregated = pd.concat(result_timeseries, axis=1)
                result_timeseries_aggregated.to_csv(self.paths.create_result_path(suffix=f"{self.name}_results_ts.csv"))

                result_messages = blocks.MessageCollectionBlockVisitor().collect_messages(self.block_registry)
                for msg in result_messages:
                    self.logger.info(msg)

            self.runtime.stop()
            self.logger.info(f"Scenario finished - runtime {self.runtime}")

    def update_scenario_status(self, status_msg: dict):
        if self.status_update is not None:
            status_msg.update(scenario=self.name)
            self.status_update(queue=self.status_queue, status_msg=status_msg)

    def calc_meta_results(self):
        # pandas creates a RuntimeWarning at division by 0 -> try/except does not work
        if self.energies.loc[("sources", "pro"), "sim"] == 0:
            self.logger.warning("Core efficiency calculation: division by zero")
        else:
            self.e_eta = self.energies.loc[("sinks", "del"), "sim"] / self.energies.loc[("sources", "pro"), "sim"]

        if self.energies.loc[("sources", "pro"), "sim"] == 0:
            self.logger.warning("Renewable share calculation: division by zero")
        else:
            self.renewable_share = (
                self.energies.loc[("renewable", "act"), "sim"] / self.energies.loc[("sources", "pro"), "sim"]
            )

        if self.energies.loc[("sinks", "del"), "sim"] == 0:
            self.logger.warning("LCOE calculation: division by zero")
        else:
            self.lcoe_total = self.aggregator.totex.dis / self.energies.loc[("sinks", "del"), "dis"]
            self.lcoe_wocs = (
                self.aggregator.totex.dis
                -
                # ToDo: check whether calculation of totex['dis'] of fleets is correct
                sum([fleet.aggregator.totex.dis for fleet in self.block_registry.get("Fleet", {}).values()])
            ) / self.energies.loc[("sinks", "del"), "dis"]

        self.npc = self.aggregator.totex.dis
        self.npv = self.aggregator.value.dis
        self.irr = npf.irr(self.aggregator.value.cashflows)
        self.mirr = npf.mirr(self.aggregator.value.cashflows, self.wacc, self.wacc)

        # print basic results
        self.logger.info(
            f"NPC {f'{self.npc:,.2f}' if pd.notna(self.npc) else '-'} {self.currency} | "
            f"NPV {f'{self.npv:,.2f}' if pd.notna(self.npv) else '-'} {self.currency} | "
            f"LCOE {f'{self.lcoe_wocs * 1e5:,.2f}' if pd.notna(self.lcoe_wocs) else '-'} {self.currency}-ct/kWh | "
            f"mIRR {f'{self.mirr * 100:,.2f}' if pd.notna(self.mirr) else '-'} %"
        )

    def generate_and_save_plot(self):
        figure = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])

        plot_traces = blocks.VisualizationBlockVisitor().create_plot_traces(self.block_registry)

        figure.add_traces(plot_traces.plot_lines, secondary_ys=plot_traces.secondary_ys)

        if self.strategy == "go":
            title = f"Global Optimum Results - {self.paths.output.name} - Scenario: {self.name}"
        elif self.strategy == "rh":
            title = (
                f"Rolling Horizon Results - {self.paths.output.name} - Scenario: {self.name} - "
                f"PH: {self.len_ph}h - CH: {self.len_ch}h"
            )
        else:
            title = f"Results - {self.paths.output.name} - Scenario: {self.name}"

        linecolor = "gray"
        gridcolor = "gray"

        figure.update_layout(
            title=title,
            plot_bgcolor="white",
            xaxis=dict(
                title="Local Time",
                showgrid=True,
                linecolor=linecolor,
                gridcolor=gridcolor,
            ),
            yaxis=dict(
                title="Power in W",
                showgrid=True,
                linecolor=linecolor,
                gridcolor=gridcolor,
            ),
            yaxis2=dict(
                title="State of Charge",
                showgrid=False,
                overlaying="y",
                side="right",
                range=[0, 1],
            ),
        )

        figure.write_html(self.paths.create_result_path(suffix=f"{self.name}.html"))
        try:
            figure.show(renderer="browser")
        except webbrowser.Error:  # webbrowser is not available on most remote machines
            pass

    def save_result_summary(self):
        """
        Saves all int, float and str attributes of run, scenario (incl. technoeconomic KPIs) and all blocks to the
        results dataframe
        :return: none
        """

        # get results of scenario
        results_scenario = pd.concat(
            [
                # get attributes of type int, float, bool and str for scenario.result_summary
                pd.Series(
                    {key: value for key, value in self.__dict__.items() if isinstance(value, (int, float, bool, str))}
                ),
                # get dict of blocks with class names
                pd.Series(
                    index=["blocks"],
                    data=str(
                        {key: value.classname for key, value in self.block_registry.get("TopLevelBlock", {}).items()}
                    ),
                ),
                # get energies dataframes results for scenario.result_summary
                utils.create_results_from_dataframe(df=self.energies, name_prefix="energy"),
                # get economic results for scenario.result_summary
                self.aggregator.write_result_summary(),
                # get RunTime results
                self.runtime.result_summary,
            ]
        )

        # apply MultiIndex
        results_scenario.index = pd.MultiIndex.from_tuples(
            tuples=[("scenario", key) for key in results_scenario.index], names=["block", "key"]
        )

        blocks_result_summary = blocks.SummaryCollectionBlockVisitor().collect_summary(self.block_registry)
        # write results from run and scenario to result_summary
        result_summary = pd.concat([results_scenario, *blocks_result_summary])

        # convert result_summary to DataFrame and save to temporary file
        pd.DataFrame(result_summary, columns=[self.name]).to_pickle(self.paths.output / f"{self.name}_summary_temp.pkl")


class PredictionHorizon:
    def __init__(self, index: int, scenario: Scenario, logger: logging.Logger):
        self.index = index
        self.scenario = scenario

        # Setup the logger as a child of the scenario logger with some additional metadata
        # about the index of the prediction horizon.
        logging_ctx_str = f"Horizon {self.index + 1} of {self.scenario.nhorizons} -"
        self._logger = logger_fcs.ContextLoggerAdapter(logger, {"context_str": logging_ctx_str})

        self._results = None

        # region time and data generation and slicing
        start = self.scenario.times.sim.start + (self.index * self.scenario.len_ch)
        self.ph = TimeSettings.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep.td,
            end=min(start + self.scenario.len_ph, self.scenario.times.sim.end),
        )

        self.ch = TimeSettings.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep.td,
            end=min(start + self.scenario.len_ch, self.scenario.times.eval.end),
        )

        # Display logger message if PH exceeds simulation end time and has to be truncated
        if self.ph.duration < self.scenario.len_ph:
            self._logger.info(msg="Prediction Horizon truncated to simulation end time")

        self._logger.info("Start: %s - CH end: %s - PH end: %s", self.ph.start, self.ch.end, self.ph.end)

        self.es = solph.EnergySystem(
            timeindex=self.ph.dti, infer_last_interval=True
        )  # initialize energy system model instance

        self.constraints = constraints.CustomConstraints(scenario=self.scenario)

        # if apriori power scheduling is necessary, calculate power schedules:
        if self.scenario.scheduler:
            self._logger.debug("Calculating power schedules for commodities with rulebased charging strategies")
            self.scenario.scheduler.calc_ph_schedule(self)
        # endregion

    @property
    def results(self):
        """Get the results of the previous execution."""
        return self._results

    def execute(self) -> None:
        """
        Perform the concrete optimization across a prediction horizon.
        """

        self._logger.info("Building oemof model")
        self._pre_horizon()
        self._logger.debug("Model build completed")

        model = self._create_model()

        self._logger.info("Model built, starting optimization")
        results = model.solve(
            solver=self.scenario.settings.solver, solve_kwargs={"tee": self.scenario.settings.debugmode}
        )
        if (results.solver.status == po.SolverStatus.ok) and (
            results.solver.termination_condition == po.TerminationCondition.optimal
        ):
            self._logger.info("Optimization completed, getting results")
            if (
                self.scenario.nhorizons == 1
            ):  # Don't store objective for multiple horizons in scenario (most RH scenarios)
                self.scenario.objective_opt = model.objective()
        elif results.solver.termination_condition == po.TerminationCondition.infeasible:
            raise OptimizationError(
                "Scenario failed: Infeasible",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )
        elif results.solver.termination_condition == po.TerminationCondition.unbounded:
            raise OptimizationError(
                "Scenario failed: Unbounded",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )
        elif results.solver.termination_condition == po.TerminationCondition.infeasibleOrUnbounded:
            raise OptimizationError(
                "Scenario failed: Infeasible or Unbounded (To solve this error try to "
                "set investment limits for blocks or for the scenario)",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )
        else:
            raise Exception(f"Optimization terminated with unknown status: {results.solver.termination_condition}")

        self._logger.debug(pprint.pformat(solph.processing.meta_results(model)))

        # Get result data slice for current CH from results and save in result dataframes for later analysis
        # Get (possibly optimized) component sizes from results to handle outputs more easily
        self._results = solph.processing.results(model)

        self._post_horizon()

    def _create_model(self) -> solph.Model:
        self._logger.info("Building optimization problem from oemof model")

        model = solph.Model(self.es, debug=self.scenario.settings.debugmode)
        self.constraints.apply_constraints(model=model)

        if self.scenario.settings.debugmode and self.index == 1:
            model.write(self.scenario.path.dump, io_options={"symbolic_solver_labels": True})

        return model

    def _pre_horizon(self) -> None:
        for block in self.scenario.block_registry.get("TopLevelBlock", {}).values():
            block.pre_horizon(self)

    def _post_horizon(self) -> None:
        for block in self.scenario.block_registry.get("TopLevelBlock", {}).values():
            block.post_horizon(self)
