#!/usr/bin/env python3

# builtin packages
import copy
import logging
import statistics

# from packages
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

# packages
import numpy as np
import pandas as pd
import simpy

# from local packages
from . import blocks

if TYPE_CHECKING:
    from . import scenario as scn


class MultiFilterStorePut(simpy.resources.base.Put):
    def __init__(self, resource, items, **kwargs):
        now = resource._env.now
        self.items = [(item, now) for item in items]
        super().__init__(resource, **kwargs)


class MultiFilterStoreGet(simpy.resources.base.Get):
    def __init__(self, resource, amount=1, filter: Callable[[Tuple[Any, float]], bool] = lambda x: True, **kwargs):
        self.amount = amount
        self.filter = filter
        super().__init__(resource, **kwargs)


class MultiFilterStore(simpy.resources.base.BaseResource):
    def __init__(self, env: simpy.Environment, capacity: int, initial=None):
        super().__init__(env, capacity)
        self.items = [(item, env.now) for item in (initial or [])]

    put = simpy.core.BoundClass(MultiFilterStorePut)
    get = simpy.core.BoundClass(MultiFilterStoreGet)

    def _do_put(self, event):
        if len(self.items) + len(event.items) <= self._capacity:
            self.items.extend(event.items)  # FIFO behavior
            event.succeed()

    def _do_get(self, event):
        filtered = list(filter(event.filter, self.items))
        if len(filtered) >= event.amount:
            selected = filtered[: event.amount]

            # Remove these from self.items (match by identity)
            for item in selected:
                self.items.remove(item)

            # Return only the items, not their timestamps
            event.succeed([item for item, timestamp in selected])


class DispatchTimer:
    def __init__(
        self,
        dti_base: pd.DatetimeIndex,
        buffer_pre: pd.Timedelta = pd.Timedelta(days=1),
        buffer_post: pd.Timedelta = pd.Timedelta(days=28),
    ):
        self.dti_base = dti_base
        self.step = pd.Timedelta(self.dti_base.freq)
        self.step_hours = self.step.total_seconds() / 3600
        self.time_start = self.dti_base.min() - buffer_pre  # ensures positive step counts even with preblocks
        self.time_end = self.dti_base.max() + self.step + buffer_post

        self.dti = pd.date_range(start=self.time_start, end=self.time_end, freq=self.dti_base.freq)

    def dt2steps(self, values: pd.Series | pd.Timedelta | pd.Timestamp):
        """
        utility method
        convert pandas datetime or timedelta values to DES steps
        """

        if pd.api.types.is_datetime64_any_dtype(values) or isinstance(values, pd.Timestamp):
            # ensure that the result is at least 1, as 0 would leave no time for any action in real life
            return np.maximum(1, np.ceil((values - self.time_start) / self.step).astype(int))
        elif pd.api.types.is_timedelta64_dtype(values) or isinstance(values, pd.Timedelta):
            return np.maximum(1, np.ceil(values / self.step).astype(int))
        elif values.empty:
            return
        else:
            raise ValueError(f"Unsupported type {type(values)} for conversion to steps")

    def steps2dt(self, steps: pd.Series, absolute: bool = False):
        """
        utility method
        convert DES steps to pandas datetime or timedelta values
        """
        td = pd.to_timedelta(steps * self.step_hours, unit="hour")
        if not absolute:
            return td
        else:
            return td + self.time_start


class DispatchEnvironment:
    """
    Interface between REVOl-E-TION scenario and the standalone FleetDispatchers
    """

    def __init__(self, scenario: "scn.Scenario"):
        self.scenario = scenario

        self.fleets = self.scenario.block_registry.get("DispatchFleet", {})
        self.time = DispatchTimer(dti_base=self.scenario.times.sim.dti)
        self.env = simpy.Environment()
        self.dispatchers = dict()
        self.kpis = dict()

        # region create individual dispatchers
        # Battery Dispatchers need to be initialized first to allow for range extension of Vehicle Dispatchers
        for name, fleet in [(n, f) for n, f in self.fleets.items() if not f.is_vehicle_fleet]:
            self.dispatchers[name] = FleetDispatcher(
                timer=self.time,
                demand=fleet.demand,
                env=self.env,
                params=FleetParams.from_obj(fleet=fleet),
                logger=self.scenario.logger,
            )
            fleet.dispatcher = self.dispatchers[name]

        for name, fleet in [(n, f) for n, f in self.fleets.items() if f.is_vehicle_fleet]:
            self.dispatchers[name] = FleetDispatcher(
                timer=self.time,
                demand=fleet.demand,
                env=self.env,
                params=FleetParams.from_obj(fleet=fleet),
                logger=self.scenario.logger,
            )
            fleet.dispatcher = self.dispatchers[name]
        # endregion

        self.env.run()

        # go through ALL dispatchers once and transfer rex processes before further processing
        for dispatcher in self.dispatchers.values():
            if dispatcher.params.is_vehicle_fleet:
                dispatcher.transfer_rex_processes()

        for dispatcher in self.dispatchers.values():
            dispatcher.generate_log(dti_output=self.scenario.times.sim.dti)
            dispatcher.calc_kpis()
            if not self.scenario.settings.largescalemode:
                path_log = self.scenario.paths.create_result_path(
                    suffix=f"{self.scenario.name}_{dispatcher.params.name}_log.feather"
                )
                dispatcher.save_data(path_log=path_log)

        for fleet in self.fleets.values():
            fleet.log = fleet.dispatcher.log
            fleet.rate_success = fleet.dispatcher.rate_success
            fleet.rate_blocked = fleet.dispatcher.rate_blocked
            fleet.rate_utilization = fleet.dispatcher.rate_utilization


@dataclass
class FleetParams:
    name: str
    subfleet_params: list
    is_vehicle_fleet: bool

    @classmethod
    def from_obj(cls, fleet: "blocks.Fleet") -> "FleetParams":
        subfleet_params = {name: SubFleetParams.from_obj(subfleet) for name, subfleet in fleet.subblocks.items()}

        params = dict(
            name=fleet.name,
            subfleet_params=subfleet_params,
            is_vehicle_fleet=fleet.is_vehicle_fleet,
        )

        return cls(**params)


@dataclass
class SubFleetParams:
    name: str
    units: list | dict[str, "blocks.FleetUnit"]
    is_electric: bool
    size_unit: Optional[float] = None
    pwr_chg: Optional[float] = None
    soc_upper: Optional[float] = None
    soc_lower: Optional[float] = None
    eff_chg: Optional[float] = None
    eff_roundtrip: Optional[float] = None
    loss_rate_per_hour: Optional[float] = None
    energy_total: Optional[float] = None
    energy_usable: Optional[float] = None
    dsoc_usable: Optional[float] = None
    pwr_chg_usable: Optional[float] = None
    rex_available: Optional[bool] = False
    rex_subfleet: Optional["blocks.SubFleet"] = None
    rex_dispatcher: Optional["FleetDispatcher"] = None

    @classmethod
    def from_obj(cls, subfleet: "blocks.SubFleet") -> "SubFleetParams":
        is_electric = subfleet.type_unit in ["ev", "mb"]
        params = dict(
            name=subfleet.name,
            units=subfleet.subblocks,
            is_electric=is_electric,
        )

        if is_electric and subfleet.subblocks:  # subfleet has units in it
            unit = subfleet.subblocks[next(iter(subfleet.subblocks))]  # representative

            soc_minmax = min(
                [unit.states.at[subfleet.scenario.times.sim.start, "soc_max"] for unit in subfleet.subblocks.values()]
            )
            soc_maxmin = max(
                [unit.states.at[subfleet.scenario.times.sim.start, "soc_min"] for unit in subfleet.subblocks.values()]
            )

            params.update(
                pwr_chg=unit.pwr_chg_max,
                size_unit=unit.sizes["storage"].preexisting,
                soc_upper=statistics.median([soc_minmax, unit.soc_target, soc_maxmin]),
                soc_lower=statistics.median([soc_minmax, unit.soc_return, soc_maxmin]),
                eff_chg=unit.eff["chg_int"],
                eff_roundtrip=unit.eff["storage_roundtrip"],
                loss_rate_per_hour=unit.loss_rate_per_hour,
            )

            if subfleet.type_unit == "ev":
                rex_available = subfleet.rex is not None
                if rex_available:
                    rex_subfleet = subfleet.scenario.block_registry.get("SubFleet", {}).get(subfleet.rex, None)
                    if rex_subfleet is None:
                        raise ValueError(f'Block "{subfleet.name}": rex subfleet "{subfleet.rex}" does not exist')
                    if not rex_subfleet.type_unit == "mb":
                        raise ValueError(
                            f'Block "{subfleet.name}": rex subfleet "{subfleet.rex}" is not a battery Subfleet'
                        )
                    if not hasattr(rex_subfleet.parent, "dispatcher") or not rex_subfleet.parent.dispatcher:
                        raise ValueError(
                            f'Block "{subfleet.name}": rex subfleet "{subfleet.rex}" is not actively dispatched'
                        )
                    rex_dispatcher = rex_subfleet.parent.dispatcher
                else:
                    rex_subfleet = None
                    rex_dispatcher = None

                params.update(
                    rex_available=rex_available,
                    rex_subfleet=rex_subfleet,
                    rex_dispatcher=rex_dispatcher,
                )

        return cls(**params)


class FleetDispatcher:
    def __init__(
        self,
        timer: DispatchTimer,
        demand: pd.DataFrame,
        env: simpy.Environment,
        params: FleetParams | SubFleetParams,
        logger: logging.Logger = None,
        factor_derate: float = 0.9,
    ):  # conservativeness factor on assumed charge power vs actually available power
        self.time = timer
        self.demand = demand
        self.env = env
        self.params = params
        self.logger = logger
        self.factor_derate = factor_derate

        self.rate_success = None
        self.rate_failure = None
        self.rate_blocked = None
        self.rate_utilization = None

        # create logger for standalone operation
        if self.logger is None:
            self.logger = logging.getLogger("null")
            self.logger.addHandler(logging.NullHandler())

        # SubFleetParams object given -> convert to single subfleet FleetParams object
        if isinstance(self.params, SubFleetParams):
            self.params = FleetParams(
                name=f"{self.params.name}_fp", subfleet_params={self.params.name: self.params}, is_vehicle_fleet=True
            )

        unit_names = [unit for sfparams in self.params.subfleet_params.values() for unit in sfparams.units]
        log_cols = pd.MultiIndex.from_tuples(
            [(name, col) for name in unit_names for col in ["atbase", "atac", "atdc", "dsoc", "consumption", "dist"]],
            names=["unit", "time"],
        )
        self.log = pd.DataFrame(index=self.time.dti, columns=log_cols)

        self.stores = dict()

        # region estimate usable energy and power
        for sfp in self.params.subfleet_params.values():
            self.stores[sfp.name] = MultiFilterStore(env=self.env, capacity=len(sfp.units), initial=list(sfp.units))

            if sfp.is_electric and sfp.units:  # subfleet has units
                sfp.energy_total = sfp.size_unit
                sfp.dsoc_usable = sfp.soc_upper - sfp.soc_lower
                if sfp.dsoc_usable <= 0:
                    raise ValueError(
                        f"Usable dSOC for subfleet {sfp.name} is zero or negative. Check SOC targets and aging."
                    )
                sfp.energy_usable = sfp.dsoc_usable * sfp.energy_total * np.sqrt(sfp.eff_roundtrip)
                sfp.pwr_chg_usable = (
                    sfp.pwr_chg
                    * sfp.eff_chg  # charger efficiency
                    * np.sqrt(sfp.eff_roundtrip)  # storage charging efficiency
                    - (sfp.loss_rate_per_hour * sfp.energy_total)  # maximum self discharge power
                ) * self.factor_derate

            else:  # non electric or no units
                sfp.energy_total = np.inf
                sfp.energy_usable = np.inf
                sfp.dsoc_usable = 1
                sfp.pwr_chg_usable = np.inf
        # endregion

        # region calculate a priori and subfleet agnostic process data in vectorized form
        self.demand.requests["step_req"] = self.time.dt2steps(values=self.demand.requests["time_req"])
        self.demand.requests["dtime_rental"] = self.demand.requests["dtime_active"] + self.demand.requests["dtime_idle"]
        self.demand.requests["steps_rental"] = self.time.dt2steps(values=self.demand.requests["dtime_rental"])
        self.demand.requests["steps_patience"] = self.time.dt2steps(values=self.demand.requests["dtime_patience"])

        self.demand.requests.sort_values(by="step_req", inplace=True, ignore_index=True)
        # endregion

        # create dataclass instances
        self.processes = {
            row.Index: DispatchProcess(
                pid=row.Index,
                dispatcher_prim=self,
                status="unprocessed",
                time_req=row.time_req,
                step_req=row.step_req,
                dtime_patience=row.dtime_patience,
                steps_patience=row.steps_patience,
                dtime_rental=row.dtime_rental,
                steps_rental=row.steps_rental,
                energy_req=row.energy_req,
                distance_req=getattr(row, "distance", None),
                subfleets=getattr(row, "subfleets", None),
            )
            for row in self.demand.requests.itertuples()
        }

    def run_standalone(self, dti_output: pd.DatetimeIndex = None):
        self.env.run()
        self.generate_log(dti_output=dti_output)

    def transfer_rex_processes(self):
        """
        create additional processes in the battery FleetDispatcher processes dict representing the rex processes
        """

        def switch_prim_rex(process, pid):
            rex_process = copy.copy(process)
            for f in fields(rex_process):
                if f.name.endswith("_prim"):
                    rex_name = f.name[:-5] + "_rex"
                    setattr(rex_process, f.name, getattr(process, rex_name))
                    setattr(rex_process, rex_name, None)
            rex_process.pid = pid
            return rex_process

        for process in self.processes.values():
            if process.result_rex is None or process.status != "success":
                continue  # no rex received, skip to next process

            pid = f"{self.params.name}_{process.pid}"
            process.dispatcher_rex.processes.update({pid: switch_prim_rex(process, pid)})

    def generate_log(self, dti_output: pd.DatetimeIndex = None):
        """
        post DES method
        convert processes to time based log
        """
        if dti_output is None:
            dti_output = self.time.dti_base

        # region convert processes to time based log
        self.log.loc[:, (slice(None), "atbase")] = True
        self.log.loc[:, (slice(None), "atac")] = False
        self.log.loc[:, (slice(None), "atdc")] = False
        self.log.loc[:, (slice(None), "dsoc")] = 0.0
        self.log.loc[:, (slice(None), "dist")] = 0.0
        self.log.loc[:, (slice(None), "consumption")] = 0.0

        for process in self.processes.values():
            if process.status != "success":
                continue  # skip, go to next process

            time_end = process.time_return - self.time.step
            power_avg = process.energy_req_prim / (process.steps_rental * self.time.step_hours)
            dist_avg = process.distance_req / process.steps_rental if process.distance_req is not None else 0.0

            for unit in process.result_prim:
                self.log.loc[process.time_dep : time_end, (unit, "atbase")] = False
                self.log.loc[process.time_dep : time_end, (unit, "atac")] = False  # todo destination charging?
                self.log.loc[process.time_dep : time_end, (unit, "atdc")] = True
                self.log.loc[process.time_dep : time_end, (unit, "consumption")] = power_avg
                self.log.loc[process.time_dep : time_end, (unit, "dist")] = dist_avg
                self.log.loc[process.time_dep, (unit, "dsoc")] = process.dsoc_prim

        self.log = self.log.loc[dti_output, :].convert_dtypes()
        # endregion

    def calc_kpis(self):
        """
        Calculate usage and failure rate

        rate_blocked counts all time units are unavailable for other requests (rental plus subsequent recharging),
        rate_utilization only counts the rental time itself, i.e. the time units are not at base.
        """
        self.rate_success = np.mean(["success" in process.status for process in self.processes.values()])

        processes_success = [process for process in self.processes.values() if process.status == "success"]

        time_blocked_total = np.sum(
            [process.num_prim * (process.dtime_rental + process.dtime_chg_prim) for process in processes_success]
        )
        time_utilized_total = np.sum([process.num_prim * process.dtime_rental for process in processes_success])
        time_total = (self.time.dti_base.max() + self.time.step) - self.time.dti_base.min()
        n_units = sum(store.capacity for store in self.stores.values())

        try:
            self.rate_blocked = time_blocked_total / time_total / n_units
            self.rate_utilization = time_utilized_total / time_total / n_units
        except TypeError:
            self.rate_blocked = 0.0
            self.rate_utilization = 0.0

    def save_data(self, path_log: str = None):
        """
        This function saves the converted log dataframe as a suitable example feather file for the energy system model.
        The resulting dataframe can also be handed to the energy system model directly in addition for faster
        delivery through execute_des.
        """
        if path_log is not None:
            # feather does not serialize a non-default index, so move the DatetimeIndex into a column
            self.log.reset_index().to_feather(Path(path_log).resolve())


@dataclass
class DispatchProcess:
    pid: str
    status: str
    dispatcher_prim: FleetDispatcher
    time_req: pd.Timestamp
    step_req: int
    energy_req: float
    dtime_rental: pd.Timedelta
    steps_rental: int
    dtime_patience: pd.Timedelta
    steps_patience: int
    processed: Optional[bool] = False
    distance_req: Optional[float] = None
    subfleets: Optional[list] = None
    num_prim: Optional[int] = None
    steps_wait: Optional[int] = None
    time_dep: Optional[pd.Timestamp] = None
    step_dep: Optional[int] = None
    time_return: Optional[pd.Timestamp] = None
    step_return: Optional[int] = None
    energy_usable: Optional[float] = None
    energy_total: Optional[float] = None
    utilization: Optional[float] = None
    dsoc_prim: Optional[float] = None
    energy_req_prim: Optional[float] = None
    dtime_chg_prim: Optional[pd.Timedelta] = None
    steps_chg_prim: Optional[int] = None
    request_prim: Optional[simpy.Process] = None
    result_prim: Optional[simpy.Resource] = None
    dispatcher_rex: Optional[FleetDispatcher] = None
    needs_rex: Optional[bool] = None
    num_rex: Optional[int] = None
    dsoc_rex: Optional[float] = None
    energy_req_rex: Optional[float] = None
    dtime_chg_rex: Optional[pd.Timedelta] = None
    steps_chg_rex: Optional[int] = None
    request_rex: Optional[simpy.Process] = None
    result_rex: Optional[simpy.Resource] = None

    def __post_init__(self):
        if not self.processed:
            self.dispatcher_prim.env.process(self.process())

    def process(self):
        """
        SimPy process that attempts to allocate resources from primary and optional REX fleets.
        Retries across stores until successful or patience is exhausted.
        """

        env = self.dispatcher_prim.env
        yield env.timeout(self.step_req)
        self.steps_wait = 0

        stores_prim = (
            {name: store for name, store in self.dispatcher_prim.stores.items() if name in self.subfleets}
            if self.subfleets is not None
            else self.dispatcher_prim.stores
        )

        while self.steps_wait <= self.steps_patience:
            for store_prim_name, store_prim in stores_prim.items():
                sfp_prim = self.dispatcher_prim.params.subfleet_params[store_prim_name]

                if not sfp_prim.units:
                    continue  # request from next subfleet if this one is empty

                if self.dispatcher_prim.params.is_vehicle_fleet:
                    self.num_prim = 1
                else:  # BatteryFleet
                    self.num_prim = np.ceil(self.energy_req / sfp_prim.energy_usable).astype(int)

                if sfp_prim.rex_available:  # rex is available for this subfleet/store
                    self.dispatcher_rex = sfp_prim.rex_dispatcher
                    sfp_rex = self.dispatcher_rex.params.subfleet_params[sfp_prim.rex_subfleet.name]
                    store_rex = self.dispatcher_rex.stores[sfp_prim.rex_subfleet.name]
                    self.energy_missing = max((self.energy_req - sfp_prim.energy_usable), 0)
                    self.num_rex = np.ceil(self.energy_missing / sfp_rex.energy_usable).astype(int)
                    energy_req_eff = self.energy_req
                else:
                    sfp_rex = None
                    store_rex = None
                    self.energy_missing = 0
                    self.num_rex = 0
                    energy_req_eff = min(self.energy_req, sfp_prim.energy_usable)

                self.energy_usable = (self.num_prim * sfp_prim.energy_usable) + (
                    self.num_rex * getattr(sfp_rex, "energy_usable", 0)
                )

                self.energy_total = (self.num_prim * sfp_prim.energy_total) + (
                    self.num_rex * getattr(sfp_rex, "energy_total", 0)
                )

                self.utilization = energy_req_eff / self.energy_usable
                self.dsoc_prim = sfp_prim.dsoc_usable * self.utilization
                self.dsoc_rex = getattr(sfp_rex, "dsoc_usable", 0) * self.utilization

                if sfp_prim.is_electric:
                    self.energy_req_prim = self.dsoc_prim * sfp_prim.energy_total * np.sqrt(sfp_prim.eff_roundtrip)
                    self.energy_req_rex = (
                        self.dsoc_rex
                        * getattr(sfp_rex, "energy_total", 0)
                        * np.sqrt(getattr(sfp_rex, "eff_roundtrip", 0))
                    )
                else:
                    self.energy_req_prim = self.energy_req
                    self.energy_req_rex = 0

                self.dtime_chg_prim = pd.to_timedelta(self.energy_req_prim / sfp_prim.pwr_chg_usable, unit="hour")
                self.dtime_chg_rex = pd.to_timedelta(
                    self.energy_req_rex / getattr(sfp_rex, "pwr_chg_usable", np.inf), unit="hour"
                )

                self.steps_chg_prim = self.dispatcher_prim.time.dt2steps(values=self.dtime_chg_prim)
                self.steps_chg_rex = self.dispatcher_prim.time.dt2steps(values=self.dtime_chg_rex)

                self.steps_usage_prim = self.steps_chg_prim + self.steps_rental
                self.steps_usage_rex = self.steps_chg_rex + self.steps_rental

                def min_steps_since_put(steps: int):
                    def _filter(item):
                        _, ts_put = item
                        return (env.now - ts_put) >= steps

                    return _filter

                self.request_prim = store_prim.get(
                    amount=self.num_prim, filter=min_steps_since_put(steps=self.steps_chg_prim)
                )

                if self.num_rex > 0:
                    self.request_rex = store_rex.get(
                        amount=self.num_rex, filter=min_steps_since_put(steps=self.steps_chg_rex)
                    )
                else:
                    self.request_rex = None

                yield env.timeout(0)  # resolve immediate triggers

                success_prim = self.request_prim.triggered
                success_rex = True if self.request_rex is None else self.request_rex.triggered

                if success_prim and success_rex:
                    self.processed = True
                    self.status = "success"
                    self.step_dep = env.now
                    self.time_dep = self.dispatcher_prim.time.steps2dt(self.step_dep, absolute=True)
                    self.result_prim = self.request_prim.value
                    self.result_rex = None if self.request_rex is None else self.request_rex.value
                    yield env.timeout(self.steps_rental)
                    self.step_return = env.now
                    self.time_return = self.dispatcher_prim.time.steps2dt(self.step_return, absolute=True)
                    store_prim.put(self.result_prim)
                    if self.request_rex is not None:
                        store_rex.put(self.result_rex)
                    return
                else:
                    self.status = "waiting"

                    self.request_prim.cancel()
                    if self.request_rex:
                        self.request_rex.cancel()

                    # ensure resources are put back after concurrent patience and request firing
                    # https://stackoverflow.com/q/75371166
                    if self.request_prim.triggered:
                        resource_prim = yield self.request_prim
                        store_prim.put(resource_prim)
                    if getattr(self.request_rex, "triggered", False):
                        resource_rex = yield self.request_rex
                        store_rex.put(resource_rex)

                    continue  # try next store

            yield env.timeout(1)
            self.steps_wait += 1

        self.status = "timeout"
        self.processed = True
