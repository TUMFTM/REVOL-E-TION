#!/usr/bin/env python3

import logging
import os
import statistics
import time

import numpy as np
import pandas as pd
import simpy

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Any, Optional

from . import blocks
from . import utils


class MultiFilterStorePut(simpy.resources.base.Put):
    def __init__(self, store, items):
        # Add timestamps on put to track when items entered
        now = store._env.now
        self.items = [(item, now) for item in items]
        super().__init__(store)


class MultiFilterStoreGet(simpy.resources.base.Get):
    def __init__(self, store, num=1, filter: Callable[[Tuple[Any, float]], bool] = lambda x: True):
        self.amount = num
        self.filter = filter
        super().__init__(store)


class MultiFilterStore(simpy.resources.base.BaseResource):
    def __init__(self,
                 env: simpy.Environment,
                 capacity: int,
                 initial=None):
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
            selected = filtered[:event.amount]

            # Remove these from self.items (match by identity)
            for item in selected:
                self.items.remove(item)

            # Return only the items, not their timestamps
            event.succeed([item for item, timestamp in selected])


class DispatchTimer:

    def __init__(self,
                 dti_base: pd.DatetimeIndex,
                 buffer_pre: pd.Timedelta = pd.Timedelta(days=1),
                 buffer_post: pd.Timedelta = pd.Timedelta(days=28),):

        self.dti_base = dti_base
        self.step = pd.Timedelta(self.dti_base.freq)
        self.step_hours = self.step.total_seconds() / 3600
        self.time_start = self.dti_base.min() - buffer_pre  # ensures positive step counts even with preblocks
        self.time_end = self.dti_base.max() + self.step + buffer_post

        self.dti = pd.date_range(start=self.time_start,
                                 end=self.time_end,
                                 freq=self.dti_base.freq)

    def dt2steps(self,
                 values: pd.Series):
        """
        utility method
        convert pandas datetime or timedelta values to DES steps
        """

        if pd.api.types.is_datetime64_any_dtype(values):
            # ensure that the result is at least 1, as 0 would leave no time for any action in real life
            return np.maximum(1, np.ceil((values - self.time_start) / self.step).astype(int))
        elif pd.api.types.is_timedelta64_dtype(values):
            return np.maximum(1, np.ceil(values / self.step).astype(int))
        else:
            raise ValueError(f'Unsupported type {values.dtype} for conversion to steps')

    def steps2dt(self,
                 steps: pd.Series,
                 absolute: bool = False):
        """
        utility method
        convert DES steps to pandas datetime or timedelta values
        """
        td = pd.to_timedelta(steps * self.step_hours, unit='hour')
        if not absolute:
            return td
        else:
            return td + self.time_start




class DispatchEnvironment:

    def __init__(self,
                 scenario: 'simulation.Scenario'):

        self.scenario = scenario

        self.groups = self.scenario.block_registry.get('DispatchGroupActive', {})
        self.time = DispatchTimer(dti_base=self.scenario.dti_sim)
        self.env = simpy.Environment()
        self.dispatchers = dict()

        # region create individual dispatchers
        # Battery Dispatchers need to be initialized first to allow for range extension of Vehicle Dispatchers
        for name, group in [(n, g) for n, g in self.groups.items() if not g.is_vehicle_group]:
            self.dispatchers[name] = GroupDispatcher(timer=self.time,
                                                     demand=group.demand,
                                                     env=self.env,
                                                     params=DispatchGroupParams.from_obj(group=group),
                                                     logger=self.scenario.logger)
            group.dispatcher = self.dispatchers[name]

        for name, group in [(n, g) for n, g in self.groups.items() if g.is_vehicle_group]:
            self.dispatchers[name] = GroupDispatcher(timer=self.time,
                                                     demand=group.demand,
                                                     env=self.env,
                                                     params=DispatchGroupParams.from_obj(group=group),
                                                     logger=self.scenario.logger)
            group.dispatcher = self.dispatchers[name]
        # endregion

        self.env.run()

        # rex process transfer is required before all processes can be evaluated
        for vehicle_dispatcher in [disp for disp in self.dispatchers.values() if isinstance(disp, VehicleDispatcher)]:
            vehicle_dispatcher.transfer_rex_processes()

        for disp in self.dispatchers.values():
            disp.postprocess(dti_output=self.scenario.dti_sim_extd)
            if not self.scenario.settings.largescalemode:
                path_processes = self.scenario.paths.create_result_path(suffix=f'{self.scenario.name}_{disp.params.name}_processes.csv')
                path_log = self.scenario.paths.create_result_path(suffix=f'{self.scenario.name}_{disp.params.name}_log.csv')
                disp.save_data(path_processes=path_processes,
                               path_log=path_log)

        for group in self.groups.values():
            group.log = group.dispatcher.log



@dataclass
class DispatchGroupParams:
    name: str
    subfleet_params: list
    is_vehicle_group: bool

    @classmethod
    def from_obj(cls,
                 group: 'blocks.DispatchGroup') -> 'DispatchGroupParams':

        subfleet_params = {name: SubFleetParams.from_obj(subfleet) for name, subfleet in group.subblocks.items()}

        params = dict(
            name=group.name,
            subfleet_params=subfleet_params,
            is_vehicle_group=group.is_vehicle_group,
        )

        return cls(**params)


@dataclass
class SubFleetParams:
    name: str
    units: list  # either list of names of dict {name:object}
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
    rex: Optional[bool] = False
    rex_subfleet: Optional[str] = None
    rex_dispatcher: Optional['GroupDispatcher'] = None

    @classmethod
    def from_obj(cls,
                 subfleet: 'blocks.SubFleet') -> 'SubFleetParams':

        is_electric = subfleet.type_unit in ['ev', 'mb']
        params = dict(
            name=subfleet.name,
            units=subfleet.subblocks,
            is_electric=is_electric,
        )

        if is_electric:
            unit = subfleet.subblocks[next(iter(subfleet.subblocks))]  # representative

            soc_minmax = min([unit.states.at[subfleet.scenario.starttime, 'soc_max'] for unit in subfleet.subblocks.values()])
            soc_maxmin = max([unit.states.at[subfleet.scenario.starttime, 'soc_min'] for unit in subfleet.subblocks.values()])

            params.update(
                pwr_chg=unit.pwr_chg_max,
                size_unit=unit.sizes['storage'].preexisting,
                soc_upper=statistics.median([soc_minmax, unit.soc_target, soc_maxmin]),
                soc_lower=statistics.median([soc_minmax, unit.soc_return, soc_maxmin]),
                eff_chg=unit.eff['chg_int'],
                eff_roundtrip=unit.eff['storage_roundtrip'],
                loss_rate_per_hour=unit.loss_rate_per_hour)

            if subfleet.type_unit == 'ev'

                rex_available = unit.rex is not None
                rex_subfleet = unit.rex if rex_available else None
                rex_dispatcher
                params.update(
                    rex
                )

        return cls(**params)


class GroupDispatcher:

    def __init__(self,
                 timer: DispatchTimer,
                 demand: pd.DataFrame,
                 env: simpy.Environment,
                 params: DispatchGroupParams | SubFleetParams,
                 logger: logging.Logger,
                 factor_derate: float = 0.9,):  # conservativeness factor on assumed charge power vs actually available power

        self.time = timer
        self.demand = demand
        self.env = env
        self.params = params
        self.logger = logger
        self.factor_derate = factor_derate

        # create logger for standalone operation
        if self.logger is None:
            self.logger = logging.getLogger('null')
            self.logger.addHandler(logging.NullHandler())

        # SubFleetParams object given -> convert to single subfleet DispatchGroupParams
        if isinstance(self.params, SubFleetParams):
            self.params = DispatchGroupParams(name=f'{self.params.name}_group',
                                              subfleet_params={self.params.name: self.params},
                                              is_vehicle_group=(self.params.type_unit in ['ev', 'mb']))

        unit_names = [unit for sfparams in self.params.subfleet_params.values() for unit in sfparams.units]
        log_cols = pd.MultiIndex.from_tuples(
            [
                (name, col)
                for name in unit_names
                for col in ['atbase', 'atac', 'atdc', 'dsoc', 'consumption', 'dist']
            ],
            names=['unit', 'time']
        )
        self.log = pd.DataFrame(index=self.time.dti, columns=log_cols)

        self.kpis = dict()
        self.stores = dict()

        # region estimate usable energy and power
        for sfp in self.params.subfleet_params.values():

            self.stores[sfp.name] = MultiFilterStore(env=self.env,
                                                     capacity=len(sfp.units),
                                                     initial=list(sfp.units))

            if sfp.is_electric:
                sfp.energy_total = sfp.size_unit
                sfp.dsoc_usable = sfp.soc_upper - sfp.soc_lower
                if sfp.dsoc_usable <= 0:
                    raise ValueError(f'Usable dSOC for subfleet {sfp.name} is zero or negative. '
                                     f'Check SOC targets and aging.')
                sfp.energy_usable = (sfp.dsoc_usable *
                                     sfp.energy_total *
                                     np.sqrt(sfp.eff_roundtrip))
                sfp.pwr_chg_usable = (
                        (sfp.pwr_chg *
                         sfp.eff_chg *  # charger efficiency
                         np.sqrt(sfp.eff_roundtrip) -  # storage charging efficiency
                         (sfp.loss_rate_per_hour * sfp.energy_total)  # maximum self discharge power
                         )
                        * self.factor_derate)

            else:  # non electric
                sfp.energy_total = np.inf
                sfp.energy_usable = np.inf
                sfp.dsoc_usable = 1
                sfp.pwr_chg_usable = np.inf
        # endregion

        # region calculate a priori and subfleet agnostic process data in vectorized form
        self.demand.requests['step_req'] = self.time.dt2steps(values=self.demand.requests['time_req'])
        self.demand.requests['dtime_rental'] = self.demand.requests['dtime_active'] + self.demand.requests['dtime_idle']
        self.demand.requests['steps_rental'] = self.time.dt2steps(values=self.demand.requests['dtime_rental'])
        self.demand.requests['steps_patience'] = self.time.dt2steps(values=self.demand.requests['dtime_patience'])

        self.demand.requests.sort_values(by='step_req',
                                         inplace=True,
                                         ignore_index=True)
        # endregion

        # create dataclass instances
        self.processes = {pid: DispatchProcess(pid=pid,
                                               dispatcher_prim=self,
                                               status='unprocessed',
                                               time_req=row['time_req'],
                                               step_req=row['step_req'],
                                               dtime_patience=row['dtime_patience'],
                                               steps_patience=row['steps_patience'],
                                               dtime_rental=row['dtime_rental'],
                                               steps_rental=row['steps_rental'],
                                               energy_req=row['energy_req'])
                          for pid, row in self.demand.requests.iterrows()}

    def run_standalone(self,
                       dti_output: pd.DatetimeIndex = None):
        self.env.run()
        self.postprocess(dti_output=dti_output)

        print(f'Mean FleetUnit usage rate: {self.kpis["rate_usage_mean"]:.2f}')
        print(f'Mean dispatch failure rate: {self.kpis["rate_failure"]:.2f}')

    def postprocess(self,
                    dti_output: pd.DatetimeIndex = None):
        """
        post DES method
        convert processes to time based log and calculate KPIs
        """
        if dti_output is None:
            dti_output = self.time.dti_base

        # calculate actual time points from steps
        for point in ['preblock_prim',
                      'preblock_rex',
                      'dep',
                      'return',
                      'reavail_prim',
                      'reavail_rex']:
            self.processes[f'time_{point}'] = self.time.steps2dt(steps=self.processes[f'step_{point}'],
                                                                 absolute=True)

        # region convert processes to time based log
        self.log.loc[:, (slice(None), 'atbase')] = True
        self.log.loc[:, (slice(None), 'atac')] = False
        self.log.loc[:, (slice(None), 'atdc')] = False
        self.log.loc[:, (slice(None), 'dsoc')] = 0.0
        self.log.loc[:, (slice(None), 'dist')] = 0.0
        self.log.loc[:, (slice(None), 'consumption')] = 0.0

        processes_exploded = self.processes.explode('units_prim')

        for process in [row for id, row in processes_exploded.iterrows() if row['status'] == 'success']:
            unit = process['units_prim']
            time_end = process['time_return'] - self.time.step
            power_avg = process['energy_req_prim'] / (process['steps_rental'] * self.time.step_hours)
            dist_avg = process['distance'] / process['steps_rental'] if 'distance' in process else 0

            self.log.loc[process['time_dep']:time_end, (unit, 'atbase')] = False
            self.log.loc[process['time_dep']:time_end, (unit, 'atac')] = False  # todo destination charging?
            self.log.loc[process['time_dep']:time_end, (unit, 'atdc')] = True
            self.log.loc[process['time_dep']:time_end, (unit, 'consumption')] = power_avg
            self.log.loc[process['time_dep']:time_end, (unit, 'dist')] = dist_avg
            self.log.loc[process['time_dep'], (unit, 'dsoc')] = process['dsoc_prim']

        self.log = self.log.loc[dti_output, :]
        # endregion

        # region calculate KPIs
        steps_total = len(dti_output)
        self.kpis['rate_usage_units'] = dict()
        for unit in self.params.units:
            steps_usage = processes_exploded.loc[processes_exploded['units_prim'] == unit, 'steps_usage_prim'].sum()
            self.kpis['rate_usage_units'][unit] = steps_usage / steps_total
        self.kpis['rate_usage_mean'] = np.mean(list(self.kpis['rate_usage_units'].values()))

        self.kpis['rate_failure'] = 1 - (self.processes['status'] == 'success').mean()
        # endregion

    def save_data(self,
                  path_processes: str = None,
                  path_log: str = None):
        """
        This function saves the converted log dataframe as a suitable example csv file for the energy system model.
        The resulting dataframe can also be handed to the energy system model directly in addition for faster
        delivery through execute_des.
        """
        if path_processes is not None:
            self.processes.to_csv(Path(path_processes).resolve())
        if path_log is not None:
            self.log.to_csv(Path(path_log).resolve())


class VehicleDispatcher(GroupDispatcher):

    def __init__(self,
                 timer: DispatchTimer,
                 demand: pd.DataFrame,
                 env: simpy.Environment,
                 params: DispatchGroupParams | SubFleetParams,
                 logger: logging.Logger = None,
                 factor_derate: float = 0.9):

        super().__init__(timer=timer,
                         demand=demand,
                         env=env,
                         params=params,
                         logger=logger,
                         factor_derate=factor_derate)

    def transfer_rex_processes(self):
        """
        copy data for rex unit usage from VehicleDispatcher process frame to rex (BatteryDispatcher) process frame
        """

        if not self.params.rex:
            return

        rex_processes = self.processes.loc[(self.processes['status'] == 'success') &
                                           (self.processes['request_rex']), :].copy()

        rex_processes['usecase'] = f'rex_{self.params.name}'

        rex_processes = rex_processes.rename(columns=lambda col: col.replace('_rex', '_temp')
                                             .replace('_prim', '_rex')
                                             .replace('_temp', '_prim'))

        self.params.rex_dispatcher.processes = pd.concat(
            objs=[getattr(self.rex_dispatcher, 'processes', None), rex_processes],
            join='inner')
        self.params.rex_dispatcher.processes.sort_values(
            by='step_preblock_prim',
            inplace=True,
            ignore_index=True)


class BatteryDispatcher(GroupDispatcher):

    def __init__(self,
                 timer: DispatchTimer,
                 demand: pd.DataFrame,
                 env: simpy.Environment,
                 params: DispatchGroupParams | SubFleetParams,
                 logger: logging.Logger = None,
                 factor_derate: float = 0.9):

        params.rex = False
        params.rex_subfleet = None
        params.rex_dispatcher = None

        super().__init__(timer=timer,
                         demand=demand,
                         env=env,
                         params=params,
                         logger=logger,
                         factor_derate=factor_derate)


@dataclass
class DispatchProcess:
    pid: str
    status: str
    dispatcher_prim: GroupDispatcher
    time_req: pd.Timestamp
    step_req: int
    energy_req: float
    dtime_rental: pd.Timedelta
    steps_rental: int
    dtime_patience: pd.Timedelta
    steps_patience: int
    num_prim: Optional[int] = None
    units_prim: Optional[list] = None
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
    success_prim: Optional[bool] = None
    result_prim: Optional[simpy.Resource] = None
    dispatcher_rex: Optional[GroupDispatcher] = None
    needs_rex: Optional[bool] = None
    num_rex: Optional[int] = None
    units_rex: Optional[list] = None
    dsoc_rex: Optional[float] = None
    energy_req_rex: Optional[float] = None
    dtime_chg_rex: Optional[pd.Timedelta] = None
    steps_chg_rex: Optional[int] = None
    request_rex: Optional[simpy.Process] = None
    success_rex: Optional[bool] = None
    result_rex: Optional[simpy.Resource] = None

    def __post_init__(self):
        self.dispatcher_prim.env.process(self.process())

    def process(self):

        env = self.dispatcher_prim.env

        yield env.timeout(self.step_req)

        for store_name, store in self.dispatcher_prim.stores.items():

            sfp_prim = self.dispatcher_prim.params.subfleet_params[store_name]

            if self.dispatcher_prim.params.is_vehicle_group:
                self.num_prim = 1
            else:  # Battery DispatchGroup
                self.num_prim = (np.ceil(self.demand.requests['energy_req'] / sfp_prim.energy_usable).astype(int))

            if sfp_prim.rex:  # rex is available for this subfleet/store
                sfp_rex = sfp_prim.rex_dispatcher.params.subfleet_params[sfp_prim.rex_subfleet]
                store_rex = sfp_prim.rex_dispatcher.stores[sfp_prim.rex_subfleet]
                self.energy_missing = (self.energy_req - sfp_prim.energy_usable).clip(lower=0)
                self.num_rex = np.ceil(self.energy_missing / sfp_rex.energy_usable).astype(int)
            else:
                sfp_rex = None
                store_rex = None
                self.energy_missing = 0
                self.num_rex = 0
                self.energy_req = self.energy_req.clip(upper=sfp_prim.energy_usable)

            self.energy_usable = (self.num_prim * sfp.energy_usable) + \
                                 (self.num_rex * getattr(sfp_rex, 'energy_usable', 0))

            self.energy_total = (self.num_prim * sfp.energy_total) + \
                                (self.num_rex * getattr(sfp_rex, 'energy_total', 0))

            self.utilization = self.energy_req / self.energy_usable
            self.dsoc_prim = sfp.dsoc_usable * self.utilization
            self.dsoc_rex = getattr(sfp_rex, 'dsoc_usable', 0) * self.utilization

            if sfp_prim.is_electric:
                self.energy_req_prim = self.dsoc_prim * sfp_prim.energy_total
                self.energy_req_rex = self.dsoc_rex * getattr(sfp_rex, 'energy_total', 0)
            else:
                self.energy_req_prim = self.energy_req
                self.energy_req_rex = 0

            self.dtime_chg_prim = pd.to_timedelta(self.energy_req_prim /
                                                  sfp.pwr_chg_usable,
                                                  unit='hour')
            self.dtime_chg_rex = pd.to_timedelta(self.energy_req_rex /
                                                 getattr(sfp_rex, 'pwr_chg_usable', np.inf),
                                                 unit='hour')

            self.steps_chg_prim = self.dispatcher_prim.time.dt2steps(values=self.dtime_chg_prim)
            self.steps_chg_rex = self.dispatcher_prim.time.dt2steps(values=dtime_chg_rex)

            self.steps_usage_prim = steps_chg_prim + self.steps_rental
            self.steps_usage_rex = steps_chg_rex + self.steps_rental

            def min_steps_since_put(steps: int):
                def _filter(item):
                    _, ts_put = item
                    return (env.now - ts_put) >= steps

                return _filter

            self.request_prim = store_prim.get(
                amount=self.num_prim,
                filter=min_steps_since_put(steps=self.steps_chg_prim))

            if self.num_rex > 0:
                self.request_rex = store_rex.get(
                    amount=self.num_rex,
                    filter=min_steps_since_put(steps=self.steps_chg_rex))

            yield env.timeout(0)  # resolve immediate triggers

            self.success_prim = request_prim.triggered
            self.success_rex = True if self.request_rex is None else self.request_rex.triggered

            if self.success_prim and self.success_rex:
                self.status = 'success'
                self.result_prim = self.request_prim.value
                self.result_rex = None if self.request_rex is None else self.request_rex.value
                yield env.timeout(process['steps_rental'])
                self.step_return = env.now
                store_prim.put(self.result_prim)
                if self.request_rex is not None:
                    store_rex.put(self.result_rex)
                break
            else:
                self.status = 'failure'
                self.request_prim.cancel()
                if self.request_rex:
                    self.request_rex.cancel()
                continue  # try next store

