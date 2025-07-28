#!/usr/bin/env python3

import logging
import os
import statistics

import numpy as np
import pandas as pd
import simpy

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import blocks
from . import utils


class MultiStoreGet(simpy.resources.base.Get):
    def __init__(self, store, num=1):
        self.amount = num
        super(MultiStoreGet, self).__init__(store)


class MultiStorePut(simpy.resources.base.Put):
    def __init__(self, store, items):
        self.items = items
        super(MultiStorePut, self).__init__(store)


class MultiStore(simpy.resources.base.BaseResource):
    def __init__(self,
                 env: simpy.Environment,
                 capacity: int):
        super(MultiStore, self).__init__(env, capacity)
        self.items = []

    put = simpy.core.BoundClass(MultiStorePut)
    get = simpy.core.BoundClass(MultiStoreGet)

    def _do_put(self, event):
        if len(self.items) + len(event.items) <= self._capacity:
            self.items.extend(event.items)
            event.succeed()

    def _do_get(self, event):
        if self.items and event.amount <= len(self.items):
            elements = self.items[:event.amount]
            self.items = self.items[event.amount:]
            event.succeed(elements)


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
        # BatteryDispatchers need to be initialized first to allow for range extension of ElectricVehicleDispatchers
        for name, group in [(n, g) for n, g in self.groups.items() if not g.is_vehicle_group]:
            self.dispatchers[name] = BatteryDispatcher(timer=self.time,
                                                       demand=group.demand,
                                                       env=self.env,
                                                       params=DispatchGroupParams.from_obj(group=group),
                                                       logger=self.scenario.logger)
            group.dispatcher = self.dispatchers[name]

        for name, group in [(n, g) for n, g in self.groups.items() if g.is_vehicle_group]:
            self.dispatchers[name] = VehicleDispatcher(timer=self.time,
                                                       demand=subfleet.demand,
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

        subfleet_params = {name: SubFleetParams.from_obj(subfleet) for name, subfleet in group.subblocks}

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
    size_unit: float
    pwr_chg: float
    is_electric: bool = None
    soc_upper: Optional[float] = 1.0
    soc_lower: Optional[float] = 0.0
    eff_chg: Optional[float] = 1.0
    eff_roundtrip: Optional[float] = 1.0
    loss_rate_per_hour: Optional[float] = 0.0
    rex: Optional[str] = None

    @classmethod
    def from_obj(cls,
                 subfleet: 'blocks.SubFleet') -> 'SubFleetParams':

        is_electric = subfleet.type_unit in ['ev', 'mb']
        unit = subfleet.subblocks[next(iter(subfleet.subblocks))]  # representative

        params = dict(
            name=subfleet.name,
            units=subfleet.subblocks,
            pwr_chg=None,
            is_electric=is_electric,
            size_unit=None,
            soc_upper=None,
            soc_lower=None,
            eff_chg=None,
            eff_roundtrip=None,
            loss_rate_per_hour=None,
            rex=None,
        )

        if is_electric:

            soc_minmax = min([unit.states.at[subfleet.scenario.starttime, 'soc_max'] for unit in subfleet.subblocks.values()])
            soc_maxmin = max([unit.states.at[subfleet.scenario.starttime, 'soc_min'] for unit in subfleet.subblocks.values()])

            params.update(
                pwr_chg=unit.pwr_chg_max,
                size_unit=unit.sizes['storage'].preexisting,
                soc_upper=statistics.median([soc_minmax, unit.soc_target, soc_maxmin]),
                soc_lower=statistics.median([soc_minmax, unit.soc_return, soc_maxmin]),
                eff_chg=unit.eff['chg_int'],
                eff_roundtrip=unit.eff['storage_roundtrip'],
                loss_rate_per_hour=unit.loss_rate_per_hour,
                rex=getattr(unit, 'rex', None)
            )

        return cls(**params)


class GroupDispatcher:

    def __init__(self,
                 timer: DispatchTimer,
                 demand: pd.DataFrame,
                 env: simpy.Environment,
                 params: DispatchGroupParams | SubFleetParams,
                 logger: logging.Logger,
                 factor_derate: float,  # conservativeness factor on assumed charge power vs actually available power
                 ):

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

        unit_names = [unit for sfparams in self.params.subfleet_params for unit in sfparams.units]
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

        self.store = MultiStore(env=self.env,
                                capacity=len(unit_names))
        for name in unit_names:
            self.store.put([name])

        self.preprocess_dispatch()

    def preprocess_dispatch(self):

        # region estimate usable energy and power
        for sfname, sfparams in self.params.subfleet_params.items():

            if sfparams.is_electric:
                sfparams.energy_total = sfparams.size_unit
                sfparams.dsoc_usable = sfparams.soc_upper - sfparams.soc_lower
                if sfparams.dsoc_usable <= 0:
                    raise ValueError(f'Usable dSOC for subfleet {sfparams.name} is zero or negative. '
                                     f'Check SOC targets and aging.')
                sfparams.energy_usable = (sfparams.dsoc_usable *
                                          sfparams.energy_total *
                                          np.sqrt(sfparams.params.eff_roundtrip))
                sfparams.pwr_chg_usable = (
                        (sfparams.params.pwr_chg *
                         sfparams.params.eff_chg *  # charger efficiency
                         np.sqrt(sfparams.params.eff_roundtrip) -  # storage charging efficiency
                         (sfparams.params.loss_rate_per_hour * sfparams.energy_total)  # maximum self discharge power
                         )
                        * sfparams.factor_derate)

            else:  # non electric
                sfparams.energy_total = np.inf
                sfparams.energy_usable = np.inf
                sfparams.dsoc_usable = 1
                sfparams.pwr_chg_usable = np.inf
        # endregion

        # region calculate a priori and subfleet agnostic process data
        self.processes = self.demand.requests.copy()

        #initialize process as unsuccessful
        self.processes['status'] = 'unprocessed'
        self.processes['units_prim'] = None
        self.processes['units_rex'] = None
        result_prim = [False]
        result_rex = [False]
        request_prim = False
        request_rex = False

        self.processes['step_req'] = self.time.dt2steps(values=self.processes['time_req'])
        self.processes['steps_patience'] = self.time.dt2steps(values=self.processes['dtime_patience'])
        self.processes['dtime_rental'] = self.processes['dtime_active'] + self.processes['dtime_idle']
        self.processes['steps_rental'] = self.time.dt2steps(values=self.processes['dtime_rental'])
        self.processes['num_prim'] = (np.ceil(self.processes['energy_req'] / self.energy_usable).astype(int))\
            if not self.params.is_vehicle_group else 1

        # preblock time is required a priori, but rex status is not known -> initially assume no rex
        self.processes['energy_usable_est'] = (
                self.energy_usable
        self.processes['dsoc_prim_']



        if self.params.rex:
            energy_missing = (self.processes['energy_req'] - self.energy_usable).clip(lower=0)
            self.processes['num_rex'] = np.ceil(energy_missing / self.params.rex_dispatcher.energy_usable).astype(int)
            self.processes['request_rex'] = self.processes['num_rex'] > 0
        else:
            self.processes['num_rex'] = 0
            self.processes['request_rex'] = False
            self.processes['energy_req'] = self.processes['energy_req'].clip(upper=self.energy_usable)

        self.processes['energy_usable'] = (
                self.energy_usable + (self.processes['num_rex'] * getattr(self.params.rex_dispatcher, 'energy_usable', 0)))
        self.processes['energy_total'] = (
                self.energy_total + (self.processes['num_rex'] * getattr(self.params.rex_dispatcher, 'energy_total', 0)))

        utilization = self.processes['energy_req'] / self.processes['energy_usable']
        self.processes['dsoc_prim'] = self.dsoc_usable * utilization
        self.processes['dsoc_rex'] = getattr(self.params.rex_dispatcher, 'dsoc_usable', 0) * utilization

        if self.params.is_electric:
            self.processes['energy_req_prim'] = self.processes['dsoc_prim'] * self.energy_total
            self.processes['energy_req_rex'] = self.processes['dsoc_rex'] * getattr(self.params.rex_dispatcher, 'energy_total', 0)
        else:
            self.processes['energy_req_prim'] = self.processes['energy_req']
            self.processes['energy_req_rex'] = 0

        self.processes['dtime_chg_prim'] = pd.to_timedelta(self.processes['energy_req_prim'] /
                                                           self.pwr_chg_usable,
                                                           unit='hour')
        self.processes['dtime_chg_rex'] = pd.to_timedelta(self.processes['energy_req_rex'] /
                                                          getattr(self.params.rex_dispatcher, 'pwr_chg_usable', np.inf),
                                                          unit='hour')

        self.processes['steps_chg_prim'] = self.time.dt2steps(values=self.processes['dtime_chg_prim'])
        self.processes['steps_chg_rex'] = self.time.dt2steps(values=self.processes['dtime_chg_rex'])

        self.processes['steps_usage_prim'] = self.processes['steps_chg_prim'] + self.processes['steps_rental']
        self.processes['steps_usage_rex'] = self.processes['steps_chg_rex'] + self.processes['steps_rental']

        self.processes['steps_preblock_prim'] = (self.processes['steps_chg_prim']
                                                 if (self.params.is_electric and self.params.is_vehicle)
                                                 else 0)
        self.processes['steps_postblock_prim'] = (self.processes['steps_chg_prim']
                                                  if (self.params.is_electric and not self.params.is_vehicle)
                                                  else 0)

        self.processes['steps_preblock_rex'] = 0
        self.processes['steps_postblock_rex'] = self.processes['steps_chg_rex']

        self.processes['step_preblock_prim'] = self.processes['step_req'] - self.processes['steps_preblock_prim']
        self.processes['step_postblock_prim'] = self.processes['step_req'] + self.processes['steps_postblock_prim']
        self.processes['step_preblock_rex'] = self.processes['step_req'] - self.processes['steps_preblock_rex']
        self.processes['step_postblock_rex'] = self.processes['step_req'] + self.processes['steps_postblock_rex']

        self.processes.sort_values(by='step_preblock_prim',
                                   inplace=True,
                                   ignore_index=True)
        # endregion

        for idx, row in self.processes.iterrows():
            self.env.process(self.define_process(id=idx))

    def process_dispatch(self,
                         id: int):
        """
        runtime DES method
        actual definition of the runtime process steps for DES
        """

        result_prim = [False]
        result_rex = [False]
        request_prim = False
        request_rex = False


        # region request primary resource(s) at preblock time
        yield self.env.timeout(self.processes.at[id, 'step_preblock_prim'])

        self.logger.debug(f'{self.params.name} process {id} preblocked at {self.env.now}')

        with self.store.get(self.processes.at[id, 'num_prim']) as request_prim:
            result_prim = (yield request_prim | self.env.timeout(self.processes.at[id, 'steps_patience']))

        self.logger.debug(f'{self.params.name} process {id} requested {self.processes.at[id, "num_prim"]}'
                          f' primary resource(s) at {self.env.now}')
        # endregion

        yield self.env.timeout(self.processes.at[id, 'steps_preblock_prim'])

        # region request rex resource(s) at actual request time
        if self.processes.at[id, 'request_rex']:
            with self.params.rex_dispatcher.store.get(self.processes.at[id, 'num_rex']) as request_rex:
                result_rex = yield request_rex | self.env.timeout(self.processes.at[id, 'steps_patience'])

            self.logger.debug(f'{self.params.name} process {id} requested '
                              f'{self.processes.at[id, "num_rex"]} secondary resource(s) '
                              f'at {self.env.now}')
        # endregion

        if (request_prim in result_prim) and (request_rex in result_rex):
            # region postprocessing if all requests are successful
            self.processes.loc[id, 'status'] = 'success'
            self.processes.loc[id, 'step_dep'] = self.env.now

            self.processes.at[id, 'units_prim'] = request_prim.value
            self.logger.debug(f'{self.params.name} process {id} received primary resource '
                              f'{self.processes.loc[id, "units_prim"]} at {self.env.now}')

            if self.processes.at[id, 'request_rex']:
                self.processes.at[id, 'units_rex'] = request_rex.value
                self.logger.debug(f'{self.params.name} process {id} received secondary resource'
                                  f' {self.processes.loc[id, "units_rex"]} at {self.env.now}')

            # cover the usage & idle time
            yield self.env.timeout(self.processes.at[id, 'steps_rental'])
            self.processes.loc[id, 'step_return'] = self.env.now

            # cover the postblock time
            yield self.env.timeout(self.processes.at[id, 'steps_postblock_prim'])
            self.processes.loc[id, 'step_reavail_prim'] = self.env.now
            yield self.env.timeout(max(0, self.processes.at[id, 'steps_postblock_rex'] -
                                       self.processes.at[id, 'steps_postblock_prim']))
            self.processes.loc[id, 'step_reavail_rex'] = self.env.now

            # put back resources
            self.store.put(result_prim[request_prim])
            self.logger.debug(
                f'{self.params.name} process {id} returned resource(s) {self.processes.at[id, "units_prim"]}'
                f' at {self.env.now}. Primary store content after return: {self.store.items}')

            if self.processes.at[id, 'request_rex']:
                self.params.rex_dispatcher.store.put(result_rex[request_rex])
                self.logger.debug(f'{self.params.name} process {id} returned secondary resource(s)'
                                  f'{request_rex.value} at {self.env.now}. '
                                  f'Secondary store content after return: '
                                  f'{self.params.rex_dispatcher.store.items}')
            # endregion

        else:
            # region postprocessing if at least one request is unsuccessful

            # record type of failure
            if (request_prim not in result_prim) and (request_rex not in result_rex):
                self.processes.loc[id, 'status'] = 'failure_both'
                self.logger.debug(f'{self.params.name} process {id} failed '
                                  f'(did not receive either resource) at {self.env.now}. '
                                  f'Primary store content after failure: {self.store.items}. '
                                  f'Secondary store content after failure: '
                                  f'{self.params.rex_dispatcher.store.items}')

            elif request_prim not in result_prim:
                self.processes.loc[id, 'status'] = 'failure_primary'
                if self.processes.at[id, 'request_rex']:
                    self.logger.debug(f'{self.params.name} process {id} failed '
                                      f'(didn´t receive primary resource(s)) at {self.env.now}. '
                                      f'Primary store content after fail: {self.store.items}. '
                                      f'Secondary store content after failure: '
                                      f'{self.params.rex_dispatcher.store.items}')
                else:
                    self.logger.debug(f'{self.params.name} process {id} failed '
                                      f'(didn´t receive primary resource(s)) at {self.env.now}. '
                                      f'Primary store content after fail: {self.store.items}')

            elif request_rex not in result_rex:
                self.processes.loc[id, 'status'] = 'failure_secondary'
                self.logger.debug(f'{self.params.name} process {id} failed '
                                  f'(didn´t receive secondary resource(s)) at {self.env.now}. '
                                  f'Primary store content after fail: {self.store.items}. '
                                  f'Secondary store content after failure: '
                                  f'{self.params.rex_dispatcher.store.items}')
            # endregion

            # region ensure resources are put back
            # https://stackoverflow.com/q/75371166
            if request_prim.triggered:
                resource_prim = yield request_prim
                self.store.put(resource_prim)
                self.logger.debug(f'{self.params.name} process {id} returned '
                                  f'primary resource {resource_prim} at {self.env.now}. '
                                  f'Primary store content after return: {self.store.items}.')

            if hasattr(request_rex, 'triggered'):
                if request_rex.triggered:
                    resource_rex = yield request_rex
                    self.params.rex_dispatcher.store.put(resource_rex)
                    self.logger.debug(f'{self.params.name} process {id} returned '
                                      f'secondary resource {resource_rex} at {self.env.now}. '
                                      f'Primary store content after return: {self.store.items}. '
                                      f'Secondary store content after return: '
                                      f'{self.params.rex_dispatcher.store.items}')
            # endregion

        self.logger.debug(f'{self.params.name} process {id} finished at {self.env.now}')

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
                 params: SubFleetParams,
                 logger: logging.Logger = None,
                 factor_derate: float = 0.9):

        if params.rex is not None:
            params.rex_subfleet = scenario.block_registry.get('SubFleet', {}).get(self.params.rex, None)
            params.rex = True

            BASE_MSG = f'Scenario "{scenario.name}" - Block "{subfleet.parent.name}" -' \
                       f'Subfleet "{subfleet.name}": selected range extender fleet "{subfleet.rex}"'

            if self.rex_subfleet is None:
                raise ValueError(f'{BASE_MSG} does not exist')
            elif not self.rex_subfleet.type_unit.lower() == 'mb':
                raise ValueError(f'{BASE_MSG} is not a Battery SubFleet')
            elif self.rex_subfleet not in scenario.block_registry.get('SubFleetDispatch', {}).values():
                raise ValueError(f'{BASE_MSG} is not dispatched and cannot be used as range extender')

            params.rex_dispatcher = params.rex_subfleet.dispatcher
        else:
            params.rex = False
            params.rex_subfleet = None
            params.rex_dispatcher = None

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
                 params: SubFleetParams,
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
