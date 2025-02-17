#!/usr/bin/env python3

import os
import statistics

import numpy as np
import pandas as pd
import simpy

from revoletion import blocks


class MultiStoreGet(simpy.resources.base.Get):
    def __init__(self, store, num=1):
        self.amount = num
        super(MultiStoreGet, self).__init__(store)


class MultiStorePut(simpy.resources.base.Put):
    def __init__(self, store, items):
        self.items = items
        super(MultiStorePut, self).__init__(store)


class MultiStore(simpy.resources.base.BaseResource):
    def __init__(self, env, capacity):
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
            elements = self.items[(len(self.items) - event.amount - 0):]
            self.items = self.items[:(len(self.items) - event.amount - 0)]
            event.succeed(elements)


class SiteDispatcher:

    def __init__(self,
                 scenario: 'simulation.Scenario'):

        self.scenario = scenario

        self.subfleets = self.scenario.subfleets_dispatch
        if not self.subfleets:
            return

        # region extend datetimeindex
        self.time_ref = self.scenario.starttime - pd.Timedelta(days=1)  # reference time for DES steps
        time_start_overhang = scenario.dti_sim_extd[-1] + scenario.dti_sim_extd.freq
        time_end_overhang = time_start_overhang + pd.Timedelta(days=28)
        self.dti = scenario.dti_sim_extd.union(pd.date_range(start=time_start_overhang,
                                                             end=time_end_overhang,
                                                             freq=scenario.dti_sim_extd.freq))
        # endregion

        self.environment = simpy.Environment()

        # region create subfleet dispatchers
        self.dispatchers = dict()
        for subfleet_name, subfleet in self.subfleets.items():
            # BatteryDispatchers need to be initialized first to allow for range extension of ElectricVehicleDispatchers
            if subfleet.type_unit in ['mb']:
                self.dispatchers[subfleet_name] = BatteryDispatcher(subfleet=subfleet,
                                                                    parent=self,
                                                                    scenario=self.scenario)
        for subfleet_name, subfleet in self.subfleets.items():
            if subfleet.type_unit in ['ev', 'icev']:
                self.dispatchers[subfleet_name] = VehicleDispatcher(subfleet=subfleet,
                                                                    parent=self,
                                                                    scenario=self.scenario)
        # endregion

        self.environment.run()

        # rex process transfer is required before all processes can be evaluated
        for vehicle_dispatcher in [disp for disp in self.dispatchers.values() if isinstance(disp, VehicleDispatcher)]:
            vehicle_dispatcher.transfer_rex_processes()

        for disp in self.dispatchers.values():
            disp.postprocess()


class SubFleetDispatcher:

    def __init__(self,
                 subfleet: blocks.SubFleet,
                 parent: SiteDispatcher,
                 scenario: 'simulation.Scenario'):

        self.subfleet = subfleet
        self.parent = parent
        self.scenario = scenario

        self.name = self.subfleet.name
        self.demand = self.subfleet.demand
        self.env = self.parent.environment

        self.subfleet.dispatcher = self

        self.units = self.subfleet.subblocks
        log_columns = pd.MultiIndex.from_tuples(
            [(unit, lbl) for unit in self.units for lbl in ['atbase', 'atac', 'atdc', 'dsoc', 'consumption', 'dist']],
            names=['unit', 'time']
        )
        self.log = pd.DataFrame(index=self.parent.dti, columns=log_columns)
        self.kpis = dict()

        factor_derate = 0.9  # conservativeness factor on assumed charge power vs actually available power

        # region estimate usable energy and power
        unit_repr = self.units[next(iter(self.units))]  # all units are equal and representative a priori

        self.energy_total = unit_repr.sizes.loc['block', 'preexisting']

        soc_minmax = min([unit.soc_max for unit in self.units.values()])
        soc_maxmin = max([unit.soc_min for unit in self.units.values()])

        soc_upper = statistics.median([soc_minmax, unit_repr.soc_target, soc_maxmin])
        soc_lower = statistics.median([soc_minmax, unit_repr.soc_return, soc_maxmin])

        self.dsoc_usable = soc_upper - soc_lower

        if self.dsoc_usable <= 0:
            raise ValueError(f'Usable dSOC for subfleet {self.subfleet.name} is zero or negative. '
                             f'Check SOC targets and aging.')

        self.energy_usable = (self.dsoc_usable *
                              self.energy_total *
                              np.sqrt(unit_repr.eff_storage_roundtrip))

        pwr_loss_max = (1 - (1 - unit_repr.loss_rate) ** self.scenario.timestep_hours * self.energy_total)
        self.pwr_chg_usable = (unit_repr.pwr_chg_max * unit_repr.eff_chg_int - pwr_loss_max) * factor_derate
        # endregion

        # region calculate a priori process data
        self.processes = self.demand.demand.copy()
        self.processes['step_req'] = dt2steps(values=self.processes['time_req'],
                                              scenario=self.scenario)

        self.processes['steps_patience'] = dt2steps(values=self.processes['dtime_patience'],
                                                    scenario=self.scenario)

        self.processes['dtime_rental'] = self.processes['dtime_active'] + self.processes['dtime_idle']
        self.processes['steps_rental'] = dt2steps(values=self.processes['dtime_rental'],
                                                  self.scenario)

        self.processes['num_prim'] = {'mb': np.ceil(self.processes['energy_req'] / self.energy_usable).astype(int),
                                      'ev': 1,
                                      'icev': 1}[self.subfleet.type_unit]

        if self.rex:
            energy_missing = self.processes['energy_req'] - self.energy_usable
            self.processes['num_rex'] = np.ceil(energy_missing / self.rex_dispatcher.energy_usable).astype(int)
            self.processes['request_rex'] = self.processes['num_rex'] > 0
        else:
            self.processes['num_rex'] = 0
            self.processes['request_rex'] = False
            self.processes['energy_req'] = self.processes['energy_req'].clip(upper=self.energy_usable)  # clip to max

        self.processes['energy_usable'] = (
                self.energy_usable + (self.processes['num_rex'] * getattr(self.rex_dispatcher, 'energy_usable', 0)))
        self.processes['energy_total'] = (
                self.energy_total + (self.processes['num_rex'] * getattr(self.rex_dispatcher, 'energy_total', 0)))

        utilization = self.processes['energy_req'] / self.processes['energy_usable']
        self.processes['dsoc_prim'] = self.dsoc_usable * utilization
        self.processes['dsoc_rex'] = getattr(self.rex_dispatcher, 'dsoc_usable', 0) * utilization

        self.processes['energy_req_prim'] = self.processes['dsoc_prim'] * self.energy_total
        self.processes['energy_req_rex'] = self.processes['dsoc_rex'] * getattr(self.rex_dispatcher, 'energy_total', 0)

        self.processes['dtime_chg_prim'] = pd.to_timedelta(self.processes['energy_req_prim'] /
                                                           self.pwr_chg_usable,
                                                           unit='hour')
        self.processes['dtime_chg_rex'] = pd.to_timedelta(self.processes['energy_req_rex'] /
                                                          getattr(self.rex_dispatcher, 'pwr_chg_usable', np.inf),
                                                          unit='hour')

        self.processes['steps_chg_prim'] = dt2steps(self.processes['dtime_chg_prim'], self.scenario)
        self.processes['steps_chg_rex'] = dt2steps(self.processes['dtime_chg_rex'], self.scenario)

        self.processes['steps_preblock_prim'] = {'ev': self.processes['steps_chg_prim'],
                                                 'icev': 0,
                                                 'mb': 0}[self.subfleet.type_unit]
        self.processes['steps_postblock_prim'] = {'ev':0,
                                                  'icev': 0,
                                                  'mb': self.processes['steps_chg_prim']}[self.subfleet.type_unit]
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

        self.store = MultiStore(env=self.parent.environment,
                                capacity=self.subfleet.num)

        for unit in self.units.values():
            self.store.put([unit.name])

        for idx, row in self.processes.iterrows():
            self.parent.environment.process(self.define_process(id=idx))

    def dt2steps(self,
                 values: pd.Series):
        """
        utility method
        convert pandas datetime or timedelta values to DES steps
        """
        if pd.api.types.is_datetime64_any_dtype(values):
            # ensure that the result is at least 1, as 0 would leave no time for any action in real life
            return np.maximum(1, np.ceil((values - self.parent.time_ref) / self.scenario.timestep_td).astype(int))
        elif pd.api.types.is_timedelta64_dtype(values):
            return np.maximum(1, np.ceil(values / self.scenario.timestep_td).astype(int))
        else:
            raise ValueError(f'Unsupported type {values.dtype} for conversion to steps')

    def steps2dt(self,
                 steps: pd.Series,
                 absolute: bool = False):
        """
        utility method
        convert DES steps to pandas datetime or timedelta values
        """
        td = pd.to_timedelta(steps * self.scenario.timestep_hours, unit='hour')
        if not absolute:
            return td
        else:
            return td + self.parent.time_ref

    def define_process(self,
                       id: int):
        """
        runtime DES method
        actual definition of the runtime process steps for DES
        """
        # region initialize process as unsuccessful
        self.processes['status'] = 'unprocessed'
        self.processes['units_prim'] = None
        self.processes['units_rex'] = None

        result_prim = [False]
        result_rex = [False]
        request_prim = False
        request_rex = False
        # endregion

        # region request primary resource(s) at preblock time
        yield self.env.timeout(self.processes.at[id, 'step_preblock_prim'])

        self.scenario.logger.debug(f'{self.name} process {id} preblocked at {self.env.now}')

        with self.store.get(self.processes.at[id, 'num_prim']) as request_prim:
            result_prim = (yield request_prim | self.env.timeout(self.processes.at[id, 'steps_patience']))

        self.scenario.logger.debug(f'{self.name} process {id} requested {self.processes.at[id, "num_prim"]}'
                                   f' primary resource(s) at {self.env.now}')
        # endregion

        yield self.env.timeout(self.processes.at[id, 'steps_preblock_prim'])

        # region request rex resource(s) at actual request time
        if self.processes.at[id, 'request_rex']:
            with self.rex_dispatcher.store.get(self.processes.at[id, 'num_rex']) as request_rex:
                result_rex = yield request_rex | self.env.timeout(self.processes.at[id, 'steps_patience'])

            self.scenario.logger.debug(f'{self.name} process {id} requested '
                                       f'{self.processes.at[id, "num_rex"]} secondary resource(s) '
                                       f'at {self.env.now}')
        # endregion

        if (request_prim in result_prim) and (request_rex in result_rex):
            # region postprocessing if all requests are successful
            self.processes.loc[id, 'status'] = 'success'
            self.processes.loc[id, 'step_dep'] = self.env.now

            self.processes.loc[id, 'units_prim'] = request_prim.value
            self.scenario.logger.debug(f'{self.name} process {id} received primary resource '
                                       f'{self.processes.loc[id, "units_prim"]} at {self.env.now}')

            if self.processes.at[id, 'request_rex']:
                self.processes.loc[id, 'units_rex'] = request_rex.value
                self.scenario.logger.debug(f'{self.name} process {id} received secondary resource'
                                           f' {self.processes.loc[id, "units_rex"]} at {self.env.now}')

            # cover the usage & idle time
            yield self.env.timeout(self.processes.at[id, 'steps_rental'])
            self.processes.loc[id, 'step_return'] = self.env.now

            # cover the postblock time
            yield self.env.timeout(self.processes.at[id, 'steps_postblock_prim'])
            self.processes.loc[id, 'step_reavail_primary'] = self.env.now
            yield self.env.timeout(self.processes.at[id, 'steps_postblock_rex'] -
                                   self.processes.at[id, 'steps_postblock_prim'])
            self.processes.loc[id, 'step_reavail_secondary'] = self.env.now

            # put back resources
            self.store.put(result_prim[request_prim])
            self.scenario.logger.debug(
                f'{self.name} process {id} returned resource(s) {self.processes.at[id, "units_prim"]}'
                f' at {self.env.now}. Primary store content after return: {self.store.items}')

            if self.processes.at[id, 'request_rex']:
                self.rex_dispatcher.store.put(result_rex[self.request_rex])
                self.scenario.logger.debug(f'{self.name} process {id} returned secondary resource(s)'
                                           f'{self.request_rex.value} at {self.env.now}. '
                                           f'Secondary store content after return: '
                                           f'{self.rex_dispatcher.store.items}')
            # endregion

        else:
            # region postprocessing if at least one request is unsuccessful

            # record type of failure
            if (request_prim not in result_prim) and (request_rex not in result_rex):
                self.processes.loc[id, 'status'] = 'failure_both'
                self.scenario.logger.debug(f'{self.name} process {id} failed '
                                           f'(did not receive either resource) at {self.env.now}. '
                                           f'Primary store content after failure: {self.store.items}. '
                                           f'Secondary store content after failure: '
                                           f'{self.rex_dispatcher.store.items}')

            elif request_prim not in result_prim:
                self.processes.loc[id, 'status'] = 'failure_primary'
                if self.processes.at[id, 'request_rex']:
                    self.scenario.logger.debug(f'{self.name} process {id} failed '
                                               f'(didn´t receive primary resource(s)) at {self.env.now}. '
                                               f'Primary store content after fail: {self.store.items}. '
                                               f'Secondary store content after failure: '
                                               f'{self.rex_dispatcher.store.items}')
                else:
                    self.scenario.logger.debug(f'{self.name} process {id} failed '
                                               f'(didn´t receive primary resource(s)) at {self.env.now}. '
                                               f'Primary store content after fail: {self.store.items}')

            elif request_rex not in result_rex:
                self.processes.loc[id, 'status'] = 'failure_secondary'
                self.scenario.logger.debug(f'{self.name} process {id} failed '
                                           f'(didn´t receive secondary resource(s)) at {self.env.now}. '
                                           f'Primary store content after fail: {self.store.items}. '
                                           f'Secondary store content after failure: '
                                           f'{self.rex_dispatcher.store.items}')
            # endregion

            # region ensure resources are put back
            # https://stackoverflow.com/q/75371166
            if request_prim.triggered:
                primary_resource = yield request_prim
                self.store.put(primary_resource)
                self.scenario.logger.debug(f'{self.name} process {id} returned '
                                           f'primary resource {primary_resource} at {self.env.now}. '
                                           f'Primary store content after return: {self.store.items}.')

            if hasattr(request_rex, 'triggered'):
                if request_rex.triggered:
                    secondary_resource = yield request_rex
                    self.rex_dispatcher.store.put(secondary_resource)
                    self.scenario.logger.debug(f'{self.name} process {id} returned '
                                               f'secondary resource {secondary_resource} at {self.env.now}. '
                                               f'Primary store content after return: {self.store.items}. '
                                               f'Secondary store content after return: '
                                               f'{self.rex_dispatcher.store.items}')
            # endregion

        self.scenario.logger.debug(f'{self.name} process {id} finished at {self.env.now}')

    def postprocess(self):
        """
        post DES method
        convert processes to time based log and calculate KPIs
        """
        # calculate actual time points from steps
        for point in ['preblock_prim', 'preblock_rex', 'dep', 'return', 'reavail_prim', 'reavail_rex']:
            self.processes[f'time_{point}'] = steps2dt(steps=self.processes[f'step_{point}'],
                                                       scenario=self.scenario,
                                                       absolute=True)

        # region convert processes to time based log
        self.log.loc[:, (slice(None), 'atbase')] = True
        self.log.loc[:, (slice(None), 'atac')] = False
        self.log.loc[:, (slice(None), 'atdc')] = False
        self.log.loc[:, (slice(None), 'dsoc')] = 0.0
        self.log.loc[:, (slice(None), 'dist')] = 0.0
        self.log.loc[:, (slice(None), 'consumption')] = 0.0

        for process in [row for id, row in self.processes.iterrows() if row['status'] == 'success']:
            for unit in process['units_prim']:

                time_end = process['time_return'] - self.scenario.timestep_td
                power_avg = process['energy_req_prim'] / (process['steps_rental'] * self.scenario.timestep_hours)
                dist_avg = process['distance'] / process['steps_rental']

                self.log.loc[process['time_dep']:time_end, (unit, 'atbase')] = False
                self.log.loc[process['time_dep']:time_end, (unit, 'atac')] = False  # todo destination charging?
                self.log.loc[process['time_dep']:time_end, (unit, 'atdc')] = True
                self.log.loc[process['time_dep']:time_end, (unit, 'consumption')] = power_avg
                self.log.loc[process['time_dep']:time_end, (unit, 'dist')] = dist_avg
                self.log.loc[process['time_dep'], (unit, 'dsoc')] = process['dsoc_prim']
        # endregion

        self.log = self.log.loc[self.scenario.dti_sim_extd, :]

        # region calculate KPIs
        steps_total = self.log.shape[0]
        # make an individual row for each used unit in a process
        processes_exploded = self.processes.explode('units_prim')

        # calculate percentage of DES (not sim, the latter is shorter) time
        # occupied by active, idle & recharge times
        for unit in list(self.subfleet.subblocks.keys()):
            processes = processes_exploded.loc[processes_exploded['commodities_primary'] == unit, :]
            steps_blocked = processes['steps_charge_primary'].sum() + processes['steps_rental'].sum()
            self.use_rate[unit] = steps_blocked / steps_total
        self.subfleet.use_rate = np.mean(list(self.use_rate.values()))

        # calculate overall percentage of failed trips
        n_success = self.processes.loc[self.processes['status'] == 'success', 'status'].shape[0]
        n_total = self.processes.shape[0]
        self.fail_rate = self.subfleet.fail_rate = 1 - (n_success / n_total)
        # endregion

    def save_data(self):
        """
        This function saves the converted log dataframe as a suitable example csv file for the energy system model.
        The resulting dataframe can also be handed to the energy system model directly in addition for faster
        delivery through execute_des.
        """
        processes_path = os.path.join(
            self.scenario.run.path_result_dir,
            f'{self.scenario.run.runtimestamp}_'
            f'{self.scenario.run.name}_'
            f'{self.scenario.name}_'
            f'{self.subfleet.name}_'
            f'processes.csv')
        self.processes.to_csv(processes_path)

        log_path = os.path.join(
            self.scenario.run.path_result_dir,
            f'{self.scenario.run.runtimestamp}_'
            f'{self.scenario.run.name}_'
            f'{self.scenario.name}_'
            f'{self.subfleet.name}_'
            f'log.csv')
        self.log.to_csv(log_path)


class VehicleDispatcher(SubFleetDispatcher):

    def __init__(self,
                 subfleet: 'blocks.VehicleFleet',
                 parent: SiteDispatcher,
                 scenario: 'simulation.Scenario'):

        if subfleet.rex is not None:
            self.rex = True
            self.rex_fleet = scenario.blocks.get(subfleet.rex, None)
            self.rex_dispatcher = self.rex_fleet.get('dispatcher', None)

            base_msg = f'Scenario "{scenario.name}" - Block "{subfleet.parent.name}" -' \
                       f'Subfleet "{subfleet.name}": selected range extender fleet "{self.rex}"'

            if self.rex is None:
                raise ValueError(f'{base_msg} does not exist')
            elif not isinstance(self.rex_fleet, blocks.BatteryFleet):
                raise ValueError(f'{base_msg} is not a BatteryFleet')
            elif len(self.rex.subfleets) > 1:  # only one subfleet allowed
                raise ValueError(f'{base_msg} must have exactly one subfleet')
            elif not self.rex.subfleets[0].data_source in ['usecases', 'demand']:
                raise ValueError(f'{base_msg} - data source"{self.rex.subfleet[0].data_source}" is not allowed. '
                                 f'Allowed values: ["usecases", "demand"]')

        else:
            self.rex = False
            self.rex_fleet = None
            self.rex_dispatcher = None

        super().__init__(subfleet=subfleet,
                         parent=parent,
                         scenario=scenario)

    def transfer_rex_processes(self):
        """
        copy data for rex unit usage from VehicleDispatcher process frame to rex (BatteryDispatcher) process frame
        """

        if not self.rex:
            return

        rex_processes = self.processes.loc[(self.processes['status'] == 'success') &
                                           (self.processes['request_rex']), :].copy()

        rex_processes['name_usecase'] = f'rex_{self.subfleet.name}'

        def swap_rex(col_name):
            if 'prim' in col_name:
                return col_name.replace('prim', 'carrier')
            elif 'rex' in col_name:
                return col_name.replace('rex', 'prim')
            return col_name
        rex_processes.columns = [swap_rex(col) for col in rex_processes.columns]
        rex_processes.drop([col for col in rex_processes.columns if 'carrier' in col], axis=1, inplace=True)

        self.rex_dispatcher.processes = pd.concat(objs=[getattr(self.rex_dispatcher, 'processes', None), rex_processes],
                                                  join='inner')
        self.rex_dispatcher.sort_values(by='step_preblock_prim',
                                        inplace=True,
                                        ignore_index=True)


class BatteryDispatcher(SubFleetDispatcher):

    def __init__(self,
                 subfleet: blocks.SubFleet,
                 parent: SiteDispatcher,
                 scenario: 'simulation.Scenario'):

        self.rex = False
        self.rex_fleet = None
        self.rex_dispatcher = None

        super().__init__(subfleet=subfleet,
                         parent=parent,
                         scenario=scenario)




