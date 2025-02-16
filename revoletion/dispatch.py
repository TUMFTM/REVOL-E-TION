#!/usr/bin/env python3

import os
import statistics

import numpy as np
import pandas as pd
import simpy

from revoletion import blocks


def dt2steps(series, sc):
    out = None  # default value
    ref_time = sc.starttime - pd.Timedelta(days=1)

    if pd.api.types.is_datetime64_any_dtype(series):
        # ensure that the result is at least 1, as 0 would leave no time for any action in real life
        out = np.maximum(1, np.ceil((series - ref_time) / sc.timestep_td).astype(int))

    elif pd.api.types.is_timedelta64_dtype(series):
        out = np.maximum(1, np.ceil(series / sc.timestep_td).astype(int))

    return out


def steps2dt(series, sc, absolute=False):
    ref_time = sc.starttime - pd.Timedelta(days=1)
    out = pd.to_timedelta(series * sc.timestep_hours, unit='hour')
    if absolute:
        out += ref_time
    return out


class SiteDispatcher:

    def __init__(self,
                 scenario: 'simulation.Scenario'):

        self.scenario = scenario

        self.subfleets = self.scenario.subfleets_dispatch
        if not self.subfleets:
            return

        # region extend datetimeindex
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
                                                                    scenario=self.scenario)
        for subfleet_name, subfleet in self.subfleets.items():
            if subfleet.type_unit in ['ev']:
                self.dispatchers[subfleet_name] = ElectricVehicleDispatcher(subfleet=subfleet,
                                                                            scenario=self.scenario)
            if subfleet.type_unit in ['icev']:
                self.dispatchers[subfleet_name] = CombustionVehicleDispatcher(subfleet=subfleet,
                                                                              scenario=self.scenario)
        # endregion

        self.environment.run()

        for dispatcher in self.dispatchers.values():
            dispatcher.postprocess()

        # todo move to subfleet dispatchers
        # reconvert time steps to actual times
        for rental_system in scenario.rental_systems.values():
            rental_system.processes['time_dep'] = steps2dt(rental_system.processes['step_dep'],
                                                           scenario,
                                                           absolute=True)
            rental_system.processes['time_return'] = steps2dt(rental_system.processes['step_return'],
                                                              scenario,
                                                              absolute=True)
            rental_system.processes['time_reavail_primary'] = steps2dt(rental_system.processes['step_reavail_primary'],
                                                                       scenario,
                                                                       absolute=True)
            if 'step_reavail_secondary' in rental_system.processes.columns:
                rental_system.processes['time_reavail_secondary'] = steps2dt(rental_system.processes[
                                                                                 'step_reavail_secondary'],
                                                                             scenario,
                                                                             absolute=True)

        # add additional rex processes from VehicleRentalSystems with rex to BatteryRentalSystems to complete process dataframe
        for rental_system in [rs for rs in scenario.rental_systems.values() if (rs.fleet.rex_fleet is not None)]:
            rental_system.transfer_rex_processes()

        # reframe logging results to resource-based view instead of process based (and save)
        for rental_system in scenario.rental_systems.values():
            rental_system.convert_process_log()
            rental_system.calc_performance_metrics()
            if run.save_results_dispatch:
                rental_system.save_data()


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


class SubFleetDispatcher:

    def __init__(self,
                 subfleet: blocks.SubFleet,
                 scenario: 'simulation.Scenario'):

        self.subfleet = subfleet
        self.scenario = scenario
        self.name = self.subfleet.name
        self.demand = self.subfleet.demand

        self.subfleet.dispatcher = self

        factor_derate = 0.9  # conservativeness factor on assumed charge power vs actually available power

        # region estimate usable energy and power
        self.units = self.subfleet.subblocks
        unit_repr = self.units[self.units.keys()[0]]  # a priori, all units are representative

        self.energy_total = unit_repr.sizes.loc['pc', 'existing']

        soc_minmax = min([unit.soc_max for unit in self.units.values()])
        soc_maxmin = max([unit.soc_min for unit in self.units.values()])

        soc_upper = statistics.median([soc_minmax, self.subfleet.soc_target, soc_maxmin])
        soc_lower = statistics.median([soc_minmax, self.subfleet.soc_return, soc_maxmin])

        self.dsoc_usable = soc_upper - soc_lower

        if self.dsoc_usable <= 0:
            raise ValueError(f'Usable dSOC for {self.subfleet.name} is zero or negative. Check SOC targets and aging.')

        self.energy_usable = (self.dsoc_usable *
                              self.energy_total *
                              np.sqrt(self.subfleet.eff_storage_roundtrip))

        pwr_loss_max = (1 - (1 - unit_repr.loss_rate) ** self.scenario.timestep_hours * unit_repr.sizes.loc['block', 'total'])
        self.pwr_chg_usable = (unit_repr.pwr_chg_max * unit_repr.eff_chg_int - pwr_loss_max) * factor_derate
        # endregion

        # region calculate a priori process data
        self.processes = self.demand.demand.copy()
        self.processes['step_req'] = dt2steps(self.processes['time_req'], self.scenario)

        self.processes['steps_patience'] = dt2steps(self.processes['dtime_patience'], self.scenario)

        self.processes['dtime_rental'] = self.processes['dtime_active'] + self.processes['dtime_idle']
        self.processes['steps_rental'] = dt2steps(self.processes['dtime_rental'], self.scenario)

        self.processes['num_prim'] = {'mb': np.ceil(self.processes['energy_req'] / self.energy_usable).astype(int),
                                      'ev': 1,
                                      'icev': 1}[self.subfleet.type_unit]
        self.processes['energy_req'] = self.processes['distance'] * self.processes['consumption']  # todo check if not already contained in demand

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


        self.processes['dtime_preblock_prim'] = {'icev': pd.to_timedelta(arg=0, unit='hour'),
                                                 'mb': pd.to_timedelta(arg=0, unit='hour'),
                                                 'ev': self.processes['dtime_chg_prim']}[self.subfleet.type_unit]
        self.processes['step_preblock_primary'] = self.processes['step_req'] - self.processes['steps_charge_primary']
        self.processes['time_preblock_primary'] = steps2dt(self.processes['step_preblock_primary'],
                                                           self.scenario,
                                                           absolute=True)



        self.block_charge_time()  # block charge time pre-rental for vehicles

        self.processes.sort_values(by='time_preblock_primary', inplace=True, ignore_index=True)
        # endregion

        self.store = MultiStore(env=self.scenario.env_dispatch,
                                capacity=self.subfleet.num)

        for unit in self.units.values():
            self.store.put([unit.name])

        for idx, row in self.processes.iterrows():
            self.processes.loc[idx, 'process_obj'] = Process(data=row,
                                                             dispatcher=self)

    def calc_performance_metrics(self):

        self.use_rate = dict()
        steps_total = self.data.shape[0]
        # make an individual row for each used unit in a process
        processes_exploded = self.processes.explode('commodities_primary')

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

    def convert_process_log(self):
        """
        This function converts the process based log from DES execution into a time based log for each unit
        as required by the energy system model as an example
        """

        commodities = list(self.subfleet.subblocks.keys())
        column_names = []
        for unit in commodities:
            column_names.extend([(unit,'atbase'), (unit,'dsoc'), (unit,'consumption'),
                                 (unit,'atac'), (unit,'atdc')])
            if isinstance(self, VehicleRentalSystem):
                column_names.extend([(unit,'dist')])
        column_index = pd.MultiIndex.from_tuples(column_names, names=['time', 'time'])

        # Initialize dataframe for time based log
        self.data = pd.DataFrame(data=0, index=self.scenario.dti_des, columns=column_index)
        for col, dtype in {(com, col): ('bool' if 'at' in col else 'float') for com, col in column_index}.items():
            self.data[col] = self.data[col].astype(dtype)
        self.data.loc[:, (slice(None), 'atbase')] = True
        self.data.loc[:, (slice(None), 'atac')] = False
        self.data.loc[:, (slice(None), 'atdc')] = False

        for process in [row for _, row in self.processes.iterrows() if row['status'] == 'success']:
            for unit in process['commodities_primary']:
                # Set Availability at base for charging
                self.data.loc[process['time_dep']:(process['time_return'] - self.scenario.timestep_td),
                (unit, 'atbase')] = False

                # set consumption power as constant while rented out
                self.data.loc[process['time_dep']:(process['time_return'] - self.scenario.timestep_td),
                (unit, 'consumption')] = (process['energy_pc_primary'] /
                                               (process['steps_rental'] * self.scenario.timestep_hours))

                # Set minimum SOC at departure makes sure that only vehicles with at least that SOC are rented out
                self.data.loc[process['time_dep'], (unit, 'dsoc')] = process['dsoc_primary']

                if isinstance(self, VehicleRentalSystem):
                    # set distance in first timestep of rental (for distance based revenue calculation)
                    self.data.loc[process['time_dep'], (unit, 'dist')] = process['distance']

        self.subfleet.data = self.data

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
        self.data.to_csv(log_path)


class ElectricVehicleDispatcher(SubFleetDispatcher):

    def __init__(self,
                 subfleet: 'blocks.VehicleFleet',
                 scenario: 'simulation.Scenario'):

        if subfleet.rex_fleet is not None:
            self.rex = True
            self.rex_fleet = scenario.blocks.get(rex_fleet, None)
            self.check_rex_fleet()
            self.rex_dispatcher = self.rex_fleet.dispatcher

        super().__init__(subfleet=subfleet,
                         scenario=scenario)


    def block_charge_time(self):


    def calc_process_data(self):
        """
        fill the processes dataframe with calculated data
        """


    def check_rex_fleet(self):
        base_msg = f'Scenario "{self.scenario.name}" -'
        f'Block "{self.subfleet.parent.name}" -'
        f'Subfleet "{self.subfleet.name}":'
        f'selected range extender fleet "{self.rex_fleet}"'

        if self.rex_fleet is None:
            raise ValueError(f'{base_msg} does not exist')
        elif not isinstance(self.rex_fleet, blocks.BatteryFleet):
            raise ValueError(f'{base_msg} is not a BatteryFleet')
        elif len(self.rex_fleet.subfleets) > 1:  # only one subfleet allowed
            raise ValueError(f'{base_msg} must have exactly one subfleet')
        elif not self.rex_fleet.subfleets[0].data_source in ['usecases', 'demand']:
            raise ValueError(f'{base_msg} - data source"{self.rex_fleet.subfleet[0].data_source}" is not allowed. '
                             f'Allowed values: ["usecases", "demand"]')

    def transfer_rex_processes(self):
        """
        This function takes processes requiring REX from the VehicleRentalSystem and adds them to the target
        BatteryRentalSystem's processes dataframe as these don't originate from the latter's demand pregeneration
        and are not logged there yet.
        """
        mask = (self.processes['status'] == 'success') & (self.processes['rex_request'])
        rex_processes = self.processes.loc[mask, :].copy()

        # convert values for target BatteryRentalSystem
        rex_processes['usecase_id'] = -1
        rex_processes['usecase_name'] = f'rex_{self.subfleet.name}'

        def swap_primary_secondary(col_name):
            if 'primary' in col_name:
                return col_name.replace('primary', 'secondary')
            elif 'secondary' in col_name:
                return col_name.replace('secondary', 'primary')
            return col_name
        rex_processes.columns = [swap_primary_secondary(col) for col in rex_processes.columns]

        # drop all secondary columns
        #rex_processes.drop([col for col in rex_processes.columns if 'secondary' in col], axis=1, inplace=True)
        rex_processes['time_preblock_primary'] = rex_processes['time_req']
        rex_processes['step_preblock_primary'] = rex_processes['step_req']

        # add rex processes to end of target BatteryRentalSystem's processes dataframe and create new sorted index
        self.subfleet.rex_fleet.rental_system.processes = pd.concat(
            [self.subfleet.rex_fleet.rental_system.processes, rex_processes],
            join='inner')
        self.subfleet.rex_fleet.rental_system.processes.sort_values(
            by='time_preblock_primary',
            inplace=True,
            ignore_index=True)


class BatteryDispatcher(SubFleetDispatcher):

    def __init__(self,
                 subfleet: blocks.SubFleet,
                 scenario: 'simulation.Scenario'):

        self.rex = False

        super().__init__(subfleet=subfleet,
                         scenario=scenario)


    def block_charge_time(self):
        self.processes['step_preblock_primary'] = self.processes['step_req']
        self.processes['time_preblock_primary'] = self.processes['time_req']


class Process:

    def __init__(self,
                 data: pd.Series,
                 dispatcher: SubFleetDispatcher):

        self.data = data
        self.dispatcher = dispatcher
        self.logger = self.disp.scenario
        self.env = self.dispatcher.scenario.dispatcher.environment

        self.primary_result = self.secondary_result = [False]  # defaults equal to insuccessful request
        self.primary_request = self.secondary_request = False
        self.status = None

        # define_process is not executed here, but only when the environment is run
        self.environment.process(self.define_process())

    def define_process(self):

        # wait until start of preblock time
        yield self.env.timeout(self.data['step_preblock_primary'])

        self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} preblocked at {self.env.now}')

        # request primary resource(s) from (Multi)Store
        with self.rental_system.store.get(self.data['num_primary']) as self.primary_request:
            self.primary_result = yield self.primary_request | self.env.timeout(
                self.data['steps_patience'])

        self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} requested {self.data["num_primary"]}'
                                   f' primary resource(s) at {self.env.now}')

        # wait until actual request time (leaving time to charge vehicle)
        if isinstance(self.rental_system, VehicleRentalSystem):
            yield self.env.timeout(self.data['steps_charge_primary'])

        # request secondary resources from other MultiStore
        if 'num_secondary' in self.data.index.values and self.data['num_secondary'] > 0:
            with (self.rental_system.fleet.rex_fleet.rental_system.store.get(self.data['num_secondary'])
                  as self.secondary_request):
                self.secondary_result = yield self.secondary_request | self.env.timeout(self.data['steps_patience'])

            self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} requested '
                                       f'{self.data["num_secondary"]} secondary resource(s) at {self.env.now}')

        # if primary or both request(s) successful
        if (self.primary_request in self.primary_result) and (self.secondary_request in self.secondary_result):
            self.rental_system.processes.loc[self.id, 'status'] = self.status = 'success'
            self.rental_system.processes.at[self.id, 'commodities_primary'] = self.primary_request.value
            self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} received primary resource'
                                       f' {self.primary_request.value} at {self.env.now}')

            if self.secondary_request:
                self.rental_system.processes.at[self.id, 'commodities_secondary'] = self.secondary_request.value
                self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} received secondary resource'
                                           f' {self.secondary_request.value} at {self.env.now}')

            self.rental_system.processes.loc[self.id, 'step_dep'] = self.env.now

            # cover the usage & idle time
            yield self.env.timeout(self.data['steps_rental'])
            self.rental_system.processes.loc[self.id, 'step_return'] = self.env.now

            # cover the recharge time for BatteryFleets
            if isinstance(self.rental_system, VehicleRentalSystem):
                self.rental_system.processes.loc[self.id, 'step_reavail_primary'] = self.env.now
                if self.secondary_request:
                    yield self.env.timeout(self.data['steps_charge_secondary'])
                    self.rental_system.processes.loc[self.id, 'step_reavail_secondary'] = self.env.now

            elif isinstance(self.rental_system, BatteryRentalSystem):
                yield self.env.timeout(self.data['steps_charge_primary'])
                self.rental_system.processes.loc[self.id, 'step_reavail_primary'] = self.env.now

            self.rental_system.store.put(self.primary_result[self.primary_request])
            self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} returned resource(s) {self.primary_request.value}'
                                       f' at {self.env.now}. Primary store content after return: {self.rental_system.store.items}')

            if self.secondary_request:
                self.rental_system.fleet.rex_fleet.rental_system.store.put(self.secondary_result[self.secondary_request])
                self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} returned secondary resource(s)'
                                           f'{self.secondary_request.value} at {self.env.now}. '
                                           f'Secondary store content after return: '
                                           f'{self.rental_system.fleet.rex_fleet.rental_system.store.items} ')

        else:  # either or both (primary/secondary) request(s) unsuccessful

            # log type of failure
            if ((self.primary_request not in self.primary_result)
                    and (self.secondary_request not in self.secondary_result)):
                self.rental_system.processes.loc[self.id, 'status'] = self.status = 'failure_both'
                self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} failed '
                                           f'(didn´t receive either resource) at {self.env.now}. '
                                           f'Primary store content after failure: {self.rental_system.store.items}. '
                                           f'Secondary store content after failure: '
                                           f'{self.rental_system.fleet.rex_fleet.rental_system.store.items}')

            elif self.primary_request not in self.primary_result:
                self.rental_system.processes.loc[self.id, 'status'] = self.status = 'failure_primary'
                if self.secondary_request:
                    self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} failed '
                                               f'(didn´t receive primary resource(s)) at {self.env.now}. '
                                               f'Primary store content after fail: {self.rental_system.store.items}. '
                                               f'Secondary store content after failure: '
                                               f'{self.rental_system.fleet.rex_fleet.rental_system.store.items}')
                else:
                    self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} failed '
                                               f'(didn´t receive primary resource(s)) at {self.env.now}. '
                                               f'Primary store content after fail: {self.rental_system.store.items}')

            elif self.secondary_request not in self.secondary_result:
                self.rental_system.processes.loc[self.id, 'status'] = self.status = 'failure_secondary'
                self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} failed '
                                           f'(didn´t receive secondary resource(s)) at {self.env.now}. '
                                           f'Primary store content after fail: {self.rental_system.store.items}. '
                                           f'Secondary store content after failure: '
                                           f'{self.rental_system.fleet.rex_fleet.rental_system.store.items}')

            # ensure resources are put back after conditional event: https://stackoverflow.com/q/75371166
            if self.primary_request.triggered:
                primary_resource = yield self.primary_request
                self.rental_system.store.put(primary_resource)
                self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} returned '
                                           f'primary resource {primary_resource} at {self.env.now}. '
                                           f'Primary store content after return: {self.rental_system.store.items}.')

            if hasattr(self.secondary_request, 'triggered'):
                if self.secondary_request.triggered:
                    secondary_resource = yield self.secondary_request
                    self.rental_system.fleet.rex_fleet.rental_system.store.put(secondary_resource)
                    self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} returned '
                                               f'secondary resource {secondary_resource} at {self.env.now}. '
                                               f'Primary store content after return: {self.rental_system.store.items}. '
                                               f'Secondary store content after return: '
                                               f'{self.rental_system.fleet.rex_fleet.rental_system.store.items}')

        self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} finished at {self.env.now}')


