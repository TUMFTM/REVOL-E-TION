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


class RentalSystem:

    def __init__(self, commodity_system, scenario):

        self.commodity_system = commodity_system
        self.scenario = scenario
        self.name = self.commodity_system.name

        self.soc_max_init = min([commodity.soc_max
                                 for commodity in self.commodity_system.commodities.values()])
        self.soc_min_init = max([commodity.soc_min
                                 for commodity in self.commodity_system.commodities.values()])

        # calculate usable energy to expect

        self.dsoc_usable_high = (statistics.median([self.soc_min_init,
                                                    self.commodity_system.soc_target_high,
                                                    self.soc_max_init]) -
                                 statistics.median([self.soc_min_init,
                                                    self.commodity_system.soc_return,
                                                    self.soc_max_init]))
        self.dsoc_usable_low = (statistics.median([self.soc_min_init,
                                                   self.commodity_system.soc_target_low,
                                                   self.soc_max_init]) -
                                statistics.median([self.soc_min_init,
                                                   self.commodity_system.soc_return,
                                                   self.soc_max_init]))

        if (self.dsoc_usable_high <= 0) or (self.dsoc_usable_low <= 0):
            raise ValueError(f'Usable dSOC for {self.commodity_system.name} '
                             f'is zero or negative. Check SOC targets and aging.')

        self.energy_usable_pc_high = (self.dsoc_usable_high *
                                      self.commodity_system.size_pc *
                                      np.sqrt(self.commodity_system.eff_storage_roundtrip))
        self.energy_usable_pc_low = (self.dsoc_usable_low *
                                     self.commodity_system.size_pc *
                                     np.sqrt(self.commodity_system.eff_storage_roundtrip))

        self.n_processes = self.processes = self.demand_daily = self.store = None
        self.use_rate = self.fail_rate = None
        self.process_objs = []

    def add_dispatch_columns(self):
        """
        Add empty columns to process frame that are iteratively filled during actual dispatch (i.e. in DES)
        """
        self.processes['commodities_primary'] = None
        if (hasattr(self.commodity_system, 'rex_cs')) and (self.commodity_system.rex_cs):
            self.processes['commodities_secondary'] = None

    def create_store(self):
        self.store = MultiStore(self.scenario.env_des, capacity=self.commodity_system.num)
        for commodity in self.commodity_system.commodities.values():
            self.store.put([commodity.name])

    def calc_performance_metrics(self):

        self.use_rate = dict()
        steps_total = self.data.shape[0]
        # make an individual row for each used commodity in a process
        processes_exploded = self.processes.explode('commodities_primary')

        # calculate percentage of DES (not sim, the latter is shorter) time
        # occupied by active, idle & recharge times
        for commodity in list(self.commodity_system.commodities.keys()):
            processes = processes_exploded.loc[processes_exploded['commodities_primary'] == commodity, :]
            steps_blocked = processes['steps_charge_primary'].sum() + processes['steps_rental'].sum()
            self.use_rate[commodity] = steps_blocked / steps_total
        self.commodity_system.use_rate = np.mean(list(self.use_rate.values()))

        # calculate overall percentage of failed trips
        n_success = self.processes.loc[self.processes['status'] == 'success', 'status'].shape[0]
        n_total = self.processes.shape[0]
        self.fail_rate = self.commodity_system.fail_rate = 1 - (n_success / n_total)

    def convert_process_log(self):
        """
        This function converts the process based log from DES execution into a time based log for each commodity
        as required by the energy system model as an example
        """

        commodities = list(self.commodity_system.commodities.keys())
        column_names = []
        for commodity in commodities:
            column_names.extend([(commodity,'atbase'), (commodity,'dsoc'), (commodity,'consumption'),
                                 (commodity,'atac'), (commodity,'atdc')])
            if isinstance(self, VehicleRentalSystem):
                column_names.extend([(commodity,'tour_dist')])
        column_index = pd.MultiIndex.from_tuples(column_names, names=['time', 'time'])

        # Initialize dataframe for time based log
        self.data = pd.DataFrame(data=0, index=self.scenario.dti_des, columns=column_index)
        for col, dtype in {(com, col): ('bool' if 'at' in col else 'float') for com, col in column_index}.items():
            self.data[col] = self.data[col].astype(dtype)
        self.data.loc[:, (slice(None), 'atbase')] = True
        self.data.loc[:, (slice(None), 'atac')] = False
        self.data.loc[:, (slice(None), 'atdc')] = False

        for process in [row for _, row in self.processes.iterrows() if row['status'] == 'success']:
            for commodity in process['commodities_primary']:
                # Set Availability at base for charging
                self.data.loc[process['time_dep']:(process['time_return'] - self.scenario.timestep_td),
                (commodity, 'atbase')] = False

                # set consumption power as constant while rented out
                self.data.loc[process['time_dep']:(process['time_return'] - self.scenario.timestep_td),
                (commodity, 'consumption')] = (process['energy_pc_primary'] /
                                               (process['steps_rental'] * self.scenario.timestep_hours))

                # Set minimum SOC at departure makes sure that only vehicles with at least that SOC are rented out
                self.data.loc[process['time_dep'], (commodity, 'dsoc')] = process['dsoc_primary']

                if isinstance(self, VehicleRentalSystem):
                    # set distance in first timestep of rental (for distance based revenue calculation)
                    self.data.loc[process['time_dep'], (commodity, 'tour_dist')] = process['distance']

        self.commodity_system.data = self.data

    def generate_process_frame(self):

        self.processes = self.commodity_system.demand.copy()
        self.processes['step_req'] = dt2steps(self.processes['time_req'], self.scenario)
        self.calc_process_data()

        # common calculations for both types of RentalSystem
        self.processes['dtime_rental'] = self.processes['dtime_active'] + self.processes['dtime_idle']
        self.processes['steps_rental'] = dt2steps(self.processes['dtime_rental'], self.scenario)

        self.processes['steps_charge_primary'] = dt2steps(self.processes['dtime_charge_primary'], self.scenario)
        if 'dtime_charge_secondary' in self.processes.columns:
            self.processes['steps_charge_secondary'] = dt2steps(self.processes['dtime_charge_secondary'], self.scenario)

        self.processes['steps_patience'] = dt2steps(self.processes['dtime_patience'], self.scenario)

        self.block_charge_time()  # block charge time pre-rental for vehicles

        self.processes.sort_values(by='time_preblock_primary', inplace=True, ignore_index=True)

    def save_data(self):
        """
        This function saves the converted log dataframe as a suitable example csv file for the energy system model.
        The resulting dataframe can also be handed to the energy system model directly in addition for faster
        delivery through execute_des.
        """
        processes_path = os.path.join(
            self.scenario.run.path_result_dir,
            f'{self.scenario.run.runtimestamp}_'
            f'{self.scenario.run.scenario_file_name}_'
            f'{self.scenario.name}_'
            f'{self.commodity_system.name}_'
            f'processes.csv')
        self.processes.to_csv(processes_path)

        log_path = os.path.join(
            self.scenario.run.path_result_dir,
            f'{self.scenario.run.runtimestamp}_'
            f'{self.scenario.run.scenario_file_name}_'
            f'{self.scenario.name}_'
            f'{self.commodity_system.name}_'
            f'log.csv')
        self.data.to_csv(log_path)


class VehicleRentalSystem(RentalSystem):

    def __init__(self, commodity_system, scenario):

        super().__init__(commodity_system, scenario)

        # replace the rex system name read in from scenario file with the actual CommoditySystem object
        if self.commodity_system.rex_cs:
            self.check_rex_inputs()
            self.commodity_system.rex_cs = self.scenario.blocks[self.commodity_system.rex_cs]

        self.dsoc_usable_rex_high = (self.commodity_system.rex_cs.soc_target_high -
                                     self.commodity_system.rex_cs.soc_return)\
            if self.commodity_system.rex_cs else 0
        self.dsoc_usable_rex_low = (self.commodity_system.rex_cs.soc_target_low -
                                    self.commodity_system.rex_cs.soc_return)\
            if self.commodity_system.rex_cs else 0

        if self.commodity_system.rex_cs:  # system can extend range
            self.energy_usable_rex_pc_high = (self.dsoc_usable_rex_high *
                                              self.commodity_system.rex_cs.size_pc *
                                              np.sqrt(self.commodity_system.rex_cs.eff_storage_roundtrip))
            self.energy_usable_rex_pc_low = (self.dsoc_usable_rex_low *
                                             self.commodity_system.rex_cs.size_pc *
                                             np.sqrt(self.commodity_system.rex_cs.eff_storage_roundtrip))
        else:  # no rex defined
            self.energy_usable_rex_pc_high = 0
            self.energy_usable_rex_pc_low = 0

        self.generate_process_frame()
        self.add_dispatch_columns()
        self.create_store()

    def block_charge_time(self):
        self.processes['step_preblock_primary'] = self.processes['step_req'] - self.processes['steps_charge_primary']
        self.processes['time_preblock_primary'] = steps2dt(self.processes['step_preblock_primary'],
                                                           self.scenario,
                                                           absolute=True)

    def calc_process_data(self):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        :return: None
        """
        self.processes['num_primary'] = 1  # always one vehicle per rental process
        self.processes['range_usable_high'] = self.energy_usable_pc_high / self.processes['consumption']
        self.processes['range_usable_low'] = self.energy_usable_pc_low / self.processes['consumption']

        if self.commodity_system.rex_cs:  # system can extend range
            # determine number of rex needed to cover missing distance and calc available energy
            self.processes['distance_missing'] = np.maximum(
                0,
                self.processes['distance'] - self.processes['range_usable_high'])
            self.processes['energy_missing'] = self.processes['distance_missing'] * self.processes['consumption']
            self.processes['num_secondary'] = np.ceil(self.processes['energy_missing'] /
                                                      self.energy_usable_rex_pc_high).astype(int)
            self.processes['rex_request'] = self.processes['num_secondary'] > 0

            self.processes['energy_usable_both'] = (self.energy_usable_pc_high +
                                                    (self.processes['num_secondary'] *
                                                     self.energy_usable_rex_pc_high))
            self.processes['energy_total_both'] = (self.commodity_system.size_pc +
                                                   (self.processes['num_secondary'] *
                                                    self.commodity_system.rex_cs.size_pc))

        else:  # no rex defined
            self.processes['num_secondary'] = 0
            self.processes['rex_request'] = False

            self.processes['energy_usable_both'] = self.energy_usable_pc_high
            self.processes['energy_total_both'] = self.commodity_system.size_pc
            # for non-rex systems, dsoc_primary is clipped to max usable dSOC (equivalent to external charging)
            self.processes['energy_req'] = self.processes['energy_req'].clip(upper=self.energy_usable_pc_high)

        # calculate different delta SOC for primary and secondary commodity due to different start SOCs (linear)
        self.processes['dsoc_primary'] = (self.dsoc_usable_high * self.processes['energy_req'] /
                                          self.processes['energy_usable_both'])
        self.processes['dsoc_secondary'] = (self.dsoc_usable_rex_high * self.processes['energy_req'] /
                                            self.processes['energy_usable_both']) * self.processes['rex_request']

        self.processes['energy_pc_primary'] = (self.processes['dsoc_primary'] *
                                               self.commodity_system.size_pc)
        self.processes['dtime_charge_primary'] = pd.to_timedelta(
            self.processes['energy_pc_primary'] / self.commodity_system.pwr_chg_des,
            unit='hour')

        if self.commodity_system.rex_cs:
            self.processes['energy_pc_secondary'] = (self.processes['dsoc_secondary'] *
                                                     self.commodity_system.rex_cs.size_pc)
            self.processes['dtime_charge_secondary'] = pd.to_timedelta(
                self.processes['energy_pc_secondary'] / self.commodity_system.rex_cs.pwr_chg_des,
                unit='hour')
        else:
            self.processes['energy_pc_secondary'] = 0
            self.processes['dtime_charge_secondary'] = None

    def check_rex_inputs(self):
        if self.commodity_system.rex_cs not in self.scenario.blocks.keys():
            raise ValueError(f'Scenario "{self.scenario.name}" -'
                             f'Block "{self.commodity_system.name}":'
                             f'Selected range extender system "{self.commodity_system.rex_cs}" '
                             f'does not exist')
        elif not isinstance(self.scenario.blocks[self.commodity_system.rex_cs], blocks.BatteryCommoditySystem):
            raise ValueError(f'Scenario "{self.scenario.name}" -'
                             f'Block "{self.commodity_system.name}":'
                             f'Selected range extender system "{self.commodity_system.rex_cs}" '
                             f'is not a BatteryCommoditySystem')
        elif not self.scenario.blocks[self.commodity_system.rex_cs].data_source in ['usecases', 'demand']:
            raise ValueError(f'Scenario "{self.scenario.name}" -'
                             f'Block "{self.commodity_system.name}":'
                             f'Selected range extender system "{self.commodity_system.rex_cs}" '
                             f'has data source"{self.commodity_system.rex_cs.data_source}". '
                             f'Allowed values: [\'usecases\', \'demand\']')

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
        rex_processes['usecase_name'] = f'rex_{self.commodity_system.name}'

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
        self.commodity_system.rex_cs.rental_system.processes = pd.concat(
            [self.commodity_system.rex_cs.rental_system.processes, rex_processes],
            join='inner')
        self.commodity_system.rex_cs.rental_system.processes.sort_values(
            by='time_preblock_primary',
            inplace=True,
            ignore_index=True)


class BatteryRentalSystem(RentalSystem):

    def __init__(self, commodity_system, scenario):

        super().__init__(commodity_system, scenario)

        self.commodity_system.rex_cs = None  # needs to be set for later check

        self.generate_process_frame()
        self.add_dispatch_columns()
        self.create_store()

    def block_charge_time(self):
        self.processes['step_preblock_primary'] = self.processes['step_req']
        self.processes['time_preblock_primary'] = self.processes['time_req']

    def calc_process_data(self):
        """
        This function fills the demand dataframe with stochastically generated values describing the rental
        requests for each day in the simulation timeframe.
        """
        self.processes['rex_request'] = False
        self.processes['num_primary'] = np.ceil(self.processes['energy_req'] / self.energy_usable_pc_high).astype(int)
        self.processes['energy_total_both'] = self.processes['num_primary'] * self.commodity_system.size_pc
        self.processes['energy_usable_both'] = self.processes['num_primary'] * self.energy_usable_pc_high
        self.processes['dsoc_primary'] = self.processes['energy_req'] / self.processes['energy_total_both']
        self.processes['energy_pc_primary'] = self.processes['dsoc_primary'] * self.commodity_system.size_pc
        self.processes['energy_primary'] = self.processes['energy_pc_primary'] * self.processes['num_primary']
        self.processes['dtime_charge_primary'] = pd.to_timedelta(
            self.processes['energy_pc_primary'] / (self.commodity_system.pwr_chg *
                                                   self.commodity_system.eff_chg *  # charger efficiency (into commodity's bus)
                                                   # storage component efficiency (both ways)
                                                   self.commodity_system.eff_storage_roundtrip),
            unit='hour')


class RentalProcess:

    def __init__(self, id, data, rental_system):

        self.id = id
        self.data = data
        self.rental_system = rental_system
        self.scenario = self.rental_system.scenario
        self.env = self.scenario.env_des

        self.rental_system.process_objs.append(self)

        self.primary_result = self.secondary_result = [False]  # defaults equal to insuccessful request
        self.primary_request = self.secondary_request = False
        self.status = None

        # initiate the simpy process function (define_process is not executed here, but only when the env is run)
        self.env.process(self.define_process())

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
            with (self.rental_system.commodity_system.rex_cs.rental_system.store.get(self.data['num_secondary'])
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

            # cover the recharge time for BatteryCommoditySystems
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
                self.rental_system.commodity_system.rex_cs.rental_system.store.put(self.secondary_result[self.secondary_request])
                self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} returned secondary resource(s)'
                                           f'{self.secondary_request.value} at {self.env.now}. '
                                           f'Secondary store content after return: '
                                           f'{self.rental_system.commodity_system.rex_cs.rental_system.store.items} ')

        else:  # either or both (primary/secondary) request(s) unsuccessful

            # log type of failure
            if ((self.primary_request not in self.primary_result)
                    and (self.secondary_request not in self.secondary_result)):
                self.rental_system.processes.loc[self.id, 'status'] = self.status = 'failure_both'
                self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} failed '
                                           f'(didn´t receive either resource) at {self.env.now}. '
                                           f'Primary store content after failure: {self.rental_system.store.items}. '
                                           f'Secondary store content after failure: '
                                           f'{self.rental_system.commodity_system.rex_cs.rental_system.store.items}')

            elif self.primary_request not in self.primary_result:
                self.rental_system.processes.loc[self.id, 'status'] = self.status = 'failure_primary'
                if self.secondary_request:
                    self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} failed '
                                               f'(didn´t receive primary resource(s)) at {self.env.now}. '
                                               f'Primary store content after fail: {self.rental_system.store.items}. '
                                               f'Secondary store content after failure: '
                                               f'{self.rental_system.commodity_system.rex_cs.rental_system.store.items}')
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
                                           f'{self.rental_system.commodity_system.rex_cs.rental_system.store.items}')

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
                    self.rental_system.commodity_system.rex_cs.rental_system.store.put(secondary_resource)
                    self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} returned '
                                               f'secondary resource {secondary_resource} at {self.env.now}. '
                                               f'Primary store content after return: {self.rental_system.store.items}. '
                                               f'Secondary store content after return: '
                                               f'{self.rental_system.commodity_system.rex_cs.rental_system.store.items}')

        self.scenario.logger.debug(f'{self.rental_system.name} process {self.id} finished at {self.env.now}')


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


def execute_des(scenario, run):

    # define a DES environment
    scenario.env_des = simpy.Environment()

    # extend datetimeindex to simulate in DES to cover any shifts & predictions necessary
    time_start_overhang = scenario.dti_sim_extd[-1] + scenario.dti_sim_extd.freq
    dtime_overhang = pd.Timedelta(days=28)
    time_end_overhang = time_start_overhang + dtime_overhang
    scenario.dti_des = scenario.dti_sim_extd.union(
        pd.date_range(start=time_start_overhang,
                      end=time_end_overhang,
                      freq=scenario.dti_sim_extd.freq))

    # create rental systems (including stochastic pregeneration of individual rental processes)
    scenario.rental_systems = dict()
    for cs in [cs for cs in scenario.commodity_systems.values() if cs.data_source in ['usecases', 'demand']]:
        if isinstance(cs, blocks.VehicleCommoditySystem):
            cs.rental_system = VehicleRentalSystem(cs, scenario)
        elif isinstance(cs, blocks.BatteryCommoditySystem):
            cs.rental_system = BatteryRentalSystem(cs, scenario)
        scenario.rental_systems[cs.name] = cs.rental_system

    # generate individual RentalProcess instances for every pregenerated process
    for rental_system in scenario.rental_systems.values():
        for idx, row in rental_system.processes.iterrows():
            # VehicleRentalSystem RentalProcesses can init additional processes in BatteryRentalSystems at runtime
            process = RentalProcess(id=idx,
                                    data=row,
                                    rental_system=rental_system)
            rental_system.processes.loc[idx, 'process_obj'] = process

    # actually run the discrete event simulation
    scenario.env_des.run()

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
    for rental_system in [rs for rs in scenario.rental_systems.values() if (rs.commodity_system.rex_cs is not None)]:
        rental_system.transfer_rex_processes()

    # reframe logging results to resource-based view instead of process based (and save)
    for rental_system in scenario.rental_systems.values():
        rental_system.convert_process_log()
        rental_system.calc_performance_metrics()
        if run.save_des_results:
            rental_system.save_data()
