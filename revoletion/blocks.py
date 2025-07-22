#!/usr/bin/env python3

from __future__ import annotations
import ast
from dataclasses import dataclass, field
import numpy as np
import oemof.solph as solph
import pandas as pd
import plotly.graph_objects as go
import pvlib
import re
import requests
from typing import Any, Optional
import windpowerlib

from abc import ABC, abstractmethod

from . import battery as bat
from . import economics as eco
from . import mobility
from . import utils


class BlockScenarioInterface(ABC):
    @abstractmethod
    def pre_scenario(self) -> None:
        """
        Trigger actions to be executed after all inits.
        """
        pass

    @abstractmethod
    def pre_horizon(self,
                    horizon: 'PredictionHorizon') -> None:
        """
        Trigger actions to be executed before each horizon.
        """
        pass

    @abstractmethod
    def post_horizon(self,
                     horizon: 'PredictionHorizon') -> None:
        """
        Trigger actions to be executed after each horizon.
        """
        pass

    @abstractmethod
    def post_scenario(self) -> None:
        """
        Trigger actions to be executed after the scenario has been run.
        """
        pass


class BaseBlock(BlockScenarioInterface):
    """
    abstract class
    """

    def init_evaluators(self):
        # add a new POI to block.pois
        pass

    def init_states(self):
        # add a new column to block.states
        pass


    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 params: dict = None,
                 parent: 'Block | Scenario' = None,
                 ):
        """
        Initialize (Sub)Block object with attributes and data structures
        """

        self.name = name
        self.scenario = scenario
        self.parent = parent

        self.classname = self.__class__.__name__  # get name of class
        self.top_level_block = True if self.parent is self.scenario else False  # distinguish top level blocks/subblocks

        self.register_block()

        # region set attributes from scenario file or parent
        params = params if params is not None else self.scenario.parameters.loc[self.name]
        for key, value in params.items():
            setattr(self, key, value)
        # endregion

        # preprocessing of invest/sizes which are set to equal will be set to the same value
        self.expansion_equal = False
        self.params_preprocessing()

        self.aggregator = eco.EcoAggregator(name=self.name,
                                            scenario=self.scenario,
                                            block=self)

        self.states = pd.DataFrame(index=self.scenario.dti_sim_extd,
                                   dtype='float64')

        self.sizes = dict()  # entries are created by EcoEvaluator

        self.evaluators = dict()
        self.init_evaluators()

        # # region get poi and state name definitions
        # # ToDo: find a more elegant way to combine POIs and state names from class hierarchy
        # # combine all previously defined POIs and state names from class hierarchy
        # definitions = [cls.get_init_definitions()
        #                for cls in self.__class__.mro()
        #                if ('get_init_definitions' in vars(cls) and  # distinguish implemented and inherited methods
        #                    cls.get_init_definitions() is not None)  # avoid None return value of @abstractmethod
        #                ]
        #
        # # if not definitions:
        # #     raise ValueError(f'Block "{self.name}" has no POIs or state names defined in its class hierarchy.')
        #
        # self.pois = {poi_key: poi_value
        #              for definition in definitions
        #              for poi_key, poi_value in
        #              definition['pois'].items()}
        # state_names = [state_name for definition in definitions for state_name in definition['state_names']]
        # if len(state_names) != len(set(state_names)):
        #     raise ValueError(f'Block "{self.name}" has duplicate state names in its class hierarchy definitions.')
        # # endregion

        # region initialize data structures
        self.subblocks = dict()

        # self.initialize_sizes()

        # self.evaluators = self.create_evaluator_objects()
        # self.aggregator.pre_scenario()  # aggregate capex preexisting

        # ToDo: Delete ccr and ls as they are now contained in evaluators
        # for attribute in set(value for poi in self.pois.values() for value in poi['params'].values()):
        #     if hasattr(self, attribute):
        #         delattr(self, attribute)

        # initialize result data structures
        self.result_summary = []  # -> list of pd.Series
        self.result_timeseries = []  # -> list of pd.DataFrames
        self.result_messages = []
        self.plot_traces = dict(powers=[],
                                states=[],
                                )
        # endregion

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

    def register_block(self):
        for cls in self.__class__.__mro__:
            if cls in (object, ABC, BlockScenarioInterface):
                continue
            name_class = cls.__name__
            # Initialize per-class registry if not present and register self
            self.scenario.block_registry.setdefault(name_class, {})[self.name] = self

        if self.top_level_block:
            self.scenario.block_registry.setdefault('TopLevelBlock', {})[self.name] = self
        else:  # is subblock
            self.parent.subblocks[self.name] = self

    def params_preprocessing(self):
        pass

    def initialize_sizes(self,
                         pois: dict = None):
        """
        Initialize the sizes DataFrame for the block
        """

        sizes = [k for k, v in self.pois.items() if ('size', 'name') in v['params'].keys()]

        if len(sizes) > len(set(sizes)):  # avoid duplicate size names in POIs
            raise ValueError(f'Block "{self.name}" has duplicate size names in its POIs')

        for size in sizes:
            self.sizes[size] = eco.Size(name=size,
                                        block=self)

            if self.sizes[size].invest and self.scenario.strategy != 'go':
                raise ValueError(f'Block "{self.name}" component size optimization '
                                 f'not implemented for any other strategy than "GO"')

            if self.sizes[size].preexisting == 0 and not self.sizes[size].invest:
                self.scenario.logger.warning(f'Block "{self.name}" - '
                                             f'component "{size}" was defined without preexisting size and does not '
                                             f'allow further investments. This may cause unintended system behavior.')

    def init_equalizable_variables(self, name_vars: list):
        name_var1, name_var2 = name_vars
        if (getattr(self, name_var1) == 'equal') and (getattr(self, name_var2) == 'equal'):
            error_msg = (f'"{self.name}" parameters {name_var1} and {name_var2} were both set to equal.'
                         f' Maximum one of these variables is allowed to be set to "equal"')
            self.scenario.logger.error(error_msg)
        elif getattr(self, name_var1) == 'equal':
            setattr(self, name_var1, getattr(self, name_var2))
        elif getattr(self, name_var2) == 'equal':
            setattr(self, name_var2, getattr(self, name_var1))

    def create_evaluator_objects(self):
        """
        Create EconomicEvaluator objects for each POI depending on the class name defined
        """

        evaluators = dict()
        for name, poi_definition in self.pois.items():
            class_obj = getattr(eco, poi_definition['class_name'], None)
            if class_obj is not None and isinstance(class_obj, type):
                evaluators[name] = class_obj(name=name,
                                             block=self,
                                             params=poi_definition['params'])
            else:
                raise ValueError(f'Class "{poi_definition["class_name"]}" not found in economics.py file - '
                                 f'Check for typos or add class.')
        return evaluators

    def pre_scenario(self):
        """
        trigger actions to be executed after all inits
        """
        for subblock in self.subblocks.values():
            subblock.pre_scenario()

    def pre_horizon(self,
                    horizon: 'PredictionHorizon'):

        for subblock in self.subblocks.values():
            subblock.pre_horizon(horizon=horizon)

    def post_horizon(self,
                     horizon: 'PredictionHorizon'):

        for subblock in self.subblocks.values():
            subblock.post_horizon(horizon=horizon)

    def post_scenario(self):

        for subblock in self.subblocks.values():
            subblock.post_scenario()

        # calculate results
        self.calc_results_economics()

        # create result outputs
        self.create_result_messages()
        self.create_plot_traces()
        self.create_result_summary()
        self.create_result_timeseries()

        # add block results to scenario's structures
        self.write_results_to_scenario()

    def calc_results_economics(self):
        # calculate economic results and write one level up
        for evaluator in self.evaluators.values():
            evaluator.aggregate()
        self.aggregator.aggregate()

    def create_result_summary(self):
        # get attributes of type int, float, bool and str for scenario.result_summary
        self.result_summary.extend([pd.Series({key: value for key, value in self.__dict__.items()
                                               if isinstance(value, (int, float, bool, str))})])

        # get energy results for scenario.result_summary
        for size in self.sizes.values():
            self.result_summary.append(size.result_summary)

        # get economic results for scenario.result_summary
        self.result_summary.append(self.aggregator.write_result_summary())

    @abstractmethod
    def create_result_timeseries(self):
        pass

    def create_result_messages(self, unit='kW'):
        for size in self.sizes.values():
            if (msg := size.result_msg) != '':
                self.result_messages.append(msg)

    @abstractmethod
    def create_plot_traces(self):
        pass

    def write_results_to_scenario(self):
        # result_summary
        # concat all result_summary and apply MultiIndex with
        self.result_summary = pd.concat(self.result_summary)
        self.result_summary.index = pd.MultiIndex.from_tuples(tuples=[(self.name, key)
                                                                      for key in self.result_summary.index],
                                                              names=['block', 'key'])

        # write block's results to scenario.result_summary
        self.scenario.result_summary.append(self.result_summary)

        # result_timeseries
        self.scenario.result_timeseries.extend(self.result_timeseries)

        # result_messages
        self.scenario.result_messages.extend(self.result_messages)

        # plot traces
        for axis in ['powers', 'states']:
            self.scenario.plot_traces[axis].extend(self.plot_traces[axis])

    def get_legend_entry(self):
        """
        Standard legend entry for simple blocks using power as their size
        """
        return f'{self.name} power (max. {self.sizes["block"].total / 1e3:.1f} kW)'


class NonElectricBlock(BaseBlock):

    def create_plot_traces(self, *_args, **_kwargs):
        """
        dummy method
        """
        pass

    def create_result_timeseries(self, *_args, **_kwargs):
        """
        dummy method
        """
        pass


class ElectricBlock(BaseBlock):
    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 flow_apriori_names: list = None,
                 params: dict = None,
                 parent: 'Block | Scenario' = None,
                 ):

        self.flow_names = set()

        super().__init__(name=name,
                         scenario=scenario,
                         params=params,
                         parent=parent)

        # empty list not possible as default argument as it is mutable
        flow_apriori_names = flow_apriori_names if flow_apriori_names is not None else []

        self.components = dict()
        self.bus_connected = None

        # ToDo: (1) remove flow_apriori_names and use flow names instead
        #       (2) remove flows_apriori and use flows instead to save memory
        self.flows_apriori = pd.DataFrame(index=self.scenario.dti_sim,
                                          columns=flow_apriori_names,
                                          dtype='float64'
                                          )

        self.flows = pd.DataFrame(index=self.scenario.dti_sim,
                                  columns=(['total'] + list(self.flow_names)),
                                  data=0.0,
                                  dtype='float64')

        self.energies = pd.DataFrame(index=(['total'] + list(self.flow_names)),
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0.0,  # cumulative property
                                     dtype=float)

        self.eff = dict()
        self.initialize_efficiencies()

    def initialize_efficiencies(self):
        for key in list(self.__dict__.keys()):  # use list() to safely modify the dict (delattr) while iterating
            if key.startswith('eff_'):
                self.eff[re.sub(r'^[^_]+_', '', key)] = getattr(self, key)
                delattr(self, key)

    def pre_horizon(self,
                    horizon: 'PredictionHorizon'):

        self.define_oemof_components(horizon=horizon)
        horizon.es.add(*self.components.values())

        super().pre_horizon(horizon=horizon)  # executes pre_horizon for subblocks

    def post_horizon(self,
                     horizon: 'PredictionHorizon'):

        super().post_horizon(horizon=horizon)  # executes post_horizon for subblocks

        self.get_horizon_results(horizon=horizon)

    def post_scenario(self):
        # ToDo: check, whether this requires SubBlock's post_scenario() execution triggered in super().post_scenario()
        self.calc_results_flows()
        self.calc_results_energies()

        super().post_scenario()

    @abstractmethod
    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        pass

    @abstractmethod
    def get_horizon_results(self,
                            horizon: 'PredictionHorizon'):
        pass

    def calc_results_flows(self):
        # total flow calculation is duplicated in StorageBlock
        self.flows['total'] = self.flows.get(key='out', default=0) - self.flows.get(key='in', default=0)

    def calc_results_energies(self):
        """
        post scenario method
        process flows and calculate energies from flows
        """
        for flow_name, flow in self.flows.items():
            energy = flow[self.scenario.dti_eval].sum() * self.scenario.timestep_hours
            self.energies.loc[flow_name, 'sim'] = energy
            if ('circular' in flow_name) and (energy != 0):
                self.scenario.logger.warning(f'Block "{self.name}" - circular flow detected - check energy results')

        self.energies['yrl'] = self.energies['sim'] / self.scenario.sim_yr_rat
        self.energies['prj'] = self.energies['yrl'] * self.scenario.prj_duration_yrs
        self.energies['dis'] = (self.energies['yrl'] *
                                self.scenario.discount_factors.loc[self.scenario.periods_prj, 'end'].sum())

    def create_result_summary(self):
        super().create_result_summary()
        # get energy results for scenario.result_summary
        self.result_summary.append(utils.create_results_from_dataframe(df=self.energies, name_prefix='energy'))

    def create_result_timeseries(self):
        """
        write flows and states to scenario.result_timeseries
        """
        if not self.scenario.settings.largescalemode:
            # write flows and states to scenario.result_timeseries
            self.flows.columns = pd.MultiIndex.from_tuples(tuples=[(self.name, col) for col in self.flows.columns],
                                                           names=['block', 'key'])

            self.states.columns = pd.MultiIndex.from_tuples(tuples=[(self.name, col) for col in self.states.columns],
                                                            names=['block', 'key'])

            self.result_timeseries.extend([self.flows.loc[self.scenario.dti_eval, :],
                                           self.states.loc[utils.extend_dti(dti=self.scenario.dti_eval,
                                                                            freq=self.scenario.timestep_td), :]])

    def create_plot_traces(self):
        self.plot_traces['powers'].append(go.Scatter(x=self.scenario.dti_eval,
                                                     y=self.flows.loc[self.scenario.dti_eval, 'total'],
                                                     mode='lines',
                                                     name=self.get_legend_entry(),
                                                     line=dict(width=2, dash=None, shape='hv'),
                                                     visible=True if self.top_level_block else 'legendonly',
                                                     )
                                          )


class SourceBlock(ElectricBlock):

    @abstractmethod
    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        pass

    @abstractmethod
    def get_horizon_results(self,
                            horizon: 'PredictionHorizon'):
        pass


    def calc_results_energies(self):
        super().calc_results_energies()
        self.scenario.energies.loc[('sources', 'pro'), :] += self.energies.loc['total', :]


class SinkBlock(ElectricBlock):

    @abstractmethod
    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        pass

    @abstractmethod
    def get_horizon_results(self,
                            horizon: 'PredictionHorizon'):
        pass

    def calc_results_energies(self):
        super().calc_results_energies()
        self.scenario.energies.loc[('sinks', 'del'), :] -= self.energies.loc['total', :]


class SystemCore(ElectricBlock):

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['acdc'] = eco.EcoEvaluator(name='acdc',
                                                   scenario=self.scenario,
                                                   block=self,
                                                   create_size=True,
                                                   flow_name='acdc',
                                                   ls=self.ls,
                                                   ccr=self.ccr,
                                                   capex_config=dict(consider_preexisting=self.capex_preexisting_acdc,
                                                                     spec=self.capex_spec,
                                                                     ),
                                                   mntex_config=dict(spec=self.mntex_spec),
                                                   opex_config=dict(spec=self.opex_spec),
                                                   )

        self.evaluators['dcac'] = eco.EcoEvaluator(name='dcac',
                                                   scenario=self.scenario,
                                                   block=self,
                                                   create_size=True,
                                                   flow_name='dcac',
                                                   ls=self.ls,
                                                   ccr=self.ccr,
                                                   capex_config=dict(consider_preexisting=self.capex_preexisting_dcac,
                                                                     spec=self.capex_spec,
                                                                     ),
                                                   mntex_config=dict(spec=self.mntex_spec),
                                                   opex_config=dict(spec=self.opex_spec),
                                                   )

    def __init__(self,
                 name : str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=None,
                         params=None,
                         parent=scenario)

    def params_preprocessing(self):

        self.expansion_equal = True if self.invest_acdc =='equal' or self.invest_dcac == 'equal' else False

        self.init_equalizable_variables(name_vars=['invest_acdc', 'invest_dcac'])
        self.init_equalizable_variables(name_vars=['size_preexisting_acdc', 'size_preexisting_dcac'])
        self.init_equalizable_variables(name_vars=['size_max_acdc', 'size_max_dcac'])

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

          dc          ac
          |-x--dcac-->|
          |           |
          |<---acdc-x-|
        """

        self.components['ac'] = solph.Bus()
        self.components['dc'] = solph.Bus()

        self.components['acdc'] = solph.components.Converter(
            inputs={self.components['ac']: solph.Flow(
                nominal_capacity=solph.Investment(ep_costs=self.evaluators['acdc'].opt.spec_ep_invest,
                                               existing=self.sizes['acdc'].preexisting,
                                               maximum=self.sizes['acdc'].expansion_max),
                variable_costs=self.evaluators['acdc'].opt.spec_ep_operation[horizon.dti_ph])},
            outputs={self.components['dc']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['dc']: self.eff['acdc']})

        self.components['dcac'] = solph.components.Converter(
            inputs={self.components['dc']: solph.Flow(
                nominal_capacity=solph.Investment(ep_costs=self.evaluators['dcac'].opt.spec_ep_invest,
                                               existing=self.sizes['dcac'].preexisting,
                                               maximum=self.sizes['dcac'].expansion_max),
                variable_costs=self.evaluators['dcac'].opt.spec_ep_operation[horizon.dti_ph])},
            outputs={self.components['ac']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['ac']: self.eff['dcac']})

        horizon.constraints.add_invest_costs(
            invest=(self.components['ac'], self.components['acdc']),
            capex_spec=self.evaluators['acdc'].capex.spec,
            invest_type='flow')

        horizon.constraints.add_invest_costs(
            invest=(self.components['dc'], self.components['dcac']),
            capex_spec=self.evaluators['dcac'].capex.spec,
            invest_type='flow')

        if self.expansion_equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            horizon.constraints.add_equal_invests([{'in': self.components['dc'], 'out': self.components['dcac']},
                                                   {'in': self.components['ac'], 'out': self.components['acdc']}])

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes['acdc'].expansion = horizon.results[(self.components['ac'],
                                                         self.components['acdc'])]['scalars']['invest']
        self.sizes['dcac'].expansion = horizon.results[(self.components['dc'],
                                                         self.components['dcac'])]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'acdc'] = horizon.results[(self.components['ac'],
                                                                  self.components['acdc'])]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'dcac'] = horizon.results[(self.components['dc'],
                                                                  self.components['dcac'])]['sequences']['flow'][horizon.dti_ch]

    def calc_results_flows(self):
        """
        post scenario method
        """
        super().calc_results_flows()
        self.flows['circular'] = self.flows[['dcac', 'acdc']].min(axis=1)

    def create_plot_traces(self):
        self.plot_traces['powers'].extend([go.Scatter(x=self.scenario.dti_eval,
                                                      y=self.flows.loc[self.scenario.dti_eval, 'dcac'],
                                                      mode='lines',
                                                      name=f'{self.name} DC-AC power (max. '
                                                           f'{self.sizes["dcac"].total / 1e3:.1f} kW)',
                                                      line=dict(width=2, dash=None, shape='hv'),
                                                      visible='legendonly',
                                                      ),
                                           go.Scatter(x=self.scenario.dti_eval,
                                                      y=self.flows.loc[self.scenario.dti_eval, 'acdc'],
                                                      mode='lines',
                                                      name=f'{self.name} AC-DC power (max. '
                                                           f'{self.sizes["acdc"].total / 1e3:.1f} kW)',
                                                      line=dict(width=2, dash=None, shape='hv'),
                                                      visible='legendonly',
                                                      )])


class RenewableSource(SourceBlock):
    """
    abstract class
    """

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['block'] = eco.EcoEvaluator(name='block',
                                                    scenario=self.scenario,
                                                    block=self,
                                                    create_size=True,
                                                    flow_name='out',
                                                    ls=self.ls,
                                                    ccr=self.ccr,
                                                    capex_config=dict(consider_preexisting=self.capex_preexisting_block,
                                                                      spec=self.capex_spec),
                                                    mntex_config=dict(spec=self.mntex_spec),
                                                    opex_config=dict(spec=self.opex_spec),
                                                    )

        self.evaluators['curt'] = eco.EcoEvaluator(name='curt',
                                                   scenario=self.scenario,
                                                   block=self,
                                                   flow_name='curt',
                                                   )

        self.evaluators['pot'] = eco.EcoEvaluator(name='pot',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  flow_name='pot',
                                                  )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'block': {'class_name': 'EconomicEvaluator',
                                    'params': {('capex', 'preexisting'): 'capex_preexisting_block',
                                               ('capex', 'spec'): 'capex_spec',
                                               ('mntex', 'spec'): 'mntex_spec',
                                               ('opex', 'spec'): 'opex_spec',
                                               ('size', 'name'): 'block',
                                               ('flow', 'name'): 'out',
                                               ('aux', 'ls'): 'ls',
                                               ('aux', 'ccr'): 'ccr'}},
                          'curt': {'class_name': 'EconomicEvaluator',
                                   'params': {('flow', 'name'): 'curt'}},
                          'pot': {'class_name': 'EconomicEvaluator',
                                  'params': {('flow', 'name'): 'pot'}}
                          },
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=None,
                         params=None,
                         parent=scenario)

        self.data = None  # todo move to a priori flows (except for wind speed and ambient temp)
        self.get_ts_data()

        self.share_curtailment = None

    @abstractmethod
    def get_ts_data(self):
        pass

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected      name_bus
          |                   |
          |<--x----name_out---|<--name_src
          |                   |
          |                   |-->name_exc
        """

        self.bus_connected = self.scenario.block_registry.get('TopLevelBlock', {})['core'].components[self.system]

        self.components['bus'] = solph.Bus()

        self.components['outflow'] = solph.components.Converter(
            inputs={self.components['bus']: solph.Flow()},
            outputs={self.bus_connected: solph.Flow()},
            conversion_factors={self.bus_connected: self.eff['block']}
        )

        # Curtailment has to be disincentivized in the optimization to force optimizer to charge storage or commodities
        # instead of curtailment. 2x cost_eps is required as SystemCore also has ccost_eps in charging direction.
        # All other components such as converters and storages only have cost_eps in the output direction.
        self.components['exc'] = solph.components.Sink(
            inputs={self.components['bus']: solph.Flow()}
        )

        self.components['src'] = solph.components.Source(
            outputs={self.components['bus']: solph.Flow(
                nominal_capacity=solph.Investment(ep_costs=self.evaluators['block'].opt.spec_ep_invest,
                                               existing=self.sizes['block'].preexisting,
                                               maximum=self.sizes['block'].expansion_max),
                fix=self.data.loc[horizon.dti_ph, 'power_spec'],
                variable_costs=self.evaluators['block'].opt.spec_ep_operation[horizon.dti_ph])}
        )

        horizon.constraints.add_invest_costs(invest=(self.components['src'], self.components['bus']),
                                             capex_spec=self.evaluators['block'].capex.spec,
                                             invest_type='flow')

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes['block'].expansion = horizon.results[(self.components['src'],
                                                          self.components['bus'])]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['outflow'],
                                                                 self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'pot'] = horizon.results[(self.components['src'],
                                                                 self.components['bus'])]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'curt'] = horizon.results[(self.components['bus'],
                                                                  self.components['exc'])]['sequences']['flow'][horizon.dti_ch]

    def calc_results_energies(self):

        super().calc_results_energies()
        # add curt and pot to scenario.energies
        self.scenario.energies.loc[('renewable', 'act'), :] += self.energies.loc['out', :]

        # pandas creates a RuntimeWarning at division by 0 -> try/except does not work
        if self.energies.loc['pot', 'sim'] == 0:
            self.scenario.logger.warning(f'Block {self.name}: Curtailment share calculation: division by zero')
        else:
            self.share_curtailment = self.energies.loc['curt', 'sim'] / self.energies.loc['pot', 'sim']

    def create_plot_traces(self):
        super().create_plot_traces()
        self.plot_traces['powers'].extend([go.Scatter(x=self.scenario.dti_eval,
                                                      y=-1 * self.flows.loc[self.scenario.dti_eval, 'curt'],
                                                      mode='lines',
                                                      name=f'{self.name} curtailed power',
                                                      line=dict(width=2, dash=None, shape='hv'),
                                                      visible='legendonly',
                                                      ),
                                           go.Scatter(x=self.scenario.dti_eval,
                                                      y=self.flows.loc[self.scenario.dti_eval, 'pot'],
                                                      mode='lines',
                                                      name=f'{self.name} potential power',
                                                      line=dict(width=2, dash=None, shape='hv'),
                                                      visible='legendonly',
                                                      )])

    def get_legend_entry(self):
        return f'{self.name} power (nom. {self.sizes["block"].total / 1e3:.1f} kW)'


class PVSource(RenewableSource):

    def get_ts_data(self):
        """
        pre scenario (init) method
        Get potential power profile from API or file, each either from Solcast or PVGIS
        """

        def calc_power_from_irradiation():
            """
            pre scenario (init) method
            calculate PV potential output power from insolation and weather data
            function is necessary for solcast input that does not contain power data
            """

            u0 = 26.9  # W/(˚C.m2) - cSi Free standing
            u1 = 6.2  # W.s/(˚C.m3) - cSi Free standing
            mod_temp = self.data['temp_air'] + (self.data['gti'] / (u0 + (u1 * self.data['speed_wind'])))

            # PVGIS temperature and irradiance coefficients for cSi panels as per Huld T., Friesen G., Skoczek A.,
            # Kenny R.P., Sample T., Field M., Dunlop E.D. A power-rating model for crystalline silicon PV modules
            # Solar Energy Materials & Solar Cells. 2011 95, 3359-3369.
            k1 = -0.017237
            k2 = -0.040465
            k3 = -0.004702
            k4 = 0.000149
            k5 = 0.000170
            k6 = 0.000005
            g = self.data['gti'] / 1000
            t = mod_temp - 25
            lng = np.zeros_like(g)
            lng[g != 0] = np.log(g[g != 0])  # ln(g) ignoring zeros

            # Faiman, D. Assessing the outdoor operating temperature of photovoltaic modules.
            # Prog. Photovolt. Res. Appl.2008, 16, 307–315
            eff_rel = (1 +
                       (k1 * lng) +
                       (k2 * (lng ** 2)) +
                       (k3 * t) +
                       (k4 * t * lng) +
                       (k5 * t * (lng ** 2)) +
                       (k6 * (t ** 2)))
            eff_rel = eff_rel.fillna(0)

            # calculate power of a 1kWp array, limited to 0 (negative values fail calculation)
            self.data['P'] = np.maximum(0, eff_rel * self.data['gti'])

        # region get data from PVGIS API
        if self.data_source == 'pvgis api':  # PVGIS API example selected
            api_startyear = self.scenario.starttime.tz_convert('utc').year
            api_endyear = self.scenario.sim_extd_endtime.tz_convert('utc').year
            api_length = api_endyear - api_startyear
            api_shift = pd.to_timedelta('0 days')

            API_MAX_YEAR = 2023
            API_MIN_YEAR = 2005
            API_MAX_LENGTH = API_MAX_YEAR - API_MIN_YEAR

            if api_length > API_MAX_LENGTH:
                raise ValueError('PVGIS API request exceeds maximum length of available data')
            elif api_endyear > API_MAX_YEAR:  # PVGIS-SARAH3 only has data up to 2023
                api_shift = (pd.to_datetime(f'{API_MAX_YEAR}-01-01 00:00:00+00:00') -
                             pd.to_datetime(f'{api_endyear}-01-01 00:00:00+00:00'))
                api_endyear = API_MAX_YEAR
                api_startyear = API_MAX_YEAR - api_length
                self.scenario.logger.warning(f'PVGIS API request exceeds available endtime - data shifted by '
                                             f'{abs(api_shift)} year{"s" if abs(api_shift) == 1 else ""} to '
                                             f'end in {API_MAX_YEAR}')
            elif api_startyear < API_MIN_YEAR:  # PVGIS-SARAH3 only has data from 2005
                api_shift = (pd.to_datetime(f'{API_MIN_YEAR}-01-01 00:00:00+00:00') -
                             pd.to_datetime(f'{api_startyear}-01-01 00:00:00+00:00'))
                api_startyear = API_MIN_YEAR
                api_endyear = API_MIN_YEAR + api_length
                self.scenario.logger.warning(f'PVGIS API request exceeds available starttime - data shifted by '
                                             f'{abs(api_shift)} year{"s" if abs(api_shift) == 1 else ""} to '
                                             f'start in {API_MIN_YEAR}')
            # Todo leap years can result in data shifting not landing at the same point in time

            optimal_tilt = True if self.tilt == 'optimal' else False
            optimal_angles = True if self.azimuth == 'optimal' else False
            if optimal_angles and not optimal_tilt:
                raise ValueError('Optimal azimuth requires optimal tilt as well')

            self.data, *_ = pvlib.iotools.get_pvgis_hourly(
                latitude=self.scenario.latitude,
                longitude=self.scenario.longitude,
                start=api_startyear,
                end=api_endyear,
                # PVGIS API is case sensitive and all inputs are lowered -> revert
                raddatabase=self.raddatabase.upper(),
                components=True,  # output solar radiation components (beam, diffuse, and reflected)
                surface_tilt=self.tilt if self.tilt != 'optimal' else 0,  # has to be numeric
                surface_azimuth=self.azimuth if self.azimuth != 'optimal' else 0,  # has to be numeric
                outputformat='json',
                usehorizon=self.horizon,
                userhorizon=self.horizon_custom,
                pvcalculation=True,
                peakpower=1,
                # PVGIS API is case sensitive and all inputs are lowered -> revert
                pvtechchoice={'crystsi': 'crystSi',
                              'cis': 'CIS',
                              'cdte': 'CdTe',
                              'unknown': 'Unknown'}[self.pvtechchoice],
                mountingplace=self.mountingplace,
                loss=0,
                trackingtype=self.trackingtype,
                optimal_surface_tilt=optimal_tilt,
                optimalangles=optimal_angles,
                url='https://re.jrc.ec.europa.eu/api/v5_3/',
                map_variables=True,
                timeout=30,  # default value
            )

            # rename column wind_speed to speed_wind
            self.data.rename(columns={'wind_speed': 'speed_wind'}, inplace=True)

            self.data.index = self.data.index.round('h')  # PVGIS does not give time slots as full hours
            self.data.index = self.data.index - api_shift
        # endregion

        # region get data from Solcast API
        elif self.data_source == 'solcast api':  # solcast API example selected
            # set api key as bearer token
            if self.scenario.settings.key_solcast_api is None:
                raise ValueError(f'Scenario {self.scenario.name} - Block {self.name}: '
                                 f'No Solcast API key specified in run arguments')

            latitude = self.scenario.latitude  # unmetered location for testing 41.89021
            longitude = self.scenario.longitude  # unmetered location for testing 12.492231

            # Avoid unintended use of metered coordinates
            if latitude != 41.89021 or longitude != 12.492231:
                raise ValueError('Remove this line if you want to proceed with metered coordinates!')

            params = dict(latitude=latitude,
                          longitude=longitude,
                          start=self.scenario.starttime,
                          end=self.scenario.sim_extd_endtime,
                          period='PT5M',
                          output_parameters=['air_temp',
                                             'albedo',
                                             'azimuth',
                                             'clearsky_dhi',
                                             'clearsky_dni',
                                             'clearsky_ghi',
                                             'clearsky_gti',
                                             'cloud_opacity',
                                             'dewpoint_temp',
                                             'dhi',
                                             'dni',
                                             'ghi',
                                             'gti',
                                             'precipitable_water',
                                             'precipitation_rate',
                                             'relative_humidity',
                                             'surface_pressure',
                                             'snow_depth',
                                             'snow_water_equivalent',
                                             'snow_soiling_rooftop',
                                             'snow_soiling_ground',
                                             'wind_direction_100m',
                                             'wind_direction_10m',
                                             'wind_speed_100m',
                                             'wind_speed_10m',
                                             'zenith'],
                          format='json',
                          array_type={0: 'fixed', 1: 'horizontal_single_axis'}[self.trackingtype],
                          time_zone='utc',
                          include_etadata=False,
                          terrain_shading=self.horizon,
                          )

            # add parameters azimuth and tilt. If not specified, Solcast uses default/optimized values
            if self.tilt != 'optimal':
                params['tilt'] = self.tilt
            if self.azimuth != 'optimal':
                # Convert to Solcast convention: (-180, 180], north=0, east=-90, south=180, west=90
                params['azimuth'] = x - 360 if (x := (-1 * self.azimuth) % 360) > 180 else x

            # get data from Solcast API
            response = requests.get(url='https://api.solcast.com.au/data/historic/radiation_and_weather',
                                    headers={'Authorization': f'Bearer {self.scenario.settingskey_solcast_api}'},
                                    params=params)

            if response.status_code != 200:
                raise ValueError(f'Block {self.name} - '
                                 f'Solcast API returned {response.status_code} instead of 200: '
                                 f'{response.json()["response_status"]["message"]}')

            self.data = pd.json_normalize(response.json()['estimated_actuals'])
            # save solcast file
            if not self.scenario.settings.largescalemode:
                self.data.to_csv(self.scenario.paths.create_result_path(suffix=f'{self.scenario.name}_'
                                                                               f'{self.name}_log_solcast_raw.csv'),
                    index=False,
                )

            # calculate period_start as only period_end is given, set as index and remove unnecessary columns
            self.data['period_start'] = pd.to_datetime(self.data['period_end']) - pd.to_timedelta(self.data['period'])
            self.data.set_index(pd.DatetimeIndex(self.data['period_start']), inplace=True)
            self.data = self.data.tz_convert(self.scenario.timezone)
            self.data.drop(columns=['period', 'period_start', 'period_end'], inplace=True)
            # rename columns according to further processing steps
            self.data.rename(columns={'air_temp': 'temp_air',
                                      'wind_speed_10m': 'speed_wind'}, inplace=True)
            # calculate specific pv power
            calc_power_from_irradiation()
        # endregion

        elif 'file' in self.data_source:
            # region get data from file
            path_input_file = self.scenario.paths.input / utils.set_extension(filename=self.filename,
                                                                                  default_extension='.csv')

            # region read input data from timeseries csv with specific power
            if self.data_source == 'file':
                self.data = utils.read_timeseries_csv(path_input_file=path_input_file,
                                                      block=self,
                                                      scenario=self.scenario,
                                                      multiheader=False,
                                                      resampling=False)
            # endregion
            elif self.data_source in ['pvgis file', 'solcast file']:
                # region get data from PVGIS file
                if self.data_source == 'pvgis file':
                    self.data, meta = pvlib.iotools.read_pvgis_hourly(path_input_file, map_variables=True)
                    self.scenario.latitude = meta['inputs']['latitude']
                    self.scenario.longitude = meta['inputs']['longitude']
                    # rename column wind_speed to speed_wind
                    self.data.rename(columns={'wind_speed': 'speed_wind'}, inplace=True)
                    self.data.index = self.data.index.round('h')  # PVGIS does not necessarily give full hour time vals
                # endregion

                # region get data from Solcast file
                elif self.data_source == 'solcast file':
                    # no lat/lon contained in solcast files
                    self.data = pd.read_csv(path_input_file)
                    self.data.rename(columns={'air_temp': 'temp_air',
                                              'wind_speed_10m': 'speed_wind'}, inplace=True)
                    self.data['period_start'] = (pd.to_datetime(self.data['period_end'], utc=True) -
                                                 pd.to_timedelta(self.data['period']))
                    self.data.set_index(pd.DatetimeIndex(self.data['period_start']), inplace=True)
                    self.data = self.data.tz_convert(self.scenario.timezone)

                    # if at least one of azimuth or tilt are specified, recalculate irradiation for new pose
                    if self.azimuth is not None or self.tilt is not None:
                        if self.azimuth is None or self.azimuth == 'optimal':
                            azimuth = 0 if self.scenario.latitude < 0 else 180  # Solcast "optimum"
                        else:
                            azimuth = self.azimuth

                        if self.tilt is None or self.tilt == 'optimal':
                            abs(self.scenario.latitude)  # Something close to Solcast "optimum"
                        else:
                            tilt = self.tilt

                        # calculate solar position for location (gets altitude from lookup table)
                        solar_position = (pvlib.location.Location(latitude=self.scenario.latitude,
                                                                  longitude=self.scenario.longitude)
                                          .get_solarposition(times=self.data.index,
                                                             method='nrel_numpy')
                                          )
                        solar_azimuth = solar_position['azimuth']
                        solar_zenith = solar_position['zenith']

                        # alternatively use solcast data, but this data is rounded to integers  # ToDo: benchmark
                        # solar_azimuth = self.data['azimuth']
                        # solar_zenith = self.data['zenith']

                        self.data['gti'] = pvlib.irradiance.get_total_irradiance(
                            surface_tilt=tilt,
                            surface_azimuth=azimuth,
                            solar_zenith=solar_zenith,
                            solar_azimuth=solar_azimuth,
                            dni=self.data['dni'],
                            ghi=self.data['ghi'],
                            dhi=self.data['dhi'],
                            dni_extra=pvlib.irradiance.get_extra_radiation(self.data.index),
                            model='haydavies',  # 'haydavies', 'reindl', 'klucher', or 'isotropic' too
                            albedo=self.data['albedo'],
                        )['poa_global']

                    self.data = self.data[['temp_air', 'speed_wind', 'gti']]
                    calc_power_from_irradiation()
                # endregion

            else:
                raise ValueError(f'Scenario {self.scenario.name} - Block {self.name}: No usable PV data input specified')

        # region resample, localize, and transform data
        # data is in W for a 1kWp PV array -> convert to specific power (if not already done e.g. for timeseries file)
        if 'power_spec' not in self.data.columns:
            self.data['power_spec'] = self.data['P'] / 1e3
        # resample to timestep, fill NaN values with previous ones (or next ones, if not available)
        self.data = self.data.resample(self.scenario.timestep).mean().ffill().bfill()
        # convert to local time
        self.data.index = self.data.index.tz_convert(tz=self.scenario.timezone)

        # only keep relevant columns and timestamps
        self.data = self.data.loc[self.scenario.dti_sim, ['power_spec', 'speed_wind', 'temp_air']]
        # endregion

        if not self.scenario.settings.largescalemode:
            self.data.to_csv(self.scenario.paths.create_result_path(suffix=f'{self.scenario.name}_{self.name}_log.csv'))

        if getattr(self, 'temp_scn', False):  # parameter only exists for instances specified in scenario.temp_air
            self.scenario.temp_air['temp_air'] = self.data['temp_air']


class WindSource(RenewableSource):

    def get_ts_data(self):
        """
        pre scenario (init) method
        get potential power profile from PVSource block or file
        """
        if self.data_source in self.scenario.block_registry.get('TopLevelBlock', {}).keys():
            # region get data from PVSource block
            self.data = self.scenario.block_registry.get('TopLevelBlock', {})[self.data_source].data.copy()
            self.data['speed_wind_adj'] = windpowerlib.wind_speed.hellman(self.data['speed_wind'], 10, self.height)

            path_turbine_data_file = self.scenario.paths.data_persist / 'turbine_data.pkl'
            turbine_data = pd.read_pickle(path_turbine_data_file)
            # smallest fully filled wind turbine in dataseta as per June 2024
            turbine_data = turbine_data.loc[turbine_data['turbine_type'] == 'E-53/800'].reset_index()

            self.data['power_original'] = windpowerlib.power_output.power_curve(
                wind_speed=self.data['speed_wind_adj'],
                power_curve_wind_speeds=ast.literal_eval(turbine_data.loc[0, 'power_curve_wind_speeds']),
                power_curve_values=ast.literal_eval(turbine_data.loc[0, 'power_curve_values']),
                density_correction=False)
            self.data['power_spec'] = self.data['power_original'] / turbine_data.loc[0, 'nominal_power']
            # endregion
        elif self.data_source == 'file':
            # region get data from file
            self.data = utils.read_timeseries_csv(path_input_file=(self.scenario.paths.input /
                                                                   utils.set_extension(filename=self.filename,
                                                                                       default_extension='.csv')),
                                                  block=self,
                                                  scenario=self.scenario)
            # endregion
        else:
            raise ValueError(f'Scenario {self.scenario.name} - Block {self.name}: No usable data input specified')

        if not self.scenario.settings.largescalemode:
            self.data.to_csv(self.scenario.paths.create_result_path(suffix=f'{self.scenario.name}_{self.name}_log.csv'))


class FixedDemand(SinkBlock):

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['block'] = eco.EcoEvaluator(name='block',
                                                    scenario=self.scenario,
                                                    block=self,
                                                    flow_name='in',
                                                    crev_config=dict(spec=self.crev_spec),
                                                    )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'block': {'class_name': 'EconomicEvaluator',
                                    'params': {('crev', 'spec'): 'crev_spec',
                                               ('flow', 'name'): 'in'}}
                          },
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=['demand'],
                         params=None,
                         parent=scenario)

        self.get_flows_apriori()

    def get_flows_apriori(self):
        self.flows_apriori.index = self.scenario.dti_sim  # ToDo: Why needs this to be set explicitly? Should be done in init()
        if self.load_profile in ['h0', 'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'l0', 'l1', 'l2']:
            def get_timeframe(date):
                month = date.month
                day = date.day
                if ((month, day) >= (11, 1)) or ((month, day) <= (3, 20)):
                    return 'Winter'
                elif (5, 15) <= (month, day) <= (9, 14):
                    return 'Summer'
                else:  # Transition months
                    return 'Transition'

            def get_daytype(date, holidays):
                if date.date() in holidays or date.weekday() == 6:
                    return 'Sunday'
                # Treat Christmas Eve and New Year's Eve as Saturdays if they are not Sundays
                elif (date.weekday() == 5) or ((date.month, date.day) in [(12, 24), (12, 31)]):
                    return 'Saturday'
                else:
                    return 'Workday'

            # Read BDEW SLP profiles
            slp = pd.read_csv(self.scenario.paths.data_persist / 'slp_bdew.csv',
                              skiprows=[0],
                              header=[0, 1, 2],
                              index_col=0)

            slp.index = pd.to_datetime(slp.index, format='%H:%M').time

            # use a fixed frequency of 15 minutes for the timeseries generation as the SLPs are given with that frequency
            freq_slp = '15min'
            dti_slp = pd.DatetimeIndex(pd.date_range(start=self.scenario.starttime.floor(freq_slp),
                                                     end=self.scenario.dti_sim.max().ceil(freq_slp),
                                                     freq=freq_slp))

            data = pd.Series(index=dti_slp, data=0, dtype='float64')

            data = data.index.to_series().apply(
                lambda x: slp.loc[x.time(), (self.load_profile.upper(), get_timeframe(x),
                                             get_daytype(x, self.scenario.holiday_dates))])

            # apply dynamic correction for household profiles
            if self.load_profile == 'h0':
                # for private households use dynamic correction as stated in VDEW manual -> round to 1/10 Watt
                num_day = data.index.dayofyear.astype('int64')
                data = round(data * (-3.92e-10 * num_day ** 4 + 3.2e-7 * num_day ** 3 -
                                     7.02e-5 * num_day ** 2 + 2.1e-3 * num_day ** 1 + 1.24),
                             ndigits=1)

            # scale load profile (given for consumption of 1MWh per year) to specified yearly consumption
            # this calculation leads to small deviations from the specified yearly consumption due to varying holidays and
            # leap years, but is the correct way as stated by the VDEW manual
            data *= (self.consumption_yrl / 1e6)

            # resample to simulation time step
            self.flows_apriori['demand'] = data.resample(self.scenario.timestep).mean().ffill().bfill()
        elif self.load_profile in ['const', 'constant']:
            self.flows_apriori['demand'] = self.consumption_yrl / (365 * 24)
        elif isinstance(self.load_profile, str):  # load_profile is a file name
            data = utils.read_timeseries_csv(path_input_file=(self.scenario.paths.input /
                                                              utils.set_extension(filename=self.load_profile,
                                                                                  default_extension='.csv')),
                                             block=self,
                                             scenario=self.scenario,
                                             )

            if data.shape[1] != 1:
                self.scenario.logger.warning(f'Input file "{utils.set_extension(self.load_profile)}" for parameter '
                                             f'"load_profile" in block "{self.name}" has more than one column. '
                                             f'Sum of all columns is calculated for load profile.')

            data = data.sum(axis=1)[self.flows_apriori.index]  # convert to series and slice to sim timeframe
            self.flows_apriori['demand'] = data
        else:
            raise ValueError(f'Parameter "load_profile" in block "{self.block.name}" is not valid')

        if not self.scenario.settings.largescalemode:
            self.flows_apriori['demand'].to_csv(
                self.scenario.paths.create_result_path(suffix=f'{self.scenario.name}_{self.name}_flow.csv')
            )

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected
          |
          |-x->name_snk
          |
        """

        self.bus_connected = self.scenario.block_registry.get('TopLevelBlock', {})['core'].components[self.system]

        self.components['snk'] = solph.components.Sink(
            inputs={self.bus_connected: solph.Flow(nominal_capacity=1,
                                                   fix=self.flows_apriori['demand'][horizon.dti_ph])}
        )

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.bus_connected,
                                                                self.components['snk'])]['sequences']['flow'][horizon.dti_ch]

    def get_legend_entry(self):
        return f'{self.name} power'


class ControllableSource(SourceBlock):

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['block'] = eco.EcoEvaluator(name='block',
                                                    scenario=self.scenario,
                                                    block=self,
                                                    create_size=True,
                                                    flow_name='out',
                                                    ls=self.ls,
                                                    ccr=self.ccr,
                                                    capex_config=dict(consider_preexisting=self.capex_preexisting_block,
                                                                      spec=self.capex_spec),
                                                    mntex_config=dict(spec=self.mntex_spec),
                                                    opex_config=dict(spec=self.opex_spec),
                                                    )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'block': {'class_name': 'EconomicEvaluator',
                                    'params': {('capex', 'preexisting'): 'capex_preexisting_block',
                                               ('capex', 'spec'): 'capex_spec',
                                               ('mntex', 'spec'): 'mntex_spec',
                                               ('opex', 'spec'): 'opex_spec',
                                               ('size', 'name'): 'block',
                                               ('flow', 'name'): 'out',
                                               ('aux', 'ls'): 'ls',
                                               ('aux', 'ccr'): 'ccr'}},
                          },
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         params=None,
                         flow_apriori_names=None,
                         parent=scenario)

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected
          |
          |<-name_gen
          |
        """

        self.bus_connected = self.scenario.block_registry.get('TopLevelBlock', {})['core'].components[self.system]

        self.components['src'] = solph.components.Source(
            outputs={self.bus_connected: solph.Flow(
                nominal_capacity=solph.Investment(ep_costs=self.evaluators['block'].opt.spec_ep_invest,
                                               existing=self.sizes['block'].preexisting,
                                               maximum=self.sizes['block'].expansion_max),
                variable_costs=self.evaluators['block'].opt.spec_ep_operation[horizon.dti_ph])}
        )

        horizon.constraints.add_invest_costs(invest=(self.components['src'], self.bus_connected),
                                             capex_spec=self.evaluators['block'].capex.spec,
                                             invest_type='flow')

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes['block'].expansion = horizon.results[(self.components['src'],
                                                          self.bus_connected)]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['src'],
                                                                 self.bus_connected)]['sequences']['flow'][horizon.dti_ch]


class GridConnection(ElectricBlock):

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['g2s'] = eco.EcoEvaluator(name='g2s',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  create_size=True,
                                                  flow_name='out',
                                                  ls=self.ls,
                                                  ccr=self.ccr,
                                                  capex_config=dict(consider_preexisting=self.capex_preexisting_g2s,
                                                                    spec=self.capex_spec),
                                                  mntex_config=dict(spec=self.mntex_spec),
                                                  )

        self.evaluators['s2g'] = eco.EcoEvaluator(name='s2g',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  create_size=True,
                                                  flow_name='in',
                                                  ls=self.ls,
                                                  ccr=self.ccr,
                                                  capex_config=dict(consider_preexisting=self.capex_preexisting_s2g,
                                                                    spec=self.capex_spec),
                                                  mntex_config=dict(spec=self.mntex_spec),
                                                  )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'g2s': {'class_name': 'EconomicEvaluator',
                                  'params': {('capex', 'preexisting'): 'capex_preexisting_g2s',
                                             ('capex', 'spec'): 'capex_spec',
                                             ('mntex', 'spec'): 'mntex_spec',
                                             ('size', 'name'): 'g2s',
                                             ('flow', 'name'): 'out',
                                             ('aux', 'ls'): 'ls',
                                             ('aux', 'ccr'): 'ccr'}},
                          's2g': {'class_name': 'EconomicEvaluator',
                                  'params': {('capex', 'preexisting'): 'capex_preexisting_s2g',
                                             ('capex', 'spec'): 'capex_spec',
                                             ('mntex', 'spec'): 'mntex_spec',
                                             ('size', 'name'): 's2g',
                                             ('flow', 'name'): 'in',
                                             ('aux', 'ls'): 'ls',
                                             ('aux', 'ccr'): 'ccr'}},
                          },
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=None,
                         params=None,
                         parent=scenario)

        self.inflows = dict()
        self.outflows = dict()

        self.peak_periods = pd.DataFrame()
        self.bus_activation = pd.DataFrame()

        self.initialize_peakshaving()

        if not self.markets:
            raise ValueError(f'Block "{self.name}": No markets defined! '
                             f'At least one market has to be defined to buy and sell energy to the grid.')

        self.subblocks = {market: GridMarket(name=market,
                                             scenario=self.scenario,
                                             params=None,
                                             parent=self)
                          for market in self.markets}
        del self.markets

    def params_preprocessing(self):

        self.expansion_equal = True if self.invest_g2s == 'equal' or self.invest_s2g == 'equal' else False

        self.init_equalizable_variables(name_vars=['invest_s2g', 'invest_g2s'])
        self.init_equalizable_variables(name_vars=['size_preexisting_g2s', 'size_preexisting_s2g'])
        self.init_equalizable_variables(name_vars=['size_max_g2s', 'size_max_s2g'])

    def initialize_peakshaving(self):
        # Create functions to extract relevant property of datetimeindex for peakshaving intervals
        periods_func = {
            'day': lambda x: x.strftime('%Y-%m-%d'),
            'week': lambda x: x.strftime('%Y-CW%W'),
            'month': lambda x: x.strftime('%Y-%m'),
            'quarter': lambda x: f"{x.year}-Q{(x.month - 1) // 3 + 1}",
            'year': lambda x: x.strftime('%Y'),
        }

        if self.peak_period not in periods_func.keys():
            raise ValueError(f'Block {self.name}: parameter "peak_period" must be one of {periods_func.keys()}')

        # Get dummies directly from the 'periods' data
        self.bus_activation = pd.get_dummies(
            self.scenario.dti_sim.to_series().map(periods_func[str(self.peak_period)])).astype(int)

        # Create a series to store peak power values
        self.peak_periods = pd.DataFrame(index=self.bus_activation.columns,
                                                columns=['power'],
                                                data=self.peak_power_init,  # cumulative variable
                                                dtype='float64')

        def process_period(period):
            dti_period = self.bus_activation[self.bus_activation[period] == 1].index
            dti_period_sim = dti_period[dti_period.isin(self.scenario.dti_eval)]  # remove non-sim timestamps

            # if interval is not part of dti_sim (happens for rh), dti is empty -> return 0
            if len(dti_period_sim) == 0:
                period_fraction = 0.0
            else:
                if period == 'day':
                    start = dti_period_sim.min().normalize()
                    end = start + pd.DateOffset(days=1) - self.scenario.timestep_td
                elif period == 'week':
                    start = dti_period_sim.min().normalize() - pd.Timedelta(days=dti_period_sim[0].weekday())
                    end = start + pd.DateOffset(weeks=1) - self.scenario.timestep_td
                elif period == 'month':
                    start = dti_period_sim.min().normalize().replace(day=1)
                    end = start + pd.DateOffset(months=1) - self.scenario.timestep_td
                elif period == 'quarter':
                    start = dti_period_sim.min().normalize().replace(day=1, month=((dti_period_sim[0].month - 1) // 3) * 3 + 1)
                    end = start + pd.DateOffset(months=3) - self.scenario.timestep_td
                elif period == 'year':
                    start = dti_period_sim.min().normalize().replace(day=1, month=1)
                    end = start + pd.DateOffset(years=1) - self.scenario.timestep_td
                else:
                    start = dti_period_sim.min()
                    end = dti_period_sim.max()

                period_fraction = len(dti_period_sim) / len(pd.date_range(start, end, freq=self.scenario.timestep_td))

            return pd.Series({'period_fraction': period_fraction,
                              'start': dti_period.min(),
                              'end': dti_period.max()})

        # Apply the function to each period in peak_periods
        self.peak_periods[['period_fraction', 'start', 'end']] = self.peak_periods.index.to_series().apply(process_period)

        self.n_peak_periods_yr = (pd.date_range(start=self.scenario.starttime,
                                                end=self.scenario.starttime + pd.DateOffset(years=1),
                                                freq=self.scenario.timestep,
                                                inclusive='left')
                                  .to_series().apply(periods_func[str(self.peak_period)])).unique().size

        self.evaluators.update({period: eco.PeakEvaluator(name=period,
                                                          block=self,
                                                          scenario=self.scenario,
                                                          opex_config=dict(spec_peak=self.opex_spec_peak),
                                                          )
                                for period in self.peak_periods.index})

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected          name_bus
          |                        |
          |---name_inflow_1--x---->|
          |<--name_outflow_1--x----|
          |                        |---(GridMarket Instance)
          |---name_inflow_2--x---->|
          |<--name_outflow_2--x----|
          |                        |---(GridMarket Instance)

                     ...

          |---name_inflow_n--x---->|
          |<--name_outflow_n--x----|
        """

        self.bus_connected = self.scenario.block_registry.get('TopLevelBlock', {})['core'].components[self.system]

        self.components['bus'] = solph.Bus()

        self.inflows = {f'{self.name}_inflow_1': solph.components.Converter(
            # Peakshaving not implemented for feed-in into grid
            inputs={self.bus_connected: solph.Flow()},
            # Size optimization
            outputs={self.components['bus']: solph.Flow(
                nominal_capacity=solph.Investment(ep_costs=self.evaluators['s2g'].opt.spec_ep_invest,
                                                  existing=self.sizes['s2g'].preexisting,
                                                  maximum=self.sizes['s2g'].expansion_max),
                variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['bus']: 1})}

        self.components.update(self.inflows)

        self.outflows = {f'{self.name}_outflow_{period}': solph.components.Converter(
            # Size optimization: investment costs are assigned to first peakshaving interval only. The application of
            # constraints ensures that the optimized grid connection sizes of all peakshaving intervals are equal
            inputs={self.components['bus']: solph.Flow(
                nominal_capacity=solph.Investment(ep_costs=(self.evaluators['g2s'].opt.spec_ep_invest if period == self.peak_periods.index[0] else 0),
                                                  existing=self.sizes['g2s'].preexisting,
                                                  maximum=self.sizes['g2s'].expansion_max)
            )},
            # Peakshaving
            outputs={self.bus_connected: solph.Flow(
                nominal_capacity=(solph.Investment(ep_costs=(self.evaluators[period].opt.spec_ep_peak
                                                             if self.peakshaving else 0),
                                                   existing=self.peak_periods.loc[period, 'power'])
                               ),
                max=(self.bus_activation.loc[horizon.dti_ph, period]))},
            conversion_factors={self.bus_connected: 1}) for period in self.peak_periods.index}

        self.components.update(self.outflows)

        horizon.constraints.add_invest_costs(invest=(self.components[f'{self.name}_inflow_1'],
                                                     self.components['bus']),
                                             capex_spec=self.evaluators['s2g'].capex.spec,
                                             invest_type='flow')
        horizon.constraints.add_invest_costs(invest=(self.components['bus'],
                                                     self.components[f'{self.name}_outflow_{self.peak_periods.index[0]}']),
                                             capex_spec=self.evaluators['g2s'].capex.spec,
                                             invest_type='flow')

        # The optimized sizes of the buses of all peakshaving intervals have to be the same as they technically
        # represent the same grid connection
        equal_investments = [{'in': self.components['bus'], 'out': outflow}
                             for outflow in self.outflows.values()]

        # If size of in- and outflow from and to the grid have to be the same size, add outflow investment(s)
        if self.expansion_equal:
            equal_investments.append({'in': self.components[f'{self.name}_inflow_1'],
                                      'out': self.components['bus']})  # currently only works without peakshaving for inflows

        # add list of variables to the scenario constraints if list contains more than one element
        # lists with one element occur, if peakshaving is deactivated and grid sizes don't have to be equal
        if len(equal_investments) > 1:
            horizon.constraints.add_equal_invests(equal_investments)

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes['g2s'].expansion = horizon.results[(self.components['bus'],
                                                        list(self.outflows.values())[0])]['scalars']['invest']
        self.sizes['s2g'].expansion = horizon.results[(list(self.inflows.values())[0],
                                                        self.components['bus'])]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'in'] = sum([horizon.results[(inflow, self.components['bus'])]['sequences']['flow'][horizon.dti_ch]
                                                    for inflow in self.inflows.values()])
        self.flows.loc[horizon.dti_ch, 'out'] = sum([horizon.results[(self.components['bus'], outflow)]['sequences']['flow'][horizon.dti_ch]
                                                     for outflow in self.outflows.values()])

        def get_peak_power(row):
            peak_power = max(row['power'],
                             horizon.results[(self.outflows[f'{self.name}_outflow_{row.name}'],
                                              self.bus_connected)]['sequences']['flow'][horizon.dti_ch].max())
            return peak_power

        self.peak_periods['power'] = self.peak_periods.apply(get_peak_power, axis=1)

    def calc_results_flows(self):
        super().calc_results_flows()
        self.flows['circular'] = self.flows[['in', 'out']].min(axis=1)

    def calc_results_energies(self):
        super().calc_results_energies()
        self.scenario.energies.loc[('sources', 'pro'), :] += self.energies.loc['out', :]
        self.scenario.energies.loc[('sinks', 'del'), :] += self.energies.loc['in', :]

    def create_result_summary(self):
        super().create_result_summary()

        peak_power_results = {}
        for period, row in self.peak_periods.iterrows():
            if row['start'] < self.scenario.sim_endtime:
                peak_power_results.update({
                    f'{period}_peak_power': row['power'],
                    f'{period}_peak_period_fraction': row['period_fraction'],
                    f'{period}_peak_opex_sim': self.evaluators[period].opex.sim
                })
        self.result_summary.append(pd.Series(peak_power_results))

    def create_result_messages(self, *_):
        super().create_result_messages(unit='kW')

        # add peak power results
        self.result_messages.extend(
            [f'{"Optimized peak" if self.peakshaving else "Peak"} power in component "{self.name}" for peak period '
             f'"{period}": {row["power"] / 1e3:.1f} kW '
             f'- OPEX in simulation period: {self.evaluators[period].opex.sim:.2f} {self.scenario.currency}'
             for period, row in self.peak_periods.iterrows() if row['start'] < self.scenario.sim_endtime]
        )

    def get_legend_entry(self):
        return (f'{self.name} power (max. {self.sizes["g2s"].total / 1e3:.1f} kW from / '
                f'{self.sizes["s2g"].total / 1e3:.1f} kW to grid)')


class GridMarket(ElectricBlock):

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['g2s'] = eco.EcoEvaluator(name='g2s',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  flow_name='out',
                                                  opex_config=dict(spec=self.opex_spec_g2s),
                                                  )

        self.evaluators['s2g'] = eco.EcoEvaluator(name='s2g',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  flow_name='in',
                                                  opex_config=dict(spec=self.opex_spec_s2g),
                                                  )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'g2s': {'class_name': 'EconomicEvaluator',
                                  'params': {('opex', 'spec'): 'opex_spec_g2s',
                                             ('flow', 'name'): 'out'}},
                          's2g': {'class_name': 'EconomicEvaluator',
                                  'params': {('opex', 'spec'): 'opex_spec_s2g',
                                             ('flow', 'name'): 'in'}},
                          },
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario,
                 params,
                 parent):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=None,
                         params=params,
                         parent=parent)

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        """
        pre horizon method

        parent_bus
            |<---x----name_src
            |
            |----x--->name_snk
            |
        """

        self.components['src'] = solph.components.Source(
            outputs={self.parent.components['bus']: solph.Flow(
                nominal_capacity=self.pwr_g2s,
                max=1 if self.pwr_g2s else None,
                variable_costs=self.evaluators['g2s'].opt.spec_ep_operation[horizon.dti_ph])
            }
        )

        self.components['snk'] = solph.components.Sink(
            inputs={
                self.parent.components['bus']: solph.Flow(
                    nominal_capacity=self.pwr_s2g,
                    max=1 if self.pwr_s2g else None,
                    variable_costs=(self.evaluators['s2g'].opt.spec_ep_operation[horizon.dti_ph]),
                )
            }
        )

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """

        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.parent.components['bus'],
                                                                self.components['snk'])]['sequences']['flow'][horizon.dti_ch]

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['src'],
                                                                 self.parent.components['bus'])]['sequences']['flow'][horizon.dti_ch]

    def get_legend_entry(self):
        powers = {power: min(self.parent.sizes[power].total,
                             (getattr(self, f'pwr_{power}')
                              if pd.notna(getattr(self, f'pwr_{power}'))
                              else self.parent.sizes[power].total))
                  for power in ['g2s', 's2g']}

        return f'{self.name} power (max. {powers["g2s"] / 1e3:.1f} kW from / {powers["s2g"] / 1e3:.1f} kW to grid)'


class StorageBlock(ElectricBlock):
    """
    abstract class
    """

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['storage'] = eco.EcoEvaluator(name='storage',
                                                      scenario=self.scenario,
                                                      block=self,
                                                      create_size=True,
                                                      ls=self.ls,
                                                      ccr=self.ccr,
                                                      capex_config=dict(consider_preexisting=self.capex_preexisting_storage,
                                                                        spec=self.capex_spec),
                                                      mntex_config=dict(spec=self.mntex_spec),
                                                      )

        self.evaluators['in'] = eco.EcoEvaluator(name='in',
                                                 scenario=self.scenario,
                                                 block=self,
                                                 flow_name='in',
                                                 opex_config=dict(spec=self.opex_spec),
                                                 )

        self.evaluators['out'] = eco.EcoEvaluator(name='out',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  flow_name='out',
                                                  )

        self.evaluators['bat_in'] = eco.EcoEvaluator(name='bat_in',
                                                     scenario=self.scenario,
                                                     block=self,
                                                     flow_name='bat_in',
                                                     )

        self.evaluators['bat_out'] = eco.EcoEvaluator(name='bat_out',
                                                      scenario=self.scenario,
                                                      block=self,
                                                      flow_name='bat_out',
                                                      )


    @staticmethod
    def get_init_definitions():
        return dict(pois={'storage': {'class_name': 'EconomicEvaluator',
                                      'params': {('capex', 'preexisting'): 'capex_preexisting_storage',
                                                 ('capex', 'spec'): 'capex_spec',
                                                 ('mntex', 'spec'): 'mntex_spec',
                                                 ('size', 'name'): 'storage',
                                                 ('aux', 'ls'): 'ls',
                                                 ('aux', 'ccr'): 'ccr'}},
                          'in': {'class_name': 'EconomicEvaluator',
                                 'params': {('opex', 'spec'): 'opex_spec',
                                            ('flow', 'name'): 'in'}},
                          'out': {'class_name': 'EconomicEvaluator',
                                  'params': {('flow', 'name'): 'out'}},
                          'bat_in': {'class_name': 'EconomicEvaluator',
                                     'params': {('flow', 'name'): 'bat_in'}},
                          'bat_out': {'class_name': 'EconomicEvaluator',
                                      'params': {('flow', 'name'): 'bat_out'}},
                          },
                    state_names=['energy', 'soc', 'soh', 'q_loss_cal', 'q_loss_cyc', 'soc_min', 'soc_max'])

    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 flow_apriori_names: list = None,
                 params: dict = None,
                 parent: 'Block | Scenario' = None,
                 ):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=flow_apriori_names,
                         params=params,
                         parent=parent)

        def calc_loss_rate_per_period(period: pd.Timedelta = pd.Timedelta(hours=1)) -> float:
            """
            convert self-discharge rate (sdr) per month of a battery storage to a loss rate (lr) per target time step.
            oemof specifies one hour as the target time step for the loss rate.
            """
            ratio_timestep = period / pd.Timedelta('30 days')  # assumption: 30 days per month
            return (1 - (1 - self.sdr) ** ratio_timestep)

        self.loss_rate_per_hour = calc_loss_rate_per_period(period=pd.Timedelta(hours=1))
        self.loss_rate_per_ts = calc_loss_rate_per_period(period=self.scenario.timestep_td)
        delattr(self, 'sdr')

        # set initial SOC
        self.states.loc[self.scenario.starttime, 'soc'] = self.soc_init
        delattr(self, 'soc_init')

        # set initial SOH
        self.states.loc[self.scenario.starttime, 'soh'] = 1 - self.q_loss_cal_init - self.q_loss_cyc_init

        # set initial calendric loss
        self.states.loc[self.scenario.starttime, 'q_loss_cal'] = self.q_loss_cal_init
        delattr(self, 'q_loss_cal_init')

        # set inital cyclic loss
        self.states.loc[self.scenario.starttime, 'q_loss_cyc'] = self.q_loss_cyc_init
        delattr(self, 'q_loss_cyc_init')

        self.states.loc[:, 'soc_min'] = (1 - self.states.loc[self.scenario.starttime, 'soh']) / 2
        self.states.loc[:, 'soc_max'] = 1 - ((1 - self.states.loc[self.scenario.starttime, 'soh']) / 2)

        # initialization of aging model after all blocks are initialized to get temp from pv blocks
        self.aging_model = None

    def pre_scenario(self):
        super().pre_scenario()
        self.aging_model = bat.BatteryPackModel(self)

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None,
                                ):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected   name_bus
             |             |
             |<-x-name_xc--|
             |             |<--->name_ess
             |-x-name_ess->|
             |             |

        """

        if params is None:
            raise ValueError(f'Block "{self.name}": Parameter "params" is required for StorageBlock method '
                             f'define_oemof_components()')

        self.components['bus'] = solph.Bus()

        self.components['inflow'] = solph.components.Converter(
            inputs={self.bus_connected: solph.Flow(
                nominal_capacity=params['inflow_nominal_capacity'],
                max=params['inflow_max'],
                fix=params['inflow_fix'],
            )},
            outputs={self.components['bus']: solph.Flow(
                variable_costs=self.scenario.cost_eps * -3  # incentivize charging of StorageBlocks vs. curtailment
            )},
            conversion_factors={self.components['bus']: self.eff['chg_int']})

        self.components['outflow'] = solph.components.Converter(
            inputs={self.components['bus']: solph.Flow()},
            outputs={self.bus_connected: solph.Flow(
                nominal_capacity=params['outflow_nominal_capacity'],
                max=params['outflow_max'],
                fix=params['outflow_fix'],
                variable_costs=self.scenario.cost_eps * 4  # disincentivize waste loop with inflow (sum must be positive)
                )
            },
            conversion_factors={self.bus_connected: self.eff['dis_int']})

        self.components['storage'] = solph.components.GenericStorage(
            inputs={self.components['bus']: solph.Flow(
                variable_costs=self.evaluators['storage'].opt.spec_ep_operation[horizon.dti_ph]
            )},
            outputs={
                self.components['bus']: solph.Flow(
                    variable_costs=self.scenario.cost_eps
                    )},
            loss_rate=self.loss_rate_per_hour,
            balanced=params['storage_balanced'],
            initial_storage_level=self.states.loc[horizon.starttime, ['soc', 'soc_min', 'soc_max']].median(),
            # crate measured "outside" of conversion factor (efficiency)
            invest_relation_input_capacity=params['invest_relation_input_capacity'],
            invest_relation_output_capacity=params['invest_relation_output_capacity'],
            inflow_conversion_factor=np.sqrt(self.eff['storage_roundtrip']),
            outflow_conversion_factor=np.sqrt(self.eff['storage_roundtrip']),
            nominal_capacity=solph.Investment(
                ep_costs=self.evaluators['storage'].opt.spec_ep_invest,
                existing=self.sizes['storage'].preexisting,
                maximum=self.sizes['storage'].expansion_max),
            max_storage_level=self.states.loc[horizon.dti_ph_extd, 'soc_max'],
            min_storage_level=self.states.loc[horizon.dti_ph_extd, 'soc_min']
        )

        horizon.constraints.add_invest_costs(invest=(self.components['storage'],),
                                             capex_spec=self.evaluators['storage'].capex.spec,
                                             invest_type='storage')

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes['storage'].expansion = horizon.results[(self.components['storage'], None)]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['outflow'],
                                                                 self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.bus_connected,
                                                                self.components['inflow'])]['sequences']['flow'][horizon.dti_ch]

        self.flows.loc[horizon.dti_ch, 'bat_out'] = horizon.results[(self.components['storage'],
                                                                     self.components['bus'])]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'bat_in'] = horizon.results[(self.components['bus'],
                                                                    self.components['storage'])]['sequences']['flow'][horizon.dti_ch]

        self.states.loc[horizon.dti_ch_extd, 'energy'] = horizon.results[(self.components['storage'], None)]['sequences']['storage_content'][horizon.dti_ch_extd]
        # divide by 0 (size=0) -> pandas returns NaN -> SOC init = NaN in next horizon -> pyomo fails -> fillna(0)
        self.states.loc[horizon.dti_ch_extd, 'soc'] = (
                self.states.loc[horizon.dti_ch_extd, 'energy'] /
                self.sizes['storage'].total).fillna(0)

        self.aging_model.age(horizon=horizon)

    def calc_results_flows(self):
        self.flows['total'] = self.flows.get(key='out', default=0) - self.flows.get(key='in', default=0)  # same as Block
        self.flows['bat_total'] = self.flows.get(key='bat_out', default=0) - self.flows.get(key='bat_in', default=0)

        self.flows['circular'] = self.flows[['in', 'out']].min(axis=1)
        self.flows['bat_circular'] = self.flows[['bat_in', 'bat_out']].min(axis=1)

    def create_plot_traces(self):
        """
        post-scenario plotting of SOC and SOH traces in timeseries plot
        """

        super().create_plot_traces()

        data_soc = self.states.loc[utils.extend_dti(dti=self.scenario.dti_eval,
                                                    freq=self.scenario.timestep_td), 'soc'].dropna()
        data_soh = self.states.loc[utils.extend_dti(dti=self.scenario.dti_eval,
                                                    freq=self.scenario.timestep_td), 'soh'].dropna()
        self.plot_traces['states'].extend([go.Scatter(x=data_soc.index,
                                                      y=data_soc,
                                                      mode='lines',
                                                      name=f'{self.name} SOC',
                                                      line=dict(width=2, dash=None),
                                                      visible='legendonly',
                                                      ),
                                           go.Scatter(x=data_soh.index,
                                                      y=data_soh,
                                                      mode='lines',
                                                      name=f'{self.name} SOH',
                                                      line=dict(width=2, dash=None),
                                                      visible='legendonly',
                                                      ),
                                           ])


class StationaryBattery(StorageBlock):

    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 ):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=None,
                         params=None,
                         parent=scenario,
                         )

    def initialize_efficiencies(self):
        self.eff['chg'] = self.eff_acdc if self.system == 'ac' else 1
        self.eff['dis'] = self.eff_dcac if self.system == 'ac' else 1

        # necessary for common efficiency definition with ElectricFleetUnit
        self.eff['chg_int'] = self.eff['chg']
        self.eff['dis_int'] = self.eff['dis']

        for attr in ['eff_acdc', 'eff_dcac']:
            delattr(self, attr)

        super().initialize_efficiencies()

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        self.bus_connected = self.scenario.block_registry.get('TopLevelBlock', {})['core'].components[self.system]
        params = {'inflow_nominal_capacity': None,
                  'outflow_nominal_capacity': None,
                  'inflow_max': None,
                  'outflow_max': None,
                  'inflow_fix': None,
                  'outflow_fix': None,
                  'invest_relation_input_capacity': self.crate_chg,
                  'invest_relation_output_capacity': self.crate_dis,
                  'storage_balanced': True if self.scenario.strategy == 'go' else False,
                  }
        super().define_oemof_components(horizon, params)

    def create_result_messages(self, *_):
        super().create_result_messages(unit='kWh')

    def get_legend_entry(self):
        return (f'{self.name} power (max. {self.sizes["storage"].total * self.crate_chg * self.eff["chg"] / 1e3:.1f} kW charge / '
                f'{self.sizes["storage"].total * self.crate_dis * self.eff["dis"] / 1e3:.1f} kW discharge)')


class Fleet(SinkBlock):

    def init_evaluators(self):
        super().init_evaluators()
        self.evaluators['f2s'] = eco.EcoEvaluator(name='f2s',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  create_size=True,
                                                  flow_name='out',
                                                  opex_config=dict(spec=self.opex_spec_f2s),
                                                  )

        self.evaluators['s2f'] = eco.EcoEvaluator(name='s2f',
                                                  scenario=self.scenario,
                                                  block=self,
                                                  create_size=True,
                                                  flow_name='in',
                                                  opex_config=dict(spec=self.opex_spec_s2f),
                                                  )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'f2s': {'class_name': 'EconomicEvaluator',
                                  'params': {('opex', 'spec'): 'opex_spec_f2s',
                                             ('flow', 'name'): 'out',
                                             ('size', 'name'): 'f2s',}},
                          's2f': {'class_name': 'EconomicEvaluator',
                                  'params': {('opex', 'spec'): 'opex_spec_s2f',
                                             ('flow', 'name'): 'in',
                                             ('size', 'name'): 's2f',}}
                          },
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario: 'Scenario'):

        super().__init__(name=name,
                         scenario=scenario,
                         flow_apriori_names=None,
                         params=None,
                         parent=scenario)

        if not self.subfleets:
            raise ValueError(f'Block "{self.name}": No subfleets defined! At least one subfleet has to be defined.')

        self.subblocks = {name: SubFleet(name=name,
                                         scenario=self.scenario,
                                         parent=self) for name in self.subfleets}
        del self.subfleets

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):
        """
        pre horizon method

        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        bus_connected        name_bus
          |<----name_outflow--x-|---(ElectricFleetUnit Instance)
          |                     |
          |-x----name_inflow--->|---(ElectricFleetUnit Instance)
          |                     |
          |                     |   (CombustionVehicle Instance)
        """

        self.components['bus'] = solph.Bus()
        self.bus_connected = self.scenario.block_registry.get('TopLevelBlock', {})['core'].components[self.system]

        self.components['inflow'] = solph.components.Converter(
            inputs={self.bus_connected: solph.Flow(
                variable_costs=self.evaluators['s2f'].opt.spec_ep_operation[horizon.dti_ph],
                nominal_capacity=self.sizes['s2f'].preexisting,
                # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
            )},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: 1}
        )

        self.components['outflow'] = solph.components.Converter(
            inputs={self.components['bus']: solph.Flow(
                variable_costs=self.evaluators['f2s'].opt.spec_ep_operation[horizon.dti_ph],
                nominal_capacity=self.sizes['f2s'].preexisting,
                # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
            )},
            outputs={self.bus_connected: solph.Flow(
                variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.bus_connected: 1}
        )

    def get_horizon_results(self,
                            horizon: 'PredictionHorizon'):
        """
        post horizon method
        """
        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['outflow'],
                                                                 self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.bus_connected,
                                                                self.components['inflow'])]['sequences']['flow'][horizon.dti_ch]

    def get_legend_entry(self):
        str_f2s = f'max. {self.sizes["f2s"].total / 1e3:.1f} kW' \
            if pd.notna(self.sizes["f2s"].total / 1e3) \
            else 'unlimited power'
        str_s2f = f'max. {self.sizes["s2f"].total / 1e3:.1f} kW' \
            if pd.notna(self.sizes["f2s"].total / 1e3) \
            else 'unlimited power'

        return f'{self.name} power ({str_f2s} from / {str_s2f} to fleet)'


class SubFleet(NonElectricBlock):

    @staticmethod
    def get_init_definitions():
        return dict(pois={},
                    state_names=[])


    def __init__(self,
                 name: str,
                 scenario,
                 parent):

        # subfleet parameters contain FleetUnit parameters
        params = scenario.parameters.loc[name]
        params_subfleet = {key: params.pop(key) if key in params else None for key in ['num',
                                                                                       'type_unit',
                                                                                       'data_source',
                                                                                       'filename',
                                                                                       'filename_mapper',
                                                                                       'rex']}

        super().__init__(name=name,
                         scenario=scenario,
                         params=params_subfleet,
                         parent=parent)

        self.demand = self.log = None

        if self.type_unit not in ['ev', 'icev', 'mb']:
            raise ValueError(f'Fleet "{self.parent.name}": Subfleet "{self.name}" - invalid type_unit "{self.type_unit}"')

        cls_fu = {'ev': ElectricVehicle,
                  'icev': CombustionVehicle,
                  'mb': MobileBattery}.get(self.type_unit)

        self.unit_names = [f'{self.name}{i}' for i in range(self.num)]
        self.subblocks = {name: cls_fu(name=name,
                                       scenario=self.scenario,
                                       params=params,
                                       parent=self) for name in self.unit_names}

        # Create demand object
        if self.data_source in ['usecases', 'demand']:
            cls_demand = {'ev': mobility.VehicleDemand,
                          'icev': mobility.VehicleDemand,
                          'mb': mobility.BatteryDemand}.get(self.type_unit)
            self.demand = cls_demand(scenario=self.scenario,
                                     subfleet=self)

        if self.data_source == 'usecases':
            self.demand.read_usecase_file()
            self.demand.sample()
            self.scenario.block_registry.setdefault('SubFleetDispatch', {})[self.name] = self
        elif self.data_source == 'demand':
            self.demand.read_demand_file()
            self.scenario.block_registry.setdefault('SubFleetDispatch', {})[self.name] = self
        elif self.data_source in ['log', 'logfile']:
            self.log = self.read_input_log()
        else:
            raise ValueError(f'Block "{self.name}": invalid data source')

        if params.get('mode_scheduling') in scenario.apriori_lvls:  # mode scheduling attr is in FleetUnit
            self.scenario.block_registry.setdefault('SubFleetScheduling', {})[self.name] = self

        if getattr(self, 'invest', False) and self.data_source in ['usecases', 'demand']:
            raise ValueError(f'Subfleet "{self.name}": investment not implemented for data source "{self.data_source}"')

    def read_input_log(self) -> pd.DataFrame:
        """
        Read in a predetermined log file for the SubFleet behavior.
        """

        df = utils.read_timeseries_csv(path_input_file=(self.scenario.paths.input /
                                                        utils.set_extension(filename=self.filename,
                                                                            default_extension='.csv')),
                                       block=self,
                                       scenario=self.scenario,
                                       multiheader=True,
                                       resampling=False)  # Normal resampling cannot be used as consumption must be
                                                          # meaned, while booleans, distances and dsocs must not.

        # Timedelta of frequency of log file
        freq_log = pd.infer_freq(df.index).lower()
        # pd.Timedelta('h') fails --> add '1' --> pd.Timedelta('1h')
        freq_log = pd.Timedelta((freq_log if freq_log[0].isdigit() else '1' + freq_log))

        # Compare Timedelta objects instead of strings to avoid problems (1h vs. 60min)
        if freq_log != self.scenario.timestep_td:
            self.scenario.logger.warning(f'Block "{self.name}": '
                                         f'log file does not match specified timestep - Resampling')

            cols = df.columns  # save orignal column sorting to apply after resampling
            cols_consumption = df.columns[df.columns.get_level_values(1) == 'consumption']
            cols_dist = df.columns[df.columns.get_level_values(1) == 'dist']
            cols_bool = df.columns.difference(cols_consumption).difference(cols_dist)
            # mean ensures equal energy consumption after downsampling, ffill and bfill fill upsampled NaN values
            df_new = pd.DataFrame()
            df_new[cols_consumption] = df[cols_consumption].resample(self.scenario.timestep).mean().ffill().bfill()
            df_new[cols_dist] = df[cols_dist].resample(self.scenario.timestep).sum().ffill().bfill()
            df_new[cols_bool] = df[cols_bool].resample(self.scenario.timestep).ffill().bfill()
            df = df_new[cols]  # ensure right sorting

        if not (self.scenario.dti_eval.isin(df.index).all()):
            raise IndexError(f'Block "{self.name}": Input timeseries data does not cover simulation timeframe')

        # extract the relevant time series
        df = df.loc[self.scenario.dti_sim_extd]  # need dsoc for last timestep

        # rename fleet units according to schema subfleet.name{idx}
        unit_names_log = sorted(df.columns.get_level_values(0).unique()[:self.num].tolist())
        unit_names_map = {log_name: f'{self.name}{idx}' for idx, log_name in enumerate(unit_names_log)}
        df.columns = df.columns.map(lambda x: (unit_names_map.get(x[0], x[0]), *x[1:]))

        return df


class FleetUnit:

    def init_evaluators(self):
        self.evaluators['glider'] = eco.FleetUnitEvaluator(name='glider',
                                                           scenario=self.scenario,
                                                           block=self,
                                                           ls=self.ls,
                                                           ccr=self.ccr,
                                                           capex_config=dict(
                                                               consider_preexisting=self.capex_preexisting_glider,
                                                               fix=self.capex_fix_glider),
                                                           mntex_config=dict(
                                                               fix=self.mntex_fix_glider),
                                                           opex_config=dict(
                                                               spec=0.0,
                                                               dist=self.opex_spec_dist),
                                                           crev_config=dict(
                                                               spec=0.0,
                                                               dist=self.crev_spec_dist,
                                                               time=self.crev_spec_time),
                                                           )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'glider': {'class_name': 'FleetUnitEvaluator',
                                     'params': {('capex', 'preexisting'): 'capex_preexisting_glider',
                                                ('capex', 'fix'): 'capex_fix_glider',
                                                ('mntex', 'fix'): 'mntex_fix_glider',
                                                ('opex', 'dist'): 'opex_spec_dist',
                                                ('crev', 'time'): 'crev_spec_time',
                                                ('crev', 'dist'): 'crev_spec_dist',
                                                ('aux', 'ls'): 'ls',
                                                ('aux', 'ccr'): 'ccr'}},
                          },
                    state_names=[])

    def __init__(self):
        self.log = None

    def pre_scenario(self):
        """
        slice log file from subfleet
        """
        self.log = self.parent.log.loc[:, (self.name, slice(None))].droplevel(0, axis=1)


class ElectricFleetUnit(StorageBlock, FleetUnit):
    """
    abstract class
    """

    def init_evaluators(self):
        StorageBlock.init_evaluators(self)
        FleetUnit.init_evaluators(self)

        self.evaluators['charger'] = eco.EcoEvaluator(name='charger',
                                                      scenario=self.scenario,
                                                      block=self,
                                                      ls=self.ls,
                                                      ccr=self.ccr,
                                                      capex_config=dict(consider_preexisting=self.capex_preexisting_charger,
                                                                        fix=self.capex_fix_charger),
                                                      )

        self.evaluators['ext_ac'] = eco.EcoEvaluator(name='ext_ac',
                                                     scenario=self.scenario,
                                                     block=self,
                                                     flow_name='ext_ac',
                                                     opex_config=dict(spec=self.opex_spec_ext_ac),
                                                     )

        self.evaluators['ext_dc'] = eco.EcoEvaluator(name='ext_dc',
                                                     scenario=self.scenario,
                                                     block=self,
                                                     flow_name='ext_dc',
                                                     opex_config=dict(spec=self.opex_spec_ext_dc),
                                                     )

    @staticmethod
    def get_init_definitions():
        return dict(pois={'charger': {'class_name': 'EconomicEvaluator',
                                      'params': {('capex', 'preexisting'): 'capex_preexisting_charger',
                                                 ('capex', 'fix'): 'capex_fix_charger',
                                                 ('aux', 'ls'): 'ls',
                                                 ('aux', 'ccr'): 'ccr'}},
                          'ext_ac': {'class_name': 'EconomicEvaluator',
                                     'params': {('opex', 'spec'): 'opex_spec_ext_ac',
                                                ('flow', 'name'): 'ext_ac'}},
                          'ext_dc': {'class_name': 'EconomicEvaluator',
                                     'params': {('opex', 'spec'): 'opex_spec_ext_dc',
                                                ('flow', 'name'): 'ext_dc'}},
                          },
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 parent: SubFleet,
                 params: dict):

        StorageBlock.__init__(self=self,
                              name=name,
                              scenario=scenario,
                              flow_apriori_names=['p_int_chg', 'p_ext_ac_chg', 'p_ext_dc_chg',
                                                  'p_int_dis', 'p_ext_ac_dis', 'p_ext_dc_dis'],
                              params=params,
                              parent=parent)

        FleetUnit.__init__(self=self)

        self.apriori = True if self.mode_scheduling in self.scenario.apriori_lvls else False

        if any([size.invest for size in self.sizes.values()]) and self.mode_scheduling in self.scenario.apriori_lvls:
            raise ValueError(f'ElectricFleetUnit "{self.name}": size optimization not '
                             f'implemented for a priori integration levels: {self.scenario.apriori_lvls}')

    def initialize_efficiencies(self):
        self.eff['chg_int'] = {'ac': self.eff_chg_ac, 'dc': self.eff_chg_dc}[self.parent.parent.system]
        self.eff['dis_int'] = {'ac': self.eff_dis_ac, 'dc': self.eff_dis_dc}[self.parent.parent.system]
        super().initialize_efficiencies()

    def pre_scenario(self):
        StorageBlock.pre_scenario(self=self)
        FleetUnit.pre_scenario(self=self)

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon',
                                params: dict = None):

        """
        pre horizon method

        parent.parent_bus     name_bus
            |<--x--name_fleet---|<-x->name_storage (handled in StorageBlock)
            |                   |
            |---x--fleet_name-->|-->name_snk (handled in StorageBlock)
            |                   |
            |                   |<--name_ext_ac-x- (external charging AC)
            |                   |
            |                   |<--name_ext_dc-x- (external charging DC)
            |
        """

        # region calc minimum soc targets before usage and max soc for myopic optimization
        dsoc_ph = self.log.loc[horizon.dti_ph_extd, 'dsoc']
        if (self.scenario.strategy == 'rh') and (self.mode_scheduling == 'oc') and isinstance(self, ElectricVehicle):
            soc_min_hor = dsoc_ph.mask(cond=dsoc_ph > 0, other=dsoc_ph + self.dsoc_buffer).clip(
                lower=self.states.loc[horizon.dti_ph_extd, 'soc_min'],
                upper=self.states.loc[horizon.dti_ph_extd, 'soc_max'])
        elif (self.scenario.strategy == 'rh') and (self.mode_scheduling == 'oc') and isinstance(self, MobileBattery):
            soc_min_hor = dsoc_ph.mask(cond=dsoc_ph > 0, other=self.soc_target).clip(
                lower=self.states.loc[horizon.dti_ph_extd, 'soc_min'],
                upper=self.states.loc[horizon.dti_ph_extd, 'soc_max'])
        else:  # a priori or global optimization
            soc_min_hor = self.states.loc[horizon.dti_ph_extd, 'soc_min']
        self.states.update({'soc_min': soc_min_hor.astype('float64')})
        # endregion

        self.bus_connected = self.parent.parent.components['bus']

        params = {'inflow_nominal_capacity': self.pwr_chg_max,
                  'outflow_nominal_capacity': self.pwr_dis_max * self.eff['dis_int'],
                  'inflow_max': None if self.apriori else self.log.loc[horizon.dti_ph, 'atbase'].astype(int),
                  'outflow_max': None if self.apriori else self.log.loc[horizon.dti_ph, 'atbase'].astype(int),
                  'inflow_fix': self.flows_apriori.loc[horizon.dti_ph, 'p_int_chg'] if self.apriori else None,
                  'outflow_fix': self.flows_apriori.loc[horizon.dti_ph, 'p_int_dis'] if self.apriori else None,
                  'invest_relation_input_capacity': None,
                  'invest_relation_output_capacity': None,
                  'storage_balanced': False,
                  }

        super().define_oemof_components(horizon=horizon,
                                        params=params)

        self.components['snk'] = solph.components.Sink(
            inputs={self.components['bus']: solph.Flow(
                nominal_capacity=1,
                fix=self.log.loc[horizon.dti_ph, 'consumption']
            )})

        self.components['bus_ext_ac'] = solph.Bus()

        self.components['src_ext_ac'] = solph.components.Source(
            outputs={self.components['bus_ext_ac']: solph.Flow(
                nominal_capacity=self.pwr_ext_ac_max,
                max=None if self.apriori else self.log.loc[horizon.dti_ph, 'atac'].astype(int),
                fix=self.flows_apriori.loc[horizon.dti_ph, 'p_ext_ac_chg'] if self.apriori else None,
                variable_costs=self.evaluators['ext_ac'].opt.spec_ep_operation[horizon.dti_ph])}
        )

        self.components['conv_ext_ac'] = solph.components.Converter(
            inputs={self.components['bus_ext_ac']: solph.Flow()},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: self.eff['chg_ac']}
        )

        self.components['bus_ext_dc'] = solph.Bus()

        self.components['src_ext_dc'] = solph.components.Source(
            outputs={self.components['bus_ext_dc']: solph.Flow(
                nominal_capacity=self.pwr_ext_dc_max,
                max=None if self.apriori else self.log.loc[horizon.dti_ph, 'atdc'].astype(int),
                fix=self.flows_apriori.loc[horizon.dti_ph, 'p_ext_dc_chg'] if self.apriori else None,
                variable_costs=self.evaluators['ext_dc'].opt.spec_ep_operation[horizon.dti_ph])}
        )

        self.components['conv_ext_dc'] = solph.components.Converter(
            inputs={self.components['bus_ext_dc']: solph.Flow()},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: 1}  # billed energy is already dc in external dc charging
        )

    def get_horizon_results(self,
                            horizon: 'PredictionHorizon'):
        """
        post horizon method
        """

        self.flows.loc[horizon.dti_ch, 'ext_ac'] = horizon.results[
            (self.components['bus_ext_ac'], self.components['conv_ext_ac'])]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'ext_dc'] = horizon.results[
            (self.components['bus_ext_dc'], self.components['conv_ext_dc'])]['sequences']['flow'][horizon.dti_ch]

        super().get_horizon_results(horizon=horizon)

    def create_plot_traces(self):
        super().create_plot_traces()

        legend_ext_ac = f'{self.name} external AC charging power (max. {self.pwr_ext_ac_max / 1e3:.1f} kW)'
        legend_ext_dc =f'{self.name} external DC charging power (max. {self.pwr_ext_dc_max / 1e3:.1f} kW)'
        self.plot_traces['powers'].extend([go.Scatter(x=self.scenario.dti_eval,
                                                      y=self.flows.loc[self.scenario.dti_eval, 'ext_ac'],
                                                      mode='lines',
                                                      name=legend_ext_ac,
                                                      line=dict(width=2, dash=None, shape='hv'),
                                                      visible='legendonly',
                                                      ),
                                           go.Scatter(x=self.scenario.dti_eval,
                                                      y=self.flows.loc[self.scenario.dti_eval, 'ext_dc'],
                                                      mode='lines',
                                                      name=legend_ext_dc,
                                                      line=dict(width=2, dash=None, shape='hv'),
                                                      visible='legendonly',
                                                      ),
                                           ])

    def get_legend_entry(self):
        return (f'{self.name} power (max. {self.pwr_chg_max / 1e3:.1f} kW charge / '
                f'{(self.pwr_dis_max * self.eff["dis_int"]) / 1e3:.1f} kW discharge)')


class CombustionVehicle(NonElectricBlock, FleetUnit):

    def init_evaluators(self):
        NonElectricBlock.init_evaluators(self=self)
        FleetUnit.init_evaluators(self=self)

    @staticmethod
    def get_init_definitions():
        return dict(pois={},
                    state_names=[])

    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 parent: SubFleet,
                 params: dict):

        NonElectricBlock.__init__(self=self,
                                  name=name,
                                  scenario=scenario,
                                  params=params,
                                  parent=parent)

        FleetUnit.__init__(self=self)

        # delete parameters not needed for CombustionVehicles
        # ToDo: specify required parameters instead of obsolete ones
        for param in ['aging', 'chemistry', 'temp_battery', 'q_loss_cal_init', 'q_loss_cyc_init',
                      'soc_init', 'soc_target', 'soc_return', 'dsoc_buffer',
                      'pwr_chg_max', 'pwr_dis_max', 'pwr_ext_ac_max', 'pwr_ext_dc_max',
                      'eff_storage_roundtrip', 'eff_chg_ac', 'eff_chg_dc', 'eff_dis_ac', 'eff_dis_dc', 'sdr']:
                if hasattr(self, param):
                    delattr(self, param)

    def pre_scenario(self):
        NonElectricBlock.pre_scenario(self=self)
        FleetUnit.pre_scenario(self=self)


class ElectricVehicle(ElectricFleetUnit):
    """
    dummy class to enable tracking
    """

    pass


class MobileBattery(ElectricFleetUnit):
    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 parent: SubFleet,
                 params: dict):
        self.opex_spec_dist = 0.0  # no distance for mobile battery
        self.opex_spec_time = 0.0  # no distance for mobile battery
        super().__init__(name=name,
                         scenario=scenario,
                         parent=parent,
                         params=params)

