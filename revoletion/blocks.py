#!/usr/bin/env python3

import ast
import io
import numpy as np
import oemof.solph as solph
import os
import pandas as pd
import plotly.graph_objects as go
import pvlib
import requests
import statistics
import windpowerlib

from revoletion import battery as bat
from revoletion import economics as eco
from revoletion import mobility
from revoletion import simulation as sim
from revoletion import utils


class Block:

    def __init__(self,
                 name: str,
                 scenario,  # todo type hint
                 flow_names: list = None,
                 state_names: list = None,
                 size_names: list = None,
                 poe_names: list = None,
                 params: dict = None,
                 parent=None,  # todo type hint
                 ):
        """
        Initialize (Sub)Block object with its attributes and data structures
        """

        self.name = name
        self.scenario = scenario
        self.parent = parent

        # region set attributes from scenario file or parent
        if self.parent is None:  # is top level block
            scenario.blocks[self.name] = self

            self.parameters = self.scenario.parameters.loc[self.name]
            for key, value in self.parameters.items():
                setattr(self, key, value)  # this sets all the parameters defined in the scenario file

        elif params is not None:  # is subblock with inherited params
            self.parent.subblocks[self.name] = self

            for key, value in params.items():
                setattr(self, key, value)

        else:  # is subblock without params defined
            raise ValueError(f'Subblock {self.name} of {self.parent.name} has no parameters defined')
        # endregion

        # region initialize data structures
        self.subblocks = dict()

        self.flows_apriori = pd.DataFrame()  # partially recalculated for every horizon
        self.flows = pd.DataFrame(index=self.scenario.dti_sim,
                                  columns=flow_names if flow_names is not None else [],
                                  data=np.nan,
                                  dtype='float64')
        self.states = pd.DataFrame(index=self.scenario.dti_sim,
                                   columns=state_names if state_names is not None else [],
                                   data=np.nan,
                                   dtype='float64')
        self.energies = pd.DataFrame(index=['total', 'in', 'out', 'del', 'pro'],
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,  # cumulative property
                                     dtype=float)

        self.sizes = pd.DataFrame()
        self.expansion_equal = False
        self.initialize_sizes(components=size_names)

        self.poes = {eco.PointOfEvaluation(name=name, block=self) for name in poe_names} \
            if poe_names is not None else dict()

        # todo move ls & ccr to poe

        self.economic_results = eco.EconomicResults(self)
        # endregion

    def initialize_sizes(self,
                         components: list = None):
        """
        Initialize the sizes DataFrame for the block
        """

        components = components if components else ['block']
        self.sizes = pd.DataFrame(index=components,
                                  columns=['total', 'preexisting', 'expansion', 'total_max', 'expansion_max'],
                                  data=np.nan,
                                  dtype='float64')
        self.sizes['invest'] = False

        for component in components:
            component_str = '' if component == 'block' else f'_{component}'
            self.sizes.loc[component, 'preexisting'] = getattr(self, f'size_preexisting{component_str}')
            self.sizes.loc[component, 'total_max'] = getattr(self, f'size_max{component_str}')
            self.sizes.loc[component, 'invest'] = getattr(self, f'invest{component_str}')
            delattr(self, f'size_preexisting{component_str}')
            delattr(self, f'size_max{component_str}')
            delattr(self, f'invest{component_str}')

        # expansion_max logic: 0=no investment, NaN=unlimited investment, float=limited investment
        self.sizes['expansion_max'] = self.sizes['total_max'] - self.sizes['preexisting']
        self.sizes.loc[~self.sizes['invest'], 'expansion_max'] = 0

        if self.sizes['invest'].any() and self.scenario.strategy != 'go':
            raise ValueError(f'Block "{self.name}" component size optimization '
                             f'not implemented for any other strategy than "GO"')


class SystemCore(Block):

    def __init__(self,
                 name : str,
                 scenario):
        super().__init__(name=name,
                         scenario=scenario,
                         size_names=['acdc', 'dcac'],
                         poe_names=['acdc', 'dcac'])



    def initialize_sizes(self,
                         components: list = None):

        self.expansion_equal = True if self.invest_acdc =='equal' or self.invest_dcac == 'equal' else False

        utils.init_equalizable_variables(block=self, name_vars=['invest_acdc', 'invest_dcac'])
        utils.init_equalizable_variables(block=self, name_vars=['size_preexisting_acdc', 'size_preexisting_dcac'])
        utils.init_equalizable_variables(block=self, name_vars=['size_max_acdc', 'size_max_dcac'])

        super().initialize_sizes(components=components)










