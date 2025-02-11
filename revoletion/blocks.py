#!/usr/bin/env python3

import os
import io
import numpy as np
import oemof.solph as solph
import pandas as pd
import plotly.graph_objects as go
import pvlib
import requests
import statistics
import windpowerlib

from revoletion import battery as bat
from revoletion import economics as eco
from revoletion import mobility
from revoletion import utils


class Block:
    """
    abstract class
    """

    def __init__(self,
                 name: str,
                 scenario: 'Scenario',
                 pois: dict = None,
                 state_names: list = None,
                 params: dict = None,
                 parent: 'Block | Scenario' = None,
                 ):
        """
        Initialize (Sub)Block object with attributes and data structures
        """

        self.name = name
        self.scenario = scenario
        self.parent = parent

        # region set attributes from scenario file or parent
        if self.parent is self.scenario:  # is top level block
            self.scenario.blocks[self.name] = self

            self.parameters = self.scenario.parameters.loc[self.name]
            for key, value in self.parameters.items():
                setattr(self, key, value)  # this sets all the parameters defined in the scenario file

        elif params is not None:  # is subblock with inherited params
            self.parent.subblocks[self.name] = self

            for key, value in params.items():
                setattr(self, key, value)

        else:  # is subblock without params defined
            raise ValueError(f'Subblock {self.name} of {self.parent.name} has no inherited parameters defined')
        # endregion

        # region initialize data structures
        self.subblocks = dict()
        self.components = dict()
        self.bus_connected = None

        self.flows_apriori = pd.DataFrame()  # partially recalculated for every horizon
        flow_names = ['total',
                      *[name for name in
                        [poi.get(('flow', 'name')) for poi in pois.values()]
                        if name is not None]]
        self.flows = pd.DataFrame(index=self.scenario.dti_sim,
                                  columns=flow_names,
                                  data=np.nan,
                                  dtype='float64')
        self.energies = pd.DataFrame(index=flow_names,
                                     columns=['sim', 'yrl', 'prj', 'dis'],
                                     data=0,  # cumulative property
                                     dtype=float)
        self.states = pd.DataFrame(index=utils.extend_dti(self.scenario.dti_sim),
                                   columns=state_names if state_names is not None else [],
                                   data=np.nan,
                                   dtype='float64')

        self.sizes = pd.DataFrame()
        self.expansion_equal = False
        self.initialize_sizes(pois=pois)

        self.aggregator = eco.EconomicAggregator(name=self.name,
                                                 block=self)
        self.evaluators = {name: eco.EconomicEvaluator(name=name,
                                                       block=self,
                                                       params=params) for name, params in pois.items()}

        self.aggregator.pre_scenario()  # aggregate capex preexisting

        # Delete ccr and ls as they are now contained in evaluators
        for attribute in set(value for poi in pois.values() for value in poi.values()):
            if hasattr(self, attribute):
                delattr(self, attribute)
        # endregion

    def initialize_sizes(self,
                         pois: dict = None):
        """
        Initialize the sizes DataFrame for the block
        """

        sizes = [name for name in [poi.get(('size', 'name')) for poi in pois.values()] if name is not None]
        self.sizes = pd.DataFrame(index=sizes,
                                  columns=['total', 'preexisting', 'expansion', 'total_max', 'expansion_max'],
                                  data=np.nan,
                                  dtype='float64')
        self.sizes['invest'] = False

        for size in sizes:
            size_str = '' if size == 'block' else f'_{size}'
            self.sizes.loc[size, 'preexisting'] = getattr(self, f'size_preexisting{size_str}', 0)
            self.sizes.loc[size, 'total_max'] = getattr(self, f'size_max{size_str}', 0)
            self.sizes.loc[size, 'invest'] = getattr(self, f'invest{size_str}', False)
            # delete attributes if available
            for attr in ['size_preexisting', 'size_max', 'invest']:
                attr_str = f'{attr}{size_str}'
                if hasattr(self, attr_str):
                    delattr(self, attr_str)

        # expansion_max logic: 0=no investment, NaN=unlimited investment, float=limited investment
        self.sizes['expansion_max'] = self.sizes['total_max'] - self.sizes['preexisting']
        self.sizes.loc[~self.sizes['invest'], 'expansion_max'] = 0

        if self.sizes['invest'].any() and self.scenario.strategy != 'go':
            raise ValueError(f'Block "{self.name}" component size optimization '
                             f'not implemented for any other strategy than "GO"')

    def pre_horizon(self,
                    horizon: 'PredictionHorizon'):
        """
        build energy system components before each horizon
        """
        self.define_oemof_components(horizon=horizon)
        horizon.es.add(*self.components.values())

        for subblock in self.subblocks.values():
            subblock.pre_horizon(horizon=horizon)

    def post_horizon(self,
                     horizon):

        for subblock in self.subblocks.values():
            subblock.post_horizon(horizon=horizon)

        self.get_horizon_results(horizon=horizon)
        self.sizes['total'] = self.sizes['preexisting'] + self.sizes['expansion']

    def post_scenario(self):

        for subblock in self.subblocks.values():
            subblock.post_scenario()

        self.flows['total'] = self.flows.get(key='out', default=0) - self.flows.get(key='in', default=0)
        self.check_bidi_flows()

        # region calculate energy values from flows
        for flow_name, flow in self.flows.items():
            self.energies.loc[flow_name, 'sim'] = flow.sum() * self.scenario.timestep_hours

        self.energies['yrl'] = utils.scale_sim2year(value=self.energies['sim'], scenario=self.scenario)
        self.energies['prj'] = utils.scale_year2prj(value=self.energies['yrl'], scenario=self.scenario)
        self.energies['dis'] = utils.scale_year2dis(value=self.energies['yrl'], scenario=self.scenario)
        # endregion

        for evaluator in self.evaluators.values():
            evaluator.post_scenario()
        self.aggregator.post_scenario()

        # write values to scenario results summary
        # ToDo: write attributes of type int, float, bool and str
        # ToDo: write entries of energies dataframe to results summary
        pass

    def check_bidi_flows(self):
        """
        post scenario method
        """
        if {'in', 'out'}.issubset(set(self.flows.columns)):
            if any(~(self.flows['in'] == 0) & ~(self.flows['out'] == 0)):
                self.scenario.logger.warning(f'Block {self.name} - simultaneous in- and outflow detected!')


class NonElectricBlock:
    """
    abstract class
    """

    def define_oemof_components(self, *_):
        """
        Dummy to be callable for all blocks
        """
        pass

    def get_horizon_results(self, *_):
        """
        Dummy to be callable for all blocks
        """
        pass

    def calc_energies(self, *_):
        """
        Dummy to be callable for all blocks
        """
        pass


class SystemCore(Block):

    def __init__(self,
                 name : str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         pois={
                             'acdc': {('capex', 'preexisting'): 'capex_preexisting_acdc',
                                      ('capex', 'spec'): 'capex_spec',
                                      ('mntex', 'spec'): 'mntex_spec',
                                      ('opex', 'spec'): 'opex_spec',
                                      ('size', 'name'): 'acdc',
                                      ('flow', 'name'): 'acdc',
                                      ('aux', 'ls'): 'ls',
                                      ('aux', 'ccr'): 'ccr'},
                             'dcac': {('capex', 'preexisting'): 'capex_preexisting_dcac',
                                      ('capex', 'spec'): 'capex_spec',
                                      ('mntex', 'spec'): 'mntex_spec',
                                      ('opex', 'spec'): 'opex_spec',
                                      ('size', 'name'): 'dcac',
                                      ('flow', 'name'): 'dcac',
                                      ('aux', 'ls'): 'ls',
                                      ('aux', 'ccr'): 'ccr'},
                         },
                         state_names=None,
                         params=None,
                         parent=scenario)

    def initialize_sizes(self,
                         pois: dict = None):

        self.expansion_equal = True if self.invest_acdc =='equal' or self.invest_dcac == 'equal' else False

        utils.init_equalizable_variables(block=self, name_vars=['invest_acdc', 'invest_dcac'])
        utils.init_equalizable_variables(block=self, name_vars=['size_preexisting_acdc', 'size_preexisting_dcac'])
        utils.init_equalizable_variables(block=self, name_vars=['size_max_acdc', 'size_max_dcac'])

        super().initialize_sizes(pois=pois)

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method
        x denotes the flow measurement point in results

          dc          ac
          |-x--dcac-->|
          |           |
          |<---acdc-x-|
        """

        self.components['ac'] = solph.Bus(label='ac')
        self.components['dc'] = solph.Bus(label='dc')

        self.components['acdc'] = solph.components.Converter(
            label='acdc',
            inputs={self.components['ac']: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.evaluators['acdc'].capex['spec_ep'],
                                               existing=self.sizes.loc['acdc', 'preexisting'],
                                               maximum=utils.conv_add_max(self.sizes.loc['acdc', 'expansion_max'])),
                variable_costs=self.evaluators['acdc'].opex['spec_ep'][horizon.dti_ph])},
            outputs={self.components['dc']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['dc']: self.eff_acdc})

        self.components['dcac'] = solph.components.Converter(
            label='dcac',
            inputs={self.components['dc']: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.evaluators['dcac'].capex['spec_ep'],
                                               existing=self.sizes.loc['dcac', 'preexisting'],
                                               maximum=utils.conv_add_max(self.sizes.loc['dcac', 'expansion_max'])),
                variable_costs=self.evaluators['dcac'].opex['spec_ep'][horizon.dti_ph])},
            outputs={self.components['ac']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['ac']: self.eff_dcac})

        horizon.constraints.add_invest_costs(
            invest=(self.components['ac'], self.components['acdc']),
            capex_spec=self.evaluators['acdc'].capex['spec'],
            invest_type='flow')

        horizon.constraints.add_invest_costs(
            invest=(self.components['dc'], self.components['dcac']),
            capex_spec=self.evaluators['dcac'].capex['spec'],
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
        self.sizes.loc['acdc', 'expansion'] = horizon.results[(self.components['ac'],
                                                               self.components['acdc'])]['scalars']['invest']
        self.sizes.loc['dcac', 'expansion'] = horizon.results[(self.components['dc'],
                                                               self.components['dcac'])]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'acdc'] = horizon.results[(self.components['ac'],
                                                                  self.components['acdc'])]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'dcac'] = horizon.results[(self.components['dc'],
                                                                  self.components['dcac'])]['sequences']['flow'][horizon.dti_ch]

    def check_bidi_flows(self):
        """
        post scenario method
        """
        if any(~(self.flows['acdc'] == 0) & ~(self.flows['dcac'] == 0)):
            self.scenario.logger.warning(f'Block {self.name} - simultaneous AC/DC and DC/AC conversion detected!')


class RenewableSource(Block):
    """
    abstract class
    """

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         pois={
                             'block': {('capex', 'preexisting'): 'capex_preexisting',
                                       ('capex', 'spec'): 'capex_spec',
                                       ('mntex', 'spec'): 'mntex_spec',
                                       ('opex', 'spec'): 'opex_spec',
                                       ('size', 'name'): 'block',
                                       ('flow', 'name'): 'out',
                                       ('aux', 'ls'): 'ls',
                                       ('aux', 'ccr'): 'ccr'},
                             'curt': {('flow', 'name'): 'curt'},
                             'pot': {('flow', 'name'): 'pot'}
                         },
                         state_names=None,
                         params=None,
                         parent=scenario)

        self.data = None  # todo move to a priori flows (except for wind speed and ambient temp)
        self.get_ts_data()

        self.scenario.renewable_sources[self.name] = self

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected      name_bus
          |                   |
          |<--x----name_out---|<--name_src
          |                   |
          |                   |-->name_exc
        """

        self.bus_connected = self.scenario.blocks['core'].components[self.system]

        self.components['bus'] = solph.Bus(label=f'{self.name}_bus')

        self.components['outflow'] = solph.components.Converter(
            label=f'{self.name}_out',
            inputs={self.components['bus']: solph.Flow()},
            outputs={self.bus_connected: solph.Flow()},
            conversion_factors={self.bus_connected: self.eff}
        )

        # Curtailment has to be disincentivized in the optimization to force optimizer to charge storage or commodities
        # instead of curtailment. 2x cost_eps is required as SystemCore also has ccost_eps in charging direction.
        # All other components such as converters and storages only have cost_eps in the output direction.
        self.components['exc'] = solph.components.Sink(
            label=f'{self.name}_exc',
            inputs={self.components['bus']: solph.Flow(variable_costs=2 * self.scenario.cost_eps)}
        )

        self.components['src'] = solph.components.Source(
            label=f'{self.name}_src',
            outputs={self.components['bus']: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.evaluators['block'].capex['spec_ep'],
                                               existing=self.sizes.loc['block', 'preexisting'],
                                               maximum=utils.conv_add_max(self.sizes.loc['block', 'expansion_max'])),
                fix=self.data.loc[horizon.dti_ph, 'power_spec'],
                variable_costs=self.evaluators['block'].opex['spec_ep'][horizon.dti_ph])}
        )

        horizon.constraints.add_invest_costs(invest=(self.components['src'], self.components['bus']),
                                             capex_spec=self.evaluators['block'].capex['spec'],
                                             invest_type='flow')

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes.loc['block', 'expansion'] = horizon.results[(self.components['src'],
                                                                self.components['bus'])]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['outflow'],
                                                                 self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'pot'] = horizon.results[(self.components['src'],
                                                                 self.components['bus'])]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'curt'] = horizon.results[(self.components['bus'],
                                                                  self.components['exc'])]['sequences']['flow'][horizon.dti_ch]


class PVSource(RenewableSource):

    def get_ts_data(self):
        """
        pre scenario (init) method
        Get potential power profile from API or file, each either from Solcast or PVGIS
        """

        if 'api' in self.data_source.lower():  # PVGIS API or Solcast API example selected

            # region get API parameters
            if self.filename:
                try:
                    api_params = pd.read_csv(
                        os.path.join(
                            self.scenario.run.path_input_data,
                            utils.set_extension(self.filename)
                        ),
                        index_col=[0],
                        na_filter=False)
                    api_params = api_params.map(utils.infer_dtype)['value'].to_dict() \
                        if api_params.index.name == 'parameter' and all(api_params.columns == 'value') else {}
                except FileNotFoundError:
                    api_params = {}
            else:
                api_params = {}
            # endregion

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
                    self.logger.warning(f'PVGIS API request exceeds available endtime - data shifted by {abs(api_shift)}'
                                        f' year{s if abs(api_shift) == 1 else ""} to end in {API_MAX_YEAR}')
                elif api_startyear < API_MIN_YEAR:  # PVGIS-SARAH3 only has data from 2005
                    api_shift = (pd.to_datetime(f'{API_MIN_YEAR}-01-01 00:00:00+00:00') -
                                 pd.to_datetime(f'{api_startyear}-01-01 00:00:00+00:00'))
                    api_startyear = API_MIN_YEAR
                    api_endyear = API_MIN_YEAR + api_length
                    self.logger.warning(f'PVGIS API request exceeds available starttime - data shifted by {abs(api_shift)}'
                                        f' year{s if abs(api_shift) == 1 else ""} to start in {API_MIN_YEAR}')
                # Todo leap years can result in data shifting not landing at the same point in time

                # revert lower() in reading data as pvgis is case-sensitive
                # ToDo: move to checker.py
                api_params['raddatabase'] = api_params.get('raddatabase', 'PVGIS-SARAH3').upper()
                api_params['pvtechchoice'] = {'crystsi': 'crystSi',
                                              'cis': 'CIS',
                                              'cdte': 'CdTe',
                                              'unknown': 'Unknown'}[api_params.get('pvtechchoice', 'crystsi')]

                self.data, *_ = pvlib.iotools.get_pvgis_hourly(
                    self.scenario.latitude,
                    self.scenario.longitude,
                    start=api_startyear,
                    end=api_endyear,
                    url='https://re.jrc.ec.europa.eu/api/v5_3/',
                    components=False,
                    outputformat='json',
                    pvcalculation=True,
                    peakpower=1,
                    map_variables=True,
                    loss=0,
                    raddatabase=api_params['raddatabase'],  # conversion above ensures that the parameter exists
                    pvtechchoice=api_params['pvtechchoice'],  # conversion above ensures that the parameter exists
                    mountingplace=api_params.get('mountingplace', 'free'),
                    optimalangles=api_params.get('optimalangles', True),
                    optimal_surface_tilt=api_params.get('optimal_surface_tilt', False),
                    surface_azimuth=api_params.get('surface_azimuth', 180),
                    surface_tilt=api_params.get('surface_tilt', 0),
                    trackingtype=api_params.get('trackingtype', 0),
                    usehorizon=api_params.get('usehorizon', True),
                    userhorizon=api_params.get('userhorizon', None),
                )
                self.data.index = self.data.index.round('h')  # PVGIS does not give time slots not as full hours
                self.data.index = self.data.index - api_shift
            # endregion

            # region get data from Solcast API
            elif self.data_source == 'solcast api':  # solcast API example selected
                # set api key as bearer token
                headers = {'Authorization': f'Bearer {self.scenario.run.key_api_solcast}'}

                params = {
                    **{'latitude': self.scenario.latitude,  # unmetered location for testing 41.89021,
                       'longitude': self.scenario.longitude,  # unmetered location for testing 12.492231,
                       'period': 'PT5M',
                       'output_parameters': ['air_temp', 'gti', 'wind_speed_10m'],
                       'start': self.scenario.starttime,
                       'end': self.scenario.sim_extd_endtime,
                       'format': 'csv',
                       'time_zone': 'utc',},
                    **{parameter: value for parameter, value in api_params.items() if value is not None}}

                url = 'https://api.solcast.com.au/data/historic/radiation_and_weather'

                # get data from Solcast API
                response = requests.get(url, headers=headers, params=params)
                # convert to csv
                self.data = pd.read_csv(io.StringIO(response.text))
                # calculate period_start as only period_end is given, set as index and remove unnecessary columns
                self.data['period_start'] = pd.to_datetime(self.data['period_end']) - pd.to_timedelta(self.data['period'])
                self.data.set_index(pd.DatetimeIndex(self.data['period_start']), inplace=True)
                self.data = self.data.tz_convert(self.scenario.timezone)
                self.data.drop(columns=['period', 'period_start', 'period_end'], inplace=True)
                # rename columns according to further processing steps
                self.data.rename(columns={'air_temp': 'temp_air', 'wind_speed_10m': 'wind_speed'}, inplace=True)
                # calculate specific pv power
                self.calc_power_solcast()
            # endregion

        else:

            # region get data from file
            path_input_file = os.path.join(
                self.scenario.run.path_input_data,
                utils.set_extension(self.filename)
            )

            # region get data from PVGIS file
            if self.data_source == 'pvgis file':
                self.data, meta, *_ = pvlib.iotools.read_pvgis_hourly(self.path_input_file, map_variables=True)
                self.scenario.latitude = meta['latitude']
                self.scenario.longitude = meta['longitude']
                self.data.index = self.data.index.round('h')  # PVGIS does not necessarily give full hour time vals
            # endregion

            # region get data from Solcast file
            elif self.data_source == 'solcast file':
                # no lat/lon contained in solcast files
                self.data = pd.read_csv(self.path_input_file)
                self.data.rename(columns={'PeriodStart': 'period_start',
                                          'PeriodEnd': 'period_end',
                                          'AirTemp': 'temp_air',
                                          'GtiFixedTilt': 'gti',
                                          'WindSpeed10m': 'wind_speed'}, inplace=True)
                self.data['period_start'] = pd.to_datetime(self.data['period_start'], utc=True)
                self.data['period_end'] = pd.to_datetime(self.data['period_end'], utc=True)
                self.data.set_index(pd.DatetimeIndex(self.data['period_start']), inplace=True)
                self.data = self.data[['temp_air', 'wind_speed', 'gti']]
                self.calc_power_solcast()
            # endregion

            else:
                raise ValueError(f'Scenario {self.scenario.name} - Block {self.name}: No usable PV data input specified')

        # region resample, localize, and transform data

        # resample to timestep, fill NaN values with previous ones (or next ones, if not available)
        self.data = self.data.resample(self.scenario.timestep).mean().ffill().bfill()
        # convert to local time
        self.data.index = self.data.index.tz_convert(tz=self.scenario.timezone)
        # data is in W for a 1kWp PV array -> convert to specific power
        self.data['power_spec'] = self.data['P'] / 1e3

        self.data = self.data[['power_spec', 'wind_speed', 'temp_air']]  # only keep relevant columns
        # endregion

    def calc_power_from_irradiation(self):
        """
        pre scenario (init) method
        calculate PV potential output power from insolation and weather data
        function is necessary for solcast input that does not contain power data
        """

        u0 = 26.9  # W/(˚C.m2) - cSi Free standing
        u1 = 6.2  # W.s/(˚C.m3) - cSi Free standing
        mod_temp = self.data['temp_air'] + (self.data['gti'] / (u0 + (u1 * self.data['wind_speed'])))

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


class WindSource(RenewableSource):

    def get_ts_data(self):
        """
        pre scenario (init) method
        get potential power profile from PVSource block or file
        """
        if self.data_source in self.scenario.blocks.keys():
            # region get data from PVSource block
            self.data = self.scenario.blocks[self.data_source].data.copy()
            self.data['wind_speed_adj'] = windpowerlib.wind_speed.hellman(self.data['wind_speed'], 10, self.height)

            path_turbine_data_file = os.path.join(self.scenario.run.path_data_immut, 'turbine_data.pkl')
            turbine_data = pd.read_pickle(path_turbine_data_file)
            # smallest fully filled wind turbine in dataseta as per June 2024
            turbine_data = turbine_data.loc[turbine_data['turbine_type'] == 'E-53/800'].reset_index()

            self.data['power_original'] = windpowerlib.power_output.power_curve(
                wind_speed=self.data['wind_speed_adj'],
                power_curve_wind_speeds=ast.literal_eval(turbine_data.loc[0, 'power_curve_wind_speeds']),
                power_curve_values=ast.literal_eval(turbine_data.loc[0, 'power_curve_values']),
                density_correction=False)
            self.data['power_spec'] = self.data['power_original'] / self.turbine_data.loc[0, 'nominal_power']
            # endregion
        elif self.data_source == 'file':
            # region get data from file
            path_input_file = os.path.join(self.scenario.run.path_input_data,
                                           utils.set_extension(self.filename))
            self.data = utils.read_input_csv(path_input_file=path_input_file,
                                             block=self,
                                             scenario=self.scenario)
            # endregion
        else:
            raise ValueError(f'Scenario {self.scenario.name} - Block {self.name}: No usable data input specified')


class FixedDemand(Block):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         pois={
                             'block': {('crev', 'spec'): 'crev_spec',
                                       ('flow', 'name'): 'in'}
                         },
                         state_names=None,
                         params=None,
                         parent=scenario)

        self.get_flows_apriori()

    def get_flows_apriori(self):
        self.flows_apriori.index = self.scenario.dti_sim_extd
        if self.load_profile in ['h0', 'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'l0', 'l1', 'l2']:
            self.get_demand_from_slp()
        elif self.load_profile in ['const', 'constant']:
            self.flows_apriori['demand'] = self.consumption_yrl / (365 * 24)
        elif isinstance(self.load_profile, str):  # load_profile is a file name
            self.get_demand_from_file()
        else:
            raise ValueError(f'Parameter "load_profile" in block "{self.block.name}" is not valid')

    def get_demand_from_file(self):
        data = utils.read_input_csv(path_input_file=os.path.join(self.scenario.run.path_input_data,
                                                                 utils.set_extension(self.load_profile)),
                                    block=self,
                                    scenario=self.scenario,
                                    )

        if data.shape[1] != 1:
            self.scenario.logger.warning(f'Input file "{utils.set_extension(self.load_profile)}" for parameter '
                                         f'"load_profile" in block "{self.block.name}" has more than one column. '
                                         f'Sum of all columns is calculated for load profile.')

        data = data.sum(axis=1)[self.flows_apriori.index]  # convert to series and slice to sim timeframe
        self.flows_apriori['demand'] = data

    def get_demand_from_slp(self):
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
        slp = pd.read_csv(os.path.join(self.scenario.run.path_data_immut, 'slp_bdew.csv'),
                          skiprows=[0],
                          header=[0, 1, 2],
                          index_col=0)

        slp.index = pd.to_datetime(slp.index, format='%H:%M').time

        # use a fixed frequency of 15 minutes for the timeseries generation as the SLPs are given with that frequency
        freq_slp = '15min'
        dti_slp = pd.DatetimeIndex(pd.date_range(start=self.scenario.starttime.floor(freq_slp),
                                                 end=max(self.scenario.dti_sim_extd).ceil(freq_slp),
                                                 freq=freq_slp))

        data = pd.Series(index=dti_slp, data=0, dtype='float64')

        data = data.index.to_series().apply(
            lambda x: slp.loc[x.time(), (self.load_profile.upper(), get_timeframe(x), get_daytype(x, self.scenario.holiday_dates))])

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

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected
          |
          |-x->name_snk
          |
        """

        self.bus_connected = self.scenario.blocks['core'].components[self.system]

        self.components['snk'] = solph.components.Sink(
            label=f'{self.name}_snk',
            inputs={self.bus_connected: solph.Flow(nominal_value=1,
                                                   fix=self.flows_apriori['demand'][horizon.dti_ph])}
        )

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.bus_connected,
                                                                self.components['snk'])]['sequences']['flow'][horizon.dti_ch]


class ControllableSource(Block):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         state_names=None,
                         pois={
                             'block': {('capex', 'preexisting'): 'capex_preexisting',
                                      ('capex', 'spec'): 'capex_spec',
                                      ('mntex', 'spec'): 'mntex_spec',
                                      ('opex', 'spec'): 'opex_spec',
                                      ('size', 'name'): 'block',
                                      ('flow', 'name'): 'out',
                                      ('aux', 'ls'): 'ls',
                                      ('aux', 'ccr'): 'ccr'},
                         },
                         params=None,
                         parent=scenario)

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected
          |
          |<-name_gen
          |
        """

        self.bus_connected = self.scenario.blocks['core'].components[self.system]

        self.components['src'] = solph.components.Source(
            label=f'{self.name}_src',
            outputs={self.bus_connected: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.evaluators['block'].capex['spec_ep'],
                                               existing=self.sizes.loc['block', 'preexisting'],
                                               maximum=utils.conv_add_max(self.sizes.loc['block', 'expansion_max'])),
                variable_costs=self.evaluators['block'].opex['spec_ep'][horizon.dti_ph])}
        )

        horizon.constraints.add_invest_costs(invest=(self.components['src'], self.bus_connected),
                                             capex_spec=self.evaluators['block'].capex['spec'],
                                             invest_type='flow')

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes.loc['block', 'expansion'] = horizon.results[(self.components['src'],
                                                                self.bus_connected)]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['src'],
                                                                 self.bus_connected)]['sequences']['flow'][horizon.dti_ch]


class GridConnection(Block):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         pois={'g2s': {('capex', 'preexisting'): 'capex_preexisting_g2s',
                                       ('capex', 'spec'): 'capex_spec',
                                       ('mntex', 'spec'): 'mntex_spec',
                                       ('size', 'name'): 'g2s',
                                       ('flow', 'name'): 'out',
                                       ('aux', 'ls'): 'ls',
                                       ('aux', 'ccr'): 'ccr'},
                               's2g': {('capex', 'preexisting'): 'capex_preexisting_s2g',
                                       ('capex', 'spec'): 'capex_spec',
                                       ('mntex', 'spec'): 'mntex_spec',
                                       ('size', 'name'): 's2g',
                                       ('flow', 'name'): 'in',
                                       ('aux', 'ls'): 'ls',
                                       ('aux', 'ccr'): 'ccr'},
                               },
                         state_names=None,
                         params=None,
                         parent=scenario)

        self.inflows = dict()
        self.outflows = dict()

        self.peakshaving_periods = pd.DataFrame()

        self.initialize_peakshaving()
        self.initialize_markets()

    def initialize_sizes(self,
                         pois: dict = None):

        self.expansion_equal = True if self.invest_g2s == 'equal' or self.invest_s2g == 'equal' else False

        utils.init_equalizable_variables(self, ['invest_s2g', 'invest_g2s'])
        utils.init_equalizable_variables(self, ['size_preexisting_g2s', 'size_preexisting_s2g'])
        utils.init_equalizable_variables(self, ['size_max_g2s', 'size_max_s2g'])

        super().initialize_sizes(pois=pois)

    def initialize_peakshaving(self):
        # Create functions to extract relevant property of datetimeindex for peakshaving intervals
        periods_func = {
            'day': lambda x: x.strftime('%Y-%m-%d'),
            'week': lambda x: x.strftime('%Y-CW%W'),
            'month': lambda x: x.strftime('%Y-%m'),
            'quarter': lambda x: f"{x.year}-Q{(x.month - 1) // 3 + 1}",
            'year': lambda x: x.strftime('%Y'),
            'None': lambda x: 'sim_duration'
        }

        # Pre-calculate peakshaving type
        peakshaving = str(self.peakshaving)

        # Get dummies directly from the 'periods' data
        self.bus_activation = pd.get_dummies(
            self.scenario.dti_sim_extd.to_series().map(periods_func[peakshaving])).astype(int)


        # Create a series to store peak power values
        self.peakshaving_periods = pd.DataFrame(index=self.bus_activation.columns,
                                                columns=['power'],
                                                data=0.0,  # cumulative variable
                                                dtype='float64')

        self.opex_spec_peak = self.opex_spec_peak if self.peakshaving is not None else 0

        # calculate the fraction of each period that is covered by the sim time (NOT sim_extd!)
        # for period in self.peakshaving_periods.index:
        #     self.peakshaving_periods.loc[period, 'period_fraction'] = utils.get_period_fraction(
        #         dti=self.bus_activation.loc[self.scenario.dti_sim][
        #             self.bus_activation.loc[self.scenario.dti_sim, period] == 1].index,
        #         period=self.peakshaving,
        #         freq=self.scenario.timestep)
        #
        #     # Get first and last timestep of each peakshaving interval -> used for rh calculation later on
        #     dti_period = self.bus_activation[self.bus_activation[period] == 1].index
        #     self.peakshaving_periods.loc[period, 'start'] = dti_period.min()
        #     self.peakshaving_periods.loc[period, 'end'] = dti_period.max()

        # slice sim dataframe from activation bus (compared to sim_extd)
        bus_activation_sim = self.bus_activation.loc[self.scenario.dti_sim]

        def process_period(period):
            # Calculate period fraction
            period_fraction = utils.get_period_fraction(
                dti=bus_activation_sim[bus_activation_sim.loc[self.scenario.dti_sim, period] == 1].index,
                period=self.peakshaving,
                freq=self.scenario.timestep
            )

            # Get first and last timestep of the peakshaving interval
            dti_period = self.bus_activation[self.bus_activation[period] == 1].index
            start = dti_period.min()
            end = dti_period.max()

            return pd.Series({'period_fraction': period_fraction,
                              'start': start,
                              'end': end})

        # Apply the function to each period in peakshaving_periods
        self.peakshaving_periods[['period_fraction', 'start', 'end']] = self.peakshaving_periods.index.to_series().apply(process_period)

        self.n_peakshaving_periods_yr = (pd.date_range(start=self.scenario.starttime,
                                                       end=self.scenario.starttime + pd.DateOffset(years=1),
                                                       freq=self.scenario.timestep,
                                                       inclusive='left')
                                         .to_series().apply(periods_func[peakshaving])).unique().size

        self.evaluators.update({period: eco.PeakEvaluator(
            name=period,
            block=self,
            params={('opex', 'spec'): 'opex_spec_peak',
                    ('flow', 'name'): f'',})
            for period in self.peakshaving_periods.index})

    def initialize_markets(self):
        # get information about GridMarkets specified in the scenario file
        markets = pd.read_csv(os.path.join(self.scenario.run.path_input_data,
                                           utils.set_extension(self.filename_markets)),
                              index_col=[0]).map(utils.infer_dtype)

        # Generate individual GridMarkets instances
        self.subblocks = {market_name: GridMarket(name=market_name,
                                                  scenario=self.scenario,
                                                  params=dict(markets.loc[:, market_name]),
                                                  parent=self)
                          for market_name in markets.columns}

    def define_oemof_components(self,
                                horizon):
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

        self.bus_connected = self.scenario.blocks['core'].components[self.system]

        self.components['bus'] = solph.Bus(label=f'{self.name}_bus')

        self.inflows = {f'{self.name}_inflow_1': solph.components.Converter(
            label=f'xc_{self.name}',
            # Peakshaving not implemented for feed-in into grid
            inputs={self.bus_connected: solph.Flow()},
            # Size optimization
            outputs={self.components['bus']: solph.Flow(
                nominal_value=solph.Investment(ep_costs=self.evaluators['s2g'].capex['spec_ep'],
                                               existing=self.sizes.loc['s2g', 'preexisting'],
                                               maximum=utils.conv_add_max(self.sizes.loc['s2g', 'expansion_max'])),
                variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components['bus']: 1})}

        self.components.update(self.inflows)

        self.outflows = {f'{self.name}_outflow_{period}': solph.components.Converter(
            label=f'{self.name}_xc_{period}',
            # Size optimization: investment costs are assigned to first peakshaving interval only. The application of
            # constraints ensures that the optimized grid connection sizes of all peakshaving intervals are equal
            inputs={self.components['bus']: solph.Flow(
                nominal_value=solph.Investment(ep_costs=(self.evaluators['g2s'].capex['spec_ep'] if period == self.peakshaving_periods.index[0] else 0),
                                               existing=self.sizes.loc['g2s', 'preexisting'],
                                               maximum=utils.conv_add_max(self.sizes.loc['g2s', 'expansion_max']))
            )},
            # Peakshaving
            outputs={self.bus_connected: solph.Flow(
                nominal_value=(solph.Investment(ep_costs=self.evaluators[period].opex['spec_ep'],
                                                existing=self.peakshaving_periods.loc[period, 'power'],)
                               if self.peakshaving else None),
                max=(self.bus_activation.loc[horizon.dti_ph, period] if self.peakshaving else None))},
            conversion_factors={self.bus_connected: 1}) for period in self.peakshaving_periods.index}

        self.components.update(self.outflows)

        horizon.constraints.add_invest_costs(invest=(self.components[f'{self.name}_inflow_1'],
                                                     self.components['bus']),
                                             capex_spec=self.evaluators['s2g'].capex['spec'],
                                             invest_type='flow')
        horizon.constraints.add_invest_costs(invest=(self.components['bus'],
                                                     self.components[f'{self.name}_outflow_{self.peakshaving_periods.index[0]}']),
                                             capex_spec=self.evaluators['g2s'].capex['spec'],
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
        self.sizes.loc['g2s', 'expansion'] = horizon.results[(self.components['bus'],
                                                              list(self.outflows.values())[0])]['scalars']['invest']
        self.sizes.loc['s2g', 'expansion'] = horizon.results[(list(self.inflows.values())[0],
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

        self.peakshaving_periods['power'] = self.peakshaving_periods.apply(get_peak_power, axis=1)


class GridMarket(Block):

    def __init__(self,
                 name: str,
                 scenario,
                 params,
                 parent):

        super().__init__(name=name,
                         scenario=scenario,
                         state_names=None,
                         pois={'g2s': {('opex', 'spec'): 'opex_spec_g2s',
                                       ('size', 'name'): 'g2s',
                                       ('flow', 'name'): 'out'},
                               's2g': {('opex', 'spec'): 'opex_spec_s2g',
                                       ('size', 'name'): 'g2s',
                                       ('flow', 'name'): 'in'},
                               },
                         params=params,
                         parent=parent)

    def define_oemof_components(self,
                                horizon):
        """
        pre horizon method

        parent_bus
            |<---x----name_src
            |
            |----x--->name_snk
            |
        """

        self.components['src'] = solph.components.Source(
            label=f'{self.name}_src',
            outputs={self.parent.components['bus']: solph.Flow(
                nominal_value=(self.pwr_g2s if not pd.isna(self.pwr_g2s) else None),
                variable_costs=self.evaluators['g2s'].opex['spec_ep'][horizon.dti_ph])
            }
        )

        self.components['snk'] = solph.components.Sink(
            label=f'{self.name}_snk',
            inputs={
                self.parent.components['bus']: solph.Flow(
                    nominal_value=(self.pwr_s2g if not pd.isna(self.pwr_s2g) else None),
                    variable_costs=(self.evaluators['s2g'].opex['spec_ep'][horizon.dti_ph]),
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


class StorageBlock:
    """
    abstract class
    """

    def __init__(self):

        self.scenario.storage_blocks[self.name] = self

        self.loss_rate = 0
        self.calc_loss_rate_per_hour()
        delattr(self, 'sdr')

        self.states.loc[self.scenario.starttime, 'soc'] = self.soc_init
        delattr(self, 'soc_init')

        self.aging_model = bat.BatteryPackModel(self)
        self.soc_min = (1 - self.states.loc[self.scenario.starttime, 'soh']) / 2  # todo move to states df
        self.soc_max = 1 - ((1 - self.states.loc[self.scenario.starttime, 'soh']) / 2)

    def calc_loss_rate_per_hour(self,
                                target: pd.Timedelta = pd.Timedelta(hours=1)) -> float:
        """
        convert self-discharge rate (sdr) per month of a battery storage to a loss rate (lr) per target time step.
        oemof specifies one hour as the target time step for the loss rate.
        """
        ratio_timestep = target / pd.Timedelta('30 days')  # assumption: 30 days per month
        self.loss_rate = 1 - (1 - self.sdr) ** ratio_timestep

    def get_horizon_results(self,
                            horizon):
        """
        post horizon method
        """
        self.sizes.loc['block', 'expansion'] = horizon.results[(self.components['storage'], None)]['scalars']['invest']

        self.flows.loc[horizon.dti_ch, 'out'] = horizon.results[(self.components['outflow'],
                                                                 self.bus_connected)]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'in'] = horizon.results[(self.bus_connected,
                                                                self.components['inflow'])]['sequences']['flow'][horizon.dti_ch]

        self.flows.loc[horizon.dti_ch, 'bat_out'] = horizon.results[(self.components['storage'],
                                                                     self.components['bus'])]['sequences']['flow'][horizon.dti_ch]
        self.flows.loc[horizon.dti_ch, 'bat_in'] = horizon.results[(self.components['bus'],
                                                                    self.components['storage'])]['sequences']['flow'][horizon.dti_ch]

        # preemptive size calculation to enable soc calculation
        self.sizes['total'] = self.sizes['preexisting'] + self.sizes['expansion']
        self.states.loc[utils.extend_dti(horizon.dti_ch), 'energy'] = horizon.results[(self.components['storage'], None)]['sequences']['storage_content'][utils.extend_dti(horizon.dti_ch)]
        self.states['soc'] = self.states['energy'] / self.sizes.loc['block', 'total']

        #self.aging_model.age(horizon=horizon)  # todo reactivate

    def add_state_traces(self):
        """
        post-scenario plotting of SOC and SOH traces in timeseries plot
        """
        legentry = f'{self.name} SOC ({self.size.loc["block", "total"]/1e3:.1f} kWh)'
        self.scenario.figure.add_trace(go.Scatter(x=self.states.index,
                                                  y=self.states['soc'],
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None)),
                                       secondary_y=True)

        legentry = f'{self.name} SOH'
        data = self.states['soh'].dropna()
        self.scenario.figure.add_trace(go.Scatter(x=data.index,
                                                  y=data,
                                                  mode='lines',
                                                  name=legentry,
                                                  line=dict(width=2, dash=None),
                                                  visible='legendonly'),
                                       secondary_y=True)


class StationaryBattery(Block, StorageBlock):

    def __init__(self,
                 name: str,
                 scenario):

        super().__init__(name=name,
                         scenario=scenario,
                         pois={
                             'block': {('capex', 'preexisting'): 'capex_preexisting',
                                       ('capex', 'spec'): 'capex_spec',
                                       ('mntex', 'spec'): 'mntex_spec',
                                       ('opex', 'spec'): 'opex_spec',
                                       ('size', 'name'): 'block',
                                       ('flow', 'name'): 'in',
                                       ('aux', 'ls'): 'ls',
                                       ('aux', 'ccr'): 'ccr'},
                             'out': {('flow', 'name'): 'out'},
                             'bat_in': {('flow', 'name'): 'bat_in'},
                             'bat_out': {('flow', 'name'): 'bat_out'},
                         },
                         state_names=['energy', 'soc', 'soh', 'q_loss_cal', 'q_loss_cyc'],
                         params=None,
                         parent=scenario)

        StorageBlock.__init__(self)

        self.eff_chg = self.eff_acdc if self.system == 'ac' else 1
        self.eff_dis = self.eff_dcac if self.system == 'ac' else 1
        for attr in ['eff_acdc', 'eff_dcac']:
            delattr(self, attr)

    def define_oemof_components(self,
                                horizon):
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

        self.bus_connected = self.scenario.blocks['core'].components[self.system]

        self.components['bus'] = solph.Bus(
            label=f'{self.name}_bus'
        )

        self.components['inflow'] = solph.components.Converter(
            label=f'xc_{self.name}',
            inputs={self.bus_connected: solph.Flow(
                variable_costs=self.evaluators['block'].opex['spec_ep'][horizon.dti_ph]
            )},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: self.eff_chg}
        )

        self.components['outflow'] = solph.components.Converter(
            label=f'{self.name}_xc',
            inputs={self.components['bus']: solph.Flow()},
            # cost_eps are needed to prevent storage from being emptied in RH
            outputs={self.bus_connected: solph.Flow(
                variable_costs=self.scenario.cost_eps
            )},
            conversion_factors={self.bus_connected: self.eff_dis}
        )

        self.components['storage'] = solph.components.GenericStorage(
            label=f'{self.name}_storage',
            inputs={self.components['bus']: solph.Flow()},
            outputs={
                self.components['bus']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            loss_rate=self.loss_rate,
            balanced={'go': True, 'rh': False}[self.scenario.strategy],
            initial_storage_level=statistics.median(
                [self.soc_min,
                 self.states.loc[horizon.starttime, 'soc'],
                 self.soc_max]
            ),
            invest_relation_input_capacity=self.crate_chg,  # crate measured "outside" of conversion factor (efficiency)
            invest_relation_output_capacity=self.crate_dis,
            inflow_conversion_factor=np.sqrt(self.eff_roundtrip),
            outflow_conversion_factor=np.sqrt(self.eff_roundtrip),
            nominal_storage_capacity=solph.Investment(
                ep_costs=self.evaluators['block'].opex['spec_ep'],
                existing=self.sizes.loc['block', 'preexisting'],
                maximum=utils.conv_add_max(self.sizes.loc['block', 'expansion_max'])),
            max_storage_level=pd.Series(
                data=self.soc_max,
                index=utils.extend_dti(horizon.dti_ph)
            ),
            min_storage_level=pd.Series(
                data=self.soc_min,
                index=utils.extend_dti(horizon.dti_ph)
            )
        )

        horizon.constraints.add_invest_costs(
            invest=(self.components['storage'],),
            capex_spec=self.evaluators['block'].capex['spec'],
            invest_type='storage')


class Fleet(Block):
    """
    abstract class
    """

    def __init__(self,
                 name: str,
                 scenario: 'Scenario'):

        super().__init__(name=name,
                         scenario=scenario,
                         pois={'sys_chg': {('opex', 'spec'): 'opex_spec_sys_chg',
                                           ('flow', 'name'): 'in'},
                               'sys_dis': {('opex', 'spec'): 'opex_spec_sys_dis',
                                           ('flow', 'name'): 'out'}},
                         state_names=None,
                         params=None,
                         parent=scenario)

        self.scenario.fleets[self.name] = self

        # region get dispatch mode and get data accordingly
        if self.mode_scheduling in self.scenario.run.apriori_lvls:  # uc, equal, soc, fcfs
            if self.mode_scheduling == 'uc':
                self.mode_dispatch = 'apriori_unlimited'
            elif isinstance(self.power_lim_static, (int, float)):
                self.mode_dispatch = 'apriori_static'
            elif self.power_lim_static is None:
                self.mode_dispatch = 'apriori_dynamic'

        elif self.scenario.strategy == 'rh':
            self.mode_dispatch = 'opt_myopic'
        elif self.scenario.strategy == 'go':
            self.mode_dispatch = 'opt_global'

        # static load management is deactivated for 'uc' mode
        if self.power_lim_static and self.mode_scheduling == 'uc':
            self.scenario.logger.warning(f'CommoditySystem "{self.name}": static load management is not implemented'
                                         f' for scheduling mode "uc". deactivating static load management')
            self.power_lim_static = None

        if self.invest and self.mode_scheduling in self.scenario.run.apriori_lvls:
            raise ValueError(f'CommoditySystem "{self.name}": commodity size optimization not '
                             f'implemented for a priori integration levels: {self.scenario.run.apriori_lvls}')

        self.unit_names = [f'{self.name}_{i}' for i in range(self.num)]
        if self.data_source == 'usecases':
            self.usecases = utils.read_usecase_file(self)
            self.demand = self.demand.sample()
        elif self.data_source == 'demand':
            self.demand = utils.read_demand_file(self)
        elif self.data_source == 'log':
            self.log = utils.read_input_log(self)
            self.unit_names = self.log.columns.get_level_values(0).unique()[:self.num].tolist()
        else:
            raise ValueError(f'Block "{self.name}": invalid data source')
        # endregion

        # region define params for subblocks
        params_units_list = ['invest',
                             'aging',
                             'dsoc_buffer',
                             'mode_dispatch',
                             'soc_init',
                             'chemistry',
                             'q_loss_cal_init',
                             'q_loss_cyc_init',
                             'pwr_chg',
                             'pwr_dis',
                             'eff_chg_ac',
                             'eff_chg_dc',
                             'eff_dis_ac',
                             'eff_dis_dc',
                             'eff_storage_roundtrip',
                             'temp_battery',
                             'size_preexisting',
                             'capex_preexisting',
                             'capex_fix_glider',
                             'capex_fix_charger',
                             'capex_spec',
                             'mntex_fix_glider',
                             'opex_spec_dist',
                             'opex_spec_ext_ac',
                             'opex_spec_ext_dc',
                             'crev_spec_time',
                             'crev_spec_dist',
                             'factor_pwr_des',
                             'ls',
                             'ccr',
                             'sdr']
        self.params_units = {key: getattr(self, key) for key in params_units_list}
        for param in params_units_list:
            delattr(self, param)
        # endregion

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon'):
        """
        pre horizon method

        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        bus_connected        name_bus
          |<-x---name_outflow---|---(FleetUnit Instance)
          |                     |
          |-x----name_inflow--->|---(FleetUnit Instance)
        """

        self.components['bus'] = solph.Bus(label=f'{self.name}_bus')
        self.bus_connected = self.scenario.blocks['core'].components[self.system]

        self.components['inflow'] = solph.components.Converter(
            label=f'xc_{self.name}',
            inputs={self.bus_connected: solph.Flow(
                variable_costs=self.evaluators['sys_chg'].opex['spec_ep'][horizon.dti_ph],
                nominal_value=self.power_lim_static,
                max=1 if self.power_lim_static else None
            )},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: 1}
        )

        self.components['outflow'] = solph.components.Converter(
            label=f'{self.name}_xc',
            inputs={self.components['bus']: solph.Flow(
                variable_costs=self.evaluators['sys_dis'].opex['spec_ep'][horizon.dti_ph],
                nominal_value=(self.power_lim_static if self.lvl_cap == 'v2s' else 0),
                max=(1 if self.power_lim_static is not None else None))},
            outputs={self.bus_connected: solph.Flow(
                variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.bus_connected: 1}
        )


class ElectricVehicleFleet(Fleet):

    def __init__(self,
                 name: str,
                 scenario: 'Scenario'):

        self.demand = mobility.VehicleFleetDemand(scenario, self)

        super().__init__(name=name,
                         scenario=scenario)

        self.subblocks = {name: ElectricVehicle(name=name,
                                                parent=self,
                                                scenario=scenario) for name in self.unit_names}
        for attr in ['unit_names', 'params_units']:
            delattr(self, attr)


class BatteryFleet(Fleet):

    def __init__(self,
                 name: str,
                 scenario: 'Scenario'):

        self.demand = mobility.BatteryFleetDemand(scenario, self)

        super().__init__(name=name,
                         scenario=scenario)

        self.subblocks = {name: MobileBattery(name=name,
                                              parent=self,
                                              scenario=scenario) for name in self.unit_names}
        for attr in ['unit_names', 'params_inherit']:
            delattr(self, attr)


class ElectricFleetUnit(Block, StorageBlock):
    """
    abstract class
    """

    def __init__(self,
                 name: str,
                 parent: Fleet,
                 scenario: 'Scenario'):

        super().__init__(name=name,
                         scenario=scenario,
                         pois={
                             'glider': {('capex', 'fix'): 'capex_fix_glider',
                                        ('mntex', 'fix'): 'mntex_fix_glider',
                                        ('opex', 'dist'): 'opex_spec_dist',
                                        ('crev', 'time'): 'crev_spec_time',
                                        ('crev', 'dist'): 'crev_spec_dist',
                                        ('flow', 'name'): 'in',  # todo reference ???
                                        ('aux', 'ls'): 'ls',
                                        ('aux', 'ccr'): 'ccr'},
                             'charger': {('capex', 'fix'): 'capex_fix_charger'},
                             'storage': {('capex', 'preeexisting'): 'capex_preexisting',
                                         ('capex', 'spec'): 'capex_spec',
                                         ('size', 'name'): 'block'},
                             'ext_ac': {('opex', 'spec'): 'opex_spec_ext_ac',
                                        ('flow', 'name'): 'ext_ac'},
                             'ext_dc': {('opex', 'spec'): 'opex_spec_ext_dc',
                                        ('flow', 'name'): 'ext_dc'},
                             'out': {('flow', 'name'): 'out'},
                             'in': {('flow', 'name'): 'in'},
                             'bat_in': {('flow', 'name'): 'bat_in'},
                             'bat_out': {('flow', 'name'): 'bat_out'},
                         },
                         state_names=['energy', 'soc', 'soh', 'q_loss_cal', 'q_loss_cyc'],
                         params=parent.params_units,
                         parent=parent)

        StorageBlock.__init__(self)

        self.eff_chg = {'ac': self.eff_chg_ac, 'dc': self.eff_chg_dc}[self.parent.system]
        self.eff_dis = {'ac': self.eff_dis_ac, 'dc': self.eff_dis_dc}[self.parent.system]
        delattr(self, 'eff_chg_ac')
        delattr(self, 'eff_chg_dc')

        # todo move to dispatch
        if self.parent.data_source in ['usecases', 'demand']:  # dispatch simulation will run
            # estimate maximum power drawn by self discharge
            self.pwr_loss_max = 1 - (1 - self.loss_rate) ** self.scenario.timestep_hours * self.sizes.loc['block', 'total']
            # downrate assumed power for a priori dispatch simulation
            self.pwr_chg_des = (self.pwr_chg * self.eff_chg - self.pwr_loss_max) * self.factor_pwr_des

    def define_oemof_components(self,
                                horizon: 'PredictionHorizon'):
        """
        pre horizon method

        parent_bus            name_bus
            |<--x--name_fleet---|<->name_storage
            |                   |
            |---x--fleet_name-->|-->name_snk
            |                   |
            |                   |<--name_ext_ac (external charging AC)
            |                   |
            |                   |<--name_ext_dc (external charging DC)
            |
        """

        self.components['bus'] = solph.Bus(label=f'{self.name}_bus')

        self.components['inflow'] = solph.components.Converter(
            label=f'mc_{self.name}',
            inputs={self.parent.components['bus']: solph.Flow(
                nominal_value=self.pwr_chg,
                max=inflow_max,
                fix=inflow_fix)},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: self.eff_chg})

        self.components['outflow'] = solph.components.Converter(
            label=f'{self.name}_mc',
            inputs={self.components['bus']: solph.Flow()},
            outputs={self.parent.bus: solph.Flow(
                nominal_value=(self.pwr_dis * self.eff_dis if self.parent.lvl_cap in ['v2v', 'v2s'] else 0),
                max=outflow_max,
                fix=outflow_fix,
                variable_costs=self.scenario.cost_eps)
            },
            conversion_factors={self.parent.bus: self.eff_dis})

        self.components['snk'] = solph.components.Sink(
            label=f'{self.name}_snk',
            inputs={self.components['bus']: solph.Flow(
                nominal_value=1,
                fix=self.data_ph['consumption']
            )})

        self.components['storage'] = solph.components.GenericStorage(
            label=f'{self.name}_ess',
            inputs={self.components['bus']: solph.Flow(variable_costs=self.parent.opex_ep_spec[horizon.dti_ph])},
            # cost_eps are needed to prevent storage from being emptied in RH
            outputs={self.components['bus']: solph.Flow(variable_costs=self.scenario.cost_eps)},
            loss_rate=self.parent.loss_rate,
            balanced=False,
            initial_storage_level=statistics.median(
                [soc_min[horizon.starttime],
                 self.storage_timeseries.loc[horizon.starttime, 'soc'],
                 soc_max[horizon.starttime]]),
            inflow_conversion_factor=np.sqrt(self.eff_storage_roundtrip),
            outflow_conversion_factor=np.sqrt(self.eff_storage_roundtrip),
            nominal_storage_capacity=solph.Investment(
                ep_costs=self.parent.capex_ep_spec,
                existing=self.size.loc['block', 'existing'],
                maximum=utils.conv_add_max(self.size.loc['block', 'additional_max'])),
            min_storage_level=soc_min,
            max_storage_level=soc_max
        )

        # always add charger -> reduce different paths of result calculations; no chargers -> power is set to 0 kW
        self.components['bus_ext_ac'] = solph.Bus(label=f'{self.name}_bus_ext_ac')

        self.components['src_ext_ac'] = solph.components.Source(
            label=f'{self.name}_src_ext_ac',
            outputs={self.components['bus_ext_ac']: solph.Flow(
                nominal_value=self.parent.pwr_ext_ac,
                max=ext_ac_max,
                fix=ext_ac_fix,
                variable_costs=self.parent.opex_ep_spec_ext_ac[horizon.dti_ph])}
        )

        self.components['conv_ext_ac'] = solph.components.Converter(
            label=f'{self.name}_conv_ext_ac',
            inputs={self.components['bus_ext_ac']: solph.Flow()},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: self.parent.eff_chg_ac}
        )

        self.components['bus_ext_dc'] = solph.Bus(label=f'{self.name}_bus_ext_dc')

        self.components['src_ext_dc'] = solph.components.Source(
            label=f'{self.name}_src_ext_dc',
            outputs={self.components['bus_ext_dc']: solph.Flow(
                nominal_value=self.parent.pwr_ext_dc,
                max=ext_dc_max,
                fix=ext_dc_fix,
                variable_costs=self.parent.opex_ep_spec_ext_dc[horizon.dti_ph])}
        )

        self.components['conv_ext_dc'] = solph.components.Converter(
            label=f'{self.name}_conv_ext_dc',
            inputs={self.components['bus_ext_dc']: solph.Flow()},
            outputs={self.components['bus']: solph.Flow()},
            conversion_factors={self.components['bus']: 1}
        )

        horizon.constraints.add_invest_costs(invest=(self.ess,),
                                             capex_spec=self.parent.capex_spec,
                                             invest_type='storage')


class Vehicle:
    pass
    # todo glider, charger, battery are pois


class ElectricVehicle(ElectricFleetUnit, Vehicle):
    pass


class CombustionVehicle(Vehicle, NonElectricBlock):
    pass


class MobileBattery(ElectricFleetUnit):
    pass









