import pandas as pd
import os


class EconomicInput:
    __slots__ = ['scenario', '_capex', '_mntex', '_opex', '_crev']  # ensure that no additional attributes are added
    def __init__(self, scenario=None, capex=None, mntex=None, opex=None, crev=None):
        self.scenario = scenario
        self._capex = capex
        self._mntex = mntex
        self._opex = opex
        self._crev = crev

        # ToDo: get dti of scenario instead
        dti = pd.DatetimeIndex(pd.date_range(start=pd.to_datetime('2024-01-01 00:00+01:00'),
                                             end=pd.to_datetime('2025-01-01 00:00+01:00'),
                                             freq='15min',
                                             inclusive='left'))

        for var_name in ['_opex', '_crev']:
            self.scalar_to_ts(var_name=var_name, dti=dti)

    def scalar_to_ts(self, var_name, dti):
        if getattr(self, var_name) is None:
            setattr(self, var_name, None)
            return

        ts = pd.DataFrame(columns=pd.MultiIndex.from_tuples(tuples=[], names=['level', 'mode']), index=dti)  # ToDo: rename levels

        for level, def_opex in getattr(self, var_name).iterrows():
            for mode, value in def_opex.items():
                if isinstance(value, str):
                    # ToDo: read csv and extract relevant time slices for scenario
                    pass
                else:
                    ts[(level, mode)] = pd.Series(value, index=dti)
        setattr(self, var_name, ts)  # write opex to class attribute

    def __getattr__(self, key):
        keys = key.split('_')
        if len(keys) not in [2, 3] or keys[0].startswith('_'):
            raise AttributeError(f'{self} has no Attribute {key}')
        elif len(keys) == 2:
            keys.append('block')
        if keys[0] in ['opex', 'crev']:  # timeseries -> MultiIndex
            return getattr(self, f'_{keys[0]}')[(keys[2], keys[1])]
        return getattr(self, f'_{keys[0]}').at[keys[2], keys[1]]

    def __setattr__(self, key, value):
        # avoid use of custom setter for initialization
        if isinstance(value, (pd.DataFrame, pd.Series, type(None))):
            super().__setattr__(key, value)
        else:
            keys = key.split('_')
            if len(keys) == 2:
                keys.append('block')
            elif len(keys) != 3:
                raise ValueError(f'{key} not found in {self}')
            print(keys)
            getattr(self, f'_{keys[0]}').loc[keys[2], keys[1]] = value


class EconomicResults:
    def __init__(self):
        self._capex = pd.DataFrame(columns=['init', 'yrl', 'prj', 'dis'])
        # ToDo: include
        #  capex_replacement     ->  cost for replacement
        #  capex_init_existing   ->  cost for existing size
        #  capex_init_additional ->  cost for additional size
        #  capex_joined_spec     ->  capex + mntex

        self._mntex = pd.DataFrame(columns=['sim', 'yrl', 'prj', 'dis'])
        self._opex = pd.DataFrame(columns=['sim', 'yrl', 'prj', 'dis'])
        self._crev = pd.DataFrame(columns=['sim', 'yrl', 'prj', 'dis'])
        self._cashflows = None
        self._totex = None
        pass


if __name__ == '__main__':
    input = EconomicInput(capex=pd.DataFrame(index=['block'], data={'existing':True, 'spec': 2.0, 'fix': 2000}),
                           mntex=pd.DataFrame(index=['block'], data={'spec': 2.0, 'fix': 2000}),
                           opex=pd.DataFrame(index=['block'], data={'spec': 2.0, 'dist': 2000}),
                           crev=pd.DataFrame(index=['block'], data={'dummy': 2.0}),
                           )

    input_test = EconomicInput(capex=pd.DataFrame(index=['s2g', 'g2s'], data=[{'existing': True, 'spec': 0.05},
                                                                               {'existing': True, 'spec': 0.05}]))

    pass
