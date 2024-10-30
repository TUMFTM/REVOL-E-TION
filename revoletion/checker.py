#!/usr/bin/env python3

import os
import pandas as pd
import re

from io import StringIO


class InputChecker:

    def __init__(self, run):

        self.run = run
        self.path_readme = os.path.join(run.path_pkg, 'README.md')
        self.settings_target = self.read_settings_from_readme()
        self.scenarios_target = self.read_scenarios_from_readme()

        self.dtype_map = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool}

    def _template(self):
        pass

    def read_scenarios_from_readme(self):
        start_str = '<!ENTRY_POINT_SCENARIOS_TABLE>'
        end_str = '<!EXIT_POINT_SCENARIOS_TABLE>'
        df = self.read_table_from_readme(start_str, end_str, multiindex=True)
        df['Block'] = df['Block'].ffill()
        return df

    def read_settings_from_readme(self):
        start_str = '<!ENTRY_POINT_SETTINGS_TABLE>'
        end_str = '<!EXIT_POINT_SETTINGS_TABLE>'
        df = self.read_table_from_readme(start_str, end_str, multiindex=False)
        df['Valid values or format'] = df['Valid values or format'].str.replace("\'", "")
        return df

    def read_table_from_readme(self, start_str, end_str, multiindex=False):

        with open(self.path_readme, 'r') as md_file:
            md_text = md_file.read()

        pattern_table = re.escape(start_str) + r'(.*?)' + re.escape(end_str)

        # Use re.DOTALL to allow `.` to match newline characters as well
        match = re.search(pattern_table, md_text, re.DOTALL)
        if match:
            md_table = match.group(1).strip()
        else:
            raise ValueError('Settings table not found in README.md')

        # remove leading and trailing whitespaces
        pattern_space = r'[ \t]*' + re.escape('|') + r'[ \t]*'
        md_table = re.sub(pattern_space,'|', md_table)

        df = pd.read_csv(StringIO(md_table),
                         sep='|',
                         header=0,
                         skiprows=[1],
                         skipinitialspace=True,
                         engine='python',
                         dtype=str)

        df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)

        return df

    def check_settings(self):

        # Check for completeness
        missing_params = [attr for attr in self.settings_target['Parameter'] if not hasattr(self.run, attr)]
        if len(missing_params) > 0:
            raise AttributeError(f'Not all required settings defined. Missing: {", ".join(missing_params)}')

        # Individual parameter checks
        for param in self.settings_target['Parameter']:
            value = getattr(self.run, param)

            # check for correct dtypes
            dtype_target_str = self.settings_target.loc[self.settings_target['Parameter'] == param, 'Type'].values[0]
            dtypes_target = tuple([self.dtype_map.get(t) for t in dtype_target_str.replace(' ', '').split(',')])
            if None in dtypes_target:
                raise ValueError(f'Invalid dtype list \'{dtype_target_str}\' defined in README for setting {param}.')
            if not isinstance(value, dtypes_target):
                raise ValueError(f'Invalid dtype for setting {param}. Expected: {dtype_target_str}')

            # Todo check for value in range

    def check_scenario(self):
        pass


class OutputChecker:

    def __init__(self):
        pass
