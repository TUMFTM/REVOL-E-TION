import json
import os
import pandas as pd

index = pd.MultiIndex.from_tuples([('scenario', 'starttime'),
                                   ('scenario', 'timestep'),
                                   ('scenario', 'sim_duration'),
                                   ('scenario', 'prj_duration'),
                                   ('scenario', 'strategy'),
                                   ('scenario', 'ph_len'),
                                   ('scenario', 'ch_len'),
                                   ('scenario', 'wacc'),
                                   ('scenario', 'blocks'),
                                   ('core', 'dcac_size'),
                                   ('core', 'acdc_size'),
                                   ('core', 'capex_spec'),
                                   ('core', 'mntex_spec'),
                                   ('core', 'opex_spec'),
                                   ('core', 'ls'),
                                   ('core', 'cdc'),
                                   ('core', 'dcac_eff'),
                                   ('core', 'acdc_eff'),
                                   ('dem', 'system'),
                                   ('dem', 'filename'),
                                   ('wind', 'size'),
                                   ('wind', 'system'),
                                   ('wind', 'filename'),
                                   ('wind', 'capex_spec'),
                                   ('wind', 'mntex_spec'),
                                   ('wind', 'opex_spec'),
                                   ('wind', 'ls'),
                                   ('wind', 'cdc'),
                                   ('wind', 'eff'),
                                   ('pv', 'size'),
                                   ('pv', 'system'),
                                   ('pv', 'data_source'),
                                   ('pv', 'latitude'),
                                   ('pv', 'longitude'),
                                   ('pv', 'filename'),
                                   ('pv', 'capex_spec'),
                                   ('pv', 'mntex_spec'),
                                   ('pv', 'opex_spec'),
                                   ('pv', 'ls'),
                                   ('pv', 'cdc'),
                                   ('pv', 'eff'),
                                   ('gen', 'size'),
                                   ('gen', 'system'),
                                   ('gen', 'capex_spec'),
                                   ('gen', 'mntex_spec'),
                                   ('gen', 'opex_spec'),
                                   ('gen', 'ls'),
                                   ('gen', 'cdc'),
                                   ('gen', 'eff'),
                                   ('grid', 'size'),
                                   ('grid', 'system'),
                                   ('grid', 'capex_spec'),
                                   ('grid', 'mntex_spec'),
                                   ('grid', 'opex_spec'),
                                   ('grid', 'ls'),
                                   ('grid', 'cdc'),
                                   ('grid', 'eff'),
                                   ('ess', 'size'),
                                   ('ess', 'system'),
                                   ('ess', 'capex_spec'),
                                   ('ess', 'mntex_spec'),
                                   ('ess', 'opex_spec'),
                                   ('ess', 'ls'),
                                   ('ess', 'cdc'),
                                   ('ess', 'chg_eff'),
                                   ('ess', 'dis_eff'),
                                   ('ess', 'chg_crate'),
                                   ('ess', 'dis_crate'),
                                   ('ess', 'init_soc'),
                                   ('ess', 'sdr'),
                                   ('ess', 'cdc'),
                                   ('ess', 'eff'),
                                   ('bev', 'size'),
                                   ('bev', 'rex_cs'),
                                   ('bev', 'int_lvl'),
                                   ('bev', 'system'),
                                   ('bev', 'filename'),
                                   ('bev', 'num'),
                                   ('bev', 'daily_mean'),
                                   ('bev', 'daily_stdev'),
                                   ('bev', 'dep_mean1'),
                                   ('bev', 'dep_stdev1'),
                                   ('bev', 'dep_mean2'),
                                   ('bev', 'dep_stdev2'),
                                   ('bev', 'dist_mean'),
                                   ('bev', 'dist_stdev'),
                                   ('bev', 'idle_mean'),
                                   ('bev', 'idle_stdev'),
                                   ('bev', 'patience'),
                                   ('bev', 'rex_patience'),
                                   ('bev', 'consumption'),
                                   ('bev', 'speed_avg'),
                                   ('bev', 'soc_min_return'),
                                   ('bev', 'soc_dep'),
                                   ('bev', 'capex_spec'),
                                   ('bev', 'mntex_spec'),
                                   ('bev', 'opex_spec'),
                                   ('bev', 'sys_chg_soe'),
                                   ('bev', 'sys_dis_soe'),
                                   ('bev', 'ls'),
                                   ('bev', 'init_soc'),
                                   ('bev', 'chg_pwr'),
                                   ('bev', 'dis_pwr'),
                                   ('bev', 'eff'),
                                   ('bev', 'chg_eff'),
                                   ('bev', 'dis_eff'),
                                   ('bev', 'cdc'),
                                   ('brs', 'size'),
                                   ('brs', 'int_lvl'),
                                   ('brs', 'system'),
                                   ('brs', 'filename'),
                                   ('brs', 'num'),
                                   ('brs', 'daily_mean'),
                                   ('brs', 'daily_stdev'),
                                   ('brs', 'idle_mean'),
                                   ('brs', 'idle_stdev'),
                                   ('brs', 'patience'),
                                   ('brs', 'soc_return_mean'),
                                   ('brs', 'soc_return_stdev'),
                                   ('brs', 'soc_dep'),
                                   ('brs', 'capex_spec'),
                                   ('brs', 'mntex_spec'),
                                   ('brs', 'opex_spec'),
                                   ('brs', 'sys_chg_soe'),
                                   ('brs', 'sys_dis_soe'),
                                   ('brs', 'ls'),
                                   ('brs', 'init_soc'),
                                   ('brs', 'chg_pwr'),
                                   ('brs', 'dis_pwr'),
                                   ('brs', 'eff'),
                                   ('brs', 'chg_eff'),
                                   ('brs', 'dis_eff'),
                                   ('brs', 'cdc')])

data = {'bev_only': ['1/1/2005', '15T', 365, 25, 'go', 48, 24, 0.09, {'core': 'SystemCore',
                                                                 'dem': 'FixedDemand',
                                                                 'wind': 'WindSource',
                                                                 'pv': 'PVSource',
                                                                 'ess': 'StationaryEnergyStorage',
                                                                 'gen': 'ControllableSource',
                                                                 'grid': 'ControllableSource',
                                                                 'bev': 'VehicleCommoditySystem'},  # scenario
                1.00E+05, 1.00E+05, 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                'AC', 'dem_example',  # dem
                1.00E+05, 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                7.50E+05, 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                5.00E+04, 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                3.00E+04, 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                1.10E+06, 'DC', 0.139, 0.142, 0.00417, 10, 1, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                3.00E+04, None, 'cc', 'AC', 'run_des', 5, 7, 2, 8, 2, 17, 2, 3.2, 0.88, 1.5, 0.5, 2, 1, 300, 25, 0.05, 1, 0, 0, 0, -0.0003, 0.00035, 10, 0.5, 11000, 11000, 1, 0.95, 0.95, 1,  # bev
                3.00E+03, 'cc', 'AC', 'run_des', 5, 6, 1, 1.5, 0.5, 1, 0.12, 0.1, 1, 0, 0, 0, -0.0003, 0.00035, 10, 0.5, 1500, 1500, 1, 0.95, 0.95, 1],
        'brs_only': ['1/1/2005', '15T', 365, 25, 'go', 48, 24, 0.09, {'core': 'SystemCore',
                                                                 'dem': 'FixedDemand',
                                                                 'wind': 'WindSource',
                                                                 'pv': 'PVSource',
                                                                 'ess': 'StationaryEnergyStorage',
                                                                 'gen': 'ControllableSource',
                                                                 'grid': 'ControllableSource',
                                                                 'brs': 'BatteryCommoditySystem'},  # scenario
                1.00E+05, 1.00E+05, 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                'AC', 'dem_example',  # dem
                1.00E+05, 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                7.50E+05, 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                5.00E+04, 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                3.00E+04, 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                1.10E+06, 'DC', 0.139, 0.142, 0.00417, 10, 1, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                3.00E+04, None, 'cc', 'AC', 'run_des', 5, 7, 2, 8, 2, 17, 2, 3.2, 0.88, 1.5, 0.5, 2, 1, 300, 25, 0.05, 1, 0, 0, 0, -0.0003, 0.00035, 10, 0.5, 11000, 11000, 1, 0.95, 0.95, 1,  # bev
                3.00E+03, 'cc', 'AC', 'run_des', 5, 6, 1, 1.5, 0.5, 1, 0.12, 0.1, 1, 0, 0, 0, -0.0003, 0.00035, 10, 0.5, 1500, 1500, 1, 0.95, 0.95, 1],
        'both': ['1/1/2005', '15T', 365, 25, 'go', 48, 24, 0.09, {'core': 'SystemCore',
                                                                  'dem': 'FixedDemand',
                                                                  'wind': 'WindSource',
                                                                  'pv': 'PVSource',
                                                                  'ess': 'StationaryEnergyStorage',
                                                                  'gen': 'ControllableSource',
                                                                  'grid': 'ControllableSource',
                                                                  'bev': 'VehicleCommoditySystem',
                                                                  'brs': 'BatteryCommoditySystem'},  # scenario
                        1.00E+05, 1.00E+05, 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                        'AC', 'dem_example',  # dem
                        1.00E+05, 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                        7.50E+05, 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                        5.00E+04, 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                        3.00E+04, 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                        1.10E+06, 'DC', 0.139, 0.142, 0.00417, 10, 1, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                        3.00E+04, 'brs', 'cc', 'AC', 'run_des', 5, 7, 2, 8, 2, 17, 2, 50, 30, 1.5, 0.5, 2, 1, 300, 25, 0.05, 1, 0, 0, 0, -0.0003, 0.00035, 10, 0.5, 11000, 11000, 1, 0.95, 0.95, 1,  # bev
                        3.00E+03, 'cc', 'AC', 'run_des', 30, 6, 1, 1.5, 0.5, 1, 0.12, 0.1, 1, 0, 0, 0, -0.0003, 0.00035, 10, 0.5, 1500, 1500, 1, 0.95, 0.95, 1]
        }  # brs

df = pd.DataFrame.from_dict(data)
df.index = index
df.reset_index(inplace=True, names=['block', 'key']) # this saves the multiindex into a column to make the index unique for json

file_path = os.path.join(os.getcwd(), 'des_test.json')
scenarios = df.to_json(file_path, orient='records', lines=True)

print(f'{file_path} created')