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
                                   ('core', 'sce'),
                                   ('core', 'sme'),
                                   ('core', 'soe'),
                                   ('core', 'ls'),
                                   ('core', 'cdc'),
                                   ('core', 'dcac_eff'),
                                   ('core', 'acdc_eff'),
                                   ('dem', 'system'),
                                   ('dem', 'filename'),
                                   ('wind', 'size'),
                                   ('wind', 'system'),
                                   ('wind', 'filename'),
                                   ('wind', 'sce'),
                                   ('wind', 'sme'),
                                   ('wind', 'soe'),
                                   ('wind', 'ls'),
                                   ('wind', 'cdc'),
                                   ('wind', 'eff'),
                                   ('pv', 'size'),
                                   ('pv', 'system'),
                                   ('pv', 'data_source'),
                                   ('pv', 'latitude'),
                                   ('pv', 'longitude'),
                                   ('pv', 'filename'),
                                   ('pv', 'sce'),
                                   ('pv', 'sme'),
                                   ('pv', 'soe'),
                                   ('pv', 'ls'),
                                   ('pv', 'cdc'),
                                   ('pv', 'eff'),
                                   ('gen', 'size'),
                                   ('gen', 'system'),
                                   ('gen', 'sce'),
                                   ('gen', 'sme'),
                                   ('gen', 'soe'),
                                   ('gen', 'ls'),
                                   ('gen', 'cdc'),
                                   ('gen', 'eff'),
                                   ('grid', 'size'),
                                   ('grid', 'system'),
                                   ('grid', 'sce'),
                                   ('grid', 'sme'),
                                   ('grid', 'soe'),
                                   ('grid', 'ls'),
                                   ('grid', 'cdc'),
                                   ('grid', 'eff'),
                                   ('ess', 'size'),
                                   ('ess', 'system'),
                                   ('ess', 'sce'),
                                   ('ess', 'sme'),
                                   ('ess', 'soe'),
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
                                   ('bev', 'int_lvl'),
                                   ('bev', 'system'),
                                   ('bev', 'filename'),
                                   ('bev', 'num'),
                                   ('bev', 'sce'),
                                   ('bev', 'sme'),
                                   ('bev', 'soe'),
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
                                   ('brs', 'sce'),
                                   ('brs', 'sme'),
                                   ('brs', 'soe'),
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

data = {'0_mg_go': ['1/1/2005', 'H', 365, 25, 'go', 48, 24, 0.09, {'core': 'SystemCore',
                                                                   'dem': 'FixedDemand',
                                                                   'wind': 'WindSource',
                                                                   'pv': 'PVSource',
                                                                   'ess': 'StationaryEnergyStorage',
                                                                   'gen': 'ControllableSource',
                                                                   'grid': 'ControllableSource'},  # scenario
                    1.00E+05, 1.00E+05, 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                    'AC', 'dem_example',  # dem
                    1.00E+05, 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                    7.50E+05, 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                    5.00E+04, 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                    3.00E+04, 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                    1.10E+06, 'DC', 0.139, 0.142, 0.00417, 0, 10, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                    3.00E+04, 'cc', 'AC', 'bev_example', 5, 0.25, 0.0075, 0, 0, 0, 10, 0.5, 11000, 11000, 1, 0.95, 0.95,
                    1,  # bev
                    3.00E+03, 'cc', 'AC', 'brs_example', 10, 0.45, 0.01, 0, 0, 0, 10, 0.5, 3600, 3600, 1, 0.95, 0.95,
                    1],  # brs
        '1_mg_rh': ['1/1/2005', 'H', 365, 25, 'rh', 48, 24, 0.09, {'core': 'SystemCore',
                                                                   'dem': 'FixedDemand',
                                                                   'wind': 'WindSource',
                                                                   'pv': 'PVSource',
                                                                   'ess': 'StationaryEnergyStorage',
                                                                   'gen': 'ControllableSource',
                                                                   'grid': 'ControllableSource'},  # scenario
                    1.00E+05, 1.00E+05, 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                    'AC', 'dem_example',  # dem
                    1.00E+05, 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                    7.50E+05, 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                    5.00E+04, 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                    3.00E+04, 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                    1.10E+06, 'DC', 0.139, 0.142, 0.00417, 0, 10, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                    3.00E+04, 'cc', 'AC', 'bev_example', 5, 0.25, 0.0075, 0, 0, 0, 10, 0.5, 11000, 11000, 1, 0.95, 0.95,
                    1,  # bev
                    3.00E+03, 'cc', 'AC', 'brs_example', 10, 0.45, 0.01, 0, 0, 0, 10, 0.5, 3600, 3600, 1, 0.95, 0.95,
                    1],  # brs
        '2_mg_opt': ['1/1/2005', 'H', 365, 25, 'go', 48, 24, 0.09, {'core': 'SystemCore',
                                                                    'dem': 'FixedDemand',
                                                                    'wind': 'WindSource',
                                                                    'pv': 'PVSource',
                                                                    'ess': 'StationaryEnergyStorage',
                                                                    'gen': 'ControllableSource',
                                                                    'grid': 'ControllableSource'},  # scenario
                     'opt', 'opt', 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                     'AC', 'dem_example',  # dem
                     'opt', 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                     'opt', 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                     'opt', 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                     'opt', 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                     'opt', 'DC', 0.139, 0.142, 0.00417, 0, 10, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                     'opt', 'cc', 'AC', 'bev_example', 5, 0.25, 0.0075, 0, 0, 0, 10, 0.5, 11000, 11000, 1, 0.95, 0.95,
                     1,  # bev
                     'opt', 'cc', 'AC', 'brs_example', 10, 0.45, 0.01, 0, 0, 0, 10, 0.5, 3600, 3600, 1, 0.95, 0.95, 1],
        # brs
        '3_mgev_go': ['1/1/2005', 'H', 365, 25, 'go', 48, 24, 0.09, {'core': 'SystemCore',
                                                                     'dem': 'FixedDemand',
                                                                     'wind': 'WindSource',
                                                                     'pv': 'PVSource',
                                                                     'ess': 'StationaryEnergyStorage',
                                                                     'gen': 'ControllableSource',
                                                                     'grid': 'ControllableSource',
                                                                     'bev': 'CommoditySystem'},  # scenario
                      1.00E+05, 1.00E+05, 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                      'AC', 'dem_example',  # dem
                      1.00E+05, 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                      7.50E+05, 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                      5.00E+04, 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                      3.00E+04, 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                      1.10E+06, 'DC', 0.139, 0.142, 0.00417, 0, 10, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                      3.00E+04, 'cc', 'AC', 'bev_example', 5, 0.25, 0.0075, 0, 0, 0, 10, 0.5, 11000, 11000, 1, 0.95,
                      0.95, 1,  # bev
                      3.00E+03, 'cc', 'AC', 'brs_example', 10, 0.45, 0.01, 0, 0, 0, 10, 0.5, 3600, 3600, 1, 0.95, 0.95,
                      1],  # brs
        '4_mgev_rh': ['1/1/2005', 'H', 365, 25, 'rh', 48, 24, 0.09, {'core': 'SystemCore',
                                                                     'dem': 'FixedDemand',
                                                                     'wind': 'WindSource',
                                                                     'pv': 'PVSource',
                                                                     'ess': 'StationaryEnergyStorage',
                                                                     'gen': 'ControllableSource',
                                                                     'grid': 'ControllableSource',
                                                                     'bev': 'CommoditySystem'},  # scenario
                      1.00E+05, 1.00E+05, 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                      'AC', 'dem_example',  # dem
                      1.00E+05, 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                      7.50E+05, 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                      5.00E+04, 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                      3.00E+04, 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                      1.10E+06, 'DC', 0.139, 0.142, 0.00417, 0, 10, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                      3.00E+04, 'cc', 'AC', 'bev_example', 5, 0.25, 0.0075, 0, 0, 0, 10, 0.5, 11000, 11000, 1, 0.95,
                      0.95, 1,  # bev
                      3.00E+03, 'cc', 'AC', 'brs_example', 10, 0.45, 0.01, 0, 0, 0, 10, 0.5, 3600, 3600, 1, 0.95, 0.95,
                      1],  # brs
        '5_mgev_opt': ['1/1/2005', 'H', 365, 25, 'go', 48, 24, 0.09, {'core': 'SystemCore',
                                                                      'dem': 'FixedDemand',
                                                                      'wind': 'WindSource',
                                                                      'pv': 'PVSource',
                                                                      'ess': 'StationaryEnergyStorage',
                                                                      'gen': 'ControllableSource',
                                                                      'grid': 'ControllableSource',
                                                                      'bev': 'CommoditySystem'},  # scenario
                       'opt', 'opt', 0.08, 0.0024, 0, 20, 1, 0.95, 0.95,  # core
                       'AC', 'dem_example',  # dem
                       'opt', 'AC', 'wind_example', 2.8, 0.084, 0, 20, 1, 0.95,  # wind
                       'opt', 'DC', 'PVGIS file', 11, 0, 'pv_example', 0.503, 0.01509, 0, 25, 1, 0.95,  # pv
                       'opt', 'AC', 0.261, 0.02088, 0.00065, 10, 1, 1,  # gen
                       'opt', 'AC', 15, 0, 0.00003, 10, 1, 1,  # grid
                       'opt', 'DC', 0.139, 0.142, 0.00417, 0, 10, 0.9, 0.9, 0.8, 0.8, 0.5, 0, 1, 1,  # ess
                       'opt', 'cc', 'AC', 'bev_example', 5, 0.25, 0.0075, 0, 0, 0, 10, 0.5, 11000, 11000, 1, 0.95, 0.95,
                       1,  # bev
                       'opt', 'cc', 'AC', 'brs_example', 10, 0.45, 0.01, 0, 0, 0, 10, 0.5, 3600, 3600, 1, 0.95, 0.95, 1]
        # brs
        }

df = pd.DataFrame.from_dict(data)
df.index = index
df.reset_index(inplace=True, names=['block', 'key']) # this saves the multiindex into a column to make the index unique for json

file_path = os.path.join(os.getcwd(), 'example.json')
scenarios = df.to_json(file_path, orient='records', lines=True)

print(f'{file_path} created')
