#!/usr/bin/env python3

import ast
import importlib.util
import numpy as np
import pandas as pd
import pandas.errors
import os


def infer_dtype(value):
    """
    Infer the data type of a value from a string representation. To be used as a .map(infer_dtype) function.
    """
    try:
        return int(value)
    except (ValueError or OverflowError):
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() in ['none', 'null', 'nan', '']:
        return None
    elif os.path.isdir(value):
        return value

    try:
        evaluated = ast.literal_eval(value)
        if isinstance(evaluated, dict):
            return evaluated
        elif isinstance(evaluated, list):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    return value.lower()


def get_period_fraction(dti, period, freq):
    """
    Calculate the fraction of a period that is covered by a datetime index. Used in peakshaving calculations for
    edges of simulation timeframe
    """
    # if interval is not part of dti_sim (happens for rh), dti is empty -> return 0
    if len(dti) == 0:
        return 0

    if period == 'day':
        start = dti.min().normalize()
        end = start + pd.DateOffset(days=1) - pd.Timedelta(freq)
    elif period == 'week':
        start = dti.min().normalize() - pd.Timedelta(days=dti[0].weekday())
        end = start + pd.DateOffset(weeks=1) - pd.Timedelta(freq)
    elif period == 'month':
        start = dti.min().normalize().replace(day=1)
        end = start + pd.DateOffset(months=1) - pd.Timedelta(freq)
    elif period == 'quarter':
        start = dti.min().normalize().replace(day=1, month=((dti[0].month - 1) // 3) * 3 + 1)
        end = start + pd.DateOffset(months=3) - pd.Timedelta(freq)
    elif period == 'year':
        start = dti.min().normalize().replace(day=1, month=1)
        end = start + pd.DateOffset(years=1) - pd.Timedelta(freq)

    period_fraction = len(dti) / len(pd.date_range(start, end, freq=freq))

    return period_fraction


def create_expenditures_dataframe():
    # previous capex_init is renamed to capex_sim to reduce number of columns of the dataframe
    expenditures = pd.DataFrame(index=['capex', 'mntex', 'opex', 'opex_ext', 'totex', 'crev'],  # ext = external charging
                                columns=['sim', 'yrl', 'prj', 'dis', 'ann'],
                                data=0,
                                dtype='float64')

    # initialize non-existing expenditures with None
    for ex_type, ex_period in [('capex', 'yrl'), ('totex', 'yrl'), ('crev', 'ann')]:
        expenditures.loc[ex_type, ex_period] = np.nan

    return expenditures


def convert_sdr(sdr: float, ts: pd.Timedelta) -> float:
    """
    This function converts the self-discharge rate (sdr) per month of a battery storage to a loss rate (lr) per timestep
    """
    # According to oemof documentation, the loss rate needs to be given for 1 hour neglecting the timestep of the model
    tsr = ts / pd.Timedelta('30 days')
    lr = 1 - (1 - sdr) ** tsr
    return lr


def conv_add_max(value):
    return value if pd.notna(value) else None


def init_tuple_variables(block, name_vars: list):
    name_var1, name_var2 = name_vars
    if (getattr(block, name_var1) == 'equal') and (getattr(block, name_var2) == 'equal'):
        error_msg = (f'"{block.name}" parameters {name_var1} and {name_var2} were both set to equal.'
                     f' Maximum one of these variables is allowed to be set to "equal"')
        block.scenario.logger.error(error_msg)
    elif getattr(block, name_var1) == 'equal':
        setattr(block, name_var1, getattr(block, name_var2))
    elif getattr(block, name_var2) == 'equal':
        setattr(block, name_var2, getattr(block, name_var1))


def extend_dti(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Extend a datetime index by one timestep to include the last timestep of the simulation timeframe.
    """
    dti_ext = dti.union(dti.shift(periods=1, freq=pd.infer_freq(dti))[-1:])
    return dti_ext


def import_module_from_path(module_name, file_path):
    """
    Import a Python module from a specific file path. Is used for timeframe mapper user input code.
    """
    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)
    # Load and execute the module
    spec.loader.exec_module(module)
    return module


def scale_sim2year(value, scenario):
    return value / scenario.sim_yr_rat


def scale_year2prj(value, scenario):
    return value * scenario.prj_duration_yrs


def scale_sim2prj(value, scenario):
    return scale_year2prj(scale_sim2year(value, scenario), scenario)


def read_demand_file(block):
    """
    Read in a CommodityDemand csv file
    """
    path_demand_file = os.path.join(block.scenario.run.path_input_data,
                                    set_extension(block.filename))
    df = pd.read_csv(path_demand_file,
                     index_col=0)
    df['time_req'] = pd.to_datetime(df['time_req'], utc=True).dt.tz_convert(block.scenario.timezone)
    df['dtime_active'] = pd.to_timedelta(df['dtime_active'])
    df['dtime_idle'] = pd.to_timedelta(df['dtime_idle'])
    df['dtime_patience'] = pd.to_timedelta(df['dtime_patience'])

    df = df.loc[df['time_req'].isin(block.scenario.dti_sim), :]

    return df


def read_input_csv(block, path_input_file, scenario, multiheader=False, resampling=True):
    """
    Properly read in timezone-aware example timeseries csv files and form correct datetimeindex
    """
    if multiheader:
        df = pd.read_csv(path_input_file, header=[0, 1])
        df.sort_index(axis=1, sort_remaining=True, inplace=True)
        df = df.set_index(pd.to_datetime(df.loc[:, ('time', 'time')], utc=True)).drop(columns='time')
    else:
        df = pd.read_csv(path_input_file)
        df = df.set_index(pd.to_datetime(df['time'], utc=True)).drop(columns='time')

    # parser in to_csv does not create datetimeindex
    df = df.tz_convert(scenario.timezone)
    if resampling:
        df = resample_to_timestep(df, block, scenario)

        if not (scenario.dti_sim.isin(df.index).all()):
            raise IndexError(f'Block "{block.name}": Input timeseries data does not cover simulation timeframe')
    return df


def read_input_log(system):
    """
    Read in a predetermined log file for the CommoditySystem behavior. Normal resampling cannot be used as
    consumption must be meaned, while booleans, distances and dsocs must not. Function has to be callable for
     ICEVSystems as well
    """

    log_path = os.path.join(system.scenario.run.path_input_data,
                            set_extension(system.filename))
    df = read_input_csv(system,
                        log_path,
                        system.scenario,
                        multiheader=True,
                        resampling=False)

    # Timedelta of frequency of log file
    freq_log = pd.infer_freq(df.index).lower()
    # pd.Timedelta('h') fails --> add '1' --> pd.Timedelta('1h')
    freq_log = pd.Timedelta((freq_log if freq_log[0].isdigit() else '1' + freq_log))

    # Compare Timedelta objects instead of strings to avoid problems (1h vs. 60min)
    if freq_log != system.scenario.timestep_td:
        system.scenario.logger.warning(f'Block "{system.name}": input data does not match specified timestep - Resampling')
        consumption_columns = list(filter(lambda x: 'consumption' in x[1], df.columns))
        bool_columns = df.columns.difference(consumption_columns)
        # mean ensures equal energy consumption after downsampling, ffill and bfill fill upsampled NaN values
        df_new = pd.DataFrame()
        df_new[consumption_columns] = df[consumption_columns].resample(system.scenario.timestep).mean().ffill().bfill()
        df_new[bool_columns] = df[bool_columns].resample(system.scenario.timestep).ffill().bfill()
        df = df_new
    if not (system.scenario.dti_sim.isin(df.index).all()):
        raise IndexError(f'Block "{system.name}": Input timeseries data does not cover simulation timeframe')

    # if the names of the commodities in the log file differ from the usual naming scheme (name of the commodity
    # system + number), the names specified in the log file names are used, with the commodity system name added
    # for unique identification.
    if system.data_source == 'log':
        com_names_log = sorted(df.columns.get_level_values(0).unique()[:system.num].tolist())
        if system.com_names != com_names_log:
            com_names_map = {log_name: f'{system.name}_{log_name}' for log_name in com_names_log}
            df.columns = df.columns.map(lambda x: (com_names_map.get(x[0], x[0]), *x[1:]))

    return df


def read_usecase_file(system):
    """
    Function reads a usecase definition csv file for DES and performs necessary normalization for each timeframe.
    Function has to be callable for ICEVSystems as well.
    """

    usecase_path = os.path.join(system.scenario.run.path_input_data,
                                set_extension(system.filename))
    df = pd.read_csv(usecase_path,
                     header=[0, 1],
                     index_col=0)
    for timeframe in df.columns.levels[0]:
        df.loc[:, (timeframe, 'rel_prob_norm')] = (df.loc[:, (timeframe, 'rel_prob')] /
                                                   df.loc[:, (timeframe, 'rel_prob')].sum())
        df.loc[:, (timeframe, 'sum_dep_magn')] = (df.loc[:, (timeframe, 'dep1_magnitude')] +
                                                  df.loc[:, (timeframe, 'dep2_magnitude')])

        # catch cases where the sum of both departure magnitudes is not one
        df.loc[:, (timeframe, 'dep1_magnitude')] = (df.loc[:, (timeframe, 'dep1_magnitude')] /
                                                    df.loc[:, (timeframe, 'sum_dep_magn')])
        df.loc[:, (timeframe, 'dep2_magnitude')] = (df.loc[:, (timeframe, 'dep2_magnitude')] /
                                                    df.loc[:, (timeframe, 'sum_dep_magn')])

        df.drop(columns=[(timeframe, 'sum_dep_magn')], inplace=True)

    return df


def resample_to_timestep(data: pd.DataFrame, block, scenario):
    """
    Resample the data to the timestep of the scenario, conserving the proper index end even in upsampling
    """

    dti = data.index
    # Add one element to the dataframe to include the last timesteps
    try:
        dti_ext = dti.union(dti.shift(periods=1, freq=pd.infer_freq(dti))[-1:])
    except pandas.errors.NullFrequencyError:
        dti_ext = dti.union(dti.shift(periods=1, freq=pd.Timedelta('15min'))[-1:])
        scenario.logger.warning(f'Block "{block.name}": Timestep of csv example data could not be inferred -'
                                f'using 15 min default')

    data_ext = data.reindex(dti_ext).ffill()

    def resample_column(column):
        if data_ext[column].dtype == bool:
            return data_ext[column].resample(scenario.timestep).ffill().bfill()
        else:
            return data_ext[column].resample(scenario.timestep).mean().ffill().bfill()

    resampled_data = pd.DataFrame({col: resample_column(col) for col in data_ext.columns})[:-1]
    return resampled_data


def transform_scalar_var(block, var_name):
    """
    Transform scalar variable to a constant pandas Series with the same index as the simulation.
    Not every block that calls this function is a child of the Block class, so function lives here
    """
    scenario = block.scenario if hasattr(block, 'scenario') else block.parent.scenario  # catch GridMarket
    attr = getattr(block, var_name)
    # In case of filename for operations cost read csv file
    if isinstance(attr, str):
        # Open csv file and use first column as index; also directly convert dates to DateTime objects
        opex = read_input_csv(block,
                              os.path.join(scenario.run.path_input_data,
                                           set_extension(attr)),
                              scenario)
        opex = opex[scenario.starttime:(scenario.sim_extd_endtime - scenario.timestep_td)]
        # Convert data column of cost DataFrame into Series
        setattr(block, var_name, opex[opex.columns[0]])
    else:  # opex_spec is given as a scalar directly in scenario file
        # Use sequence of values for variable costs to unify computation of results
        setattr(block, var_name, pd.Series(attr, index=scenario.dti_sim_extd))


def set_extension(filename, default_extension='.csv'):
    """
    Add a default extension to a filename if none is given. If the filename already has an extension, it is kept.
    """
    base, ext = os.path.splitext(filename)
    if not ext:
        filename = base + default_extension
    return filename


def lognormal_params(mean, stdev):
    mu = np.log(mean ** 2 / np.sqrt((mean ** 2) + (stdev ** 2)))
    sig = np.sqrt(np.log(1 + (stdev ** 2) / (mean ** 2)))
    return mu, sig