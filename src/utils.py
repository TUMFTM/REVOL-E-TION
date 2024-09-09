#!/usr/bin/env python3

import ast
import pandas as pd
import pandas.errors
import os


def infer_dtype(value):
    try:
        return int(value)
    except ValueError:
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

    try:
        evaluated = ast.literal_eval(value)
        if isinstance(evaluated, dict):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    return value.lower()


def get_period_fraction(dti, period, freq):

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


def convert_sdr(sdr: float, ts: pd.Timedelta) -> float:
    """
    This function converts the self-discharge rate (sdr) per month of a battery storage to a loss rate (lr) per timestep
    """
    # According to oemof documentation, the loss rate needs to be given for 1 hour neglecting the timestep of the model
    tsr = ts / pd.Timedelta('30 days')
    lr = 1 - (1 - sdr) ** tsr
    return lr


def scale_sim2year(value, scenario):
    return value / scenario.sim_yr_rat


def scale_year2prj(value, scenario):
    return value * scenario.prj_duration_yrs


def read_input_csv(block, path_input_file, scenario, multiheader=False, resampling=True):
    """
    Properly read in timezone-aware input csv files and form correct datetimeindex
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
    return df


def resample_to_timestep(data: pd.DataFrame, block, scenario):
    """
    Resample the data to the timestep of the scenario, conserving the proper index end even in upsampling
    :param data: The input dataframe with DatetimeIndex
    :param block: Object of a type defined in blocks.py
    :param scenario: The current scenario object
    :return: resampled dataframe
    """

    dti = data.index
    # Add one element to the dataframe to include the last timesteps
    try:
        dti_ext = dti.union(dti.shift(periods=1, freq=pd.infer_freq(dti))[-1:])
    except pandas.errors.NullFrequencyError:
        dti_ext = dti.union(dti.shift(periods=1, freq=pd.Timedelta('15min'))[-1:])
        scenario.logger.warning(f'Block \"{block.name}\": Timestep of csv input data could not be inferred -'
                                f'using 15 min default')

    data_ext = data.reindex(dti_ext).ffill()

    def resample_column(column):
        if data_ext[column].dtype == bool:
            return data_ext[column].resample(scenario.timestep).ffill().bfill()
        else:
            return data_ext[column].resample(scenario.timestep).mean().ffill().bfill()

    resampled_data = pd.DataFrame({col: resample_column(col) for col in data_ext.columns})[:-1]
    return resampled_data


def transform_scalar_var(block, var_name, scenario, run):
    scenario_entry = getattr(block, var_name)
    # In case of filename for operations cost read csv file
    if isinstance(scenario_entry, str):
        # Open csv file and use first column as index; also directly convert dates to DateTime objects
        dirname = block.parent.__class__.__name__ if hasattr(block, 'parent') else block.__class__.__name__
        opex = read_input_csv(block,
                              os.path.join(run.path_input_data, dirname, set_extension(scenario_entry)),
                              scenario)
        opex = opex[scenario.starttime:(scenario.sim_endtime - scenario.timestep_td)]
        # Convert data column of cost DataFrame into Series
        setattr(block, var_name, opex[opex.columns[0]])
    else:  # opex_spec is given as a scalar directly in scenario file
        # Use sequence of values for variable costs to unify computation of results
        setattr(block, var_name, pd.Series(scenario_entry, index=scenario.dti_sim))


def set_extension(filename):
    default_ext = '.csv'
    base, ext = os.path.splitext(filename)
    if not ext:
        filename = base + default_ext
    return filename
