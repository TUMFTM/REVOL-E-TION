import ast
import pandas as pd


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
    elif value.lower() in ['none', 'null', 'nan']:
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