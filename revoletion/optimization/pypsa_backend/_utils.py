import pandas as pd


def normalize_datetime_index(datetime_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized_dti = datetime_index.tz_localize(tz=None)
    return normalized_dti


def get_datetime_index_time_step_in_hours(datetime_index: pd.DatetimeIndex) -> float:
    """Retrive the time step size of a datetime index in units of hour.
    This helper is needed since the `freq` attribue of a `pd.DatetimeIndex` might not always be populated.
    """

    # Convert to a series where each row contains the dti entry and its time difference to its previous entry.
    dti_diff_series = datetime_index.to_series().diff().dropna()

    # Convert the time differences to hours.
    dti_diff_hours_series = dti_diff_series.dt.total_seconds() / 3600.0

    # Use the max and min difference to determine whether the dti is regular, i.e., each dti entry
    # is equally spaced apart.
    max_hours_diff = dti_diff_hours_series.max()
    min_hours_diff = dti_diff_hours_series.min()

    if max_hours_diff != min_hours_diff:
        raise RuntimeError(
            f"Cannot determine interval of datetime index: irregular datetime index (max={max_hours_diff}; min={min_hours_diff})"
        )

    return max_hours_diff
