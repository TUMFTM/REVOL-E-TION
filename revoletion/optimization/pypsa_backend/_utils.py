import logging
import typing

import pandas as pd
import pytz

_LOGGER = logging.getLogger(__name__)


_T = typing.TypeVar("_T", bound=pd.DatetimeIndex | pd.DataFrame | pd.Series)


def normalize_dti_or_df(dti_or_df: _T) -> _T:
    utc_dti_or_df = dti_or_df.tz_convert(tz="UTC")
    normalized_dti_or_df = utc_dti_or_df.tz_localize(tz=None)
    return normalized_dti_or_df


def denormalize_dti_or_df(dti_or_df: _T, tz: str | pytz.BaseTzInfo) -> _T:
    utc_dti_or_df = dti_or_df.tz_localize(tz="UTC")
    denormalized_dti_or_df = utc_dti_or_df.tz_convert(tz=tz)
    return denormalized_dti_or_df
