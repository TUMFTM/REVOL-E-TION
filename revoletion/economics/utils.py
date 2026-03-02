import numbers
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytz

from revoletion.utils import read_timeseries_csv, set_extension


class OccursAt(Enum):
    BEGIN = 1.0
    MID = 0.5
    END = 0.0


@dataclass(frozen=True)
class CostTypeDefinition:
    label: str
    occurs_at: OccursAt | None


class Depreciation(Enum):
    LINEAR = "linear"


def discount(
    future_value: float,
    periods: int | npt.NDArray[np.integer],
    discount_rate: float,
    occurs_at: OccursAt,
) -> float | npt.NDArray:
    """
    calculate the present value of a future value in some periods at a discount rate per period
    """
    return future_value / ((1 + discount_rate) ** (periods - occurs_at.value))


def acc_discount(
    nominal_value: float | pd.Series,
    observation_horizon: int | pd.Series,
    discount_rate: float,
    occurs_at: OccursAt,
) -> float:
    """
    calculate the accumulated present value of a periodical, nominally repeating cashflow in the future
    (from present to the observation horizon) at a discount rate per period
    """
    q = 1 + discount_rate
    discount_factor = (q**occurs_at.value) * (1 - (q**-observation_horizon)) / discount_rate
    return nominal_value * discount_factor


def annuity(present_value: float, observation_horizon: int, discount_rate: float, occurs_at: OccursAt) -> float:
    """
    calculate the annuity (the equivalent periodical, nominally recurring value to generate the same
    NPV) of a present value pv over an observation horizon at a discount rate per period. occurs_at denotes whether
    the expense or value occurs at the beginning (making the annuity an annuity due) or end of the period.
    """
    if observation_horizon < 1:
        raise ValueError(f"Observation horizon must be at least 1, got {observation_horizon}")

    if discount_rate == 0:  # leads to division by zero
        return present_value / observation_horizon

    q = 1 + discount_rate
    return present_value * discount_rate / ((1 - (q**-observation_horizon)) * (q**occurs_at.value))


def calc_wacc(
    share_equity: float,  # share of equity in capital structure
    rate_debt: float,  # interest rate on debt
    rate_market: float = 0.07,  # expected return on market
    rate_riskfree: float = 0.03,  # risk-free return rate
    rate_tax: float = 0.25,  # corporate tax rate
    rate_inflation: float = 0.02,  # expected inflation rate
    volatility_relative: float = 1,  # volatility of stock price relative to market
) -> tuple[float, float]:
    """
    This function calculates the nominal (including inflation) weighted average cost of capital (WACC) using the
    Capital Asset Pricing Model (CAPM) for equity cost.
    """
    share_debt = 1 - share_equity
    cost_equity = rate_riskfree + volatility_relative * (rate_market - rate_riskfree)  # CAPM
    wacc_nominal = share_debt * rate_debt * (1 - rate_tax) + share_equity * cost_equity
    wacc_real = wacc_nominal * ((1 + wacc_nominal) / (1 + rate_inflation))  # fisher formula
    return wacc_nominal, wacc_real


def calc_lifetime_remaining(project_duration: int, ls: float, init_age: float) -> float:
    """
    Calculate the remaining lifetime of a component based on the project duration, lifespan and age of the component.
    The remaining lifetime is 0 if the component is replaced during the project duration.
    """
    return (-project_duration - init_age) % ls


def calc_residual_value(
    lifetime_remaining_frac: float | npt.NDArray,
    depreciation: Depreciation = Depreciation.LINEAR,
    residual_at_ls: float = 0,
) -> float | npt.NDArray:
    """
    Calculate the residual value of a component based on the fraction of the remaining lifespan.
    A remaining lifespan fraction of 1 is considered as 0 as the component is not replaced anymore.
    """
    if depreciation == Depreciation.LINEAR:
        return lifetime_remaining_frac * (1 - residual_at_ls) + residual_at_ls
    else:
        raise NotImplementedError(f"Depreciation method {depreciation} is not implemented")


def transform_scalar_var(
    value: str | float, dti: pd.DatetimeIndex, data_dir: Path, allow_scalar: bool = False
) -> pd.Series | float:
    """
    Transform a value holding either the path to a csv file containing a timeseries or a scalar
    to a pandas Series with the same DatetimeIndex as the simulation.
    """
    if isinstance(value, numbers.Number):  # value is given as scalar
        if allow_scalar:
            return value
        else:
            return pd.Series(index=dti, data=np.full(len(dti), value, dtype=float), name="cost")

    elif isinstance(value, str):  # value contains filename
        filepath = set_extension(filename=data_dir / value, default_extension=".csv")
        if not filepath.is_file():
            raise FileNotFoundError(f"Timeseries file {filepath} not found.")

        tz = pytz.timezone(str(dti.tz)) if dti.tz is not None else None

        try:
            df = read_timeseries_csv(
                path_input_file=filepath,
                timezone=tz,
                multiheader=False,
                resampling_dti=dti,
            )

            if df.shape[1] != 1:
                print(f"Input data in {value} contains more than one column - only first column is used.")

            return df.iloc[:, 0]  # return only first column

        except IndexError:
            raise IndexError(f"Failed to load timeseries data from {value}.")

    else:
        raise ValueError(f"Value must be either a scalar or a Path object, got {type(value).__name__}")
