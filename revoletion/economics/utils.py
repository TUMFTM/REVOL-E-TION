from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd


class OccursAt(Enum):
    BEGIN = 1.0
    MID = 0.5
    END = 0.0


class DEPRECIATION(Enum):
    LINEAR = "linear"


def discount(
    future_value: float,
    periods: int | npt.NDArray[np.integer],
    discount_rate: float,
    occurs_at: OccursAt,
) -> float:
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
    q = 1 + discount_rate
    try:
        return present_value * discount_rate / ((1 - (q**-observation_horizon)) * (q**occurs_at.value))
    except ZeroDivisionError:  # observation_horizon = 0
        return (
            present_value / observation_horizon
        )  # ToDo: check this. Error cause leads to next error in except statement


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
    depreciation: DEPRECIATION = DEPRECIATION.LINEAR,
    residual_at_ls: float = 0,
) -> float | npt.NDArray:
    """
    Calculate the residual value of a component based on the fraction of the remaining lifespan.
    A remaining lifespan fraction of 1 is considered as 0 as the component is not replaced anymore.
    """
    if depreciation == DEPRECIATION.LINEAR:
        return lifetime_remaining_frac * (1 - residual_at_ls) + residual_at_ls
    else:
        raise NotImplementedError(f"Depreciation method {depreciation} is not implemented")


def calc_capex_factors(
    project_duration: int,
    ls: int,
    init_age: int = 0,
    init_included: bool = True,
    residual_at_ls: float = 0.0,
) -> npt.NDArray:
    """
    Calculate the capex factors for each year of the project duration based on the lifespan and remaining lifespan fraction of the component.
    The capex factor is 1 in the year of replacement and 0 otherwise. In the first year, the capex factor is equal to the remaining lifespan fraction.
    """
    for param_name in ["project_duration", "ls", "init_age"]:
        param_value = locals()[param_name]
        if param_value < 0:
            raise ValueError(f"{param_name} cannot be negative")

    if init_included and init_age != 0:
        raise ValueError(f"If init_include is True, init_age_years must be 0, got {init_age}")

    capex = np.zeros(project_duration + 1, dtype=float)

    invest_first = 0 if init_included else ls - init_age

    invest_periods = np.arange(invest_first, project_duration, ls)

    capex[invest_periods] = 1

    if residual_at_ls > 0:
        residual_periods = invest_periods[1:] if init_included else invest_periods
        capex[residual_periods] -= residual_at_ls

    capex[-1] = calc_residual_value(
        lifetime_remaining_frac=calc_lifetime_remaining(
            project_duration=project_duration,
            ls=ls,
            init_age=init_age,
        )
        / float(ls),
        depreciation=DEPRECIATION.LINEAR,
        residual_at_ls=residual_at_ls,
    )

    return capex


if __name__ == "__main__":
    print(calc_capex_factors(project_duration=10, ls=5, init_age=0, init_included=True))
    print(calc_capex_factors(project_duration=10, ls=12, init_age=0, init_included=True))
    print(calc_capex_factors(project_duration=10, ls=12, init_age=2, init_included=False))
