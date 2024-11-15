#!/usr/bin/env python3

def discount(future_value: float,
             periods: int,
             discount_rate: float) -> float:
    """
    This function calculates the present value of a future value in some periods at a discount rate per period
    """
    q = 1 + discount_rate
    present_value = future_value / (q ** periods)
    return present_value


def acc_discount(nominal_value: float,
                 observation_horizon: int,
                 discount_rate: float,
                 occurs_at='beginning') -> float:
    """
    This function calculates the accumulated present value of a periodical, nominally repeating cashflow in the future
    (from present to the observation horizon) at a discount rate per period
    """
    p_delta = {'beginning': 0, 'middle': 0.5, 'end': 1}[occurs_at]
    present_value = sum([discount(nominal_value, period + p_delta, discount_rate)
                         for period in range(observation_horizon)])
    return present_value


def join_capex_mntex(capex: float,
                     mntex: float,
                     lifespan: int,
                     discount_rate: float) -> float:
    """
    This function adjusts a component's capex to include accumulated present maintenance cost for time based maintenance
    """
    capex_adjusted = capex + acc_discount(mntex, lifespan, discount_rate, occurs_at='beginning')
    return capex_adjusted


def annuity(present_value: float,
            observation_horizon: int,
            discount_rate: float,
            occurs_at='beginning') -> float:
    """
    This function calculates the annuity (the equivalent periodical, nominally recurring value to generate the same
    NPV) of a present value pv over an observation horizon at a discount rate per period. occurs_at denotes whether
    the expense or value occurs at the beginning (making the annuity an annuity due) or end of the period.
    """
    q = 1 + discount_rate
    due_factor = {'beginning': q, 'end': 1}[occurs_at]
    try:
        annuity = present_value * discount_rate / ((1 - (q ** -observation_horizon)) * due_factor)
    except ZeroDivisionError:
        annuity = present_value / observation_horizon
    return annuity


def invest_periods(lifespan: int,
                   observation_horizon: int) -> list:
    """
    This function returns a list of period numbers to invest into a component (i.e. buy or replace it),
    given its lifespan and the observation horizon
    """
    return [period for period in range(observation_horizon) if period % lifespan == 0]


def capex_sum(capex_init: float,
              cost_change_ratio: float,
              lifespan: int,
              observation_horizon: int) -> float:
    """
    This function calculates the total (non-discounted) capital expenses for a component that has to be replaced
    after its lifespan during the observation horizon and changes in price at a cost change ratio every period.
    """
    capex_total = sum([capex_init * (cost_change_ratio ** period)
                       for period in invest_periods(lifespan, observation_horizon)])
    return capex_total


def capex_present(capex_init: float,
                  cost_change_ratio: float,
                  discount_rate: float,
                  lifespan: int,
                  observation_horizon: int) -> float:
    """
    This function calculates the present (discounted) capital expenses for a component that has to be replaced
    after its lifespan during the observation horizon and changes in price at a cost change ratio every period.
    """
    capex_present = sum([discount(capex_init * (cost_change_ratio ** period), period, discount_rate)
                         for period in invest_periods(lifespan, observation_horizon)])
    return capex_present


def annuity_due_capex(capex_init: float,
                      lifespan: int,
                      observation_horizon: int,
                      discount_rate: float,
                      cost_change_ratio: float) -> float:
    """
    This function calculates the annuity due of a recurring (every ls years) and price changing (annual ratio ccr)
    investment (the equivalent yearly sum to generate the same NPV) over a horizon of hor years
    """
    present_value = capex_present(capex_init=capex_init,
                                  cost_change_ratio=cost_change_ratio,
                                  discount_rate=discount_rate,
                                  lifespan=lifespan,
                                  observation_horizon=observation_horizon)
    annuity_due_capex = annuity(present_value=present_value,
                                observation_horizon=observation_horizon,
                                discount_rate=discount_rate,
                                occurs_at='beginning')
    return annuity_due_capex
