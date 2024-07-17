'''
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
September 19th, 2021

--- Contributors ---
none

--- Detailed Description ---
This script defines various functions used by main.py for economic calculations

--- Input & Output ---
see individual functions

--- Requirements ---
none

--- File Information ---
coding:     utf-8
license:    GPLv3
'''


def discount(value, deltat, discrate):
    """
    This function calculates the present value of an actual cost in the future (in year deltat)
    """
    pc = value / ((1 + discrate) ** deltat)  # used to be (deltat + 1) - why?
    return pc


def acc_discount(value, ts, discrate):
    """
    This function calculates the accumulated present cost of a yearly cost in the future (from now to ts years ahead)
    """
    apc = 0
    for year in range(0, int(ts)):
        apc += discount(value, year, discrate)
    return apc


def adj_ce(ce: float, me: float, ls: int, discrate: float) -> float:
    """
    This function adjusts a component's capex (ce) to include discounted present cost for time based maintenance (pme)
    """
    ace = ce + acc_discount(me, ls, discrate)
    return ace


def ann(ce, hor, discrate):
    """
        This function calculates the annuity of an initial investment ce (the equivalent yearly sum to generate the same
        NPV) over a horizon of hor years
    """
    a = ce * (discrate * (1 + discrate) ** hor) / ((1 + discrate) ** hor - 1)
    return a


def ann_recur(ce, ls, hor, discrate, cost_decr):
    """
        This function calculates the annuity of a recurring (every ls years) and cheapening (annual ratio cost_decr <1)
        investment (the equivalent yearly sum to generate the same NPV) over a horizon of hor years
    """
    a = ann(ce, hor, discrate) * ((1 - ((1-cost_decr)/(1+discrate))**hor) / (1 - ((1-cost_decr)/(1+discrate))**ls))
    return a


def repllist(ls, hor):
    """
        This function calculates a list of years to replace the component in, given its lifespan ls and the observation
        horizon hor
    """
    year = 0
    rul = ls
    repyrs = []
    while year < hor:
        if rul == 0:
            repyrs.append(year)
            rul = ls
        rul -= 1
        year +=1
    return repyrs


def tce(ce, cdr, ls, hor):
    """
        This function calculates the total (non-discounted) capital expenses for a component that has to be replaced
        every ls years during the observation horizon hor and varies in price at a ratio (cdr) every year.
    """
    tce = ce + sum([ce * (cdr ** yr) for yr in repllist(ls, hor)])
    return tce


def pce(ce, cdr, discrate, ls, hor):
    """
        This function calculates the present (discounted) capital expenses for a component that has to be replaced
        every ls years during the observation horizon hor and varies in price at a ratio (cdr) every year.
    """
    pce = ce + sum([discount(ce * (cdr ** yr), yr, discrate) for yr in repllist(ls, hor)])
    return pce
