#!/usr/bin/env python3

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


def ann(pv, hor, discrate):
    """
        This function calculates the annuity due (the equivalent yearly sum to generate the same
        NPV) of an initial investment of present value pv over a horizon of hor years
    """
    q = 1 + discrate
    a = pv / (((1 - (q ** -hor)) / discrate) * q)  # fails at discrate 0 --> Todo: replace zero by eps
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
        year += 1
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


def ann_recur(ce, ls, hor, discrate, ccr):
    """
        This function calculates the annuity due of a recurring (every ls years) and price changing (annual ratio ccr)
        investment (the equivalent yearly sum to generate the same NPV) over a horizon of hor years
    """
    replacements = repllist(ls, hor)
    npc_repl = ce  # initial value, first capital expense is at BEGINNING of year 0, i.e. without discounting
    for repl in replacements:
        ce_repl = ce * (ccr ** repl)
        npc_repl += discount(ce_repl, repl, discrate)
    a = ann(npc_repl, hor, discrate)
    return a
