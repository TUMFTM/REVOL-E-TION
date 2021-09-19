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
    This function calculates the present cost of an actual cost in the future (in year deltat)
    """
    pc = value / ((1 + discrate) ** deltat)  # used to be (deltat + 1) - why?
    return pc


def acc_discount(value, ts, discrate):
    """
    This function calculates the accumulated present cost of a yearly cost in the future (from now to ls years ahead)
    """
    apc = 0
    for year in range(0, ts):
        apc += discount(value, year, discrate)
    return apc


def adj_ce(ce, me, ls, discrate):
    """
    This function adjusts a component's capex (ce) to include discounted present cost for time based maintenance (pme)
    """
    ace = ce + acc_discount(me, ls, discrate)
    return ace