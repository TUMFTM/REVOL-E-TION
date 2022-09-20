"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
February 3rd, 2022

--- Contributors ---
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022

--- Detailed Description ---
This script defines various functions used by main.py for orderly getting results from the different operating strats

--- Input & Output ---
see individual functions

--- Requirements ---
none

--- File Information ---
coding:     utf-8
license:    GPLv3
"""


import logging
import time
import pylightxl as xl
import os






def save_results_err(sim):
    """
    Dump error message in result excel file if optimization did not succeed
    """

    if sim['dump']:
        logging.warning("Error occurred, save model data")

    #    while '/' in file:
    #        file = re.sub(r'^.*?/', '', file)

        results_filepath = os.path.join(sim['resultpath'], 'results_' + os.path.basename(sim['settings_file']))

        if sim['run'] == 0:
            db = xl.Database()  # create a blank db
        else:
            db = xl.readxl(fn=results_filepath)

        # add a blank worksheet to the db
        db.add_ws(ws=sim['sheet'])

        # header of ws
        if sim['op_strat'] == 'go':
            ws_title = 'Global Optimum Results (' + sim['settings_file'] + ' - Sheet: ' + sim['sheet'] + ')'
        if sim['op_strat'] == 'rh':
            ws_title = 'Rolling Horizon Results (' + sim['settings_file'] + ' - Sheet: ' + sim['sheet'] + ', ' + str(
                sim['rh_ph']) + 'h, CH: ' + str(sim['rh_ch']) + 'h)'

        db.ws(ws=sim['sheet']).update_index(row=1, col=1, val=ws_title)

        # add sim name
        db.ws(ws=sim['sheet']).update_index(row=3, col=1, val='Logfile:')
        db.ws(ws=sim['sheet']).update_index(row=3, col=2, val=sim['name'])

        # write error message
        db.ws(ws=sim['sheet']).update_index(row=5, col=1,
                                     val='ERROR - Optimization could NOT succeed for these simulation settings')

        # write out the db
        xl.writexl(db=db, fn=results_filepath)
    return None







