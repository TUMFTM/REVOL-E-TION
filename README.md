# MG_EV_Opti

"""
--- Tool name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de
September 2nd, 2021

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021

--- Detailed Description ---
This script models an energy system representing a minigrid for rural electrification in sub-saharan Africa and its
interaction with electric vehicles. It transforms the energy system graph into a (mixed-integer) linear program and
transfers it to a solver. The results are saved and the most important aspects visualized.

--- Input & Output ---
The script requires input data in the code block "Input". 
Additionally, several .csv-files for timeseries data are required.

--- Requirements ---
This tool requires oemof. Install by "pip install 'oemof.solph>=0.4,<0.5'"
All input data files need to be located in the same directory as this file

--- File Information ---
coding:     utf-8
license:    GPLv3

"""