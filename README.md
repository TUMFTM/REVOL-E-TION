# MG_EV_Opti

"""
--- Toolset name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Department of Mobility Systems Engineering
School of Engineering and Design
Technical University of Munich
philipp.rosner@tum.de
Created September 2nd, 2021

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022
Hannes Henglein, B.Sc. - Semester Thesis in progress
Marc Alsina Planelles, B.Sc. - Master Thesís in progress


--- Detailed Description ---
This is a toolset to simulate electric vehicles operating in minigrids for rural electrification in emerging economies such as sub-saharan
Africa and optimize component sizes to show least-cost and/or least-emission options. It models a stand-alone minigrid using graph-based
representation, generates a pyomo model and hands it to a solver. Results are summarized in the 

--- Input & Output ---
The script requires input data in the code block "Input". 
Additionally, several .csv-files for timeseries data are required.

--- Requirements ---
This code was designed to run on Python 3.8 under Windows 10 
Required packages can be installed (a clean environment to start with is recommended) via 
	pip install -r requirements.txt

A (MI)LP solver is required as well. By default, the very fast Gurobi solver is selected. It is available with a free academic license.
A very good (albeit slightly slower) and easier to install open-source option is "CBC" which is available at
    http://ampl.com/dl/open/cbc/cbc-win64.zip
The unpacked executable of the solver should be packed in the same folder as this script.

All input data files need to be located in ./scenarios

--- Steps to run the code ---
1.	Make sure to have installed the correct Python version and packages
2.	Get the cbc solver executable from http://ampl.com/dl/open/cbc/cbc-win64.zip and place it in your working directiory
3.	Generate the following folders in your working directory (they are not under version control):
		./lp_models
		./results
		./scenarios
		./scenarios/pvgis_data
		./logfiles
4.	Get pvgis data from https://re.jrc.ec.europa.eu/pvg_tools/en/#HR or the repository holder
5.	Get exemplary vehicle and demand data from the repository holder
6.	Run main.py	
"""