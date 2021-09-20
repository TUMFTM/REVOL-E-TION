# MG_EV_Opti

"""
--- Toolset name ---
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer - MGEVOpti

--- Created by ---
Philipp Rosner, M.Sc.
Institute of Automotive Technology
Technical University of Munich
philipp.rosner@tum.de

--- Contributors ---
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021

--- Detailed Description ---
This is a toolset to simulate electric vehicles operating in minigrids for rural electrification in emerging economies such as sub-saharan
Africa and optimize component sizes to show least-cost and/or least-emission options. It models a stand-alone minigrid using graph-based
representation, generates a pyomo model and hands it to a solver. Results are summarized in the 

--- Input & Output ---
The script requires input data in the code block "Input". 
Additionally, several .csv-files for timeseries data are required.

--- Requirements ---
This code was designed to run on Python 3.8 under Windows 10 
Packages required to run this code are:
	oemof.solph >= 0.4.4
	oemof.tools
	pyomo >=5.7.1
	pandas
	numpy
	matplotlib
	os
	logging
	dateutil
	datetime

A (MI)LP solver is required as well. By default, the very fast Gurobi solver is selected. It is available with a free academic license.
A very good (albeit slightly slower) and easier to install open-source option is "CBC" which is available at
    http://ampl.com/dl/open/cbc/cbc-win64.zip
The unpacked executable should be packed in the same folder as this script.

All input data files need to be located in ./scenarios

Currently, due to a bug in pyomo, the following file has to be altered after installation as a quickfix:
    C:/Users/USER/AppData/Roaming/Python/Python38/site-packages/oemof/solph/models.py
The change occurs in lines 204ff. and comments out these operations
    status = solver_results["Solver"][0]["Status"].key
    termination_condition = solver_results["Solver"][0]["Termination condition"].key
replacing them with
    status = "ok"
    termination_condition = "optimal"
which disables a specific solver error occuring when looking for the .key method

--- Steps to run the code ---
1.	Make sure to have installed the correct Python and Package versions listed above
2.	Get the cbc solver executable from http://ampl.com/dl/open/cbc/cbc-win64.zip and place it in your working directiory
3.	Generate the following folders in your working directory (they are not under version control):
		./lp_models
		./results
		./scenarios
		./scenarios/pvgis_data
4.	Get pvgis data from https://re.jrc.ec.europa.eu/pvg_tools/en/#HR or the repository holder
5.	Get exemplary vehicle and demand data from the repository holder
6.	Run main.py	


"""