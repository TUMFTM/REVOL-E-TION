# MG_EV_Opti
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer

### Created by   
Philipp Rosner, M.Sc.  
Institute of Automotive Technology  
Department of Mobility Systems Engineering  
School of Engineering and Design  
Technical University of Munich  
philipp.rosner@tum.de  
Created September 2nd, 2021

### Contributors  
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021  
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022  
Hannes Henglein, B.Sc. - Semester Thesis submitted 10/2022  
Marc Alsina Planelles, B.Sc. - Master Thesis submitted 10/2022

### Detailed Description  
This is a toolset to simulate electric vehicles operating in minigrids for rural electrification in emerging economies such as sub-saharan
Africa and optimize component sizes to show least-cost and/or least-emission options. It models a stand-alone minigrid using graph-based
representation, generates a pyomo model and hands it to a solver. Results are summarized in the

### Requirements  
This code was designed to run on Python 3.10.6 under Windows 10. Other operating systems are untested.

#### Packages
A clean Python environment to start with is recommended.  
Required packages can be installed via ```pip install -r requirements.txt``` 

#### Solver
The toolset is distributed with the open-source CBC solver as a default selection. All other solvers supported by pyomo
are also applicable in it. While CBC does work flawlessly, the commercial Gurobi solver enables a significant speed
advantage, especially when working with a large number of enabled system blocks or long term simulation. It is available
at https://www.gurobi.com/ with a free academic license model.

#### Steps to run the code
1.	Make sure to have installed the correct Python interpreter and packages
3.	Generate the following folders in your working directory (they are not under version control):
		./lp_models
		./results
		./scenarios
		./scenarios/pvgis_data
		./logfiles
4.	Get pvgis data from https://re.jrc.ec.europa.eu/pvg_tools/en/#HR or the repository holder
5.	Get exemplary vehicle and demand data from the repository holder
6.	Run main.py	

#### Input data
All input data files need to be located in ./input
Additionally, several .csv-files for timeseries data are required.

### Model output
enter description here


