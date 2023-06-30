# MG_EV_Opti
Minigrid (MG) & Electric Vehicle (EV) Interaction optimizer

#### Created by 
Philipp Rosner, M.Sc.  
Institute of Automotive Technology  
Department of Mobility Systems Engineering  
School of Engineering and Design  
Technical University of Munich  
philipp.rosner@tum.de  
Created September 2nd, 2021

#### Contributors  
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021  
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022  
Hannes Henglein, B.Sc. - Semester Thesis submitted 10/2022  
Marc Alsina Planelles, B.Sc. - Master's Thesis submitted 10/2022  
Juan Forero Yacaman - Bachelor's Thesis submitted 04/2023  
Elisabeth Spiegl - Bachelor's Thesis submitted 06/2023  
Hannes Henglein, B.Sc. - Master's Thesis ongoing  
Alejandro Hernando Armengol, B.Sc. - Master's Thesis ongoing

## Detailed Description  
This is a toolset to simulate electric vehicles operating in minigrids for rural electrification in emerging economies such as sub-saharan
Africa and optimize component sizes to show least-cost and/or least-emission options. It models a stand-alone minigrid using graph-based
representation, generates a pyomo model and hands it to a solver. Results are summarized in the

![System diagram](./images/system_diagram.png)

#### Definitions
| Term      | Description                                                                                                                                                                                                                                                                |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Run       | A run is defined by a single Microsoft Excel file, containing one sheet with run-wide information ("global_settings") such as solver and parallel mode as well as one sheet per scenario to be executed with actual energy system parameters and links to timeseries files |
| Scenario  | A scenario is defined by a single Excel worksheet and some timeseries inputs. It models an energy system, that is then sized and dispatched using the sim_os selected                                                                                                      |
| Horizon   | A scenario can be dispatched in multiple operating strategies, one of which is called "rolling horizon" and is similar to an MPC controller. In this case, each                                                                                                            |
| Block     | A block is a set of components representing a real-world system (e.g. a PV array in combination with its controller and converter). These can be toggled on or off, except for the core block containing the AC and DC buses as well as the converter(s) inbetween them.   |
| Component | A component is an oemof building block that is either a source, a sink, a bus, a transformer or a storage.                                                                                                                                                                 |


#### Input data
The toolset requires multiply types of input data, more specifically parameters and timeseries data. The former is 
directly defined in a Microsoft Excel file serving as the main input, while the latter are referenced from the Excel 
file by name and have to be located in the appropriate folder within /inputs.
- Mobile commodity system data: csv file containing columns "X_misoc", "X_consumption" and "X_atbase" with X being the name of the commodity.

#### Model output
enter description here

## Requirements  
This tool was designed to run under Windows 10 and Ubuntu 22.04 LTS. While portability was built in, other operating
systems are untested.

#### Environment
The tool was developed using Python 3.11, which is recommended to use with a clean virtual environment to start with.
All required packages are listed in ```requirements.txt``` and can be installed by entering  
```pip install -r requirements.txt``` into a terminal for the correct environment.

#### MILP Solver
The toolset is distributed with the open-source CBC solver for mixed integer linear programming (MILP) problems by 
default. All other solvers supported by pyomo are also applicable in it. On windows, this should work right out of the
box, while on Linux, an installation is required, e.g. using ```sudo apt-get install coinor-cbc coinor-libcbc-dev```
might be necessary (command is for Debian based distibutions - others on https://github.com/coin-or/Cbc). While CBC 
does work flawlessly, the commercial Gurobi solver enables a significant speed advantage, especially when working with
a large number of enabled system blocks or long term simulation. It is available at https://www.gurobi.com/ with a free
academic license model.

#### Plotly Server Handling
The toolset is using plotly to visualize the dispatch of the energy systems. However, once plotly is configured to open
the plots directly, it fails to close the handler properly (tested with plotly 5.14.1). To fix the resulting warning,
modify the last few lines of the "open_html_in_browser" function within the "_base_renderers.py script" in plotly/io
contained in your environment files:
    '''with HTTPServer(("127.0.0.1", 0), OneShotRequestHandler) as server:
			browser.open("http://127.0.0.1:%s" % server.server_port, new=new, autoraise=autoraise)
			server.handle_request()'''
in order to properly close the server handler



