# REVOL-E-TION
Resilient Electric Vehicle Optimization model for Local Energy TransitION

This toolset is designed to optimize and estimate technoeconomic potentials of electric vehicle integration into 
local energy systems such as mini- and microgrids, company sites, apartment blocks or single homes.

#### Originally created by 
Philipp Rosner, M.Sc.
Research Associate Institute of Automotive Technology
Department of Mobility Systems Engineering  
TUM School of Engineering and Design  
Technical University of Munich  
philipp.rosner@tum.de  
September 2nd, 2021

#### Contributors
Brian Dietermann, M.Sc. - Research Associate 06/2022-  
Marcel Brödel, M.Sc. - Research Associate 01/2024-

David Eickholt, B.Sc. - Semester Thesis submitted 07/2021  
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022  
Hannes Henglein, B.Sc. - Semester Thesis submitted 10/2022  
Marc Alsina Planelles, B.Sc. - Master's Thesis submitted 10/2022  
Juan Forero Yacaman - Bachelor's Thesis submitted 04/2023  
Elisabeth Spiegl - Bachelor's Thesis submitted 06/2023  
Hannes Henglein, B.Sc. - Master's Thesis ongoing  
Alejandro Hernando Armengol, B.Sc. - Master's Thesis submitted 10/2023

## Installation

This toolset was designed to run under Windows 10, Ubuntu 22.04 LTS and MacOS 14 Sonoma.
While portability is generally built in, other operating systems are untested.

#### Source Code, licensing & Distribution
The toolset is developed and internally distributed via LRZ Gitlab prior to being published open source via the FTM EV Lab's GitLab. It is easiest to clone the respective repository at https://gitlab.lrz.de/ftm-electric-powertrain/mg_ev_opti to obtain a working copy of the source code.

<mark> Prior to open source publication of the toolset, any distribution outside FTM researchers or their immediately affiliated students is prohibited. </mark> 

#### Environment & Packages
The toolset was developed using Python 3.11, which is recommended to use with a clean virtual environment to start with.
All required packages are listed in ```requirements.txt``` and can be installed by entering ```pip install -r requirements.txt``` into a terminal for the correct environment.

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
modify the last few lines of the ```open_html_in_browser``` function within the ```_base_renderers.py``` script in plotly/io
contained in your environment files:
```
with HTTPServer(("127.0.0.1", 0), OneShotRequestHandler) as server:
    browser.open("http://127.0.0.1:%s" % server.server_port, new=new, autoraise=autoraise)
    server.handle_request()
```
in order to properly close the server handler

## Basic Usage

Running the toolset needs closely defined scenarios to simulate and simulation settings to operate on. These are provided in two separate .csv-files to be specified when starting a simulation, either via a Graphical User Interface (GUI, this variant also enables a path choice of where to store results) automatically coming up when starting the toolset without any parameters given. Alternatively, the names of the files can be specified in the terminal command as follows, given the base folder is selected:
```
python main.py <scenario_file_name>.csv <settings_file_name>.csv
```
Formatting of the scenario and settings files can be taken from the example files. It is recommended to mostly use the parallel computing option for multiple rolling horizon scenarios.


## Detailed Description  
This is a toolset to simulate electric vehicles operating in minigrids for rural electrification in emerging economies such as sub-saharan
Africa and optimize component sizes to show least-cost and/or least-emission options. It models a stand-alone minigrid using graph-based
representation, generates a pyomo model and hands it to a solver. Results are summarized in the

![System diagram](./images/system_diagram.png)

#### General Terms & Definitions
| Term                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Run                    | A run is defined by a single csv scenario file and a settings csv file, the latter containing run-wide information such as solver and parallel and the former the actual scenario to be simulated/optimized with actual energy system parameters and links to timeseries files                                                                                                                                                                                                                                          |
| Scenario               | A scenario is defined by a column in the scenario csv file and some timeseries inputs. It models an energy system, that is then sized and dispatched using the strategy selected                                                                                                                                                                                                                                                                                                                                        |
| Strategy               | A strategy is a set of rules or an optimization algorithm that determines how the energy system is operated. The toolset supports two strategies: "go" for global optimum and "rh" for rolling horizon optimization.                                                                                                                                                                                                                                                                                                    |
| Horizon                | A scenario can be simulated in multiple strategies, one of which is called "rolling horizon" and is similar to an MPC controller. In this case, the total simulation time is split up into prediction horizons that are simulated consecutively. However, to guarantee some overlap between the horizons, only the so called control horizon is actually taken for overall result calculation. It has to be equal to or shorter than the prediction horizon. ![Rolling Horizon principle](./images/rolling_horizon.png) |
| Block                  | A block is a set of components representing a real-world system (e.g. a PV array in combination with its controller and converter). These are classes, the instances of which to be actually simulated can be defined in the scenario file as a dictionary, except for the core block containing the AC and DC buses as well as the converter(s) inbetween them. Multiple instances of one block can coexist in a model.                                                                                                |
| Component              | A component is an oemof element that is either a source, a sink, a bus, a converter or a storage.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| InvestBlock            | This is the parent class for all blocks that are possibly being sized in optimization (i.e. where an investment decision has to be taken) and for which entering ```'opt'``` as the size in the scenario file will trigger that size optimization. However, this is only possible in the ```'go'``` strategy                                                                                                                                                                                                            |
| SystemCore             | The central components of the energy system, i.e. the AC and DC buses and the two unidirectional transformers between them                                                                                                                                                                                                                                                                                                                                                                                              |
| FixedDemand            | An undeferrable power demand such as households.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| PVSource               | A photovoltaic array providing power as per its potential (coming from a Solcast or PVGIS file or the PVGIS API). Unused potential is curtailed through a sink within the block. .                                                                                                                                                                                                                                                                                                                                      |
| WindSource             | A wind turbine providing power as per its potential (defined within a csv file over time). Unused potential is curtailed through a sink within the block.                                                                                                                                                                                                                                                                                                                                                               |
| ControllableSource     | A freely controllable power source that is unlimited in energy, e.g. a fossil fuel generator.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| VehicleCommoditySystem | A fleet of electric vehicles, the behavior of which is defined by a csv file (the log). This csv file can be generated within a discrete event simulation incorporated in REVOL-E-TION that can also model range extension using a BatteryCommoditySystem.                                                                                                                                                                                                                                                              |
| BatteryCommoditySystem | A fleet of mobile batteries, the behavior of which is defined by a csv file (the log). This csv file can be generated within a discrete event simulation incorporated in REVOL-E-TION that can model multiple use cases as well as range extension for a VehicleCommoditySystem.                                                                                                                                                                                                                                        |
| RentalSystem           | The mother class for execution of the Discrete Event Simulation (DES) simulating the mobility/energy demand and its allocation to different commodities within a service system.                                                                                                                                                                                                                                                                                                                                        |
| RentalProcess          | A single rental of a vehicle or battery manifesting in blocking one or more primary (within the own CommoditySystem) and/or secondary commodities (in a linked CommoditySystem through range extension).                                                                                                                                                                                                                                                                                                                |


#### Input data
The toolset requires multiple types of input data, more specifically settings, parameters and timeseries data. The
former are defined in a short json file under /input/settings and define general simulation and execution settings for
the Run. The Scenarios to be executed and their parameters are defined in a second json file in /input/scenarios,
an example of which is distributed with the toolset including its generator script, while the latter are
referenced in the scenarios json file by stem name (without extension) and have to be located in the appropriate folder
within /inputs. 

The scalar parameters used are named with the block they belong to, followed by the parameter's name or abbreviation.
The following abbreviations and parameter names are used frequently:

| Name / Abbreviation              | Description                                                                                                                                                                                                                                                                                         |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| strategy                         | Operational Strategy or Energy Management Algorithm. Determines whether the optimization problem is solved in one shot (global optimum - "go") or in multiple segments (rolling horizon- "rh"). In theory, other approaches such as rule-based or artificial intelligence strategies are thinkable. |
| go                               | Global Optimum:                                                                                                                                                                                                                                                                                     |
| rh                               | Rolling Horizon:                                                                                                                                                                                                                                                                                    |
| ph                               | Prediction Horizon: Length of the slices of input data being fed to the optimizer in rolling horizon optimization equalling the timeframe of input and demand data that can be predicted reliably in operando.                                                                                      |
| ch                               | Control Horizon: Length of the the control horizon. This defines the time after which a new ControlHorizon and PredictionHorizon with the corresponding optimization is started.                                                                                                                    |
| wacc                             | Weighted average cost of capital: discount rate for future expenses/revenues and energies (usually suufix _dis signifies values that have been accumulatively discounted                                                                                                                            |
| size                             | Size of the component. Defines the energy capacity of storage components (Wh) or the power of other components (W)                                                                                                                                                                                  |
| system                           | Defines the bus (AC/DC) the current component is connected to                                                                                                                                                                                                                                       |
| filename                         | Filename of the corresponding input file which defines costs or available power or mobility demand etc.                                                                                                                                                                                             |
| capex_spec                       | Capital expenses per unit of size (W in case of nominal power or Wh in case of storage blocks)                                                                                                                                                                                                      |
| mntex_spec                       | Time-based Maintenance expenses of the block per unit of size and year.                                                                                                                                                                                                                             |
| opex_spec                        | Operational expenses including throughput based maintenance of the block per unit of energy (typically Wh). This might include sales to a market with time flexibility as negative cost, which are then omitted from customer revenues.                                                             |
| crev_spec                        | Customer revenue (i.e. where the system operator can define a price). Can be scalar or string leading to a timeseries csv file.                                                                                                                                                                     |
| ls                               | Lifespan of a component in years. After this time it will be replaced                                                                                                                                                                                                                               |
| cdc                              | Cost decrease ratio of the block's nominal price per year to be considered for replacement after its lifespan                                                                                                                                                                                       |
| eff                              | Efficiency. Often used for converters and/or blocks representing lossy processes.                                                                                                                                                                                                                   |
| chg                              | Prefix: charging                                                                                                                                                                                                                                                                                    |
| dis                              | Prefix: discharging                                                                                                                                                                                                                                                                                 |
| crate                            | Maximum/Minimum C-Rate of storage components                                                                                                                                                                                                                                                        |
| sdr                              | Self discharge rate of storage components per month                                                                                                                                                                                                                                                 |
| rex (CommoditySystem)            | Range Extension through a BatteryCommoditySystem providing extra storage to a VehicleCommoditySystem.                                                                                                                                                                                               |
| int_lvl (CommoditySystem)        | Charging Integration Level. Can reach from UC (Uncoordinated charging, equal to starting a charge at max power as soon as the commodity rejoins the system) to fully optimized V2G.                                                                                                                 |

The required timeseries must be provided as csv files in the following formats:

- FixedDemand blocks: A csv file with one or two columns, one of which must be labeled "power_w" and contain values in W.
	If a second is contained, it must be labeled "time" and contain datetime values in a pandas-readable format
- PVSource block: Depending on the input type selected in the scenario definition csv file  
- Mobile commodity system: csv file containing columns "X_misoc", "X_consumption" and "X_atbase" with X being the name of the commodity. In the case of VehicleCommoditySystem, the column "X_tour_dist" is also required.

#### Model output
<mark>enter description here</mark>

## Requirements

REVOL-E-TION relies heavily on single core computing power for each scenario and especially in 'go' strategy uses a lot of memory. To avoid memory limitations, it is advised to limit the number of parallel scenarios to be executed in the settings file depending on the hardware used. 




