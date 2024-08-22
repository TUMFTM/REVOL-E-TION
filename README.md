# REVOL-E-TION
### Resilient Electric Vehicle Optimization model for Local Energy TransitION

REVOL-E-TION is an energy system model toolset designed to optimize integration of electric vehicle fleets into 
local energy systems such as mini- and microgrids, company sites, apartment blocks or single homes and estimate
the resulting technoeconomic potentials in terms of costs and revenues within the energy (and optionally also the
mobility system). It is built as a wrapper on top of the [oemof](https://oemof.org) energy system model framework. 

#### Created by 
Philipp Rosner, M.Sc. and Brian Dietermann, M.Sc.  
Institute of Automotive Technology  
Department of Mobility Systems Engineering  
TUM School of Engineering and Design  
Technical University of Munich  
philipp.rosner@tum.de  
September 2nd, 2021

#### Contributors
Marcel Brödel, M.Sc. - Research Associate 01/2024-

David Eickholt, B.Sc. - Semester Thesis submitted 07/2021  
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022  
Hannes Henglein, B.Sc. - Semester Thesis submitted 10/2022  
Marc Alsina Planelles, B.Sc. - Master's Thesis submitted 10/2022  
Juan Forero Yacaman - Bachelor's Thesis submitted 04/2023  
Elisabeth Spiegl - Bachelor's Thesis submitted 06/2023  
Hannes Henglein, B.Sc. - Master's Thesis ongoing  
Alejandro Hernando Armengol, B.Sc. - Master's Thesis submitted 10/2023

## Licensing
REVOL-E-TION is licensed under the <mark> License still to be chosen </mark>.  
The full license text can be found in the LICENSE file in the root directory of the repository.

<mark> Prior to open source publication of the toolset, any distribution outside FTM researchers or their immediately affiliated students is prohibited. </mark> 

## Compatibility
REVOL-E-TION is designed to run under Windows 10, Ubuntu 22.04 LTS and MacOS 14 Sonoma, each with Python 3.11.
While portability is generally built in, other operating systems are untested.

## Installation
#### Step 1: Getting the source code
REVOL-E-TION is available at the institute's [GitHub](https://github.com/TUMFTM/REVOL-E-TION) and can be cloned from there.

#### Step 2: Environment & Packages
REVOL-E-TION is developed and tested with Python 3.11, which is recommended to use with a clean virtual or conda environment to start with.
All required packages are listed in ```requirements.txt``` and can be installed through ```pip install -r requirements.txt```, given the correct environment is active.

#### Step 3: MILP Solver
REVOL-E-TION requires a [pyomo compatible](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers) Mixed Integer Linear Programming (MILP) solver (as does oemof).
The open-source [cbc](https://github.com/coin-or/Cbc/releases/latest) solver works well.
The proprietary [Gurobi](https://www.gurobi.com/downloads/) solver is recommended however, as it is faster in execution, especially for large problems and offers a free academic license.

#### Step 4: Basic Usage
Running REVOL-E-TION requires closely defined scenario(s) to simulate and simulation settings to operate on as well as a target directory for the results.
The former two are to be given as two .csv-files within the ```./input/scenarios``` and ```./input/settings``` directories respectively when starting a simulation.
To define the parameters, a Graphical User Interface (GUI) comes up automatically when executing ```main.py``` without any parameters given.
Alternatively, the names of the files can be specified in the terminal execution command as follows, given the base directory is selected:
```
python main.py <scenario_file_name>.csv <settings_file_name>.csv <relative_path_to_result_parent_directory>
```
Example scenario and settings files are provided in the respective directories.
The results will be saved in the specified directory in a subfolder named after the scenario file with a run-time.
Formatting of the scenario and settings files can be taken from the example files and is explained below.

## Description  
REVOL-E-TION is a scalable generator for (mixed integer) linear energy system models of local energy systems with or without electric vehicle fleets.
It can be used to optimize component sizes and/or dispatch behavior of the system to achieve least cost in the simulation timeframe.
Simulation results are later extrapolated and discounted to a project timeframe to estimate the technoeconomic potential of the system in the long run.
Please note that this split between simulation and extrapolation improves computational effort, but creates possibly unwanted incentives for the optimizer (e.g. preferring low initial cost but operationally expensive power sources), especially when sizing components.

REVOL-E-TION groups oemof components and buses into blocks representing real-world systems (e.g. a PV array) for easy application.
Electric vehicles (in fact any mobile storage devices) are modeled individually within a block called a CommoditySystem.
Their behavior (i.e. when they depart and arrive again, how much energy they use in between and whether they can be charged externally) is described in a so called log file.
Log files can be created using the integrated Discrete Event Simulation (DES), which is also capable of modeling range extension through a Battery CommoditySystem as well as multiple use cases in different time frames (e.g. summer/winter) for the commoditites.

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




