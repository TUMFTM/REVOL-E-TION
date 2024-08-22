# REVOL-E-TION ![System diagram](./images/revol-e-tion_icon.svg)

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

## Description  
REVOL-E-TION is a scalable generator for (mixed integer) linear energy system models of local energy systems with or without electric vehicle fleets.
It can be used to optimize component sizes and/or dispatch behavior of the system to achieve least cost in the simulation timeframe.
Simulation results are later extrapolated and discounted to a project timeframe to estimate the technoeconomic potential of the system in the long run.
Please note that this split between simulation and extrapolation improves computational effort, but creates possibly unwanted incentives for the optimizer (e.g. preferring low initial cost but operationally expensive power sources), especially when sizing components.

REVOL-E-TION groups oemof components and buses into blocks representing real-world systems (e.g. a PV array) for easy application.
Electric vehicles (in fact any mobile storage devices) are modeled individually within a block called a CommoditySystem.
Their behavior (i.e. when they depart and arrive again, how much energy they use in between and whether they can be charged externally) is described in a so called log file.
Log files can be created using the integrated Discrete Event Simulation (DES), which is also capable of modeling range extension through a Battery CommoditySystem as well as multiple use cases in different time frames (e.g. summer/winter) for the commoditites.

The following system diagram shows the basic structure including one example of each block class (blocks are indicated by dashed lines):
![System diagram](./images/system_diagram.png)

## Installation
REVOL-E-TION is designed to run under Windows 10, Ubuntu 22.04 LTS and MacOS 14 Sonoma, each with Python 3.11.
While portability is generally built in, other operating systems are untested.

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
Running REVOL-E-TION requires closely defined scenario(s) to simulate and common simulation settings to operate on as well as a target directory for the results.
The former two are to be given as two .csv-files within the ```./input/scenarios``` and ```./input/settings``` directories respectively when starting a simulation.
To select the files, a Graphical User Interface (GUI) comes up automatically when executing ```main.py``` without any parameters given.
Alternatively, the names of the files can be specified in the terminal execution command as follows, given the base directory is selected:
```
python main.py <scenario_file_name>.csv <settings_file_name>.csv <relative_path_to_result_parent_directory>
```
Example scenario and settings files are provided in the respective directories.
The results will be saved in the specified directory in a subfolder named after the scenario file with a run-time.
Formatting of the scenario and settings files can be taken from the example files and is explained below.
Some parameters in the scenario files reference to other files specified by file name within ```./input/<name of block class>```.
This mostly applies to timeseries data.
Furthermore, to describe the mapping of different timeframes defining behavior of CommoditySystems (see below), modification of the ```mapper_timeframe.py``` code file might be necessary to fit the scenario as this is not simply and flexibly done in parameter files.

## General Terms & Definitions
The following table details common terms occuring in further descriptions and the code:

| Term                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Run                    | A single execution of REVOL-E-TION defined by a single scenario file (possibly containing multiple scenario definitions as columns) and settings file each. Common information and methods valid for all scenarios are defined in a SimulationRun object.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Scenario               | A set of parameters (i.e. an energy system) to be simulated and/or optimized. It is defined by a column in the scenario file and some timeseries inputs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Strategy               | The (energy management) strategy to dispatch the energy system's blocks generally (see CommoditySystem integration levels later on for differences). REVOL-E-TION supports two strategies: "go" for single shot global optimization and "rh" for rolling horizon (i.e. time slotted myopic) optimization similar to Model Predictive Control (MPC). The former is a special case of the latter with just a single horizon. Component size optimization is only applicable in "go".                                                                                                                                                                                                                                                                                                                                                                                               |
| Horizon                | A single optimization process by the solver, whether as the only one of the scenario ("go" strategy) or as a slice of the simulation timeframe ("rh" strategy). In the latter case, the total simulation time is split up into prediction horizons (each of which is represented in the code by a PredictionHorizon object) that are simulated consecutively. However, overlap between the horizons is necessary to ensure feasibilty of dispatch. Therefore, only the first part of the prediction horizon (called control horizon) is actually used for overall result calculation, while the rest is discarded. The simulation timeframe is automatically split up into horizons as per the defined length of prediction and control horizons from the scenario file. See the following diagram for clarification: ![Rolling Horizon principle](./images/rolling_horizon.png) |
| Block                  | A set of oemof components representing a real-world system including necessary converters and buses. Each is represented by an instance of the Block parent class with further child classes. The blocks present in the energy system are defined in the scenario file as a dictionary, except for the core block (of class SystemCore) containing the AC and DC buses as well as the converter(s) inbetween them. Multiple instances of one block can coexist in a model. The possible types (i.e. Classes) are laid out in the following chapter.                                                                                                                                                                                                                                                                                                                              |
| Component              | A component is an oemof element that is either a source, a sink, a bus, a converter or a storage.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## Classes of Blocks
The following table details the classes of blocks that can be specified in the "blocks" parameter of the scenario in the scenario file with respective names for the instances. Block instances cannot be named "scenario". Each instance requires a certain set of parameters dependent on its class. These are specified in the following chapter.

| Class Name              | Description                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| InvestBlock             | Parent class for all blocks that are possibly being sized in optimization (i.e. where an investment decision has to be taken) and for which entering ```'opt'``` as the size in the scenario file will trigger that size optimization. However, this option is only compatible with the "go" strategy. Class contains mostly methods for result evaluation.                                                                                           |
| RenewableInvestBlock    | Parent class for all InvestBlocks representing a renewable source with a curtailment sink component and variable potential power (defined as a fraction of nominal power to enable sizing) over time. Child of class InvestBlock.                                                                                                                                                                                                                     |
| SystemCore              | Class for collection of central energy system components (AC and DC buses and the two unidirectional transformers between them). Child of class Block. This is present once and only once in every energy system defined in REVOL-E-TION under the name "core". For pure AC or DC systems, the respective core cost and size parameters can be set to zero to have no effect on the result.                                                           |
| FixedDemand             | Class for undeferrable (i.e. inflexible) power demand such as households. Child of class Block. Power timeseries is defined in a csv file.                                                                                                                                                                                                                                                                                                            |
| PVSource                | Class for photovoltaic array. Child of class RenewableInvestBlock. Power potential is defined either in a [Solcast](https://solcast.com/) or [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/) formatted csv file (both pre-downloaded) or the [PVGIS API](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en) (requires active internet connection). |
| WindSource              | Class for wind turbine. Child of class RenewableInvestBlock. Power potential is defined either in a csv timeseries file or from PVSource data containing wind speed, which is then converted to power for a specific turbine height. For the latter option, a PVSource block must exist.                                                                                                                                                              |
| ControllableSource      | Class for independently controllable power sources (e.g. fossil generator, hydro power plant) that is unlimited in energy. Child of class InvestBlock.                                                                                                                                                                                                                                                                                                |
| GridConnection          | Class for utility grid and/or energy market connection. Child of class InvestBlock. A GridConnection instance can contain multiple GridMarkets. Definition of GridMarket instances happens in a separate csv file.                                                                                                                                                                                                                                    |
| StationaryEnergyStorage | Class for stationary battery energy storage systems. Child of class InvestBlock. A posteriori aging (i.e. capacity reduction) estimation is possible and will be taken into the next horizon as a reduced available SOC range. Storage modelling is done linearly without SOC or temperature based limits of charge or discharge power.                                                                                                               |
| CommoditySystem         | Parent class for fleets of mobile battery based devices (e.g. electric vehicles or battery rental). Child of class InvestBlock. Behavior is defined in a csv file (the log) that can either be given or generated within the integrated Discrete Event Simulation from stochastic behavioral parameters to be given in a usecase definition csv file for different timeframes defined in ```mapper_timeframe.py``` in code.                           |
| VehicleCommoditySystem  | Class for fleet of electric vehicles, possibly with range extension using a BatteryCommoditySystem (see [publication](https://ieeexplore.ieee.org/document/9905305)). Child of class CommoditySystem.                                                                                                                                                                                                                                                 |
| BatteryCommoditySystem  | Class for fleet of mobile batteries. Child of class CommoditySystem.                                                                                                                                                                                                                                                                                                                                                                                  |

## Scenario Input Parameters
REVOL-E-TION requires multiple types of input data, more specifically settings, parameters and timeseries data. The
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

## Discrete Event Simulation (DES)
RentalSystem           | The mother class for execution of the Discrete Event Simulation (DES) simulating the mobility/energy demand and its allocation to different commodities within a service system.                                                                                                                                                                                                                                                                       |
| RentalProcess          | A single rental of a vehicle or battery manifesting in blocking one or more primary (within the own CommoditySystem) and/or secondary commodities (in a linked CommoditySystem through range extension).



