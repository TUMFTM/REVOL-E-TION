# REVOL-E-TION ![Icon](./images/revol-e-tion_icon.svg)

## Resilient Electric Vehicle Optimization model for Local Energy TransitION

REVOL-E-TION is an energy system model toolbox designed to optimize integration of electric vehicle fleets into
local energy systems such as mini- and microgrids, company sites, apartment blocks or single homes and estimate
the resulting technoeconomic potentials in terms of costs and revenues within the energy (and optionally also the
mobility system). It is built as a wrapper on top of the [oemof](https://oemof.org) energy system model framework.

## Created by
Philipp Rosner, M.Sc. and Brian Dietermann, M.Sc.<br>
Institute of Automotive Technology<br>
Department of Mobility Systems Engineering<br>
TUM School of Engineering and Design<br>
Technical University of Munich<br>
philipp.rosner@tum.de<br>
September 2nd, 2021<br>

#### Contributors
Marcel Brödel, M.Sc. - Research Associate 01/2024-<br>
David Eickholt, B.Sc. - Semester Thesis submitted 07/2021<br>
Marcel Brödel, B.Sc. - Semester Thesis submitted 05/2022<br>
Hannes Henglein, B.Sc. - Semester Thesis submitted 10/2022<br>
Marc Alsina Planelles, B.Sc. - Master's Thesis submitted 10/2022<br>
Juan Forero Yacaman - Bachelor's Thesis submitted 04/2023<br>
Elisabeth Spiegl - Bachelor's Thesis submitted 06/2023<br>
Alejandro Hernando Armengol, B.Sc. - Master's Thesis submitted 10/2023<br>
Hannes Henglein, B.Sc. - Master's Thesis submitted 01/2024<br>
Florian Melzig, B.Sc. - Master's Thesis submitted 10/2024<br>
Florian Honeder, B.Sc. - Semester Thesis submitted 09/2025<br>
Jan-Niklas Weghorn, B.Sc. - IDP submitted 09/2025<br>
Jan-Niklas Weghorn, B.Sc. - Master's Thesis submitted 04/2026<br>

## Table of Contents
- [Licensing](#licensing)
- [Related publications](#related-publications)
- [Description](#description)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Common Problems & Troubleshooting](#common-problems--troubleshooting)
- [General Terms & Definitions](#general-terms--definitions)
- [Scenario Input Parameters](#scenario-input-parameters)
- [EV-Specific submodules](#ev-specific-submodules)
  - [Discrete Event Simulation (DES)](#discrete-event-simulation-des)
  - [A priori Power Scheduling](#a-priori-power-scheduling)
  - [A Posteriori Aging Model (also available for StationaryBattery)](#a-posteriori-aging-model-also-available-for-stationarybattery)
- [Outputs](#outputs)

## Licensing
REVOL-E-TION is licensed under the Apache 2.0 open source license.<br>
The full license text can be found in the LICENSE file in the root directory of the repository.

## Related publications
- P. Rosner and M. Lienkamp, "Unlocking the Joint Potential of Electric Mobility and Rural Electrification - A Concept for Improved Integration using Modular Batteries," 2022 IEEE PES/IAS PowerAfrica, Kigali, Rwanda, 2022, pp. 1-5, https://doi.org/10.1109/PowerAfrica53997.2022.9905305.
- P. Rosner, B. Dietermann, M. Brödel and M. Lienkamp, "REVOL-E-TION: A Flexible and Scalable Model to Optimally Integrate Bidirectional EV Fleets in Local Energy Systems", Poster, 2024 Vehicle2Grid Conference, Münster, Germany, 2024, https://doi.org/10.13140/RG.2.2.19632.16648
- P. Rosner, B. Dietermann (shared first authors), M. Brödel, A. Paper and M. Lienkamp, "REVOL-E-TION: A Flexible and Scalable Investment Optimization Toolbox for Local Energy Systems Incorporating Electric Vehicle Fleets", 2025, SoftwareX, https://doi.org/10.1016/j.softx.2025.102178
- B. Dietermann, P. Rosner, N. Nachtigall and M. Lienkamp, "Enabling Residential Electric Car-Sharing Models Through Optimum Local Energy System Integration: A Concept", 2025, 5th International Conference on Electrical, Computer, Communications and Mechatronics Engineering (ICECCME), https://doi.org/10.1109/ICECCME64568.2025.11277639

## Description
REVOL-E-TION is a scalable generator for (mixed integer) linear energy system models of local energy systems with or without electric vehicle fleets.
It can be used to optimize component sizes and/or dispatch behavior of the system to achieve the least cost in the simulation timeframe.
Simulation results are later extrapolated and discounted to a project timeframe to estimate the technoeconomic potential of the system in the long run.
Please note that this split between simulation and extrapolation improves computational effort, but creates possibly unwanted incentives for the optimizer (e.g. preferring low initial cost but operationally expensive power sources), especially when sizing components.

REVOL-E-TION groups oemof components and buses into blocks representing real-world systems (e.g. a PV array) for easy application.
Electric vehicles (in fact, any mobile storage devices) as well as Internal Combustion Engine Vehicles are modeled individually as instances of class FleetUnit within a Fleet block.
Their behavior (i.e. when they depart and arrive again, how much energy they use in between and whether they can be charged externally) is described in a so-called log file.
Log files can be created using the integrated Discrete Event Simulation (DES), which is also capable of modeling range extension through mobile batteries as well as multiple use cases in different time frames (e.g. summer/winter) for the FleetUnits.

The following system diagram shows the basic structure including one example of each block class (blocks are indicated by dashed lines):<br>

<div style="text-align: center;">
  <img src="./images/structure.svg" alt="Structure Diagram" style="width: 100%; max-width: 100%; height: auto; background-color: white;">
</div>

## Installation
REVOL-E-TION is designed to run under Windows 11, Ubuntu 22.04 LTS and macOS 15 Sequoia.
While portability is generally built in, other operating systems are untested.

#### Step 1: Getting the source code
REVOL-E-TION is available on [GitLab](https://gitlab.lrz.de/energysystemmodelling/revol-e-tion) and can be cloned from there using
```bash
git clone https://gitlab.lrz.de/energysystemmodelling/revol-e-tion.git
```

#### Step 2: Install package and dependencies locally
Dependencies are listed in the ```pyproject.toml``` file in the root directory of the repository (where ```README.md``` and ```pyproject.toml``` are located).
There, three different groups are defined:
- The default group contains all dependencies required to run REVOL-E-TION.
- The 'dev' group contains additional dependencies required for development such as code formatting and linting tools.
- The 'tests' group contains additional dependencies required for testing such as pytest and coverage.
- 
Depending on your use case, you can choose to install only the default dependencies (if you just want to run the package) or also the dev and tests dependencies (if you want to develop the package).
This manual describes the installation using three different package managers: uv or conda / pip.

##### a) uv

To install REVOL-E-TION using uv, navigate to the root directory of the repository in your terminal and execute the following command:
```bash
uv sync
```

This command installs the package in editable mode and all dependencies of the default group.
To install all additional dependency groups add ```--all-groups``` to the previous command or select the groups you want to install using ```--group <group_name>```.

##### b) conda / pip

It is recommended to create and activate a clean virtual environment for the installation of REVOL-E-TION.

This can be done using conda:
```bash
conda create -n <name_of_virtual_environment> python=3.12
conda activate <name_of_virtual_environment>
```
or alternatively with a pip virtual environment using the following command:
```bash
python -m venv <path_to_virtual_environment>
source <path_to_virtual_environment>/bin/activate
```

Navigate to package's root directory in your terminal.
Then install the package and its dependencies using one of the following commands depending on the chosen mode of installation:
###### i. Standard Installation
This copies the package into your (virtual environment’s) site-packages directory:
```bash
pip install .
```
After pulling new changes from the repository, the package has to be reinstalled using the same command to take the changes into account.

###### ii. Editable Installation (recommended for development)

This links the package to your local source code, so any changes (you make or pulled from the repository) are immediately reflected without reinstalling:
```bash
pip install -e . --group dev --group tests
```
Use the editable mode if you plan to modify the code during development. The previous command also installs additional dependencies required for development and testing, which are not necessary for running the package but required for development.

#### Step 3: MILP Solver
REVOL-E-TION requires a Mixed Integer Linear Programming (MILP) solver (as does oemof).
Three solvers are supported, selected via ```--solver```: ```cbc```, ```gurobi``` and ```highs```.
The open-source [HiGHS](https://highs.dev/) solver requires no separate installation, as it is installed automatically as part of the package dependencies.
The open-source [cbc](https://github.com/coin-or/Cbc/releases/latest) solver works well, but has to be installed separately.
The proprietary [Gurobi](https://www.gurobi.com/downloads/) solver is recommended however, as it is faster in execution, especially for large problems and offers a free academic license.
If [Gurobi](https://www.gurobi.com/downloads/) is used, the version of Gurobi and the license file have to match. The python package gurobipy is NOT required to run REVOL-E-TION.
To ensure this get the version of both your Gurobi license and installation (```grbgetkey --version```).

## Basic Usage
> ⚠️ **Important** ⚠️
>
> **When using REVOL-E-TION, all input data such as timeseries and the scenario file should be stored in a separate directory and not within the package's source code.**  
> **Do not store any custom files within the source code of the package.**
> **This includes the example directory in particular.**

### 1. Running REVOL-E-TION as package
REVOL-E-TION can be run using one of two terminal commands, given the correct virtual environment is activated:
1. Call to the main module: ```python -m revoletion <arguments>``` (best for local execution on host machine, e.g. through a run configuration in PyCharm)
2. Call to the entry point: ```revoletion <arguments>``` (best for remote execution on a server as it works irrespective of the current working directory as long as the correct environment is active)

<details style="margin-bottom: 1em;">
<summary style="
  border: 2px solid #333333;
  padding: 10px;
  background-color: #f0f0f0;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
    ⚙️
  </span>
  Overview of Arguments
</summary>

| Argument                        | Short form | Long form          | Default value                                                                | Description                                                                                                                                                                                                                        | Valid input                                                                                                                                           |
|---------------------------------|------------|--------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Scenario file path (mandatory)  | -scn       | --scenario         | If not given, a GUI window opens to select the scenario file                 | File path to scenario file. If 'example' is provided, the example project included in REVOL-E-TION is executed and ```--input``` is neglected. If not provided, a graphical selection dialog opens automatically to select a file. | string with path (absolute or relative to current working directory) of scenario file or 'example'                                                    |
| Input directory path            | -in        | --input            | Directory of the scenario file provided in ```--scenario```                  | Directory path to input data files.                                                                                                                                                                                                | string with directory path (absolute or relative to current working directory)                                                                        |
| Output directory path           | -out       | --output           | Directory "results" in the current working directory (created automatically) | Directory path to save output data.                                                                                                                                                                                                | string with directory path (absolute or relative to current working directory)                                                                        |
| Rerun previous run              | -rer       | --rerun            | None                                                                         | Rerun scenarios of a previous run which were not completed successfully (due to unexpected termination of SimulationRun or non-deterministic infeasibilities). Specify a path or 'latest' to rerun latest run in output directory. | False, 'latest' or string with directory path (absolute or relative to output directory defined in ```--output```) containing results of previous run |
| Solver                          | -slv       | --solver           | 'gurobi'                                                                     | Solver to be used for optimization. All three solvers are supported by both backends.                                                                                                                                              | 'cbc', 'gurobi', 'highs'                                                                                                                              |
| Solver optimality tolerance     | -otol      | --optimality_tol   | 1e-9                                                                         | Solver tolerance on reduced costs, i.e. the smallest per-unit objective difference the solver still resolves. Has to stay well below the scenario's ```cost_eps```, otherwise the tie breaking epsilon lies inside the solver's noise floor and circular flows survive in the dispatch. Pass 0 to leave the solver at its own default (coarser than ```cost_eps```, e.g. 1e-6 for Gurobi).                | float, 0 to keep the solver default                                                                                                                   |
| Optimization backend            | -bnd       | --backend          | 'oemof'                                                                      | Framework used to model the optimization problem. The pypsa backend is experimental and does not yet have feature parity with the oemof backend.                                                                                    | 'oemof', 'pypsa'                                                                                                                                      |
| Number of Processes             | -np        | --n_processes      | 1                                                                            | Number of parallel processed (i.e. cores) scenarios.                                                                                                                                                                               | integer, is limited to maximum thread count of CPU automatically                                                                                      |
| Large scale execution mode      | -ls        | --largescalemode   | False                                                                        | Boolean controlling output saving and display detail. Timeseries parameters are omitted in large scale mode.                                                                                                                       | True, False                                                                                                                                           |
| Debugmode                       | -db        | --debugmode        | False                                                                        | Boolean controlling whether to print solver progress information during the solving process. This is resource intensive and should therefore be avoided unless explicitly necessary.                                               | True, False                                                                                                                                           |
| Rerun only infeasible scenarios | -rin       | --rerun_infeasible | True                                                                         | Rerun infeasible scenarios, for which the solver did not find an optimal solution (may lead to the same result again) Neglected for ```--rerun False```                                                                            | True, False                                                                                                                                           |
| Solcast API key                 | -ksc       | --key_solcast_api  | None                                                                         | API key to use for the proprietary Solcast PV/Wind data API                                                                                                                                                                        | string                                                                                                                                                |
</details>

The scenario file is a CSV table.
Its exact API is described in the section "Scenario Input Parameters".
A runnable example scenario file is provided in the example directory.
Some parameters in the scenario file reference to other files specified by file name, mostly for timeseries data.
These are searched within the input directory specified.

Furthermore, to describe the mapping of different timeframes defining behavior of Fleets (see below), modification of the ```mapper_timeframe_example.py``` code file might be necessary to fit the scenario as this is not simply and flexibly done in parameter files.
The filename of the modified file has to be given in the scenario file under the key ```filename_mapper``` and the file has to be placed in the input directory specified in ```--input```.

Concerning computational effort, REVOL-E-TION relies heavily on single core computing power for each scenario and uses significant memory, especially in the 'go' strategy.
To avoid memory limitations, it is advised to limit the number of parallel scenarios to be executed using ```--n_processes``` depending on the available hardware.

To run the provided example project, execute the following command in the terminal:
```bash
cd PATH/TO/REVOL-E-TION_REPOSITORY/example/
revoletion -scn scenarios.csv
```

### 2. Running REVOL-E-TION in Python
REVOL-E-TION can also be used as a module in your own code.
An example notebook executing REVOL-E-TION from Python is provided in ```.revoletion/example/run_example.ipynb```.


## Common Problems & Troubleshooting
| Error message                                                                                                                    | Cause                                                                                                             | Solution                                                                                                                                                                                                                                                                                                                                                                                           |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Errors while installing the required Python packages                                                                             | various                                                                                                           | Make sure to start with a clean Python environment using Python >=3.12. We recommend conda for dependency resolution.                                                                                                                                                                                                                                                                              |
| Gurobi related installation & execution errors                                                                                   | Faulty Gurobi installation, missing Gurobi license                                                                | Check your Gurobi installation by executing ```gurobi-cl``` in a shell. If the command itself fails, Gurobi itself is not installed properly. In this case, follow the [Gurobi Install and Troubleshooting Guide](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer). Make sure that installed versions of ```Gurobi``` and ```gurobipy``` match.  |
| ```IndexError: Block "X": Input timeseries data does not cover simulation timeframe```                                           | A timeseries data file provided does not cover the complete simulation period or resampling it has failed         | Specify a different simulation period or select a different timeseries input file. Make sure the entire simulation timeframe (possibly including overhanging data for the last prediction horizons)                                                                                                                                                                                                |
| ```Class "X" not found in blocks.py file```                                                                                      | The class name specified in the blocks dictionary in the scenario file is not specified in REVOL-E-TION           | Check the blocks dictionary string in the scenario csv file for typos.                                                                                                                                                                                                                                                                                                                             |
| ```Scenario failed: Infeasible or Unbounded (To solve this error try to set investment limits for blocks or for the scenario)``` | Depending on the costs specified the optimization problem might be unbounded as infinite investment is beneficial | Redefine the provided cost structure (i.e. reduce prices for energy feed-in or increase CAPEX or OPEX for energy generation)                                                                                                                                                                                                                                                                       |
| Any other error messages or errors                                                                                               | various                                                                                                           | REVOL-E-TION prints specific error messages in most cases which help you to understand the cause of the error. Nevertheless, REVOL-E-TION is still in development and therefore might contain bugs and errors that are not covered yet. If you encounter a bug or an error message that you do not understand, please feel free to open an issue on GitHub or contact the developers.              |


## General Terms & Definitions
The following table details common terms occurring in further descriptions and the code:

| Term      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Project   | An application of REVOL-E-TION in a specific setting. All files (scenarios file, input data files, etc.) should be contained in a single directory. ```./example``` is an example project.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Run       | A single execution of REVOL-E-TION defined by a single scenario file (possibly containing multiple scenario definitions as columns). Common information and methods valid for all scenarios are defined in a SimulationRun object.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Scenario  | A set of parameters (i.e. an energy system) to be simulated and/or optimized. It is defined by a column in the scenario file and optional some additional timeseries inputs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Strategy  | The (energy management) strategy to dispatch the energy system's blocks generally. REVOL-E-TION supports two strategies: "go" for single shot global optimization and "rh" for rolling horizon (i.e. time slotted myopic) optimization similar to Model Predictive Control (MPC). The former is a special case of the latter with just a single horizon. Component size optimization is only applicable in "go".                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Horizon   | A single optimization process by the solver, whether as the only one of the scenario ("go" strategy) or as a slice of the simulation timeframe ("rh" strategy). In the latter case, the total simulation time is split up into prediction horizons (each of which is represented in the code by a PredictionHorizon object) that are simulated consecutively. However, overlap between the horizons is necessary to ensure feasibility of dispatch. Therefore, only the first part of the prediction horizon (called control horizon) is actually used for overall result calculation, while the rest is discarded. The simulation timeframe is automatically split up into horizons as per the defined length of prediction and control horizons from the scenario file. See the following diagram for clarification: ![Rolling Horizon principle](./images/rolling_horizon.png) |
| Block     | A set of oemof components representing a real-world system including necessary converters and buses. Each is represented by an instance of the Block parent class with further child classes. The blocks present in the energy system are defined in the scenario file as a dictionary, except for the core block (of class SystemCore) containing the AC and DC buses as well as the converter(s) inbetween them. Multiple instances of one block can coexist in a model. The possible types (i.e. Classes) are laid out in the following chapter.                                                                                                                                                                                                                                                                                                                               |
| Component | A component is an oemof element that is either a source, a sink, a bus, a converter or a storage.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

## Scenario Input Parameters
Scenarios are defined in a CSV file, the so-called scenario file.
This file can contain multiple scenarios, each of which is defined by a column in the file.
Each column then holds all required parameters for the scenario, which are structured by blocks, leading to the following formatting of the scenario file:

| Block              | Key               | ```<name_scenario_1>```  | ```<name_scenario_2>```  | ... | ```<name_scenario_n>```  |
|--------------------|-------------------|--------------------------|--------------------------|-----|--------------------------|
| ```<block_name>``` | ```<parameter>``` | ```<value_scenario_1>``` | ```<value_scenario_2>``` | ... | ```<value_scenario_n>``` |
| ...                | ...               | ...                      | ...                      | ... | ...                      |

In the first row, 'Block' and 'Key' are fixed column names. All other column names define the names of distinct scenarios.
To ignore a scenario during the simulation process, add a '#' in front of the scenario's name.
The first column defines the name of the block to which the parameter in the second column applies.
This name is defined in the scenarios ```blocks``` parameter for all blocks except for the scenario itself and the SystemCore which have to be named 'scenario' and 'core', respectively.

Although the scenario itself is no block in REVOL-E-TION's code logic, it is treated as a block in the scenario file.
A scenario file starts with a header row containing the fixed column names 'Block' and 'Key', followed by the names of the scenarios as column names.
The first rows represent the parameters of the scenario itself, which are not block specific.
The block name for the scenario always is 'scenario'.
The 'scenario' section is followed by the parameters for each block present in the scenario.
To ensure scalability of the scenario file, the parameters of a block are only read in, if the block is present in the scenarios parameter ```blocks```.
This allows to define scenarios with different blocks in the same scenario file.
In addition to the blocks defined in the ```blocks``` parameter of the scenario exactly one instance of class SystemCore is automatically created and named 'core'.
Although the SystemCore must not be specified in the ```blocks``` dict, its parameters have to be defined in the scenario file (block name: 'core').
Instances of type GridMarket and SubFleet are also not specified in the scenario's ```blocks``` parameter, but in the parameters of their parent classes GridConnection and Fleet, respectively.

Complex (multidimensional) parameters are mostly defined through links to other files (by filename) in the scenario file.
As all string values specified in the scenario definition file are converted to lower case, all files have to be named in lower case to be read in properly.
The following table specifies each parameter for each possible block class in the scenario file.
If and only if a block of a certain class exists within the scenario, these parameters are required and read in.
Therefore, not every scenario file contains all possible parameters.
An example scenario file is provided in the ```.revoletion/example``` directory.

A further explanation of each block and its parameters is given in the expandable boxes below.


<!ENTRY_POINT_SCHEMA_TABLES>

<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
    📋
  </span>
  Scenario
</summary>

Each run can contain multiple scenario objects.
A scenario object holds several parameters that are used by multiple blocks in the scenario.
After a successful optimization it also contains the aggregated techno-economic results such as energy throughput, costs, revenues as well as LCOE, NPC, and NPV.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `context` | Context |  |  |  |  |
| `simulation_config` | Simulation Config |  |  |  |  |
| `system_core` | System Core |  |  |  |  |
| `blocks` | Blocks | dict |  | All blocks present in the scenario except for the SystemCore, which is added automatically, in the format {block_name: class_name}. Non valid names are 'run', 'scenario' and 'core' (default name for block of class SystemCore). | "{'custom block name': 'class name of block'}" |
| `block_configs` | Block Configurations | dict |  | All blocks present in the scenario except for the SystemCore, which is added automatically, in the format {block_name: class_name}. Non valid names are 'run', 'scenario' and 'core' (default name for block of class SystemCore). |  |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
    ⇄
  </span>
  SystemCore
</summary>

Collection of central energy system components (AC and DC buses and the two unidirectional transformers between them).
This is present once and only once in every energy system defined in REVOL-E-TION under the name "core".
For pure AC or DC systems, the respective core cost and size parameters can be set to zero to have no effect on the result.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `id` | System Core ID | str |  |  |  |
| `size_acdc_preexisting` | Preexisting AC/DC size | float or str |  | Installed power of the AC/DC converter in the SystemCore in W. Set either size_acdc_preexisting or size_dcac_preexisting to 'equal' to set both preexisting converter sizes to the same value. | [0, inf[ or 'equal' |
| `capex_acdc_preexisting` | Consideration of preexisting AC/DC size in capex | bool |  | Trigger whether to consider preexisting component size specified in size_acdc_preexisting in initial capex calculation. Replacement capex are unaffected. | True, False |
| `size_acdc_max` | Maximum size of AC/DC converter | float or str or None |  | Maximum size of the AC/DC converter of the SystemCore including preexisting size specified in size_acdc_preexisting. To enable unlimited investment set this parameter to None. Set either size_acdc_max or size_dcac_max to 'equal' to set both converters' maximum investments to the same value. | [0, inf[ or None or 'equal' |
| `invest_acdc` | Investment into AC/DC converter | bool or str |  | Enable additional investment into the AC/DC converter of the SystemCore. Set either invest_acdc or invest_dcac to 'equal' to force the same expansion for both converters. | True, False |
| `size_dcac_preexisting` | Existing DC/AC size | float or str |  | Installed power of the DC/AC converter in the SystemCore in W. Set either size_acdc_preexisting or size_dcac_preexisting to 'equal' to set both preexisting converter sizes to the same value. | [0, inf[ or 'equal' |
| `capex_dcac_preexisting` | Consideration of preexisting DC/AC size in capex | bool |  | Consider existing DC/AC size in initial capex calculation. | True, False |
| `size_dcac_max` | Maximum size of DC/AC converter | float or str or None |  | Maximum size of the DC/AC converter of the SystemCore including preexisting size specified in `size_dcac_preexisting`. To enable unlimited investment set this parameter to None. Set either `size_acdc_max` or `size_dcac_max` to 'equal' to set both converters' maximum investments to the same value. | [0, inf[ or None or 'equal' |
| `invest_dcac` | Investment into DC/AC converter | bool or str |  | Enable additional investment into the DC/AC converter of the `SystemCore`. Set either `invest_acdc` or `invest_dcac` to 'equal' to force the same expansion for both converters. | True, False |
| `capex_spec` | Specific capital expenditures | float |  | Specific capital expenditures for each of the converters in the `SystemCore`: cost in currency per installed power (cumulative size of both converters) in W. | [0, inf[ |
| `mntex_spec` | Specific maintenance expenditures | float |  | Specific maintenance expenditures for each of the converters in the `SystemCore`: cost in currency per year per installed power (cumulative size of both converters) in W. | [0, inf[ |
| `opex_spec` | Specific operational expenditures | float |  | Specific operational expenditures for each of the converters in the `SystemCore` cost in currency per converted energy in Wh. Energy is measured at each converter's inflow. Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or [0, inf[ |
| `opex_spec_deficit` | Specific operational expenditures of the deficit sources | float |  | Specific operational expenditures of the unlimited deficit sources on the AC and DC bus of the `SystemCore`: cost in currency per drawn energy in Wh. The deficit sources keep the energy system solvable if no other component can cover the demand; a warning is raised whenever energy is drawn from them. These costs only penalize the use of the deficit sources within the optimization and are not part of the economic results. Optional: if not given, it defaults to 1, which is far above any realistic energy price and therefore keeps the deficit sources the optimizer's last resort. Can be given as float or filename of a csv file containing a timeseries. | string with filename or [0, inf[ |
| `ls` | Lifespan | float |  | Lifespan of the block in years after which it will be replaced. | [1, inf[ |
| `ccr` | Cost change ratio | float |  | Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan. | [0, inf[ |
| `eff_dcac` | DC/AC efficiency | float |  | Efficiency of the DC/AC converter in the `SystemCore`. | [0, 1] |
| `eff_acdc` | AC/DC efficiency | float |  | Efficiency of the AC/DC converter in the `SystemCore`. | [0, 1] |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
    🏠
  </span>
  FixedDemand
</summary>

Undeferrable (i.e. inflexible) power demand such as households.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `capex_metering_preexisting` | Consideration of preexisting metering capital expensditures | bool |  | Consider existing metering and operational capex in cost calculation. | True, False |
| `capex_fix_metering` | Fixed capital expenditures for metering infrastructure | float |  | Fixed maintenance expenditures: total cost in currency per year, irrespective of actual demand | [0.0, inf[ |
| `mntex_fix_metering` | Fixed maintenance expenditures for metering infrastructure and operations | float |  | Fixed maintenance expenditures: total cost in currency per year, irrespective of actual demand | [0.0, inf[ |
| `load_profile` | Load Profile | str |  | Load profile for the fixed demand. Can be given as reference to a TimeSeries containing the fixed demand of the block, as a constant string load, or as one of the standard load profiles by BDEW. Standard profiles include 'const', 'H0', 'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'L0', 'L1', 'L2', 'H25', 'G25', 'L25', 'P25', 'S25'. | Ref[TimeSeries], {'const', 'H0', 'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'L0', 'L1', 'L2', 'H25', 'G25', 'L25', 'P25', 'S25'} |
| `consumption_yrl` | Yearly consumption | float |  | Yearly consumption in Wh. Neglected if a filename is provided in load_profile. | [0, inf[ |
| `crev_spec` | Specific customer revenue | float |  | Specific customer revenue for consumed energy in currency per Wh. Can be given as float or file of a csv file containing a timeseries. | Ref[TimeSeries] or float |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
    ☀️
  </span>
  PVSource
</summary>

Photovoltaic array. Power potential is defined either using Solcast (pre-downloaded CSV file or API), PVGIS (pre-downloaded CSV file or API), or a timeseries CSV file.
Although the Solcast API requires an active subscription plan, there is a limited free plan for researchers.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `size_block_preexisting` | Preexisting size | float |  | Installed peak power of the pv array in in W. | [0, inf[ |
| `capex_block_preexisting` | Consideration of preexisting block size in capex | bool |  | Trigger whether to consider preexisting component size specified in `size_block_preexisting` in initial capex calculation. Replacement capex are unaffected. | True, False |
| `size_block_max` | Maximum size | float or None |  | Maximum size of `PVSource` including preexisting size specified in `size_block_preexisting`. To enable unlimited investment set this parameter to None. | [0, inf[ or None |
| `invest_block` | Investment | bool |  | Enable additional investment into the PV system. | True, False |
| `data_source` | Data source | str |  | Data source for pv power. This can be an API (PVGIS or Solcast) or a file containing data (PVGIS, Solcast or custom file). If Solcast API is chosen a valid API key has to specified in the run's arguments. A custom file has to include the columns 'time' (timezone aware timestamps), 'power_spec' (specific power in W per Wp), 'speed_wind' (in m/s), 'temp_air' (air temperature in °C). | 'pvgis api', 'solcast api', 'pvgis file', 'solcast file', 'file' |
| `filename` | Filename | None |  | Name of a PVGIS, Solcast, or custom csv file if data_source is set to 'pvgis file', 'solcast file', or 'file', respectively. Otherwise set to None. | Ref[TimeSeries] or None |
| `capex_spec` | Specific capital expenditures | float |  | Specific capital expenditures: cost in currency per installed peak power in W | [0, inf[ |
| `mntex_spec` | Specific maintenance expenditures | float |  | Specific maintenance expenditures: cost in currency per year per installed peak power in W | [0, inf[ |
| `opex_spec` | Specific operational expenditures | float |  | Specific operational expenditures: cost in currency per generated energy in Wh. Can be given as float or filename of a csv file containing a timeseries | Ref[TimeSeries] or [0, inf[ |
| `ls` | Lifespan | float |  | Lifespan of the block in years after which it will be replaced | [1, inf[ |
| `ccr` | Cost change ratio | float |  | Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan | [0, inf[ |
| `eff_block` | Efficiency | float |  | Efficiency of the PV array, taking into account all losses occurring from insulation up to the point of feeding power into the bus to which the block is connected. | [0, 1] |
| `azimuth` | Surface azimuth | float or None |  | Clockwise from north (north=0, east=90, south=180, west=270). Ignored for tracking systems. Only considered if any API or 'Solcast file' is specified in data_source. None is equal to energy yield optimum. | [0, 360[ or None. |
| `tilt` | Surface tilt angle | float or None |  | Tilt angle from horizontal plane. Ignored for two-axis tracking. Horizontal=0, Vertical=90. Only considered if any API or 'Solcast file' is specified in data_source. None sets the tilt angle to the specified location's latitude. | [0, 90] or None |
| `trackingtype` | Tracking type | int or None |  | Type of sun tracking. 0=fixed, 1=single horizontal axis aligned north-south, 2=two-axis tracking, 3=vertical axis tracking, 4=single horizontal axis aligned east-west, 5=single inclined axis aligned north-south. For data_source 'Solcast API' only 0 and 1 are valid. Ignored for any other data_source than 'PVGIS API' and 'Solcast API'. | 0, 1, 2, 3, 4, 5 |
| `horizon_custom` | User horizon | list or None |  | Optional user specified elevation of horizon in degrees for 'PVGIS API', at equally spaced angular positions starting clockwise from north. Only valid if horizon is True. Not possible in combination with activated azimuth or tilt set to 'optimal'. Ignored for any other data_source than 'PVGIS API'. | list of floats (has to be specified surrounded by " ") e.g. "[45, 30, 0, 0]" or None |
| `database` | Radiation database | str or None |  | Name of the radiation database for 'PVGIS-API'. Dependent on location and chosen simulation timeframe. 'PVGIS-SARAH' for Europe, Africa and Asia or 'PVGIS-NSRDB' for the Americas between 60°N and 20°S, 'PVGIS-ERA5' and 'PVGIS-COSMO' for Europe (including high-latitudes), and 'PVGIS-CMSAF' for Europe and Africa (will be deprecated). | 'PVGIS-SARAH2', 'PVGIS-SARAH3', 'PVGIS-NSRDB', 'PVGIS-ERA5', 'PVGIS-COSMO', 'PVGIS-CMSAF' |
| `type_cell` | PV technology | str |  | PV technology for 'PVGIS API'. | 'crystSi', 'CIS', 'CdTe', 'Unknown' |
| `mountingplace` | Mounting place | str |  | Type of mounting for PV system for 'PVGIS API'. Options: free = free-standing, building = building-integrated. | 'free', 'building' |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
    💨
  </span>
  WindSource
</summary>

Wind turbine. Power potential is defined either in a csv timeseries file or retrieved from PVSource data containing wind speed, which is then converted to power for a specific turbine height.
For the latter option, a PVSource block must exist.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `size_block_preexisting` | Preexisting size | float |  | Installed rated power of wind turbine in W | [0, inf[ |
| `capex_block_preexisting` | Consideration of preexisting block size in capex | bool |  | Trigger whether to consider existing component size in initial capex calculation. Replacement capex are unaffected. | True, False |
| `size_block_max` | Maximum size | float or None |  | Maximum size of WindSource including existing size. To enable unlimited investment set this parameter to None | [0, inf[ or None |
| `invest_block` | Investment | bool |  | Enable additional investment into the wind turbine | True, False |
| `data_source` | Data source | str or None |  | Data source for wind power. Wind power can either be given as a separate csv file or calculated from a PVSource block's data. | Ref[TimeSeries] or a string with the name of a block of class PVSource |
| `height` | Height | float |  | Hub height of the wind turbine in meters | [0, inf[ |
| `filename` | Filename | None |  | Filename of csv file containing wind power data including the columns 'time' (timezone aware timestamps) and 'power_spec' (specific power in W per rated power in W). Only considered if 'file' is given in data_source. | Ref[TimeSeries] or None |
| `capex_spec` | Specific capital expenditures | float |  | Specific capital expenditures: cost in currency per installed rated power in W | [0, inf[ |
| `mntex_spec` | Specific maintenance expenditures | float |  | Specific maintenance expenditures: cost in currency per year per installed rated power in W | [0, inf[ |
| `opex_spec` | Specific operational expenditures | float |  | Specific operational expenditures: cost in currency per generated energy in Wh. Can be given as float or filename of a csv file containing a timeseries. | Ref[TimeSeries] or float |
| `ls` | Lifespan | float |  | Lifespan of the block in years after which it will be replaced | [1, inf[ |
| `ccr` | Cost change ratio | float |  | Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan. | [0, inf[ |
| `eff_block` | Efficiency | float |  | Efficiency of the wind turbine. | [0, 1] |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
     🔩
  </span>
  ControllableSource
</summary>

Independently controllable power sources (e.g. fossil generator, hydro power plant) that is unlimited in energy.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `size_block_preexisting` | Preexisting size | float |  | Installed rated power of the source in W | [0, inf[ |
| `capex_block_preexisting` | Consideration of preexisting block size in capex | bool |  | Trigger whether to consider preexisting component size specified in `size_block_preexisting` in initial capex calculation. Replacement capex are unaffected. | True, False |
| `size_block_max` | Maximum size | float or None |  | Maximum size of ControllableSource including preexisting size specified in size_block_preexisting. To enable unlimited investment set this parameter to None. | [0, inf[ or None |
| `invest_block` | Investment | bool |  | Enable additional investment into the power source | True, False |
| `capex_spec` | Specific capital expenditures | float |  | Specific capital expenditures: cost in currency per installed power in W | [0, inf[ |
| `mntex_spec` | Specific maintenance expenditures | float |  | Specific maintenance expenditures: cost in currency per year per installed peak power in W | [0, inf[ |
| `opex_spec` | Specific operational expenditures | float |  | Specific operational expenditures: cost in currency per generated energy in Wh. Can be given as float or filename of a csv file containing a timeseries | Ref[TimeSeries] or [0, inf[ |
| `ls` | Lifespan | float |  | Lifespan of the block in years after which it will be replaced | [1, inf[ |
| `ccr` | Cost change ratio | float |  | Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan | [0, inf[ |
| `eff_block` | Efficiency | float |  | Efficiency of the source | [0, 1] |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
     🔌⚡
  </span>
  GridConnection
</summary>

Physical grid connection. A GridConnection instance requires one or multiple GridMarkets.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `size_g2s_preexisting` | Preexisting connection power from public grid to local site | float or str |  | Installed power for the power flow from the public grid to the local site (Grid2Site) in W. Set either `size_g2s_preexisting` or `size_s2g_preexisting` to 'equal' to set both directions' sizes to the same value. | [0, inf[ or 'equal' |
| `capex_g2s_preexisting` | Consideration of preexisting AC/DC size in capex | bool |  | Trigger whether to consider preexisting component size specified in size_g2s_preexisting in initial capex calculation. Replacement capex are unaffected. | True, False |
| `size_g2s_max` | Maximum size of Grid2Site | float or str or None |  | Maximum size of Grid2Site including preexisting size specified in size_g2s_preexisting. To enable unlimited investment set this parameter to None. Set either size_g2s_max or size_s2g_max to 'equal' to set both directions' maximum investments to the same value. | [0, inf[ or None or 'equal' |
| `invest_g2s` | Investment into Grid2Site | bool or str |  | Enable additional investment into the maximum power from the grid to the local site. To ensure the same additional power for both directions set one invest variable to 'equal'. | True, False |
| `size_s2g_preexisting` | Existing maximum power from local site to grid | float or str |  | Installed power for the power flow from the local site to the grid in W. To set both directions' existing powers to the same value set one size to 'equal'. | [0, inf[ or 'equal' |
| `capex_s2g_preexisting` | Consider existing block size in capex | bool |  | Trigger whether to consider existing component size in initial capex calculation. Replacement capex are unaffected. | True, False |
| `size_s2g_max` | Maximum size of Site2Grid | float or str or None |  | Maximum size of Site2Grid including existing size. To enable unlimited investment set this parameter to None. To set both directions' maximum investments to the same value set one maximum investment to 'equal'. | [0, inf[ or None or 'equal' |
| `invest_s2g` | Investment into Site2Grid | bool or str |  | Enable additional investment into the maximum power from the local site to the grid. To ensure the same additional power for both directions set one invest variable to 'equal'. | True, False |
| `peakshaving` | Activation of peak shaving | bool |  | Trigger whether to consider peak power costs in the optimization (leads to peak shaving). Peak power costs will always be considered in the post-processing regardless the parameter specified here. | True, False |
| `peak_period` | Peak power cost period | str |  | Peak power cost period. | 'day', 'week', 'month', 'quarter', 'year' |
| `peak_period_start` | Peak power cost period start | str |  | Start of the peak power periods. If 'calendar' is chosen, peak periods start at the beginning of the calendar period (e.g. at 01/01 for yearly peak periods). If 'simulation' is chosen, the first peak period starts at the simulation start time. | 'calendar', 'simulation' |
| `peak_period_measurement` | Peak power measurement period | str |  | Measurement period for the peak power. To determine the peak power the mean power of this measurement period is used. | Formats compatible with pd.to_timedelta() such as 15min, 1h, 1D. |
| `peak_power_init` | Initial peak power | float |  | Initial peak power per peak power period in W. Can be used in Rolling Horizon simulations to avoid overly reduced power consumption from the grid in first horizons of a peak period. | [0, inf[ |
| `opex_spec_peak` | Specific operational expenditures for peak power | float |  | Specific operational expenditures for maximum power drawn from the public grid per timestep in cost in currency per peak power in W per peak power period specified in peak_period. Resulting costs are always considered in post-processing, but are only taken into account by the optimizer, if peakshaving is set to 'True'. | [0, inf[ |
| `capex_spec` | Specific capital expenditures | float |  | Specific capital expenditures: cost in currency per installed power (cumulative power of both directions) in W of the grid connection. | [0, inf[ |
| `mntex_spec` | Specific maintenance expenditures | float |  | Specific maintenance expenditures: cost in currency per year per installed power (cumulative power of both directions) in W of the grid connection. | [0, inf[ |
| `ls` | Lifespan | float |  | Lifespan of the block in years after which it will be replaced. | [1, inf[ |
| `ccr` | Cost change ratio | float |  | Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan. | [0, inf[ |
| `eff_block` | Efficiency | float |  | Efficiency of the grid connection. | [0, 1] |
| `markets` | Markets | list |  | List containing names of GridMarket instances which are connected to the GridConnection. | "['name_of_market1', 'name_of_market2']" |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
     📈
  </span>
  GridMarket
</summary>

Virtual GridMarket connected to a specific physical GridConnection.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `res_only` | Renewable energy sources only | bool |  | If activated, selling energy to the grid is restricted to energy produced by renewable energies blocks (PVSource, WindSource) in the current timestep and energy stored in a storage with activated res_only parameter. | True, False |
| `opex_spec_g2s` | Specific operational expenditures for public grid to local site | float |  | Specific operational expenditures for buying energy: cost in currency per energy in Wh. Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or ]-inf, inf[ |
| `opex_spec_s2g` | Specific operational expenditures for local site to public grid | float |  | Specific operational expenditures for selling energy: cost in currency per energy in Wh (set this parameter to a negative number to earn money for feeding in energy). Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or ]-inf, inf[ |
| `pwr_s2g` | Power limit from public grid to local site | float or None |  | Power limit considered for the power flow from the public grid to the local site in W. If no additional limit for the market but only the limits of the physical grid connection should be taken into account, set to None. | [0, inf[ or None |
| `pwr_g2s` | Power limit from public grid to local site | float or None |  | Power limit considered for the power flow from the local site to the public grid in W. If no additional limit for the market but only the limits of the physical grid connection should be taken into account, set to None. | [0, inf[ or None |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
     🔋
  </span>
  StationaryBattery
</summary>

Stationary battery energy storage systems.
A posteriori aging (i.e. capacity reduction) estimation is possible and will be taken into the next horizon as a reduced available SOC range.
Storage modelling is done linearly without SOC or temperature based limits of charge or discharge power.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `size_storage_preexisting` | Preexisting size | float |  | Installed nominal capacity of the storage in Wh. | [0, inf[ |
| `capex_storage_preexisting` | Consideration of preexisting block size in capex | bool |  | Trigger whether to consider preexisting component size specified in size_storage_preexisting in initial capex calculation. Replacement capex are unaffected. | True, False |
| `size_storage_max` | Maximum size | float or None |  | Maximum size of StationaryBattery including preexisting size specified in size_storage_preexisting. To enable unlimited investment set this parameter to None. | [0, inf[ or None |
| `invest_storage` | Investment | bool |  | Enable additional investment into the storage capacity. | True, False |
| `res_only` | Renewable energy sources only | bool |  | If activated, only energy from renewable sources (PVSource, WindSource) can be stored in the storage. This allows to feed energy from the storage into GridMarket instances with activated res_only parameter. | True, False |
| `balanced` | Balanced Storage Content | bool |  | If activated, the storage's energy content at the start of the simulation has to be identical to the energy content at the end of the simulation. The parameter is neglected for Rolling Horizon optimization. | True, False |
| `aging` | Consideration of battery aging | bool |  | Battery aging calculation after each horizon. Aging results are taken into account for the next horizon by limiting the available SOC range. Maximum power is not reduced. | True, False |
| `chemistry` | Cell Chemistry | str |  | Cell chemistry of the storage to select the correct aging model for aging calculation. | 'nmc', 'lfp' |
| `temp_battery` | Battery temperature | float or None |  | Battery temperature used as stress factor in aging model. Can be set to a constant value, defined using the timeseries of a PVSource block as this contains a temperature timeseries, or set to None to inherit the temperature specified in temp_air of the Scenario. | string with name of block of class PVSourceModel or ]-inf, inf[ |
| `capex_spec` | Specific capital expenditures | float |  | Specific capital expenditures: cost in currency per installed nominal storage capacity in Wh. | [0, inf[ |
| `mntex_spec` | Specific maintenance expenditures | float |  | Specific maintenance expenditures: cost in currency per year per installed nominal storage capacity in Wh. | [0, inf[ |
| `opex_spec` | Specific operational expenditures | float |  | Specific operational expenditures: cost in currency per energy stored in the storage in Wh. Energy is measured at storage inflow. Can be given as float or filename of a csv file containing a timeseries. | Ref[TimeSeries] or [0, inf[ |
| `ls` | Lifespan | float |  | Lifespan of the block in years after which it will be replaced. | [1, inf[ |
| `eff_storage_roundtrip` | Roundtrip efficiency | float |  | Storage roundtrip efficiency. Charge and discharge efficiency is calculated using sqrt(eff_roundtrip). | [0, 1] |
| `eff_acdc` | Efficiency of the AC/DC converter | float |  | Efficiency of the AC/DC converter connecting the block to the AC bus. This parameter is neglected if the block is connected to the DC bus. | [0, 1] |
| `eff_dcac` | Efficiency of the DC/AC converter | float |  | Efficiency of the DC/AC converter connecting the block to the AC bus. This parameter is neglected if the block is connected to the DC bus. | [0, 1] |
| `crate_chg` | Charge C-rate | float |  | Maximum C-rate for charging. | [0, inf[ |
| `crate_dis` | Discharge C-rate | float |  | Maximum C-rate for discharging. | [0, inf[ |
| `soc_init` | Initial SOC | float |  | Initial SOC of the storage at simulation start. | [0, 1] |
| `soc_min` | Minimum SOC | float or None |  | Lower limit of the usable SOC window of the storage. Set to None to use the full window (0). The lower limit imposed by aging is applied on top: the higher of both limits is used. | [0, 1] or None |
| `soc_max` | Maximum SOC | float or None |  | Upper limit of the usable SOC window of the storage. Set to None to use the full window (1). The upper limit imposed by aging is applied on top: the lower of both limits is used. | [0, 1] or None |
| `q_loss_cal_init` | Initial capacity loss due to calendric aging | float |  | Initial capacity loss of the storage at simulation start due to calendric aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init). | [0, 1] |
| `q_loss_cyc_init` | Initial capacity loss due to cyclic aging | float |  | Initial cyclic loss of the storage at simulation start due to calendric aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init). | [0, 1] |
| `sdr` | Self discharge rate | float |  | Self discharge rate of storage components per month (30 days). | [0, inf[ |
| `ccr` | Cost change ratio | float |  | Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan. | [0, inf[ |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
     🚚🚗
  </span>
  Fleet
</summary>

Fleet consisting of one or several SubFleets.

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `data_source` | Data source | str |  | Define whether the fleet's usage should be (a) generated through mobility and dispatch simulation when given a usecase file, (b) generated through dispatch simulation only given a demand file, (c) read directly from a time based log file, forgoing a priori simulations, or (d) replayed from an event file written by a previous run, which forgoes the dispatch simulation and reproduces its result exactly. Fleets linked through range extension have to use the same data source, as their dispatch cannot be resolved independently. | 'usecases', 'demand', 'log', 'events' |
| `filename` | Filename of input file | None |  | Filename of the file containing (a) usecase definition for DES, (b) sampled demand, (c) a time based log, or (d) an event table, according to ```data_source```. Base search path is the scenario file's path, unless explicitly specified. | string with filename or None |
| `filename_mapper` | Filename of TimeframeMapper file | None |  | Filename of the file containing the mapping function assigning timeframes to individual days (e.g. weekday/weekend) for the Group's DES with or without the ending '.py'. The file itself has to be placed in the input directory. Base search path is the scenario file's path, unless explicitly specified. | Ref[TimeSeries] of python file with or without '.py' |
| `pwr_lim_f2s` | Power limit of fleet to site | float or str or None |  | Maximum power flow from Fleet to the local site (Fleet2Site) in W. To enable unlimited power flow set this parameter to None. Set to 'equal' to use the same value as pwr_lim_s2f. | [0, inf[ or None or 'equal' |
| `pwr_lim_s2f` | Power limit of site to fleet | float or None |  | Maximum power flow from the local site to Fleet (Site2Fleet) in W. To enable unlimited power flow set this parameter to None. | [0, inf[ or None |
| `opex_spec_f2s` | Specific operational expenditures for fleet charging | float |  | Specific operational expenditures for Fleet charging: cost in currency per energy charged into Fleet in Wh. This can be used to simulate different operators for fleets and local energy grid. Negative costs can lead to unwanted behavior (e.g. wasting energy)! Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or [0, inf[ |
| `opex_spec_s2f` | Specific operational expenditures for fleet discharging | float |  | Specific operational expenditures for Fleet discharging: cost in currency per energy discharged from Fleet in Wh. This can be used to simulate different operators for fleets and local energy grid. Negative costs can lead to unwanted behavior (e.g. wasting energy)! Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or [0, inf[ |
| `subfleets` | Subfleets | list |  | List of names of subfleets in Fleet in no particular order. Each of these subfleets must exist as such in the scenario file. |  |

</details>


<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
     🚗🚗
  </span>
  SubFleet
</summary>

SubFleet consisting of initially identical FleetUnits (Electric Vehicle, Internal Combustion Engine Vehicle, Mobile Battery).
Behavior can either be given or generated within the integrated Discrete Event Simulation from stochastic behavioral parameters (use case definition in CSV file and python script with timeframe mapper)

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
| `name` | Block name | str |  | Name of the block |  |
| `id` | Block ID | str |  | Unique ID of the block. Only used internally. |  |
| `enabled` | Block enabled | bool |  | Whether the block is enabled or not. | True, False |
| `bus` | Bus | str |  | The bus (AC or DC) the block is connected to. | 'ac', 'dc' |
| `revoletion_model` | Revoletion model | str |  | The revoletion model schema the block is connected to. This is only used internally. |  |
| `num` | Number of fleet units | int |  | Number of fleet units within the SubFleet. | [1, inf[ |
| `type_unit` | Fleet unit type | str |  | Type of Fleet units contained in the Subfleet. 'ev': Electric Vehicle, 'icev': Internal Combustion Engine Vehicle, 'mb': Mobile Battery | 'ev', 'icev', 'mb' |
| `size_storage_preexisting` | Preexisting size of storage | float | `type_unit` == 'icev' | Installed nominal capacity of the Fleet unit's storage in Wh per single Fleet unit for all Fleet units within the Subfleet. Parameter is neglected if type_unit is set to 'icev' | [0.0, inf[ |
| `size_storage_max` | Maximum size of storage | float or None | `type_unit` == 'icev' | Maximum size of storage per Fleet unit including preexisting size specified in size_storage_preexisting. To enable unlimited investment set this parameter to None | [0, inf[ or None |
| `invest_storage` | Investment into storage | bool | `type_unit` == 'icev' | Enable additional investment into the Fleet units' storages | True, False |
| `capex_storage_preexisting` | Consideration of preexisting block size in capex | bool |  | Trigger whether to consider preexisting component size specified in size_storage_preexisting in initial capex calculation. Replacement capex are unaffected. | True, False |
| `capex_fix_glider` | Capital expenditures for base vehicle | float |  | Fixed capital expenditures for each of the Fleet units in the SubFleet, irrespective of storage size, representing the base vehicle | [0.0, inf[ |
| `capex_glider_preexisting` | Consideration of preexisting glider in capex | bool |  | Trigger whether to consider preexisting glider capex specified in capex_fix_glider in initial capex calculation. Replacement capex are unaffected. | True, False |
| `capex_fix_charger` | Capital expenditures for charger | float | `type_unit` == 'icev' | Fixed capital expenditures for each of the Fleet units in the SubFleet, irrespective of storage size, representing the charger | [0.0, inf[ |
| `capex_charger_preexisting` | Consideration of preexisting charger in capex | bool | `type_unit` == 'icev' | rigger whether to consider preexisting charger capex specified in capex_fix_charger in initial capex calculation. Replacement capex are unaffected. | True, False |
| `ccr` | Cost change ratio | float |  | Cost change ratio of the block's (glider, storage, and charger) nominal price per year to be considered for replacement after its lifespan | [0.0, 1.0] |
| `ls` | Lifespan | float |  | Lifespan of the block (glider, storage, and charger) in years after which it will be replaced | [1.0, inf[ |
| `mntex_spec` | Specific maintenance expenditures | float |  | Specific maintenance expenditures: cost in currency per year per installed power (cumulative power of both directions) in W of the grid connection. | [0, inf[ |
| `mntex_fix_glider` | Fixed maintenance expenditures for glider | float |  | Fixed maintenance expenditures: cost in currency per year per Fleet unit, irrespective of traction battery size | [0.0, inf[ |
| `opex_spec` | Specific operational expenditures | float |  | Specific operational expenditures for each of the converters in the `SystemCore` cost in currency per converted energy in Wh. Energy is measured at each converter's inflow. Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or [0, inf[ |
| `opex_spec_dist` | Specific operational expenditures per distance | float |  | Specific operational expenditures per distance: cost in currency per driven distance in km | [0.0, inf[ |
| `crev_spec_time` | Specific customer revenues per time | float |  | Specific customer revenues per time: Revenues from vehicle utilization specified as revenue in currency per used time in hours. Total revenue is calculated by summing up time, distance, and energy revenue. Only rentals to external customers are invoiced: serving another SubFleet as a range extender is an internal service of the same operator and yields no revenue, while its costs still accrue. | [0.0, inf[ |
| `crev_spec_dist` | Specific customer revenues per distance | float |  | Specific customer revenues per distance: Revenues from vehicle utilization specified as revenue in currency per driven distance in km. Total revenue is calculated by summing up time, distance, and energy revenue. Only rentals to external customers are invoiced, see ```crev_spec_time```. | [0.0, inf[ |
| `crev_spec_energy` | Specific customer revenues per energy | float |  | Specific customer revenues per energy: Revenues from vehicle utilization specified as revenue in currency per energy consumed by the customer during the rental in Wh. Total revenue is calculated by summing up time, distance, and energy revenue. Only rentals to external customers are invoiced, see ```crev_spec_time```. | [0.0, inf[ |
| `opex_spec_ext_ac` | Specific operational expenditures for external AC charging | float | `type_unit` == 'icev' | Specific operational expenditures for external AC charging: cost in currency per charged energy in Wh. Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or [0, inf[ |
| `opex_spec_ext_dc` | Specific operational expenditures for external DC charging | float | `type_unit` == 'icev' | Specific operational expenditures for external DC charging: cost in currency per charged energy in Wh. Can be given as float or reference to a TimeSeries. | Ref[TimeSeries] or [0, inf[ |
| `capex_spec` | Specific capital expenditures for storage | float |  | Specific capital expenditures for Fleet unit's storages: cost in currency per installed nominal storage capacity in Wh | [0.0, inf[ |
| `rex` | Range extender SubFleet | None | `type_unit` == 'icev' | Name of a Mobile Battery SubFleet which can be used as Range Extender. Neglected, if (a) DES is not activated or (b) unit_type is not 'ev' | string with name of SubFleet |
| `mode_scheduling` | Scheduling Mode | str |  | Scheduling Mode for charging the SubFleet's Fleet units: Available options are uncoordinated charging ('uc'), three different rulebased strategies (equal distribution of the available power - 'equal', first come first served - 'fcfs', soc based charging - 'soc') and optimized charging ('oc'). Bidirectional charging is only available for 'oc' | 'uc', 'equal', 'fcfs', 'soc', 'oc' |
| `forecast_hours` | Forecast hours | float | `type_unit` == 'icev' | Neglected, if mode_scheduling is 'oc': Defines how much time in advance a trip can be seen by the charging scheduler in order to adjust the target SOC based on soc_target_high and soc_target_low. Feature currently not enabled | [0.0, inf[ |
| `aging` | Consideration of battery aging | bool | `type_unit` == 'icev' | Trigger whether to calculate battery aging for the Fleet unit's storage after each horizon. Aging results are taken into account for the next horizon by limiting the available SOC range. Maximum power is not reduced | True, False |
| `chemistry` | Cell chemistry | str | `type_unit` == 'icev' | Cell chemistry of the storage to select the correct aging model for aging calculation | 'nmc', 'lfp' |
| `temp_battery` | Battery temperature | float or None | `type_unit` == 'icev' | Battery temperature used as stress factor in aging model. Can be set to a constant value, defined using the timeseries of a PVSource block as this contains a temperature timeseries, or set to None to inherit the temperature specified in temp_air of the Scenario | string with name of block of class SubFleet or ]-inf, inf[ |
| `q_loss_cal_init` | Initial capacity loss due to calendric aging | float | `type_unit` == 'icev' | Initial capacity loss of the storage at simulation start due to calendric aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init) | [0.0, 1.0] |
| `q_loss_cyc_init` | Initial capacity loss due to cyclic aging | float | `type_unit` == 'icev' | Initial cyclic loss of the storage at simulation start due to cyclic aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init) | [0.0, 1.0] |
| `soc_init` | Initial SOC | float | `type_unit` == 'icev' | Initial SOC of the storage at simulation start | [0.0, 1.0] |
| `soc_min` | Minimum SOC | float or None | `type_unit` == 'icev' | Lower limit of the usable SOC window of the storage. Set to None to use the full window (0). The lower limit imposed by aging is applied on top: the higher of both limits is used | [0, 1] or None |
| `soc_max` | Maximum SOC | float or None | `type_unit` == 'icev' | Upper limit of the usable SOC window of the storage. Set to None to use the full window (1). The upper limit imposed by aging is applied on top: the lower of both limits is used | [0, 1] or None |
| `soc_target` | Target SOC | float | `type_unit` == 'icev' | Target SOC up to which DES charges a Fleet unit until it can be used for the next trip | [0.0, 1.0] |
| `soc_return` | Return SOC | float | `type_unit` == 'icev' | Minimum SOC after the trip. For DES this is used to calculate (a) the usable range of a battery and (b) the numbers of necessary Range Extender batteries if available | [0.0, 1.0] |
| `dsoc_buffer` | Delta SOC buffer | float | `type_unit` == 'icev' | Buffer given as SOC to compensate for aging occurring during the simulation and self discharge during a trip. Parameter is neglected for every other simulation paradigm than Rolling Horizon | [0.0, 1.0] |
| `pwr_chg_max` | Maximum charging power | float | `type_unit` == 'icev' | Maximum charging power at the local energy system in W | [0.0, inf[ |
| `pwr_dis_max` | Maximum discharging power | float or str | `type_unit` == 'icev' | Maximum discharging power at the local energy system in W. Set to 'equal' to use the same value as pwr_chg_max. | [0, inf[ or 'equal' |
| `pwr_ext_ac` | Maximum power for external AC charging | float | `type_unit` == 'icev' | Maximum charging power at an external AC charger in W | [0.0, inf[ |
| `pwr_ext_ac_max` | pwr_ext_ac_max | float | `type_unit` == 'icev' | TODO: Required by revoletion but not found in current version | ]0.0, inf[ |
| `pwr_ext_dc` | Maximum power for external DC charging | float | `type_unit` == 'icev' | Maximum charging power at an external DC charger in W | [0.0, inf[ |
| `pwr_ext_dc_max` | pwr_ext_dc_max | float | `type_unit` == 'icev' | TODO: Required by revoletion but not found in current version | ]0.0, inf[ |
| `eff_storage_roundtrip` | Storage roundtrip efficiency | float | `type_unit` == 'icev' | Roundtrip efficiency of the Fleet unit's storage measured at the connection between the Fleet unit's bus and the storage. Charge and discharge efficiency is calculated using sqrt(eff_storage_roundtrip) | [0.0, 1.0] |
| `eff_chg_ac` | AC charging efficiency | float | `type_unit` == 'icev' | Efficiency of the Fleet unit's On-Board-Charger (OBC) in charging direction. Taken into account for AC charging at the local grid (if Fleet's system is set to AC) and external AC charging | [0.0, 1.0] |
| `eff_chg_dc` | DC charging efficiency | float | `type_unit` == 'icev' | Efficiency of a DC charging station in charging direction. Taken into account for DC charging at the local grid only, as losses at public charging stations are not relevant for an fleet operator but the charging station operator only | [0.0, 1.0] |
| `eff_dis_ac` | AC discharging efficiency | float | `type_unit` == 'icev' | Efficiency of the commodity's On-Board-Charger (OBC) in discharging direction. Taken into account for DC charging at the local grid (if Fleet's system is set to AC) and external AC charging | [0.0, 1.0] |
| `eff_dis_dc` | DC discharging efficiency | float | `type_unit` == 'icev' | Efficiency of a DC charging station in discharging direction. Taken into account for DC discharging at the local grid only, as losses at public charging stations are not relevant for an fleet operator but the charging station operator only | [0.0, 1.0] |
| `sdr` | Self discharge rate | float | `type_unit` == 'icev' | Self discharge rate of storage component related to its nominal capacity per month (30 days) | [0.0, inf[ |

</details>

<!EXIT_POINT_SCHEMA_TABLES>


## EV-Specific submodules
Since REVOL-E-TION's main purpose is optimum EV integration, this is where focus lies in modeling detail, resulting in the DES and a priori power scheduling submodules.
The following sections give more detail on these submodules:

### Discrete Event Simulation (DES)
The DES takes stochastically defined mobility or energy demand in multiple use cases and timeframes, samples an actual demand and assigns it to a Fleet (more specifically to the individual FleetUnit within it) in a first-come-first-serve approach.
This enables to model the mobility/energy demand independently of the (Sub)Fleet's size.
However, the DES is run a priori to the dispatch/sizing optimization, so no consideration of whether the resulting FleetUnit dispatch is beneficial to the energy system is taken and limitations through battery aging and/or other limitations have to be covered by safety margins in dispatch to achieve a feasible solution to the energy system's optimization problem.
Through the sampling, the DES also makes REVOL-E-TION outputs non-deterministic once it is activated even though the actual core optimization is deterministic.
A run can be made reproducible by writing out its dispatch events and feeding them back in through the 'events' data source, see "Dispatch events and the time based log" below.
The DES is run separately for every scenario in a simulation run, but only once per scenario, as some Fleets might be linked and can therefore not be simulated independently.
It is built on the [simpy](https://simpy.readthedocs.io/en/latest/) library and is implemented in the ```dispatch.py``` file.
A modification is made to simpy to enable processes to require multiple resources at once, which is necessary for mobile battery use cases requiring high energy.

The core element of the DES is a so-called environment which is populated with processes that each represent one rental of a FleetUnit from a store holding those FleetUnits.
The request time for each rental is sampled from a dual normal distribution over time of day defined for each timeframe and use case in the use case definition csv file.
An example use case file is distributed with REVOL-E-TION for both Vehicles and mobile batteries (for explanation see chapter "Classes of Blocks") in the respective input directories.
Inside the code for the DES, a FleetDispatcher instance is created for each Fleet in the scenario, holding one store of FleetUnits per SubFleet, as well as a DispatchProcess instance for each request.

Each process not only covers the actual rental time (which itself is made up of an active part and idle time, both of which are sampled stochastically), but also a charging time to give the energy system enough time to recharge the FleetUnit.
This charging time follows from the energy the upcoming rental requires and from the FleetUnit's charging power, and it is placed before the rental: a FleetUnit can only be taken from its store once it has been back at base for at least that long.
The same holds for range extension FleetUnits with their own energy share, so a rental only starts once both the primary and the range extension FleetUnits have been resting long enough.
Since the charging time is derived from the request rather than from the FleetUnit's state, a FleetUnit without a traction battery of its own, which relies entirely on range extension, is available again immediately upon its return.
Note that the dispatch KPI "rate_blocked" attributes the charging time to the rental that caused it and therefore counts it after the return rather than before the departure.

Since vehicle SubFleets can be linked to mobile battery SubFleets through the scenario parameter "rex" for range extension, a process of a vehicle Fleet can require FleetUnits from its own store and from the linked battery Fleet's store at the same time.
Such a process then has a primary and a range extension FleetUnit, both of which need to be available for the process to be successful.
Once the environment has been run, every successful process that took range extension FleetUnits along is copied into the linked battery Fleet's dispatcher with the two roles swapped, so that the batteries also appear as rented within their own Fleet.
These copies are flagged as range extension, which is what distinguishes an internal service to another SubFleet of the same operator from an external rental in the economic evaluation.

Once all stores have been populated with FleetUnits and all processes have been created, the environment is run and determines which processes are successful (i.e. get their required FleetUnits) and which ones fail.

#### Dispatch events and the time based log
The result of the DES is an event table, implemented in the ```events.py``` file: essentially the demand list after dispatch, with one row per rented FleetUnit and process holding the requested time, the actual departure and return, the FleetUnit taken, the consumed energy, the driven distance and the Delta SOC.
Requests that could not be served are kept as rows without a FleetUnit and without dispatched values, which makes the table a complete record of the dispatch and allows the dispatch KPIs (success, utilization and blocked rate) to be derived from it.
Values are stored per event rather than per timestep, so the table is independent of the simulation timestep, and departure and return are resolved exactly.
A row is additionally flagged if the rental serves another SubFleet as a range extender rather than an external customer, which is what keeps the operator from invoicing themselves (see "crev_spec_time" in the parameter tables above).

The energy system model works per timestep, so each FleetUnit materializes its own time based log from its events.
This contains columns of availability in the energy system (called "atbase"), energy consumption while not at base, availability of external AC and DC charging, the Delta SOC ("dsoc") of a rental, the trip distance and the range extension flag ("rex").
Event values are distributed evenly over the rental period, as the DES resolves no intra-rental detail.
The distance is needed for vehicle SubFleets because even with a constant consumption it is no more traceable from the energy consumption due to the possibility of range extension.
Only if the dispatch of the FleetUnits is left to the optimizer (as opposed to the a priori power scheduling) myopically (i.e. in the "rh" strategy"), the 'dsoc' column is actually transferred to a hard minimum SOC constraint for the optimizer, as all other cases handle this intrinsically.

A Fleet reading a time based log from file ("log" data source) keeps that log as it is and reconstructs its events from the occupancy it records.
This reconstruction is necessarily incomplete: a log holds no record of requests that were never served, none of the recharging after a return, and no range extension flag, so such a Fleet reports no dispatch KPIs and all of its rentals are treated as external.
Two rentals following each other without an idle timestep in between are indistinguishable in a log and are recovered as a single event.
Replaying an event file ("events" data source) has none of these limitations and reproduces the dispatch of the run it was written by exactly, which is the way to obtain deterministic results despite the stochastic sampling of the DES.

### A Priori Power Scheduling
Dependent on the chosen scheduling method ("mode_scheduling") of a SubFleet a charging schedule for the SubFleet's FleetUnits is calculated before the linear optimization starts (a priori).
This approach is applied to FleetUnits in SubFleets with optimization level of all types of uncoordinated charging ('uc') and rule-based strategies ('equal', 'fcfs', 'soc').
All optimization levels causing an a priori calculation of the FleetUnit's charging power require a unidirectional ('ud') integration level ("lvl_cap").<br>
In addition to the optimization level, rule-based systems may implement a static load management system by defining the maximum available power using the "power_lim_static" parameter.
If "power_lim_static" is set to 'None', static load management is disabled, and the system defaults to dynamic load management.
For all FleetUnits within uncoordinated or rule-based SubFleets in addition to the charging power when being plugged in at the local energy system the required charging power on-route at external charging infrastructure is calculated.
The precomputed charging schedules are then enforced in the linear optimization model by applying them as fixed power constraints to the relevant model flows.

#### General Approach of A Priori Power Scheduling
The a priori power scheduling process is iterative and operates on a per-timestep basis as the State of Charge (SOC) at each timestep is influenced by the previous timestep.
Within a single timestep in a first step the charging powers for all FleetUnits plugged in to the local energy system (column 'atbase' in log file is set to True) are calculated.
Initially, the charging power calculation for uncoordinated FleetUnits connected to the local energy system is determined, followed by rule-based FleetUnits with static load management.
In the next step the remaining available power within the local energy system is allocated to all FleetUnits being part of the dynamic load management.
The linear optimization algorithm subsequently handles the allocation of any excess power.
Lastly, the external AC and DC charging power is calculated for FleetUnits for which this infrastructure is currently available (indicated by the 'atac' or 'atdc' columns being True in the log file).

#### Load management system
For rule-based FleetUnit instances within a Fleet instance, a static power limit can be defined in the scenario file, which restricts the cumulative charging power of all FleetUnits within the Fleet.
If no static power limit is set, the Fleet's FleetUnits will be part of a dynamic load management.
In this case, the available power within the local energy system is distributed among all FleetUnits that are part of the dynamic load management.
The available power is calculated by subtracting the required power of FixedDemand blocks and the power already allocated to all uncoordinated FleetUnits and rule-based FleetUnits in Fleets with static load management from the maximum available power.
The maximum available power within the local grid is calculated by summing up the output powers of GridConnection, PVSource, WindSource, and ControllableSource blocks.
All Fleets without a static power limit that include rule-based FleetUnits have to be of the same integration level as they are all controlled by the same dynamic load management system; the rule-based charging strategy is applied as described.
Furthermore, if this integration level is supposed to be 'equal', all SubFleets have to be connected to the same bus via their corresponding Fleet.
As a dynamic load management system requires knowledge about the available power within the local energy system for each timestep, it is not possible to combine a dynamic load management system with a StationaryBattery block.
This would require an a priori calculation of the StationaryBattery's SOC which is not possible in a straight-forward way, due to different prioritization of power sources based on their current opex and the efficiency of the SystemCore.

#### Charge Scheduling Modes  ("mode_scheduling")
- uncoordinated charging: only a single FleetUnit neglecting effects caused by other FleetUnits and the local energy system's limitations is taken into account to determine its charging power.
  - 'uc':Once plugged in to the local energy system, the FleetUnit charges at the maximum charging power specified for the FleetUnit in the scenario file until the target SOC is reached.
         Charging power is determined for a single FleetUnit in isolation, neither considering the influence of other FleetUnits nor any type of load management system.
- rule-based charging strategies: multiple FleetUnits (within the same Fleet for static load management or across several Fleets for dynamic load management) are considered to determine their charging power.
  - 'equal': Charging power is evenly distributed among all FleetUnits within the Fleet. Any surplus power, resulting from a FleetUnit reaching its target SOC, is redistributed among the remaining FleetUnits until all available power is allocated.
  - 'fcfs': FleetUnits are prioritized by their plug-in time, with the earliest plug-in receiving the highest priority. In cases where plug-in times are identical, FleetUnits are prioritized alphabetically by name.
  - 'soc': FleetUnits are prioritized based on their SOC, with the lowest SOC receiving the highest priority. For identical SOC levels, priorities are assigned alphabetically by the FleetUnits' names.
- optimized charging: The charging power of the FleetUnits is optimized by the linear optimization algorithm without any a priori calculation.

#### External charging
The power for all FleetUnits with available external charging infrastructure (column 'atac' or 'atdc' in log file is set to True) is calculated.
- AC charging: Once AC charging gets available, the required energy until the return to the local energy system is calculated.
  If the current SOC does not ensure a return SOC above the specified minimum return SOC (does not consider self-discharge), AC charging is activated. The specified charging power is then applied until the target SOC is reached.
- DC charging: If DC charging is available and the SOC at the next timestep with a charging possibility is below 5 %, DC charging is activated for the current timestep. The maximum SOC for DC charging is 80 % neglecting "soc_target".

### A Posteriori Aging Model (also available for StationaryBattery)
Simple semi-empirical aging models are implemented for a blocks containing electric storages (ElectricFleetUnit and StationaryBattery).
More specifically, these are the Naumann model ([publication 1](https://doi.org/10.1016/j.est.2018.01.019) and [publication 2](https://doi.org/10.1016/j.jpowsour.2019.227666)) for LFP batteries and the [Schmalstieg model](https://doi.org/10.1016/j.jpowsour.2014.02.012) for NMC batteries.
They are both contained within the ```battery.py``` file and work on superposing calendric and cycle aging.
The latter is based on cycling parameters (e.g. depths of discharge) determined using a rainflow algorithm for each horizon.

Results of the aging model (if activated through the scenario file parameter "aging") are evaluated and the capacity degradation applied as a restricted usable SOC window after every horizon.
This results in a dependency of the aging model output on the horizon resolution (i.e. length) with the extreme case of the "go" strategy, where only the State of Health at the end of the simulation timeframe is evaluated.
Please note that the lifetime used for economic extrapolation of simulation results is not connected to the aging model and sizing the battery block on its results is infeasible due to it being run a posteriori and nonlinearly.
Therefore, the aging model's output does not have any influence on the economic results of the simulation, but is only of an informative character.

## Outputs
REVOL-E-TION creates a uniquely named (containing the runtimestamp and the scenario file name) result directory for every run.
There, the following files are saved (some of them optionally):
- The log file of the run named ```<runtimestamp>_<scenario_file_name>_log.csv```. This contains all terminal log messages of the run, including errors and warnings.
- A single result summary file named ```<runtimestamp>_<scenario_file_name>_summary.csv``` containing all noncomplex (int, float, bool or string) attributes of the SimulationRun, Scenario, and all blocks in the scenario.
- A single result summary file named ```<runtimestamp>_<scenario_file_name>_summary.pkl```. This is the same as the summary csv file but in pickle format which eases further processing.
- A single scenario status file named ```<runtimestamp>_<scenario_file_name>_status.csv``` containing the current status ('started', 'fully initialized', 'completed horizon x out of y', 'successful', 'failed') and occurring exception tracebacks of all scenarios in the run. It is constantly updated during the run and can be used for monitoring purposes and filtering in the analysis of the results.
- (Optional) One result timeseries file per scenario named ```<runtimestamp>_<scenario_file_name>_<scenario_name>_results_ts.feather```. This contains all timeseries results of the energy system for every timestep, facilitating easy plotting.
- (Optional) One sampled demand file per Fleet whose usage is generated from use cases, named ```<runtimestamp>_<scenario_file_name>_<scenario_name>_<fleet_name>_demand.feather```. This holds the requests the DES was run on and can be fed back in through the 'demand' data source to skip the sampling stage. Input files are accepted in both csv and feather format.
- (Optional) One time based log file per dispatched Fleet named ```<runtimestamp>_<scenario_file_name>_<scenario_name>_<fleet_name>_log.feather```. This is what the energy system model is run on and can be fed back in through the 'log' data source.
- (Optional) One event file per dispatched Fleet named ```<runtimestamp>_<scenario_file_name>_<scenario_name>_<fleet_name>_events.feather```. This is the dispatch result the log is derived from and can be fed back in through the 'events' data source to replay the run exactly. It supersedes the log as a reuse format, as it additionally holds the requests that could not be served, the range extension flag and the exact departure and return times. Fleets whose usage is read from file are not dispatched and therefore write no event file.
- (Optional) One plot file per scenario named ```<runtimestamp>_<scenario_file_name>_<scenario_name>.html``` containing an interactive line plot of the dispatch and state variables (i.e. SOCs and SOHs) of every block in the scenario.
