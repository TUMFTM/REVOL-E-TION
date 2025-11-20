#!/usr/bin/env python3

import argparse
import importlib.resources
import warnings
from pathlib import Path

try:
    import tkinter as tk
    import tkinter.filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    warnings.warn("tkinter is not available in this environment. GUI file selection will be disabled.")


import revoletion.example

from . import rl, simulation
from .logger import configure_root_logger
from .optimization import OptimizationBackend, Solver
from .run import SimulationRun
from .scenario import SimulationPaths
from .simulation import SimulationSettings


def main():
    parser = argparse.ArgumentParser()

    # Add subparsers to handle subcommands `optimize`.
    subparsers = parser.add_subparsers(
        title="subcommands", dest="command", required=True, help="Choose which workflow to run"
    )

    simulation_settings_template = (
        SimulationSettings()
    )  # use this to only define default values once in SimulationSettings

    def filter_bool(value):
        if value in ["True", "true"]:
            return True
        elif value in ["False", "false"]:
            return False
        else:
            return value

    optimize_parser = subparsers.add_parser(
        name="optimize",
        help="Run system optimization including investments",
    )

    optimize_parser.add_argument("-scn", "--scenario", type=str, default=None, help="Path to the scenario CSV file")
    optimize_parser.add_argument("-in", "--input", type=str, default=None, help="Path to the input data directory")
    optimize_parser.add_argument("-out", "--output", type=str, default=None, help="Path to the results directory")
    optimize_parser.add_argument(
        "-msc", "--multiscenario", type=filter_bool, default=True, help="Combine multiple scenarios in a single run."
    )
    optimize_parser.add_argument(
        "-rer",
        "--rerun",
        type=filter_bool,
        default=None,
        help="Directory name of run including failed scenarios which should be rerun",
    )
    optimize_parser.add_argument(
        "-slv",
        "--solver",
        type=Solver,
        default=simulation_settings_template.solver,
        help="Pyomo compatible solver to be used for the optimization problem.",
        choices=list(Solver),
    )
    optimize_parser.add_argument(
        "-bnd",
        "--backend",
        type=OptimizationBackend,
        default=simulation_settings_template.backend,
        help="Set the backend with which the optimization problem is modeled.",
        choices=list(OptimizationBackend),
    )
    optimize_parser.add_argument(
        "-np",
        "--n_processes",
        type=int,
        default=simulation_settings_template.n_processes,
        help="Number of processes (i.e. cores) to use in parallel operation",
    )
    optimize_parser.add_argument(
        "-ls",
        "--largescalemode",
        type=filter_bool,
        default=simulation_settings_template.largescalemode,
        help="Omit detailed output data (generated input timeseries, system graphs, "
        "result timeseries, and timeseries plots)",
    )
    optimize_parser.add_argument(
        "-db",
        "--debugmode",
        type=filter_bool,
        default=simulation_settings_template.debugmode,
        help="Generate debug output and dump .lp model file for external solving",
    )
    optimize_parser.add_argument(
        "-rin",
        "--rerun_infeasible",
        type=filter_bool,
        default=simulation_settings_template.rerun_infeasible,
        help="Rerun infeasible or unbounded scenarios",
    )
    optimize_parser.add_argument(
        "-ksc",
        "--key_solcast_api",
        type=str,
        default=simulation_settings_template.key_solcast_api,
        help="API key for Solcast API",
    )

    dispatch_settings_template = simulation.DispatchSettings()

    dispatch_parser = subparsers.add_parser(name="dispatch")

    dispatch_parser.add_argument(
        "-np",
        "--n_processes",
        type=int,
        default=dispatch_settings_template.n_processes,
        help="Number of processes used for training of the RL agent",
    )

    dispatch_parser.add_argument(
        "--algo",
        type=rl.AgentAlgorithm,
        default=dispatch_settings_template.agent_algorithm,
        choices=list(rl.AgentAlgorithm),
        help="Algorithm used for dispatch",
    )

    dispatch_parser.add_argument("-scn", "--scenario", type=str, help="Path to the scenario CSV file")
    dispatch_parser.add_argument(
        "-db",
        "--debugmode",
        type=filter_bool,
        default=dispatch_settings_template.debugmode,
    )
    dispatch_parser.add_argument("-in", "--input", type=str, default=None, help="Path to the input data directory")
    dispatch_parser.add_argument("-out", "--output", type=str, default=None, help="Path to the results directory")

    dispatch_parser.add_argument(
        "-mp", "--models-path", type=str, help="Path to a folder from which models can be loaded and saved."
    )

    args = parser.parse_args()

    if args.command == "optimize":
        _optimize_cmd(args)
    elif args.command == "dispatch":
        _dispatch_cmd(args)
    else:
        parser.error(f"Invalid subcommand: {args.command}")


def _optimize_cmd(args: argparse.Namespace) -> None:
    # check boolean arguments
    for arg_name in ["multiscenario", "largescalemode", "debugmode", "rerun_infeasible"]:
        arg = getattr(args, arg_name)
        if not isinstance(arg, bool):
            raise ValueError(f'Argument --{arg_name} must be a boolean value, got "{arg}" of type {type(arg).__name__}')

    # validate that backend and solver are compatible.
    if not args.backend.is_compatible_solver(args.solver):
        raise ValueError(f"Solver {args.solver} is not supported by the {args.backend} backend")

    if args.backend == OptimizationBackend.PYPSA:
        warnings.warn(
            "The PyPSA backend is currently experimental and does not yet have feature parity with the oemof backend. Bugs are expected."
        )

    # region interpret scenario file path
    scenarios_example = False
    # Option 1: No scenario file argument passed -> select via GUI
    if args.scenario is None:
        if not TKINTER_AVAILABLE:
            raise FileNotFoundError("No scenario file provided and tkinter is unavailable.")

        root = tk.Tk()
        root.withdraw()  # hide small tk-window
        root.attributes("-topmost", True)  # make sure all tk windows appear in front of other windows
        path_scenario = tk.filedialog.askopenfilename(
            initialdir=Path.cwd(),
            title="Select scenario file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        root.destroy()  # clean up the root window

        if not path_scenario:
            raise FileNotFoundError("No scenario file selected")
    # Option 2: Example file in example project in package directory (works from anywhere)
    elif args.scenario == "example":
        scenarios_example = True
        with importlib.resources.as_file(importlib.resources.files(revoletion.example)) as example_dir:
            path_scenario = example_dir / "scenarios_example.csv"
    # Option 3: Full absolute or relative (to working directory) file path
    else:
        path_scenario = Path(args.scenario)
    # endregion

    settings = SimulationSettings(
        solver=args.solver,
        backend=args.backend,
        n_processes=args.n_processes,
        largescalemode=args.largescalemode,
        debugmode=args.debugmode,
        rerun_infeasible=args.rerun_infeasible,
        key_solcast_api=args.key_solcast_api,
    )

    paths = SimulationPaths.from_plain_paths(
        scenario=path_scenario,
        input=None if not args.input or scenarios_example else Path(args.input),
        output=None if not args.output or scenarios_example else Path(args.output),
        rerun=None if not args.rerun else args.rerun,
    )

    # Configure the level of the logger according to `debugmode` and setup handlers.
    configure_root_logger(paths.log, args.debugmode)

    simulation_run = SimulationRun(paths=paths, settings=settings)
    simulation_run.execute(plot=False)


def _dispatch_cmd(args: argparse.Namespace) -> None:
    paths = SimulationPaths.from_plain_paths(
        scenario=args.scenario,
        input=None if not args.input else Path(args.input),
        output=None if not args.output else Path(args.output),
    )
    scenario_factory = simulation.DispatchScenarioFactory(paths)

    # Configure the level of the logger according to `debugmode` and setup handlers.
    configure_root_logger(debugmode=args.debugmode)

    settings = simulation.DispatchSettings(
        n_processes=args.n_processes,
        agent_algorithm=args.algo,
        debugmode=args.debugmode,
        models_path=None if args.models_path is None else Path(args.models_path),
    )

    dispatch_horizon = simulation.DispatchHorizon(
        scenario_factory,
        settings,
    )
    dispatch_horizon.execute()


if __name__ == "__main__":
    main()
