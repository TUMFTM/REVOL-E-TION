#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path

from .logger import configure_root_logger
from .optimization import OptimizationBackend, Solver
from .run import SimulationRun
from .scenario import SimulationPaths
from .simulation import SimulationSettings


def main():
    parser = argparse.ArgumentParser()

    settings_template = SimulationSettings()  # use this to only define default values once in SimulationSettings

    def filter_bool(value):
        if value in ["True", "true"]:
            return True
        elif value in ["False", "false"]:
            return False
        else:
            return value

    parser.add_argument("-scn", "--scenario", type=str, default=None, help="Path to the scenario CSV file")
    parser.add_argument("-in", "--input", type=str, default=None, help="Path to the input data directory")
    parser.add_argument("-out", "--output", type=str, default=None, help="Path to the results directory")
    parser.add_argument(
        "-rer",
        "--rerun",
        type=filter_bool,
        default=None,
        help="Directory name of run including failed scenarios which should be rerun",
    )
    parser.add_argument(
        "-slv",
        "--solver",
        type=Solver,
        default=settings_template.solver,
        help="Compatible solver to be used for the optimization problem.",
        choices=list(Solver),
    )
    parser.add_argument(
        "-otol",
        "--optimality_tol",
        type=float,
        default=settings_template.optimality_tol,
        help="Solver tolerance on reduced costs. Has to stay below the scenario's cost_eps, "
        "otherwise the tie breaking epsilon is inside the solver's noise floor and circular "
        "flows survive in the dispatch. Pass 0 to leave the solver at its own default.",
    )
    parser.add_argument(
        "-bnd",
        "--backend",
        type=OptimizationBackend,
        default=settings_template.backend,
        help="Set the backend with which the optimization problem is modeled.",
        choices=list(OptimizationBackend),
    )

    parser.add_argument(
        "-np",
        "--n_processes",
        type=int,
        default=settings_template.n_processes,
        help="Number of processes (i.e. cores) to use in parallel operation",
    )
    parser.add_argument(
        "-ls",
        "--largescalemode",
        type=filter_bool,
        default=settings_template.largescalemode,
        help="Omit detailed output data (generated input timeseries, system graphs, "
        "result timeseries, and timeseries plots)",
    )
    parser.add_argument(
        "-db",
        "--debugmode",
        type=filter_bool,
        default=settings_template.debugmode,
        help="Generate debug output and dump .lp model file for external solving",
    )
    parser.add_argument(
        "-rin",
        "--rerun_infeasible",
        type=filter_bool,
        default=settings_template.rerun_infeasible,
        help="Rerun infeasible or unbounded scenarios",
    )
    parser.add_argument(
        "-ksc", "--key_solcast_api", type=str, default=settings_template.key_solcast_api, help="API key for Solcast API"
    )

    args = parser.parse_args()

    # check boolean arguments
    for arg_name in ["largescalemode", "debugmode", "rerun_infeasible"]:
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
        try:
            import tkinter as tk
            import tkinter.filedialog
        except ImportError:
            raise FileNotFoundError(
                "No scenario file provided and tkinter is unavailable in this environment. "
                "Please provide a scenario file path or run this script in an environment with tkinter installed."
            )

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
    # Option 2: Full absolute or relative (to working directory) file path
    else:
        path_scenario = Path(args.scenario)
    # endregion

    settings = SimulationSettings(
        solver=args.solver,
        # 0 is the CLI's way of saying "leave the solver alone", see the argument's help
        optimality_tol=args.optimality_tol or None,
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
    configure_root_logger(paths.log, args.debugmode, args.largescalemode)

    simulation_run = SimulationRun(paths=paths, settings=settings)
    simulation_run.execute()


if __name__ == "__main__":
    main()
