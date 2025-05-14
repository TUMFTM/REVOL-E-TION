#!/usr/bin/env python3

import argparse
import os
import warnings

try:
    import tkinter as tk
    import tkinter.filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    warnings.warn(
        "tkinter is not available in this environment. GUI file selection will be disabled."
    )


from revoletion import simulation as sim


class DefaultFileLocationWarning(UserWarning):
    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-scn",
        "--scenario",
        type=str,
        default=None,
        help="Path to the scenario CSV file",
    )
    parser.add_argument(
        "-in",
        "--inputdir",
        type=str,
        default=None,
        help="Path to the input data directory",
    )
    parser.add_argument(
        "-out",
        "--outputdir",
        type=str,
        default=None,
        help="Path to the results directory",
    )
    parser.add_argument(
        "-slv",
        "--solver",
        type=str,
        default="gurobi",
        help="Pyomo compatible solver to be used for the optimization problem.",
    )
    parser.add_argument(
        "-np",
        "--n_processes",
        type=int,
        default=1,
        help="Number of processes (i.e. cores) to use in parallel operation",
    )
    parser.add_argument(
        "-ls",
        "--largescalemode",
        type=bool,
        default=False,
        help="Omit detailed output data (generated input timeseries, system graphs, "
        "result timeseries, and timeseries plots)",
    )
    parser.add_argument(
        "-db",
        "--debugmode",
        type=bool,
        default=False,
        help="Generate debug output and dump .lp model file for external solving",
    )
    parser.add_argument(
        "-rer",
        "--rerun",
        type=str,
        default=False,
        help="Directory name of run including failed scenarios which should be rerun",
    )
    parser.add_argument(
        "-rin",
        "--rerun_infeasible",
        type=str,
        default=True,
        help="Rerun infeasible or unbounded scenarios",
    )
    parser.add_argument(
        "-ksc",
        "--key_solcast_api",
        type=str,
        default=None,
        help="API key for Solcast API",
    )

    args = parser.parse_args()

    path_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_cwd = os.getcwd()

    scenarios_example = False

    # region interpret scenario file path
    # Option 1: No scenario file argument passed -> select via GUI
    if args.scenario is None:
        if not TKINTER_AVAILABLE:
            raise FileNotFoundError(
                "No scenario file provided and tkinter is unavailable."
            )

        root = tk.Tk()
        root.withdraw()  # hide small tk-window
        root.lift()  # make sure all tk windows appear in front of other windows
        path_scenario = tk.filedialog.askopenfilename(
            initialdir=path_cwd,
            title="Select scenario file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path_scenario:
            raise FileNotFoundError("No scenario file selected")
    # Option 2: Full absolute or relative file path (works from anywhere)
    elif os.path.isfile(args.scenario):
        path_scenario = args.scenario
    # Option 3: File name in the working directory (works from within project directory only)
    elif os.path.isfile(os.path.join(path_cwd, args.scenario)):
        path_scenario = os.path.join(path_cwd, args.scenario)
    # Option 4: Example file in example project in package directory (works from anywhere)
    elif args.scenario in ["example", "ex"]:
        scenarios_example = True
        path_scenario = os.path.join(path_pkg, "example", "scenarios_example.csv")
        warnings.warn(
            f'Using example scenario file "{args.scenario}", data, and output directory from '
            f"REVOL-E-TION - disregard if this is intended",
            DefaultFileLocationWarning,
        )
    else:
        raise FileNotFoundError(
            f"Scenario file or path not interpretable: {args.scenario}"
        )
    # endregion

    # region interpret input directory path
    # Option 1: Example file in example project in package directory (works from anywhere)
    if scenarios_example:
        path_input = os.path.dirname(path_scenario)
    # Option 2: No input directory argument passed -> select via GUI
    elif args.inputdir is None:
        if not TKINTER_AVAILABLE:
            raise FileNotFoundError(
                "No input directory provided and tkinter is unavailable."
            )
        path_input = tk.filedialog.askdirectory(
            initialdir=path_cwd, title="Select input data directory"
        )
        if not path_input:
            raise NotADirectoryError("No input data directory selected")
    # Option 3: Full absolute or relative file path (works from anywhere)
    elif os.path.isdir(args.inputdir):
        path_input = args.inputdir
    # Option 4: Subdirectory of working directory (works from within project directory only)
    elif os.path.isdir(os.path.join(path_cwd, args.inputdir)):
        path_input = os.path.join(path_cwd, args.inputdir)
    else:
        raise NotADirectoryError(
            f"Input directory path not interpretable: {args.inputdir}"
        )
    # endregion

    # region interpret output directory path
    # Option 1: Example file in example project in package directory (works from anywhere)
    if scenarios_example:
        path_output = os.path.join(path_pkg, "results")
    # Option 2: No output directory argument passed -> select via GUI
    elif args.outputdir is None:
        if not TKINTER_AVAILABLE:
            raise FileNotFoundError(
                "No output directory provided and tkinter is unavailable."
            )
        path_output = tk.filedialog.askdirectory(
            initialdir=path_cwd, title="Select output data directory"
        )
        if not path_output:
            raise NotADirectoryError("No output data directory selected")
    # Option 3: Full absolute or relative file path (works from anywhere)
    elif os.path.isdir(args.outputdir):
        path_output = args.outputdir
    # Option 4: Subdirectory of working directory (works from within project directory only)
    elif os.path.isdir(os.path.join(path_cwd, args.outputdir)):
        path_output = os.path.join(path_cwd, args.outputdir)
    else:
        raise NotADirectoryError(
            f"Output directory path not interpretable: {args.outputdir}"
        )
    # endregion

    sim.SimulationRun(
        path_scenarios=path_scenario,
        path_input=path_input,
        path_output=path_output,
        solver=args.solver,
        n_processes=args.n_processes,
        largescalemode=args.largescalemode,
        debugmode=args.debugmode,
        rerun=args.rerun,
        rerun_infeasible=args.rerun_infeasible,
        key_solcast_api=args.key_solcast_api,
    )


if __name__ == "__main__":
    main()
