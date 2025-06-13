#!/usr/bin/env python3

import argparse
from importlib.resources import files
from pathlib import Path
import tkinter as tk
import tkinter.filedialog
import warnings

from revoletion import simulation as sim


class DefaultFileLocationWarning(UserWarning):
    pass


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-scn', '--scenario',
                        type=str,
                        default=None,
                        help='Path to the scenario CSV file')
    parser.add_argument('-in', '--inputdir',
                        type=str,
                        default=None,
                        help='Path to the input data directory')
    parser.add_argument('-out', '--outputdir',
                        type=str,
                        default=None,
                        help='Path to the results directory')
    parser.add_argument('-slv', '--solver',
                        type=str,
                        default='gurobi',
                        help='Pyomo compatible solver to be used for the optimization problem.')
    parser.add_argument('-np', '--n_processes',
                        type=int,
                        default=1,
                        help='Number of processes (i.e. cores) to use in parallel operation')
    parser.add_argument('-ls', '--largescalemode',
                        type=bool,
                        default=False,
                        help='Omit detailed output data (generated input timeseries, system graphs, '
                             'result timeseries, and timeseries plots)')
    parser.add_argument('-db', '--debugmode',
                        type=bool,
                        default=False,
                        help='Generate debug output and dump .lp model file for external solving')
    parser.add_argument('-rer', '--rerun',
                        type=str,
                        default=False,
                        help='Directory name of run including failed scenarios which should be rerun')
    parser.add_argument('-rin', '--rerun_infeasible',
                        type=str,
                        default=True,
                        help='Rerun infeasible or unbounded scenarios')
    parser.add_argument('-ksc', '--key_solcast_api',
                        type=str,
                        default=None,
                        help='API key for Solcast API')

    args = parser.parse_args()

    path_pkg = files(__package__)
    path_cwd = Path.cwd()

    scenarios_example = False

    # region interpret scenario file path
    # Option 1: No scenario file argument passed -> select via GUI
    if args.scenario is None:
        root = tk.Tk()
        root.withdraw()  # hide small tk-window
        root.lift()  # make sure all tk windows appear in front of other windows
        path_scenario = tk.filedialog.askopenfilename(initialdir=path_cwd,
                                                      title=f'Select scenario file',
                                                      filetypes=(('CSV files', '*.csv'),
                                                                 ('All files', '*.*')))
        if not path_scenario:
            raise FileNotFoundError(f'No scenario file selected')
    # Option 2: Example file in example project in package directory (works from anywhere)
    elif args.scenario  == 'example':
        scenarios_example = True
        path_scenario = path_pkg / 'example' / 'scenarios_example.csv'
        warnings.warn(f'Using example project provided with REVOL-E-TION. '
                      f'Outputs will be saved to the directory "results" in the current working directory.\n'
                      f'-> Disregard if this is intended',
                      DefaultFileLocationWarning)
    # Option 3: Full absolute or relative (to working directory) file path
    elif (path_scenario := Path(args.scenario)).is_file():
        path_scenario = path_scenario if path_scenario.is_absolute() else Path.cwd() / path_scenario
    else:
        raise FileNotFoundError(f'Scenario file or path not interpretable: {args.scenario}')
    # endregion

    # region interpret input directory path
    # Option 1: Example file in example project in package directory (works from anywhere)
    if scenarios_example or args.inputdir is None:
        path_input = path_scenario.parent
    # Option 2: Full absolute or relative (to working directory) path
    elif (path_input := Path(args.inputdir)).is_dir():
        path_input = path_input if path_input.is_absolute() else Path.cwd() / path_input
    else:
        raise NotADirectoryError(f'Input directory path not interpretable: {args.inputdir}')
    # endregion

    # region interpret output directory path
    # Option 1: Example file in example project in package directory (works from anywhere)
    if scenarios_example:
        path_output = path_cwd / 'results'
    # Option 2: No output directory argument passed -> create results folder in current working directory
    elif args.outputdir is None:
        path_output = path_cwd / 'results'
        warnings.warn(f'Outputs will be saved to the directory "results" in the current working directory.\n'
                      f'-> Disregard if this is intended',
                      DefaultFileLocationWarning)
    # Option 3: Full absolute or relative (to working directory) path
    elif (path_output := Path(args.outputdir)).is_dir():
        path_output = path_output if path_output.is_absolute() else Path.cwd() / path_output
    else:
        raise NotADirectoryError(f'Output directory path not interpretable: {args.outputdir}')
    # endregion

    sim.SimulationRun(path_scenarios=path_scenario,
                      path_input=path_input,
                      path_output=path_output,
                      solver=args.solver,
                      n_processes=args.n_processes,
                      largescalemode=args.largescalemode,
                      debugmode=args.debugmode,
                      rerun=args.rerun,
                      rerun_infeasible=args.rerun_infeasible,
                      key_solcast_api=args.key_solcast_api)


if __name__ == '__main__':
    main()
