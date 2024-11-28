#!/usr/bin/env python3

import argparse
import os
import tkinter as tk
import tkinter.filedialog
import warnings

from revoletion import simulation as sim


class DefaultFileLocationWarning(UserWarning):
    pass


def get_path(path_cwd, path_pkg, arg, file_type):
    # Option 1: No argument passed -> select file via GUI
    if arg is None:
        root = tk.Tk()
        root.withdraw()  # hide small tk-window
        root.lift()  # make sure all tk windows appear in front of other windows

        # get file
        default_dir = os.path.join(path_cwd, 'example')
        file_path = tk.filedialog.askopenfilename(initialdir=default_dir,
                                                  title=f'Select {file_type} file',
                                                  filetypes=(('CSV files', '*.csv'),
                                                             ('All files', '*.*')))
        if not file_path:
            raise FileNotFoundError(f'No {file_type} file selected')
    # Option 2: Full absolute or relative file path (works from anywhere)
    elif os.path.isfile(arg):
        file_path = arg
    # Option 3: File name in the working directory (works from within project directory only)
    elif os.path.isfile(os.path.join(path_cwd, arg)):
        file_path = os.path.join(path_cwd, arg)
    # Option 4: Example file in example project in package directory (works from anywhere)
    elif os.path.isfile(os.path.join(path_pkg, 'example', arg)):
        file_path = os.path.join(path_pkg, 'example', arg)
        warnings.warn(f'Using default {file_type} file \"{arg}\" from REVOL-E-TION package directory '
                      f'- disregard if this is intended', DefaultFileLocationWarning)
    else:
        raise FileNotFoundError(f'{file_type[0].upper()}{file_type[1:]} file or path not interpretable: {arg}')

    return file_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-scn', '--scenario',
                        type=str,
                        help='Path to the scenario CSV file')
    parser.add_argument('-set', '--settings',
                        type=str,
                        help='Path to the settings CSV file')
    parser.add_argument('-rer', '--rerun',
                        type=str,
                        default=False,
                        help='Directory name of run including failed scenarios which should be rerun')
    parser.add_argument('-exe', '--execute',
                        type=str,
                        default=True,
                        help='Immediately execute run')

    args = parser.parse_args()

    path_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_cwd = os.getcwd()

    path_scenario = get_path(path_cwd, path_pkg, args.scenario, 'scenarios')
    path_settings = get_path(path_cwd, path_pkg, args.settings, 'settings')

    sim.SimulationRun(path_scenarios=path_scenario,
                      path_settings=path_settings,
                      rerun=args.rerun,
                      execute=args.execute)


if __name__ == '__main__':
    main()
