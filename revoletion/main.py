#!/usr/bin/env python3

import os
import sys
import tkinter as tk
import tkinter.filedialog
import warnings

from revoletion import simulation as sim


class DefaultFileLocationWarning(UserWarning):
    pass


def read_arguments(path_cwd, path_pkg):

    scenario_input = sys.argv[1]
    settings_input = sys.argv[2]

    # Option 1: full file path (works from anywhere in the system)
    if os.path.isfile(scenario_input):
        scenarios_file_path = scenario_input
    # Option 2: File name in the working directory (works from parent of 'input' directory only)
    elif os.path.isfile(os.path.join(path_cwd, 'input', 'scenarios', scenario_input)):
        scenarios_file_path = os.path.join(path_cwd, 'input', 'scenarios', scenario_input)
    # Option 3: File name in package directory (works from anywhere, but should only contain example files)
    elif os.path.isfile(os.path.join(path_pkg, 'input', 'scenarios', scenario_input)):
        scenarios_file_path = os.path.join(path_pkg, 'input', 'scenarios', scenario_input)
        warnings.warn(f'Using scenario file {scenario_input} from REVOL-E-TION package directory '
                      f'- disregard if this is intended, e.g. for running an example scenario',
                      DefaultFileLocationWarning)
    else:
        raise FileNotFoundError(f'Scenario file or path not interpretable: {scenario_input}')

    # Option 1: full file path (works from anywhere in the system)
    if os.path.isfile(settings_input):
        settings_file_path = settings_input
    # Option 2: File name in the working directory (works from parent of 'input' directory only)
    elif os.path.isfile(os.path.join(path_cwd, 'input', 'settings', settings_input)):
        settings_file_path = os.path.join(path_cwd, 'input', 'settings', settings_input)
    # Option 3: File name in package directory (works from anywhere, but should only contain example files)
    elif os.path.isfile(os.path.join(path_pkg, 'input', 'settings', settings_input)):
        settings_file_path = os.path.join(path_pkg, 'input', 'settings', settings_input)
        warnings.warn(f'Using settings file {settings_input} from package directory - disregard if this is intended,'
                      f' e.g. for running an example settings configuration', DefaultFileLocationWarning)
    else:
        raise FileNotFoundError(f'Settings file or path not interpretable: {settings_input}')

    return scenarios_file_path, settings_file_path


def select_arguments(path_cwd):

    root = tk.Tk()
    root.withdraw()  # hide small tk-window
    root.lift()  # make sure all tk windows appear in front of other windows

    # get scenarios file
    scenarios_default_dir = os.path.join(path_cwd, 'input', 'scenarios')
    scenarios_file_path = tk.filedialog.askopenfilename(initialdir=scenarios_default_dir,
                                                        title="Select scenario file",
                                                        filetypes=(("CSV files", "*.csv"),
                                                                   ("All files", "*.*")))
    if not scenarios_file_path:
        raise FileNotFoundError('No scenario file selected')

    # get settings file
    settings_default_dir = os.path.join(path_cwd, 'input', 'settings')
    settings_file_path = tk.filedialog.askopenfilename(initialdir=settings_default_dir,
                                                       title="Select settings file",
                                                       filetypes=(("CSV files", "*.csv"),
                                                                  ("All files", "*.*")))
    if not settings_file_path:
        raise FileNotFoundError('No settings file selected')

    return scenarios_file_path, settings_file_path


def main():
    path_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_cwd = os.getcwd()
    if len(sys.argv) == 1:  # no arguments passed
        path_scenario, path_settings = select_arguments(path_pkg)
    elif len(sys.argv) == 3:  # two arguments passed
        path_scenario, path_settings = read_arguments(path_cwd, path_pkg)
    else:
        raise ValueError('Invalid number of arguments - please provide either none (GUI input) '
                         'or two arguments: scenarios file name or path and settings file name or path')

    sim.SimulationRun(path_scenario=path_scenario,
                      path_settings=path_settings,
                      execute=True)


if __name__ == '__main__':
    main()
