#!/usr/bin/env python3

import os
import sys
import tkinter as tk
import tkinter.filedialog
import warnings

from revoletion import simulation as sim


# raise UserWarnings about infeasibility as errors to catch them properly
warnings.simplefilter(action='error', category=UserWarning)

# only print FutureWarnings once (in theory)
# Set to 'ignore' to suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def read_arguments(path_pkg):

    if os.path.isfile(sys.argv[1]):
        scenarios_file_path = sys.argv[1]
    elif os.path.isfile(os.path.join(path_pkg, 'input', 'scenarios', sys.argv[1])):
        scenarios_file_path = os.path.join(path_pkg, 'input', 'scenarios', sys.argv[1])
    else:
        raise FileNotFoundError(f'Scenario file or path not found: {sys.argv[1]}')

    if os.path.isfile(sys.argv[2]):
        settings_file_path = sys.argv[2]
    elif os.path.isfile(os.path.join(path_pkg, 'input', 'settings', sys.argv[2])):
        settings_file_path = os.path.join(path_pkg, 'input', 'settings', sys.argv[2])
    else:
        raise FileNotFoundError(f'Settings file or pathnot found: {sys.argv[2]} not found')

    return scenarios_file_path, settings_file_path


def select_arguments(path_pkg):

    root = tk.Tk()
    root.withdraw()  # hide small tk-window
    root.lift()  # make sure all tk windows appear in front of other windows

    # get scenarios file
    scenarios_default_dir = os.path.join(path_pkg, 'input', 'scenarios')
    scenarios_file_path = tk.filedialog.askopenfilename(initialdir=scenarios_default_dir,
                                                        title="Select scenario file",
                                                        filetypes=(("CSV files", "*.csv"),
                                                                   ("All files", "*.*")))
    if not scenarios_file_path:
        raise FileNotFoundError('No scenario file selected')

    # get settings file
    settings_default_dir = os.path.join(path_pkg, 'input', 'settings')
    settings_file_path = tk.filedialog.askopenfilename(initialdir=settings_default_dir,
                                                       title="Select settings file",
                                                       filetypes=(("CSV files", "*.csv"),
                                                                  ("All files", "*.*")))
    if not settings_file_path:
        raise FileNotFoundError('No settings file selected')

    return scenarios_file_path, settings_file_path


def main():
    path_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) == 1:  # no arguments passed
        path_scenario, path_settings = select_arguments(path_pkg)
    elif len(sys.argv) == 3:  # two arguments passed
        path_scenario, path_settings = read_arguments(path_pkg)
    else:
        raise ValueError('Invalid number of arguments - please provide either none (GUI input) '
                         'or two arguments: scenarios file name or path and settings file name or path')

    sim.SimulationRun(path_scenario=path_scenario,
                      path_settings=path_settings,
                      execute=True)


if __name__ == '__main__':
    main()
