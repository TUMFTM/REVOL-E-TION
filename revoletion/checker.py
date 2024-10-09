#!/usr/bin/env python3

def check_settings_complete(run):
    target = ['solver', 'parallel', 'max_process_num', 'path_input_data', 'path_output_data', 'save_results',
              'save_des_results', 'print_results', 'save_plots', 'show_plots', 'save_system_graphs', 'dump_model',
              'debugmode']
    missing = [attr for attr in target if not hasattr(run, attr)]

    if len(missing) > 0:
        raise AttributeError(f'Not all required settings defined. Missing: {", ".join(missing)}')
