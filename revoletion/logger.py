#!/usr/bin/env python3

import logging
import multiprocessing as mp
import sys


class OptimizationSuccessfulFilter(logging.Filter):
    def filter(self,
               record):
        # Filter out log messages from the root logger
        return not (record.name == 'root' and record.msg == 'Optimization successful...')


def _get_logger_level(settings: 'SimulationSettings'):
    """
    Determine the logger level based on the settings.
    """
    return logging.DEBUG if settings.debugmode else logging.INFO


def _get_logger(settings: 'SimulationSettings',
                name: str = None,
                ):
    logger = logging.getLogger(name)

    # needs to be set here as root logger does not filter messages from the queue
    logger.setLevel(_get_logger_level(settings))

    # supress pyomo warnings
    logging.getLogger('pyomo.core').setLevel(logging.ERROR)

    # deactivate logging messages from gurobipy as it is not part of REVOL-E-TION's dependencies
    logging.getLogger('gurobipy').disabled = True

    return logger


def get_root_logger(paths: 'SimulationPaths',
                    settings: 'SimulationSettings'):

    logger = _get_logger(settings=settings,
                         name='root')

    # define log formatter
    log_formatter = logging.Formatter(f'%(levelname)-{len("WARNING")}s  '
                                      # f'%(name)-{max([len(el) for el in list(self.scenario_names) + ["root"]])}s  '
                                      f'%(name)-{10}s  '
                                      f'%(message)s')

    # define root logger handler for console output
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    log_stream_handler.addFilter(OptimizationSuccessfulFilter())
    logger.addHandler(log_stream_handler)

    # define root logger handler for file output
    log_file_handler = logging.FileHandler(paths.log)
    log_file_handler.setFormatter(log_formatter)
    log_file_handler.addFilter(OptimizationSuccessfulFilter())
    logger.addHandler(log_file_handler)

    return logger


def get_process_logger_sequential(name: str,
                                  settings: 'SimulationSettings',
                                  ):

    logger = _get_logger(name=name,
                         settings=settings,
                         )

    return logger


def get_process_logger_parallel(name: str,
                                settings: 'SimulationSettings',
                                log_queue: mp.Queue,
                                ):

    logger = _get_logger(name=name,
                         settings=settings,
                         )

    logger.propagate = False  # prevent inheritance of handlers from the root logger and duplicated messages

    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(queue_handler)

    return logger


def read_mplogger_queue(queue: mp.Queue):
    main_logger = logging.getLogger('main')

    while True:
        record = queue.get()
        if record is None:
            break
        main_logger.handle(record)
