#!/usr/bin/env python3

import logging
import multiprocessing as mp
import sys
from pathlib import Path

from typing_extensions import override


class OptimizationSuccessfulFilter(logging.Filter):
    def filter(self, record):
        # Filter out log messages from the root logger
        return not (record.name == "root" and record.msg == "Optimization successful...")


class EmptyContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "context_str"):
            record.context_str = ""
        return True


class ContextPaddingFilter(logging.Filter):
    max_length = 0

    def filter(self, record):
        record.context_str = f"{record.context_str:<{self.max_length}}"
        return True


def _get_logger_level(debugmode: bool):
    """
    Determine the logger level based on the settings.
    """
    return logging.DEBUG if debugmode else logging.INFO


def _configure_third_party_loggers() -> None:
    # supress pyomo warnings
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)

    # deactivate logging messages from gurobipy as it is not part of REVOL-E-TION's dependencies
    logging.getLogger("gurobipy").disabled = True


def configure_root_logger(log_file: Path, debugmode: bool = False) -> None:
    """
    Configure the `revoletion` root logger with its level according to `debugmode` and
    output handlers to console and the file `log_file`.

    Args:
        log_file: File where the logs are writting to.
        debugmode: Flag to control whether debugging is enabled. If True the log level is set to 'debug' else 'info'.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(_get_logger_level(debugmode))

    # Pad the level name column to the maximum level name length.
    level_name_len = len("WARNING")
    # define log formatter
    log_formatter = logging.Formatter(fmt=f"%(levelname)-{level_name_len + 2}s%(context_str)s%(message)s")

    # define root logger handler for console output
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    log_stream_handler.addFilter(OptimizationSuccessfulFilter())
    log_stream_handler.addFilter(EmptyContextFilter())
    log_stream_handler.addFilter(ContextPaddingFilter())
    root_logger.addHandler(log_stream_handler)

    # define root logger handler for file output
    log_file_handler = logging.FileHandler(log_file)
    log_file_handler.setFormatter(log_formatter)
    log_file_handler.addFilter(OptimizationSuccessfulFilter())
    log_file_handler.addFilter(EmptyContextFilter())
    log_stream_handler.addFilter(ContextPaddingFilter())
    root_logger.addHandler(log_file_handler)

    _configure_third_party_loggers()


def configure_process_logger_parallel(log_queue: mp.Queue, debugmode: bool):
    """
    Setup logging in a multiprocessing worker.

    Ensures that all log messages are sent to the main process and not logged inside the worker process.

    Args:
        log_queue: Queue to sent the log messages to.
    """
    logger = logging.getLogger()
    # For multiprocessing environments using the `spawn` method, the root logger is not inherited
    # from the parent. If the log level is not set in the worker, no log messages are forwarded to the parent.
    logger.setLevel(_get_logger_level(debugmode))
    logger.handlers.clear()

    # Ensure that no duplicate log messages appear.
    logger.propagate = False

    queue_handler = logging.handlers.QueueHandler(log_queue)

    logger.addHandler(queue_handler)

    _configure_third_party_loggers()


def read_mplogger_queue(queue: mp.Queue):
    logger = logging.getLogger()
    while True:
        record = queue.get()
        if record is None:
            break
        logger.handle(record)


class ContextLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    @override
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"]["context_str"] = self.extra["context_str"]
        return msg, kwargs
