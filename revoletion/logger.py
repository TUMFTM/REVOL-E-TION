#!/usr/bin/env python3

import datetime
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import override


@dataclass
class ColumnLayout:
    max_width: int = 0
    padding_right: int = 0
    align: str = "<"  # "<" left, ">" right

    @property
    def total_width(self):
        return self.max_width + self.padding_right

    def pad(self, value: str):
        return f"{str(value):{self.align}{self.total_width}}"


class OptimizationSuccessfulFilter(logging.Filter):
    def filter(self, record):
        # Filter out log messages from the root logger
        return not (record.name == "root" and record.msg == "Optimization successful...")


class LogFormatter(logging.Formatter):
    format_timestamp = ColumnLayout(
        max_width=19, padding_right=2, align="<"
    )  # "YYYY-MM-DD HH:MM:SS" has a fixed length of 19 characters
    format_levelname = ColumnLayout(
        max_width=8, padding_right=2, align="<"
    )  # WARNING and CRITICAL are the longest level name with 8 characters
    format_scenarioname = ColumnLayout(max_width=0, padding_right=3, align="<")

    def format(self, record):
        record.timestamp = datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        record._timestamp_str = self.format_timestamp.pad(record.timestamp)

        # Ensure levelname exists
        record.levelname = getattr(record, "levelname", "")
        record._levelname_str = self.format_levelname.pad(record.levelname)

        # Ensure scenarioname exists
        record.scenarioname = getattr(record, "scenarioname", "")
        record._scenarioname_str = self.format_scenarioname.pad(record.scenarioname)

        # Build horizon string dynamically
        n_horizon = getattr(record, "n_horizon", None)
        n_horizon_total = getattr(record, "n_horizon_total", None)
        if n_horizon is not None and n_horizon_total is not None:
            record._horizon_str = f"Horizon {n_horizon} of {n_horizon_total} - "
        else:
            record._horizon_str = ""

        return super().format(record)


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
        log_file: File where the logs are writing to.
        debugmode: Flag to control whether debugging is enabled. If True the log level is set to 'debug' else 'info'.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(_get_logger_level(debugmode))

    # Pad the level name column to the maximum level name length.
    # define log formatter
    log_formatter_stdout = LogFormatter(fmt="%(_levelname_str)s%(_scenarioname_str)s%(_horizon_str)s%(message)s")
    log_formatter_file = LogFormatter(
        fmt="%(_timestamp_str)s%(_levelname_str)s%(_scenarioname_str)s%(_horizon_str)s%(message)s"
    )

    # define root logger handler for console output
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter_stdout)
    log_stream_handler.addFilter(OptimizationSuccessfulFilter())
    root_logger.addHandler(log_stream_handler)

    # define root logger handler for file output
    log_file_handler = logging.FileHandler(log_file)
    log_file_handler.setFormatter(log_formatter_file)
    log_file_handler.addFilter(OptimizationSuccessfulFilter())
    root_logger.addHandler(log_file_handler)

    _configure_third_party_loggers()


def configure_process_logger_parallel(log_queue: mp.Queue, debugmode: bool):
    """
    Setup logging in a multiprocessing worker.

    Ensures that all log messages are sent to the main process and not logged inside the worker process.

    Args:
        log_queue: Queue to send the log messages to.
        debugmode: Flag to control whether debugging is enabled. If True the log level is set to 'debug' else 'info'.
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
    def __init__(self, logger, extra: dict | None = None):
        if extra is None:
            extra = {}

        # unwrap LoggerAdapter
        if isinstance(logger, logging.LoggerAdapter):
            base_logger = logger.logger
            parent_extra = dict(logger.extra)  # copy
        else:
            base_logger = logger
            parent_extra = {}

        # Merge:
        # - keep parent fields
        # - overwrite with new ones if keys collide
        merged_extra = {**parent_extra, **extra}

        super().__init__(base_logger, merged_extra)

    @override
    def process(self, msg, kwargs):
        # Merge adapter extras with per-call extras
        call_extra = kwargs.get("extra", {})
        kwargs["extra"] = {**self.extra, **call_extra}
        return msg, kwargs
