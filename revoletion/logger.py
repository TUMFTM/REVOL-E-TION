#!/usr/bin/env python3

import logging


def setup_logger(name, log_queue, run):
    logger = logging.getLogger(name)
    logging.getLogger('pyomo.core').setLevel(logging.ERROR)  # supress pyomo warnings
    if log_queue:
        logger.setLevel(logging.DEBUG if run.debugmode else logging.INFO)
        formatter = logging.Formatter('%(message)s')

        queue_handler = logging.handlers.QueueHandler(log_queue)
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)

    else:
        logger.setLevel(run.logger.level)
        for handler in run.logger.handlers:
            logger.addHandler(handler)

    return logger


def read_mplogger_queue(queue):
    main_logger = logging.getLogger('main')

    while True:
        record = queue.get()
        if record is None:
            break
        main_logger.handle(record)
