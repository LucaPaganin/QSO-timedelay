import logging
import sys


def configure_logger(logger, logfile):
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    # create stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.ERROR)
    # create formatters and add them to the handlers
    logfile_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(logfile_formatter)
    stdout_handler.setFormatter(logging.Formatter('%(message)s'))
    stderr_handler.setFormatter(logging.Formatter('%(message)s'))
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

