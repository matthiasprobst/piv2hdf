import logging
import pathlib
import os
from logging.handlers import RotatingFileHandler
from typing import Union

name = __package__

# Initialize logger, set high level to prevent ipython debugs. File level is
# set below
DEFAULT_LOGGING_LEVEL = logging.INFO
logger = logging.getLogger(__package__)
logger.setLevel(DEFAULT_LOGGING_LEVEL)

_formatter = logging.Formatter(
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d_%H:%M:%S')

if os.environ.get('PIV2HDF_LOG_DIR'):
    log_directory = pathlib.Path(os.environ['PIV2HDF_LOG_DIR'])
    log_directory.mkdir(parents=True, exist_ok=True)
    _file_handler = RotatingFileHandler(log_directory / f'{name}.log', maxBytes=int(5e6), backupCount=2)
    _file_handler.setLevel(DEFAULT_LOGGING_LEVEL)
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(DEFAULT_LOGGING_LEVEL)
_stream_handler.setFormatter(_formatter)

logger.addHandler(_stream_handler)


def set_loglevel(level: Union[str, int]):
    """Set the log level"""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
