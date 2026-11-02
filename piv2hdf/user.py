"""User specific constants such as paths"""

import pathlib
import os

import appdirs

from ._version import __version__

USER_DIRECTORY = pathlib.Path(os.environ.get('PIV2HDF_USER_DIR') or appdirs.user_data_dir('piv2hdf', version=__version__))

TEST_DATA_DIRECTORY = pathlib.Path(__file__).parent.parent / 'tests/resources'
TEMPORARY_USER_DIRECTORY = USER_DIRECTORY / 'tmp'
