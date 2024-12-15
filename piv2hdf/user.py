"""User specific constants such as paths"""

import pathlib

import appdirs

from ._version import __version__

USER_DIRECTORY = pathlib.Path(appdirs.user_data_dir('piv2hdf', version=__version__))

TEST_DATA_DIRECTORY = pathlib.Path(__file__).parent.parent / 'tests/resources'
TEMPORARY_USER_DIRECTORY = USER_DIRECTORY / 'tmp'
TEMPORARY_USER_DIRECTORY.mkdir(parents=True, exist_ok=True)
USER_DIRECTORY.mkdir(parents=True, exist_ok=True)
