"""piv2hdf: A package to convert PIV data to HDF5 files."""

import atexit
import pathlib
import shutil
import warnings
from typing import Dict

import h5rdmtoolbox as h5tbx
import yaml

from . import openpiv, cache
from ._logger import logger, set_loglevel
from ._version import __version__
from .config import set_config, set_pivattrs, get_config, reset_pivattrs
from .convention import cv
from .interface import PIVSnapshot, PIVPlane, PIVMultiPlane
from .user import USER_DIRECTORY

__all__ = ['PIVSnapshot', 'PIVPlane', 'PIVMultiPlane', 'logger', 'set_loglevel',
           'get_config', 'set_config', 'set_pivattrs', 'reset_pivattrs',
           '__version__']

h5tbx.use(None)

__this_dir__ = pathlib.Path(__file__).parent
_data_dir = __this_dir__ / 'data'


class Contacts:
    """A class to register contacts for the PIV convention."""

    def __init__(self):
        # read from user_dir
        self.contacts_filename = USER_DIRECTORY / 'contacts.yaml'
        if self.contacts_filename.exists():
            with open(self.contacts_filename, 'r') as f:
                self.registered_contacts = yaml.safe_load(f)
        else:
            self.registered_contacts = {}
        self.save()

    def __contains__(self, item):
        return item in self.registered_contacts

    def __getitem__(self, item):
        return self.registered_contacts[item]

    def __repr__(self):
        return f'Contacts({self.registered_contacts})'

    def __len__(self):
        return len(self.registered_contacts)

    def save(self):
        """Write the yaml file to user directory"""
        with open(self.contacts_filename, 'w') as f:
            yaml.safe_dump(self.registered_contacts, f)

    def register(self, **orcid: Dict):
        """Add a new contact. Provide one or multiple ORCID(s) as a dictionary."""
        self.registered_contacts.update(orcid)
        self.save()

    def clear(self):
        """removes all contacts"""
        self.registered_contacts = {}
        self.save()


contacts = Contacts()


@atexit.register
def clean_temp_data():
    """cleaning up the tmp directory"""
    for fname in cache.tmp_filenames:
        try:
            fname.unlink()
        except OSError:
            warnings.warn(f'Could not delete file {fname}. Consider deleting it manually.')
    for fdir in cache.tmp_dirnames:
        try:
            shutil.rmtree(fdir)
        except OSError:
            warnings.warn(f'Could not delete directory {fdir}. Consider deleting it manually.')
