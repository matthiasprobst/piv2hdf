"""Publish conversion outputs only after writing, callbacks and validation succeed."""
import inspect
import os
from functools import wraps
from pathlib import Path
from tempfile import mkstemp


def atomic_conversion(func):
    signature = inspect.signature(func)

    @wraps(func)
    def convert(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        converter = bound.arguments['self']
        destination = bound.arguments.get('hdf_filename')
        if destination is None:
            destination = converter._default_filename()
        destination = Path(destination)
        if destination.is_dir():
            destination = destination / f'{destination.stem}.hdf'
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staging = mkstemp(prefix=f'.{destination.stem}-', suffix='.hdf', dir=destination.parent)
        os.close(descriptor)
        staging = Path(staging)
        old_filename = converter.hdf_filename
        bound.arguments['hdf_filename'] = staging
        try:
            func(*bound.args, **bound.kwargs)
            os.replace(staging, destination)
        except BaseException:
            converter.hdf_filename = old_filename
            raise
        finally:
            staging.unlink(missing_ok=True)
        converter.hdf_filename = destination
        return destination

    return convert
