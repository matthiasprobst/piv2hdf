try:
    from importlib.metadata import version as _version
except ImportError as e:
    raise ImportError('Most likely you have python<3.8 installed or in case you '
                      'ran `pytest` you may not have installed it for the correct '
                      'python version. At least 3.8 is required.')

__version__ = _version('piv2hdf')
