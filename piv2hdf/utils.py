import pathlib
from itertools import count
from typing import Union, Tuple

import numpy as np
import yaml
from pint import Unit, Quantity

from . import cache, user

_filecounter = count()
_dircounter = count()
time_dimensionality = Unit('s').dimensionality
length_dimensionality = Unit('m').dimensionality


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def make_bold(string):
    """make string bold"""
    return f"{bcolors.BOLD}{string}{bcolors.ENDC}"


def read_structXML_wrapper(xmlFile):
    """
    This quick helper loads an existing vtr and converts it to a vtk dataset
    to use in mayavi. Add it with pipeline.add_dataset.

    Parameters
    ----------
    xmlFile : str
        Input stl data to load.

    Returns
    -------
    data : vtkRectilinearGrid
        Loaded XML data file.
    """
    try:
        import vtk
    except ImportError:
        ImportError('Package "vtk" not installed. Please install it first.')
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(xmlFile)
    reader.Update()
    data = reader.GetOutput()
    return data


def is_time(unit):
    """
    Returns true if unit is a time unit, e.g. 'ms' or 's'
    """
    if unit == 's':
        return True
    else:
        return unit.dimensionality == time_dimensionality


def is_length(unit):
    """
    Returns true if unit is a time unit, e.g. 'ms' or 's'
    """
    if unit == 'm':
        return True
    else:
        return unit.dimensionality == length_dimensionality


def generate_temporary_filename(prefix='tmp', suffix: str = '') -> pathlib.Path:
    """generates a temporary filename in user tmp file directory

    Parameters
    ----------
    prefix: str, optional='tmp'
        prefix string to put in front of name
    suffix: str, optional=''
        suffix (including '.')

    Returns
    -------
    tmp_filename: pathlib.Path
        The generated temporary filename
    """
    _filename = user.TEMPORARY_USER_DIRECTORY / f"{prefix}{next(_filecounter)}{suffix}"
    while _filename.exists():
        _filename = user.TEMPORARY_USER_DIRECTORY / f"{prefix}{next(_filecounter)}{suffix}"
    cache.tmp_filenames.append(_filename)
    return _filename


def generate_temporary_directory(prefix='tmp') -> pathlib.Path:
    """generates a temporary directory in user tmp file directory

    Parameters
    ----------
    prefix: str, optional='tmp'
        prefix string to put in front of name

    Returns
    -------
    tmp_filename: pathlib.Path
        The generated temporary filename
    """
    _dir = user.TEMPORARY_USER_DIRECTORY / f"{prefix}{next(_dircounter)}"
    while _dir.exists():
        _dir = user.TEMPORARY_USER_DIRECTORY / f"{prefix}{next(_dircounter)}"
    _dir.mkdir(parents=True)
    cache.tmp_dirnames.append(_dir)
    return _dir


def validate_directory(directory_path: pathlib.Path) -> pathlib.Path:
    """Raise an error if directory does not exist or passed
    path is not a directory"""
    directory_path = pathlib.Path(directory_path)
    if not directory_path.exists():
        raise NotADirectoryError(f'Directory not found: {directory_path}')
    if not directory_path.is_dir():
        raise NotADirectoryError(f'Not a directory {directory_path}')
    return directory_path


def _get_uint(max_val):
    """get uint type based on max value"""
    if max_val < 2 ** 8:
        return 'uint8'
    if max_val < 2 ** 16:
        return 'uint16'
    if max_val < 2 ** 32:
        return 'uint32'
    return 'uint64'


def get_uint_type(data: Union[np.ndarray, int, float]) -> str:
    """estimate uint dtype based on data"""
    if isinstance(data, (int, float)):
        return _get_uint(data)
    return _get_uint(data.max())


def parse_z(z: Union[str, int, float]) -> Tuple[Union[int, float], str]:
    """Parses coordinate, which can be a number or a string

    Parameters
    ----------
    z: Union[str, int, float]
        The z-coordinate input. If a number (int, float), then unit is assumed to be [m].
        If a string is provided, it is processed by `pint.Quantity`

    Returns
    -------
    mag, unit: Tuple[Union[int, float], str]
        Magnitude and unit derived based on the input.
    """
    if isinstance(z, str):
        z_quantity = Quantity(z)  # .to('m')
        mag = z_quantity.magnitude
        unit = str(z_quantity.units)
    elif isinstance(z, (int, float)):
        unit = 'm'
        mag = z
    else:
        raise TypeError(f"z coordinate data must be a number or a string but got {type(z)}")
    return mag, unit


# def create_dataset(*, h5, name, **kwargs):
#     dtype = kwargs.pop('dtype', DEFAULT_CONFIG['dtypes'].get(name, None))
#     return h5.create_dataset(name=name, dtype=dtype, **kwargs)


def read_translation_yaml_file(filename):
    with open(filename, 'r') as f:
        pivview_translation = yaml.safe_load(f)
    return pivview_translation


def write_final_interrogation_window_size_to_h5_group(h5grp, interrogation_window_size, dtype=None):
    if isinstance(interrogation_window_size, int):
        interrogation_window_size = (interrogation_window_size, interrogation_window_size)
    if dtype is None:
        dtype = 'int16'
    if not isinstance(interrogation_window_size, (list, tuple)):
        raise TypeError(
            f'interrogation_window_size must be int or list or tuple, got {type(interrogation_window_size)}')

    if not len(interrogation_window_size) in (2, 3):
        raise ValueError(f'interrogation_window_size must have length 2, got {len(interrogation_window_size)}')

    for prefix, iws in zip(('x', 'y'), interrogation_window_size):
        iwsds = h5grp.create_dataset(f'{prefix}_final_iw_size',
                                     data=iws,
                                     dtype=dtype)
        iwsds.attrs.update({
            # 'standard_name': f'{prefix}_final_interrogation_window_size',
            'long_name': f'the final interrogation window size in {prefix}-direction',
            'units': 'pixel'})


def write_final_interrogation_overlap_size_to_h5_group(h5grp, overlap_size, dtype=None):
    if isinstance(overlap_size, int):
        overlap_size = (overlap_size, overlap_size)
    if dtype is None:
        dtype = 'int16'
    if not isinstance(overlap_size, (list, tuple)):
        raise TypeError(f'overlap_size must be int or list or tuple, got {type(overlap_size)}')

    if not len(overlap_size) == 2:
        raise ValueError(f'overlap_size must have length 2, got {len(overlap_size)}')

    for prefix, iws in zip(('x', 'y'), overlap_size):
        iwods = h5grp.create_dataset(f'{prefix}_final_overlap',
                                     data=iws,
                                     dtype=dtype)
        iwods.attrs.update({
            # 'standard_name': f'{prefix}_final_interrogation_window_overlap_size',
            'long_name': f'the final interrogation window size in {prefix}-direction',
            'units': 'pixel'})
