"""consts for PIVview module"""

from os import environ

from pathlib import Path

PIVVIEW_SOFTWARE = dict(name='pivview', version=999,
                        url='https://www.pivtec.com',
                        description='PIVTEC PIVview')

DIM_NAMES = ('z', 'reltime', 'y', 'x', 'ix', 'iy')
DIM_NAMES_TIMEAVERAGED = ('z', 'y', 'x', 'ix', 'iy')
DEFAULT_DATASET_LONG_NAMES = {'x': 'x-coordinate',
                              'y': 'y-coordinate',
                              'z': 'z-coordinate',
                              'ix': 'x-pixel-coordinate',
                              'iy': 'y-pixel-coordinate',
                              'reltime': 'reltime'}
IGNORE_ATTRS = ('mean_dx', 'mean_dy', 'mean_dz',
                'rms_dx', 'rms_dy', 'rms_dz',
                'coord_min', 'coord_max', 'coord_units', 'time_step', 'time_step_idx',
                'CLASS', 'NAME', 'DIMENSION_LIST', 'REFERENCE_LIST', 'COORDINATES')

MULTIPLANE_TITLE = 'PIV multi-plane file generated from PIVview netCDF4 files.'
PIVPLANE_TITLE = 'PIV plane file generated from PIVview netCDF4 files.'
PIVSNAPSHOT_TITLE = 'PIV snapshot file generated from a single PIVview netCDF4 file.'

PIV_FILE_TYPE_NAME = {'PIVSnapshot': 'snapshot',
                      'PIVPlane': 'single_plane',
                      'PIVMultiPlane': 'multi_plane'}

try:
    NCDF2PAR = Path(environ.get('ncdf2par'))
except TypeError:
    NCDF2PAR = None
