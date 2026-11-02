import json
import os
import pathlib
import re
from datetime import datetime
from typing import Dict, Tuple, Union, List, Optional

import numpy as np
import xarray as xr
from netCDF4 import Dataset as ncDataset
from h5rdmtoolbox import get_ureg

from piv2hdf import logger
from . import parameter as pivview_parameter
from .. import flags
from ..processing import mask_vectors
from ..config import get_config
from ..interface import PIVFile, UserDefinedHDF5Operation, PIVData
from ..utils import is_time, get_uint_type

__this_file__ = pathlib.Path(__file__).resolve()


def _find_common_entries(dictionaries):
    common_dict = {}

    # Iterate over the keys of the first dictionary
    for key in dictionaries[0]:
        # Check if all dictionaries contain the key
        if all(key in dictionary for dictionary in dictionaries):
            # Check if the values are the same for all dictionaries
            if all(d[key] == dictionaries[0][key] for d in dictionaries):
                # Check if the value is a nested dictionary
                if isinstance(dictionaries[0][key], dict):
                    # Recursively find common entries in nested dictionaries
                    nested_dicts = [d[key] for d in dictionaries]
                    common_dict[key] = _find_common_entries(nested_dicts)
                else:
                    common_dict[key] = dictionaries[0][key]

    return common_dict


def _process_pivview_root_attributes(root_attrs):
    """Create new or renames or deletes dictionary entries"""
    root_attrs['recording_date'] = root_attrs['creation_date']

    del root_attrs['creation_date']
    if 'filename' in root_attrs:
        del root_attrs['filename']
    if 'image_bkgd_file' in root_attrs:
        del root_attrs['image_bkgd_file']  # should be dataset!
    if 'image_bkgd2_file' in root_attrs:
        del root_attrs['image_bkgd2_file']  # should be dataset!
    if 'image_mask_file' in root_attrs:
        del root_attrs['image_mask_file']  # should be dataset or can always be reconstructed from piv_flags!
    return root_attrs


def process_pivview_nc_data(nc_file: pathlib.Path, interpolate: bool,
                            masking: str,
                            timestep: float, time_unit: str = 's',
                            z_source: str = 'coord_min',
                            compute_dwdz: bool = False,
                            build_coord_datasets: bool = True
                            ) -> Tuple[Dict, Dict, Dict]:
    """
    Reads data and attributes from netCDF file. Results are stored in dictionary. Interpolation
    to fill "holes"/masked areas is applied if asked. Data arrays x, y, z and time are created.
    Array shape is changed from (1, ny, nx) and (1, ny, nx, nv) to (ny, nx). Thus, new variables are created:
        velocity (1, ny, nx, 2[3]) --> u (ny, nx), v (ny, nx) [, w (ny, nx)]
        piv_data (1, ny, nx, 2[3]) --> dx (ny, nx), dy (ny, nx) [, dz (ny, nx)]
        piv_peak1 (1, ny, nx, 3) --> piv_peak1_dx (ny, nx), piv_peak1_dy (ny, nx), piv_peak1_corr
        piv_peak2 (see above)
        piv_peak3 (see above)

    Parameters
    ----------
    nc_file : Path
        path to nc file
    interpolate : bool
        Linearly fill masked holes within the valid points' convex hull. Outside
        that hull, or with insufficient donors, holes remain NaN. Raw fields and
        original flags are retained separately.
    masking : str
        Required mode: strict accepts ACTIVE; sepeaks also accepts REPLACED;
        slack also accepts FILTERED, INTERPOLATED and MANUALEDIT. Invalid and
        unknown bits are rejected. The reader's module-level default is slack.
    timestep : float
        Relative recording time, expressed in ``time_unit``.
    time_unit : str, default='s'
        Time unit understood by Pint; the numeric timestep is not converted.
    z_source: str or tuple, optional='coord_min'
        A list/tuple ``(value, unit)`` or xarray DataArray can supply z manually.
        It is also possible to detect the z-location from the nc-file (experience shows
        that this was not always correctly set by the experimenter...). Alternative,
        the z-location is tried to determine by the file name.
        Automatic detection from nc file can be used scanning one of the following
        attributes which then are to be passed as string for this parameter:
            * 'coord_min'
            * 'coord_max'
            * 'origin_offset'
            * 'file'
    compute_dwdz: bool, default=False
        For stereo data, derive dwdz from in-plane gradients by incompressible
        continuity. This requires dudx and dvdy; applicability to the experiment
        must be established separately. The assumption is recorded in metadata.
    build_coord_datasets: bool, optional=True
        Include spatial and pixel coordinates. Keep True for data passed to the
        normal snapshot/plane writers. Relative time is included in either case.

    Returns
    -------
    piv_data_array_dict : dict
        Dictionary containing the arrays
    nc_root_attributes : dict
        Selected source root metadata (content, creation date and software)
    variable_attributes : dict
        Attribute dictionary of dataset variables

    Notes
    -----
    Credits to M. Elfner (ITS, Karlsruhe Institute of Technology)

    """

    # TODO get rid of nc_root_attributes

    def _build_meshgrid_xy(coord_min, coord_max, width, height):
        """generates coord meshgrid from velocity attribute stating min/max of coordinates x, y"""
        _x = np.linspace(coord_min[0], coord_max[0], width)
        _y = np.linspace(coord_min[1], coord_max[1], height)
        return np.meshgrid(_x, _y)

    # read netCDF4 data (array, root attributes and variable attributes)
    with ncDataset(nc_file, "r") as nc_rootgrp:
        dims = nc_rootgrp.dimensions
        d, h, w, vd = (dims['data_array_depth'].size, dims['data_array_height'].size,
                       dims['data_array_width'].size, dims['vec_dims'].size)

        if d > 1:
            logger.critical('File with depth not implemented')
            raise NotImplementedError('File with depth not implemented')

        root_attributes = {key: nc_rootgrp.getncattr(key) for key in
                           ('file_content', 'creation_date', 'software')}
        nc_root_attributes = _process_pivview_root_attributes(
            {attr: nc_rootgrp.getncattr(attr) for attr in nc_rootgrp.ncattrs()})

        variable_attributes = {}

        # Variable information.
        nc_data_array_dict = nc_rootgrp.variables

        if nc_root_attributes['outlier_interpolate'] and masking != 'slack':
            logger.debug('Outlier masking does not conform with pivview settings in nc '
                         f'(outlier_interpolate={nc_root_attributes["outlier_interpolate"]} vs {masking}) - '
                         f'averages might differ')
        elif nc_root_attributes['outlier_try_other_peak'] and masking != 'sepeaks':
            logger.debug('Outlier masking does not conform with pivview settings in nc '
                         f'(outlier_interpolate={nc_root_attributes["outlier_interpolate"]} vs {masking}) - '
                         f'averages might differ')

        # processed
        piv_data_array_dict = {}

        for v in nc_data_array_dict.keys():
            variable_attributes[v] = {key: nc_data_array_dict[v].getncattr(key) for key in
                                      nc_data_array_dict[v].ncattrs()}

        if 'piv_flags' in nc_data_array_dict:
            pivflags = np.asarray(nc_data_array_dict['piv_flags'])[0, ...]  # dtype int8

            # optimize memory usage
            if pivflags.dtype == 'int8':
                dtype_pivflags = 'uint8'
            elif pivflags.dtype == 'int16':
                dtype_pivflags = 'uint16'
            elif pivflags.dtype == 'int32':
                dtype_pivflags = 'uint32'
            else:
                raise ValueError(f'Invalid dtype for piv_flags: {pivflags.dtype}')
            piv_data_array_dict['piv_flags'] = pivflags.astype(dtype_pivflags)
        else:
            logger.warning(f'PIV flags not found, masking disabled, mode {masking} not used')
            mask = np.ones((h, w)).astype(bool)

        for k, v in nc_data_array_dict.items():
            if k == 'piv_data':
                for i, name in zip(range(v.shape[-1]), ('dx', 'dy', 'dz')):
                    data = v[0, :, :, i]
                    piv_data_array_dict[name] = data
                    variable_attributes[name] = {'units': 'pixel'}
            elif k == 'velocity':
                for i, name in zip(range(v.shape[-1]), ('u', 'v', 'w')):
                    data = v[0, :, :, i]
                    piv_data_array_dict[name] = data
                    variable_attributes[name] = {'units': variable_attributes['velocity']['units']}
            elif k == 'piv_flags':
                variable_attributes[k].update({'units': ''})  # , 'standard_name': 'piv_flags'
                continue  # We take the flags from above to save computation time
            elif k in ('piv_peak1', 'piv_peak2', 'piv_peak3'):
                for i, suffix, units in zip(range(v.shape[-1]),
                                            ('_dx', '_dy', '_corr'),
                                            ('pixel', 'pixel', '')):
                    data = v[0, :, :, i]
                    name = f'{k}{suffix}'
                    piv_data_array_dict[name] = data
                    variable_attributes[name] = {'units': units}
            else:
                data = v[0, ...]
                piv_data_array_dict[k] = data
                if k == 'piv_correlation_coeff':
                    variable_attributes[k].update({'units': ''})
                elif k == 'piv_snr_data':
                    variable_attributes[k].update({'units': ''})
                elif k == 'piv_3c_residuals':
                    variable_attributes[k].update({'units': '',
                                                   'long_name': 'least square residual for z_velocity',
                                                   'comment': 'Residuals from least-squares fit to determined '
                                                              'out-of-plane component. It is a measure of quality of '
                                                              'the vector reconstruction and should be lower than 0.5 '
                                                              'pixel'})
                elif 'velocity gradient' in variable_attributes[k].get('long_name', ''):
                    variable_attributes[k].setdefault('units', '1/s')

        xm, ym = _build_meshgrid_xy(nc_data_array_dict['velocity'].coord_min,
                                    nc_data_array_dict['velocity'].coord_max, w, h)
        mask_vectors(piv_data_array_dict, variable_attributes, masking,
                     interpolate=interpolate, coordinates=(xm, ym))

        # Check if source for w gradients is available, if so compute.
        # CAREFUL with indexing: Meshgrids are YX indexed by default!
        if 'w' in piv_data_array_dict or build_coord_datasets:
            fr, to = nc_data_array_dict['velocity'].coord_min, nc_data_array_dict['velocity'].coord_max
            px_fr, px_to = nc_data_array_dict['piv_data'].coord_min, nc_data_array_dict['piv_data'].coord_max

        if 'w' in piv_data_array_dict:
            # grid spacing is assumed to be homogeneous - is the case for PIV measurements!
            dx, dy = (to[:2] - fr[:2]) / (np.array([w, h]) - 1)
            if not np.all(np.isfinite([dx, dy])) or dx == 0 or dy == 0:
                raise ValueError('Velocity gradients require finite, nonzero grid spacing')
            gradient_factor = get_ureg().Quantity(
                1, f"({variable_attributes['velocity']['units']})/({nc_root_attributes['length_conversion_units']})"
            ).to('1/s').magnitude

            piv_data_array_dict['dwdx'] = np.gradient(piv_data_array_dict['w'], dx, axis=1) * gradient_factor
            piv_data_array_dict['dwdy'] = np.gradient(piv_data_array_dict['w'], dy, axis=0) * gradient_factor

            _gradient_unit = '1/s'
            variable_attributes['dwdx'] = {'units': _gradient_unit,
                                           'long_name': 'velocity gradient dw/dx'}
            variable_attributes['dwdy'] = {'units': _gradient_unit,
                                           'long_name': 'velocity gradient dw/dy'}

            if compute_dwdz:
                if 'dudx' in piv_data_array_dict.keys() and 'dvdy' in piv_data_array_dict.keys():
                    logger.info("Gradient \"dwdz\" calculated from continuity equation assuming incompressible flow!")
                    dudx = get_ureg().Quantity(piv_data_array_dict['dudx'], variable_attributes['dudx']['units']).to('1/s').magnitude
                    dvdy = get_ureg().Quantity(piv_data_array_dict['dvdy'], variable_attributes['dvdy']['units']).to('1/s').magnitude
                    piv_data_array_dict['dwdz'] = -dudx - dvdy
                    variable_attributes['dwdz'] = {
                        'units': _gradient_unit,
                        'long_name': 'velocity gradient dw/dz assuming incompressible flow',
                        'assumption': 'incompressible flow',
                        'derived_from': ['dudx', 'dvdy'],
                        # 'standard_name': 'z_derivative_of_w_velocity_assuming_incompressibility',
                        'comment': 'Calculated from continuity equation using dud and dvdy and assuming '
                                   'incompressible flow!'
                    }
                else:
                    raise ValueError('Continuity-derived dwdz requires dudx and dvdy')

        piv_data_array_dict['reltime'] = timestep
        variable_attributes['reltime'] = {'long_name': 'Recording time since start.',
                                          'units': time_unit}
        if not is_time(variable_attributes['reltime']['units']):
            raise ValueError(f'Invalid time unit: {time_unit}')

        # x,y,z,t are not part of PIVview netCDF variables
        if build_coord_datasets:  # can speed up computation when plane or case HDF files are generated
            # the velocity dataset has the attribute coord_min and coord_max from which the coordinates can be derived:
            piv_data_array_dict['x'] = np.linspace(fr[0], to[0], w)
            piv_data_array_dict['y'] = np.linspace(fr[1], to[1], h)
            variable_attributes['x'] = {'units': nc_root_attributes['length_conversion_units']}
            variable_attributes['y'] = {'units': nc_root_attributes['length_conversion_units']}
            if not px_fr[0] >= 0:
                raise ValueError(f'Invalid pixel coordinate: {px_fr[0]}')
            if not px_fr[1] >= 0:
                raise ValueError(f'Invalid pixel coordinate: {px_fr[1]}')

            piv_data_array_dict['ix'] = np.linspace(px_fr[0], px_to[0], w).astype(get_uint_type(px_to[0]))
            piv_data_array_dict['iy'] = np.linspace(px_fr[1], px_to[1], h).astype(get_uint_type(px_to[0]))
            variable_attributes['ix'] = {'long_name': 'pixel x-location of vector',
                                         'units': 'pixel'}
            variable_attributes['iy'] = {'long_name': 'pixel y-location of vector',
                                         'units': 'pixel'}

            # Z position information
            z_unit = nc_root_attributes['length_conversion_units']  # default unit of z comes from file
            if isinstance(z_source, str):
                if z_source == 'coord_min':
                    z = fr[2]
                elif z_source == 'coord_max':
                    z = to[2]
                elif z_source == 'origin_offset':
                    z = nc_root_attributes['origin_offset_z']
                elif z_source == 'file':
                    try:
                        z = float(
                            re.findall(
                                r'([-+\d].*)', os.path.split(os.path.dirname(nc_file))[-1])[0])
                    except Exception as e:
                        logger.warning(f'Z level detection failed due to: {e}')
                        z = 0
            elif isinstance(z_source, xr.DataArray):
                z = z_source.values
                z_unit = z_source.units
            elif isinstance(z_source, tuple) or isinstance(z_source, list):
                z, z_unit = z_source
            else:
                logger.warning('Z level detection failed: Invalid mode')
                z = 0

            piv_data_array_dict['z'] = z
            variable_attributes['z'] = {'units': z_unit}

        # correct shape if needed to (nz, nt, ny, nx) or (nz, nt, ny, nx, nv) respectively
        for k, v in piv_data_array_dict.items():
            if k not in ('x', 'y', 'z', 'reltime'):
                if v.ndim > 2:  # single var dataset or vector dataset
                    piv_data_array_dict[k] = v[0, ...]
                else:  # mask, coordinates
                    piv_data_array_dict[k] = v

    return piv_data_array_dict, root_attributes, variable_attributes


class PIVViewNcFile(PIVFile):
    """Interface class to a PIVview netCDF4 file for a 2D recording (using .par parameter files)

    Parameters
    ----------
    filename : pathlib.Path
        path to the nc file
    parameter : Union[None, pivview_parameter.PIVviewParamFile, pathlib.Path], optional
        path to the parameter file, by default None
    user_defined_hdf5_operations : optional UserDefinedHDF5Operation or List[UserDefinedHDF5Operation]
        Injects code that is executed after the HDF5 file is created, by default None
    kwargs : dict
        additional keyword arguments
    """
    suffix: str = '.nc'
    __parameter_cls__ = pivview_parameter.PIVviewParamFile

    def __init__(self,
                 filename: pathlib.Path,
                 parameter: Optional[Union[pivview_parameter.PIVviewParamFile, pathlib.Path]] = None,
                 user_defined_hdf5_operations: Optional[
                     Union[UserDefinedHDF5Operation, List[UserDefinedHDF5Operation]]] = None,
                 compute_dwdz: bool = False,
                 **kwargs):
        # if user_defined_hdf5_operations is None:
        #     user_defined_hdf5_operations = add_standard_name_operation
        self.compute_dwdz = compute_dwdz
        super().__init__(filename, parameter, user_defined_hdf5_operations=user_defined_hdf5_operations, **kwargs)

    def read(self,
             relative_time: float,
             **kwargs) -> PIVData:
        """reads and processes nc data

        Parameters
        ----------
        relative_time : float
            relative time of the snapshot read
        kwargs:
            build_coord_datasets : bool, optional=True
                whether to build coordinate datasets, by default True

        Returns
        -------
        nc_data : dict
            dictionary of data arrays
        nc_root_attr : dict
            dictionary of root attributes
        nc_variable_attr : dict
            dictionary of variable attributes
        """

        build_coord_datasets = kwargs.get("build_coord_datasets", True)
        from . import config as pivview_config
        masking = pivview_config['masking']
        interpolation = pivview_config['interpolation']
        z_source = pivview_config['z_source']
        nc_data, nc_root_attr, nc_variable_attr = process_pivview_nc_data(self.filename,
                                                                          z_source=z_source,
                                                                          timestep=relative_time,
                                                                          masking=masking,
                                                                          interpolate=interpolation,
                                                                          compute_dwdz=self.compute_dwdz,
                                                                          build_coord_datasets=build_coord_datasets)
        # nc_root_attr['filename'] = nc_root_attr.pop('long_name')
        # unique_flags = np.unique(piv_data.data['piv_flags'][:])
        if 'piv_flags' in nc_data:
            nc_variable_attr['piv_flags']['flag_meaning'] = json.dumps({flag.value: flag.name for flag in flags.Flags})
        nc_root_attr.update(pivview_masking=masking, pivview_interpolation=interpolation,
                            assumed_incompressible=self.compute_dwdz)
        return PIVData(nc_data, nc_variable_attr, nc_root_attr)

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: Union[datetime, List[datetime]],  # iso-format of datetime!
               z: Union[str, float, None] = None,
               **kwargs) -> pathlib.Path:
        """Serialize this adapter through the shared snapshot writer."""
        from ..writer import write_snapshot
        return write_snapshot(self.read(relative_time, **kwargs), hdf_filename,
                              self.filename, recording_dtime, z)


class PIVViewStereoNcFile(PIVViewNcFile):
    """Read PIVview stereo 2D3C results using a .cfg parameter file.

    Three velocity components are measured on a plane, not throughout a volume.
    Reading and masking options are shared with ``PIVViewNcFile``.
    """
    __suffix__ = '.nc'
    __parameter_cls__ = pivview_parameter.PivViewConfigFile
