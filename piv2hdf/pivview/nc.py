import json
import os
import pathlib
import re
from pathlib import Path
from typing import Dict, Tuple, Union, List, Optional

import h5py
import numpy as np
import xarray as xr
from netCDF4 import Dataset as ncDataset
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from piv2hdf import logger
from . import parameter as pivview_parameter
from .const import *
from .. import flags
from ..config import get_config
from ..interface import PIVFile, UserDefinedHDF5Operation
from ..time import create_recording_datetime_dataset
from ..utils import (is_time, get_uint_type,
                     parse_z)

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
    Array shape is changed from (1, ny, nx) and (1, ny, nx, nv) to (ny, nx). Thus new variables are
    created:
        velocity (1, ny, nx, 2[3]) --> u (ny, nx), v (ny, nx) [, w (ny, nx)]
        piv_data (1, ny, nx, 2[3]) --> dx (ny, nx), dy (ny, nx) [, dz (ny, nx)]
        piv_peak1 (1, ny, nx, 3) --> piv_peak1_dx (ny, nx), piv_peak1_dy (ny, nx), piv_peak1_corr
        piv_peak2 (see above)
        piv_peak3 (see above)

    Parameters
    ----------
    nc_file : Path
        path to nc file
    interpolate : bool, optional=True
        Use space interpolation (linear) in each timeframe to patch holes
        left out after masking.
    masking : str, optional='sepeaks'
        Masking mode from piv flags. Can be slack or sepeaks or strict.
            strict: only uses true good values
            slack: also uses interpolated values.
            sepeaks: uses strict and second corr. peaks.
        The default is 'sepeaks'.
    timestep : float
        Time step in [s] relative to measurement start. If astropy-quantity
        is passed, unit is taken from there.
    time_unit : str, optional='s'
        unit taken for time if time was not of type astropy.quantity.
    z_source: str or tuple, optional='coord_min'
        The z-location can be specified manually by passing a astropy quantity
        or a list/tuple in form  (value: float, unit: str).
        It also possible to detect the z-location from the nc-file (experience shows
        that this was not always correctly set by the experimenter...). Alternative,
        the z-location is tried to determined by the file name.
        Automatic detection from nc file can be used scanning one of the following
        attributes which then are to be passed as string for this parameter:
            * 'coord_min'
            * 'coord_max'
            * 'origin_offset'
            * 'file'
    compute_dwdz: bool, optional=False
        Computing the out-of-plane gradient dwdz evaluating the continuity equation.
        Only valid for incompressible flows!!!
    build_coord_datasets: bool, optional=True
        Whether to generate x,y,z,t datasets. For single snapshot HDF built this should
        be set True. For plane or case generation except the first snapshot, this can be
        set to False and thus reduce computation time.

    Returns
    -------
    piv_data_array_dict : dict
        Dictionary containing the arrays
    ncRootAttributes : dict
        Attribute dictionary of root variables
    variable_attributes : dict
        Attribute dictionary of dataset variables

    Notes
    -----
    Credits to M. Elfner (ITS, Karlsruhe Institute of Technology)

    """

    # TODO get rid of ncRootAttributes

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
        ncRootAttributes = _process_pivview_root_attributes(
            {attr: nc_rootgrp.getncattr(attr) for attr in nc_rootgrp.ncattrs()})

        variable_attributes = {}

        # Variable information.
        nc_data_array_dict = nc_rootgrp.variables

        if ncRootAttributes['outlier_interpolate'] and masking != 'slack':
            logger.debug('Outlier masking does not conform with pivview settings in nc '
                         f'(outlier_interpolate={ncRootAttributes["outlier_interpolate"]} vs {masking}) - '
                         f'averages might differ')
        elif ncRootAttributes['outlier_try_other_peak'] and masking != 'sepeaks':
            logger.debug('Outlier masking does not conform with pivview settings in nc '
                         f'(outlier_interpolate={ncRootAttributes["outlier_interpolate"]} vs {masking}) - '
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
                elif 'velocity gradient' in variable_attributes[k]['long_name']:
                    variable_attributes[k].update({'units': f"1/{variable_attributes['velocity']['units'][-1]}"})

        if interpolate:
            # While there exist some base functions assuming reg. grids and / or
            # using splines, a true interpolation considering distances is the way
            # to go here. Any other, faster methods rely on splines which are not
            # bounded!
            # Qhull on location with valid data
            if 'x' and 'y' not in piv_data_array_dict.keys():
                xm, ym = _build_meshgrid_xy(nc_data_array_dict['velocity'].coord_min,
                                            nc_data_array_dict['velocity'].coord_max,
                                            w, h)
            xv = xm.ravel()[mask.ravel()]
            yv = ym.ravel()[mask.ravel()]
            xi = xm.ravel()[~mask.ravel()]
            yi = ym.ravel()[~mask.ravel()]
            tri = Delaunay(np.stack((xv, yv)).T)

            # interpolate, create and evaluate, write
            for k, v in piv_data_array_dict.items():

                # skip iteration if k in the following variables
                if k in ('x', 'y', 'valid', 'piv_flags'):
                    continue

                data = v.copy()
                interpolator = LinearNDInterpolator(tri, data.ravel()[mask.ravel()])
                interpolated_result = interpolator(xi, yi)
                data[~mask] = interpolated_result

                piv_data_array_dict[k] = data

        # Check if source for w gradients is available, if so compute.
        # CAREFUL with indexing: Meshgrids are YX indexed by default!
        if 'w' in piv_data_array_dict or build_coord_datasets:
            fr, to = nc_data_array_dict['velocity'].coord_min, nc_data_array_dict['velocity'].coord_max
            px_fr, px_to = nc_data_array_dict['piv_data'].coord_min, nc_data_array_dict['piv_data'].coord_max

        if 'w' in piv_data_array_dict:
            # grid spacing is assumed to be homogeneous - is the case for PIV measurements!
            dx, dy = (to[:2] - fr[:2]) / (np.array([w, h]) - 1)

            piv_data_array_dict['dwdx'] = np.gradient(piv_data_array_dict['w'], dx, axis=1)
            piv_data_array_dict['dwdy'] = np.gradient(piv_data_array_dict['w'], dy, axis=0)

            _gradient_unit = f"1/{variable_attributes['velocity']['units'][-1]}"
            variable_attributes['dwdx'] = {'units': _gradient_unit,
                                           'long_name': 'velocity gradient dw/dx'}
            variable_attributes['dwdy'] = {'units': _gradient_unit,
                                           'long_name': 'velocity gradient dw/dy'}

            if compute_dwdz:
                if 'dudx' in piv_data_array_dict.keys() and 'dvdy' in piv_data_array_dict.keys():
                    logger.info("Gradient \"dwdz\" calculated from continuity equation assuming incompressible flow!")
                    piv_data_array_dict['dwdz'] = -piv_data_array_dict['dudx'] - piv_data_array_dict['dvdy']
                    variable_attributes['dwdz'] = {
                        'units': _gradient_unit,
                        'long_name': 'velocity gradient dw/dxz assuming incompressible flow',
                        'standard_name': 'z_derivative_of_w_velocity_assuming_incompressibility',
                        'comment': 'Calculated from continuity equation using dud and dvdy and assuming '
                                   'incompressible flow!'
                    }
                else:
                    logger.error(
                        "Could not compute dwdz based on continuity as du/dx and dv/dy are missing. Continuing ..."
                    )

        piv_data_array_dict['reltime'] = timestep
        variable_attributes['reltime'] = {'long_name': 'Recording time since start.',
                                          'units': time_unit}
        if not is_time(variable_attributes['reltime']['units']):
            raise AttributeError(f'Time unit is incorrect: {variable_attributes["t"]["unit"]}')

        # x,y,z,t are not part of PIVview netCDF variables
        if build_coord_datasets:  # can speed up computation when plane or case HDF files are generated
            # the velocity dataset has the attribute coord_min and coord_max from which the coordinates can be derived:
            piv_data_array_dict['x'] = np.linspace(fr[0], to[0], w)
            piv_data_array_dict['y'] = np.linspace(fr[1], to[1], h)
            variable_attributes['x'] = {'units': ncRootAttributes['length_conversion_units']}
            variable_attributes['y'] = {'units': ncRootAttributes['length_conversion_units']}
            assert px_fr[0] >= 0
            assert px_fr[1] >= 0
            piv_data_array_dict['ix'] = np.linspace(px_fr[0], px_to[0], w).astype(get_uint_type(px_to[0]))
            piv_data_array_dict['iy'] = np.linspace(px_fr[1], px_to[1], h).astype(get_uint_type(px_to[0]))
            variable_attributes['ix'] = {'long_name': 'pixel x-location of vector',
                                         'units': 'pixel'}
            variable_attributes['iy'] = {'long_name': 'pixel y-location of vector',
                                         'units': 'pixel'}

            # Z position information
            z_unit = ncRootAttributes['length_conversion_units']  # default unit of z comes from file
            if isinstance(z_source, str):
                if z_source == 'coord_min':
                    z = fr[2]
                elif z_source == 'coord_max':
                    z = to[2]
                elif z_source == 'origin_offset':
                    z = ncRootAttributes['origin_offset_z']
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
                 **kwargs):
        # if user_defined_hdf5_operations is None:
        #     user_defined_hdf5_operations = add_standard_name_operation
        super().__init__(filename, parameter, user_defined_hdf5_operations=user_defined_hdf5_operations, **kwargs)

    def read(self,
             relative_time: float,
             *,
             build_coord_datasets=True) -> Tuple[Dict, Dict, Dict]:
        """reads and processes nc data

        Parameters
        ----------
        relative_time : float
            relative time of the snapshot read
        build_coord_datasets : bool, optional
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
        from . import config as pivview_config
        masking = pivview_config['masking']
        interpolation = pivview_config['interpolation']
        z_source = pivview_config['z_source']
        nc_data, nc_root_attr, nc_variable_attr = process_pivview_nc_data(self.filename,
                                                                          z_source=z_source,
                                                                          timestep=relative_time,
                                                                          masking=masking,
                                                                          interpolate=interpolation,
                                                                          build_coord_datasets=build_coord_datasets)
        # nc_root_attr['filename'] = nc_root_attr.pop('long_name')
        # unique_flags = np.unique(nc_data['piv_flags'][:])
        nc_variable_attr['piv_flags']['flag_meaning'] = json.dumps({flag.value: flag.name for flag in flags.Flags})
        return nc_data, nc_root_attr, nc_variable_attr

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: str,  # iso-format of datetime!
               z: Union[str, float, None] = None) -> pathlib.Path:
        """converts the snapshot into an HDF file"""
        nc_data, nc_root_attr, nc_variable_attr = self.read(relative_time=relative_time)
        if z is not None:
            nc_data['z'], nc_variable_attr['z']['units'] = parse_z(z)
        ny, nx = nc_data['y'].size, nc_data['x'].size
        # building HDF file
        if hdf_filename is None:
            _hdf_filename = Path.joinpath(self.filename.parent, f'{self.filename.stem}.hdf')
        else:
            _hdf_filename = hdf_filename
        with h5py.File(_hdf_filename, "w") as main:
            main.attrs['plane_directory'] = str(self.filename.parent.resolve())
            main.attrs['software'] = json.dumps(PIVVIEW_SOFTWARE)  # TODO is the compliant with EngMeta?
            main.attrs['title'] = 'piv snapshot data'

            # process piv_parameters. there must be a parameter file at the parent location
            # piv_param_grp = main.create_group(PIV_PARAMETER_GRP_NAME)
            # self.write_parameters(piv_param_grp)
            if recording_dtime is not None:
                ds_rec_dtime = create_recording_datetime_dataset(main, recording_dtime, name='time')

            for i, cname in enumerate(('x', 'y', 'ix', 'iy')):
                ds = main.create_dataset(
                    name=cname,
                    shape=nc_data[cname].shape,
                    maxshape=nc_data[cname].shape,
                    chunks=nc_data[cname].shape,
                    data=nc_data[cname], dtype=nc_data[cname].dtype,
                    compression=get_config('compression'),
                    compression_opts=get_config('compression_opts'))
                for k, v in nc_variable_attr[cname].items():
                    ds.attrs[k] = v
                ds.make_scale(DEFAULT_DATASET_LONG_NAMES[cname])

            for i, cname in enumerate(('z', 'reltime')):
                ds = main.create_dataset(cname, data=nc_data[cname])
                for k, v in nc_variable_attr[cname].items():
                    ds.attrs[k] = v
                ds.make_scale(DEFAULT_DATASET_LONG_NAMES[cname])

            # Data Arrays
            _shape = (ny, nx)
            for k, v in nc_data.items():
                if k not in DIM_NAMES:
                    ds = main.create_dataset(
                        name=k,
                        shape=_shape,
                        dtype=v.dtype,
                        maxshape=_shape,
                        chunks=_shape,
                        compression=get_config('compression'),
                        compression_opts=get_config('compression_opts'))
                    ds[:] = v
                    for ic, c in enumerate((('y', 'iy'), ('x', 'ix'))):
                        ds.dims[ic].attach_scale(main[c[0]])
                        ds.dims[ic].attach_scale(main[c[1]])
                    if recording_dtime is None:
                        ds.attrs['COORDINATES'] = ['reltime', 'z']
                    else:
                        ds.attrs['COORDINATES'] = ['reltime', 'z', ds_rec_dtime.name]

                    if k in nc_variable_attr:
                        for attr_key, attr_val in nc_variable_attr[k].items():
                            if attr_key not in IGNORE_ATTRS:
                                ds.attrs[attr_key] = attr_val
            # # pivflags explanation:
            # unique_flags = np.unique(main['piv_flags'][:])
            # main['piv_flags'].attrs['flag_meaning'] = json.dumps(
            #     {str(PIVviewFlag(u)): int(u) for u in unique_flags})
        return hdf_filename


class PIVViewStereoNcFile(PIVViewNcFile):
    """Interface class to a PIVview netCDF4 file for stereo recordings. Stereo
    recordings uses different configuration files (.cfg)"""
    __suffix__ = '.nc'
    __parameter_cls__ = pivview_parameter.PivViewConfigFile
