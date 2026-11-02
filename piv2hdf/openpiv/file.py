import json
import pathlib
import warnings
from datetime import datetime
from typing import Dict, Union, List, Optional

import numpy as np
import pandas as pd

from .const import *
from .parameter import OpenPIVParameterFile
from ..flags import flag_translation_dict, Flags
from ..interface import PIVFile, UserDefinedHDF5Operation, PIVData

__this_dir__ = pathlib.Path(__file__)


class OpenPIVResultData(PIVFile):
    """OpenPIV Result data, not a file!"""
    suffix: str = None
    parameter = OpenPIVParameterFile
    _parameter: OpenPIVParameterFile = None

    def __init__(self, data: Dict,
                 parameter: Union[None, OpenPIVParameterFile, pathlib.Path] = None,
                 **kwargs):
        raise NotImplementedError('In-memory OpenPIV conversion is not implemented; use OpenPIVFile')

    def read(self, recording_time: float, **kwargs):
        raise NotImplementedError('Use OpenPIVFile')

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: Union[datetime, List[datetime]],
               z: Union[float, None] = None,
               **kwargs) -> pathlib.Path:
        raise NotImplementedError("Not yet implemented")


class OpenPIVFile(PIVFile):
    """Open PIV File interface class

    Note, that .txt file from openPIV will not have all datasets, when ou use the openPIV.tools.save()
    method.
    """
    suffix: str = '.txt'
    __parameter_cls__ = OpenPIVParameterFile

    def __init__(self,
                 *args,
                 user_defined_hdf5_operations: Optional[
                     Union[UserDefinedHDF5Operation, List[UserDefinedHDF5Operation]]] = None,
                 coordinate_units='pixel', velocity_units='m/s', scaling_factor_units='pixel/m',
                 **kwargs):
        self.coordinate_units = coordinate_units
        self.velocity_units = velocity_units
        self.scaling_factor_units = scaling_factor_units
        super().__init__(*args, user_defined_hdf5_operations=user_defined_hdf5_operations, **kwargs, )

    def read(self, relative_time: float, **kwargs) -> PIVData:
        """Read data from file."""
        from h5rdmtoolbox import get_ureg
        scale = float(self._parameter.param_dict['scaling_factor'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('OpenPIV scaling_factor must be finite and positive')

        data = pd.read_csv(self.filename, delimiter='\t', na_values='     nan')

        _ix = data["# x"].to_numpy()
        _iy = data["y"].to_numpy()

        nx = len(np.unique(_ix))
        ix = _ix[:nx]
        iy = _iy[::nx]
        nx = len(ix)
        ny = len(iy)
        if nx * ny != len(data) or not np.array_equal(_ix, np.tile(ix, ny)) or not np.array_equal(_iy, np.repeat(iy, nx)):
            raise ValueError('OpenPIV rows must form a complete rectilinear grid ordered by y then x')
        ureg = get_ureg()
        if self.coordinate_units == 'pixel':
            factor = ureg.Quantity(scale, self.scaling_factor_units)
            x = (ureg.Quantity(ix, 'pixel') / factor).to('m').magnitude
            y = (ureg.Quantity(iy, 'pixel') / factor).to('m').magnitude
        else:
            x = ureg.Quantity(ix, self.coordinate_units).to('m').magnitude
            y = ureg.Quantity(iy, self.coordinate_units).to('m').magnitude

        data_dict = {k: v.to_numpy().reshape((ny, nx)) for k, v in data.items() if k not in ('# x', 'y')}
        data_dict['ix'] = ix
        data_dict['iy'] = iy
        data_dict['x'] = x
        data_dict['y'] = y
        data_dict['reltime'] = np.asarray(relative_time)

        variable_attributes = {'ix': dict(units='pixel'),
                               'iy': dict(units='pixel'),
                               'x': dict(units='m'),
                               'y': dict(units='m'),
                               'u': dict(units='m/s'),
                               'v': dict(units='m/s'),
                               'reltime': dict(units='s'),
                               'sig2noise': dict(units=''),
                               }
        for component in ('u', 'v', 'w'):
            if component in data_dict:
                data_dict[component] = ureg.Quantity(data_dict[component], self.velocity_units).to('m/s').magnitude
        if self.coordinate_units != 'pixel':
            data_dict['ix'], data_dict['iy'] = np.arange(nx), np.arange(ny)
            variable_attributes['ix']['units'] = variable_attributes['iy']['units'] = ''
        flag_name = None
        if 'flags' in data_dict:
            flag_name = 'flags'
        elif 'flag' in data_dict:
            flag_name = 'flag'
        elif 'piv_flag' in data_dict:
            flag_name = 'piv_flag'
        elif 'piv_flags' in data_dict:
            flag_name = 'piv_flags'

        if flag_name:
            if 'openpiv' in flag_translation_dict:
                if not np.all(np.isin(data_dict[flag_name], list(flag_translation_dict['openpiv']))):
                    raise ValueError('Unknown OpenPIV flag value; expected 0, 1 or 2')

                # assume no flags larger than 2**8
                data_dict[flag_name] = data_dict[flag_name].astype(np.uint8)

                # convert flags to package meaning
                _flag_bools = {k: data_dict[flag_name] == k for k in flag_translation_dict['openpiv']}
                for k, v in flag_translation_dict['openpiv'].items():
                    data_dict[flag_name][_flag_bools[k]] = v.value
            variable_attributes[flag_name] = dict(units=' ',
                                                  flag_meaning=json.dumps({flag.value: flag.name for flag in Flags}))

        if 'mask' in data_dict:
            if flag_name is None:
                flag_name = 'piv_flags'
                data_dict[flag_name] = np.full((ny, nx), Flags.ACTIVE.value, dtype=np.uint8)
                variable_attributes[flag_name] = dict(units='')
            data_dict[flag_name][data_dict.pop('mask').astype(bool)] = Flags.MASKED.value

        if flag_name and flag_name != 'piv_flags':
            data_dict['piv_flags'] = data_dict.pop(flag_name)
            variable_attributes['piv_flags'] = variable_attributes.pop(flag_name)
        for name in data_dict:
            variable_attributes.setdefault(name, {'units': '', 'long_name': name.replace('_', ' ')})

        if 'w' in data_dict:
            variable_attributes['w'] = dict(units='m/s')

        from ..processing import mask_vectors
        mask_vectors(data_dict, variable_attributes, 'slack')
        return PIVData(data_dict,
                       variable_attributes,
                       {'software': json.dumps(OPENPIV_SOFTWARE),
                        'openpiv_input_coordinate_units': self.coordinate_units,
                        'openpiv_input_velocity_units': self.velocity_units,
                        'openpiv_scaling_factor_units': self.scaling_factor_units})

    def write_parameters(self, param_grp):
        super().write_parameters(param_grp)
        if 'piv_scaling_factor' in param_grp:
            param_grp['piv_scaling_factor'].attrs['units'] = self.scaling_factor_units
        return param_grp

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: Optional[Union[datetime, List[datetime]]] = None,
               z: Union[float, None] = None,
               **kwargs) -> pathlib.Path:
        """Serialize this adapter through the shared snapshot writer."""
        from ..writer import write_snapshot
        return write_snapshot(self.read(relative_time, **kwargs), hdf_filename,
                              self.filename, recording_dtime, z)


def get_files(folder, suffix='.txt', parameter=None) -> List[OpenPIVFile]:
    """get OpenPIVFile instances from folder"""
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'{folder} is not a folder.')
    if not folder.exists():
        raise FileNotFoundError(f'{folder} does not exist.')
    openpiv_txt_filenames = sorted(folder.glob(f'*{suffix}'))
    if not openpiv_txt_filenames:
        raise FileNotFoundError(f'No {suffix} files found in {folder}')
    folder = openpiv_txt_filenames[0].parent
    if parameter is None:
        parameter_filename_candiates = sorted(folder.glob('*.par'))
        if not parameter_filename_candiates:
            raise FileNotFoundError(f'No OpenPIV parameter file found in {folder}')
        if len(parameter_filename_candiates) > 1:
            warnings.warn(f'Multiple parameter files found in {folder}. Using {parameter_filename_candiates[0]}.')
        parameter = parameter_filename_candiates[0]
    if not isinstance(parameter, (str, pathlib.Path)):
        par = OpenPIVParameterFile.from_windef(parameter)
        par.save(folder / 'openpiv_parameters.par')
        parameter = folder / 'openpiv_parameters.par'
    return [OpenPIVFile(f, parameter=parameter) for f in openpiv_txt_filenames]
