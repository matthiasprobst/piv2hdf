import json
import pathlib
import warnings
from datetime import datetime
from typing import Dict, Union, List, Optional

import h5rdmtoolbox as h5tbx
import numpy as np
import pandas as pd

from .const import *
from .parameter import OpenPIVParameterFile
from ..config import get_config
from ..flags import flag_translation_dict, Flags
from ..interface import PIVFile, UserDefinedHDF5Operation, PIVData
from ..utils import get_uint_type, parse_z

__this_dir__ = pathlib.Path(__file__)


class OpenPIVResultData(PIVFile):
    """OpenPIV Result data, not a file!"""
    suffix: str = None
    parameter = OpenPIVParameterFile
    _parameter: OpenPIVParameterFile = None

    def __init__(self, data: Dict,
                 parameter: Union[None, OpenPIVParameterFile, pathlib.Path] = None,
                 **kwargs):
        _tmp_filename = pathlib.Path('__tmp__.txt')
        with open(_tmp_filename, 'w'):
            pass
        super().__init__(_tmp_filename, parameter, **kwargs)
        pathlib.Path(_tmp_filename).unlink()
        self.data = data  # data like x,y,u,v,mask etc. as it directly comes from openpiv

    def read(self, recording_time: float, **kwargs):
        pass

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
                 **kwargs):
        super().__init__(*args, user_defined_hdf5_operations=user_defined_hdf5_operations, **kwargs, )

    def read(self, relative_time: float, **kwargs) -> PIVData:
        """Read data from file."""
        px_mm_scale = float(self._parameter.param_dict['scaling_factor'])  # pixel/mm

        data = pd.read_csv(self.filename, delimiter='\t', na_values='     nan')

        _ix = data["# x"].to_numpy()
        _iy = data["y"].to_numpy()

        i = -1
        for i, x in enumerate(_ix[0:-1]):
            if (x - _ix[i + 1]) > 0:
                break
        ix = _ix[:i + 1]
        iy = _iy[::i + 1]
        nx = len(ix)
        ny = len(iy)
        x = ix / px_mm_scale  # pixel/mm * 1/pixel = mm
        y = iy / px_mm_scale  # pixel/mm * 1/pixel = mm

        data_dict = {k: v.to_numpy().reshape((ny, nx)) for k, v in data.items() if k not in ('# x', 'y')}
        data_dict['ix'] = ix.astype(get_uint_type(ix))
        data_dict['iy'] = iy.astype(get_uint_type(iy))
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

                # assume no flags larger than 2**8
                data_dict[flag_name] = data_dict[flag_name].astype(np.uint8)

                # convert flags to package meaning
                _flag_bools = {k: data_dict[flag_name] == k for k in flag_translation_dict['openpiv']}
                for k, v in flag_translation_dict['openpiv'].items():
                    data_dict[flag_name][_flag_bools[k]] = v.value
            variable_attributes[flag_name] = dict(units=' ',
                                                  flag_meaning=json.dumps({flag.value: flag.name for flag in Flags}))

        if 'mask' in data_dict:
            data_dict[flag_name][data_dict.pop('mask').astype(bool)] = Flags.MASKED.value

        if 'w' in data_dict:
            variable_attributes['w'] = dict(units='m/s')

        return PIVData(data_dict,
                       variable_attributes,
                       {'software': json.dumps(OPENPIV_SOFTWARE)})

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: Optional[Union[datetime, List[datetime]]] = None,
               z: Union[float, None] = None,
               **kwargs) -> pathlib.Path:
        """converts the snapshot into an HDF file"""
        pivdata = self.read(relative_time)
        if z is not None:
            pivdata.data['z'], pivdata.data_attrs['z']['units'] = parse_z(z)
        # building HDF file
        if hdf_filename is None:
            _hdf_filename = pathlib.Path.joinpath(self.filename.parent, f'{self.filename.stem}.hdf')
        else:
            _hdf_filename = hdf_filename

        with h5tbx.File(_hdf_filename, "w") as main:
            main.attrs['title'] = 'piv snapshot data'
            for ak, av in pivdata.root_attrs.items():
                main.attrs[ak] = av
            main.attrs['plane_directory'] = str(self.filename.parent.resolve())

            # # process piv_parameters. there must be a parameter file at the parent location
            # piv_param_grp = main.create_group(PIV_PARAMETER_GRP_NAME)
            #
            # self.write_parameters(piv_param_grp)

            # if recording_dtime is not None:
            #     ds_rec_dtime = create_recording_datetime_dataset(main, recording_dtime, name='time')
            var_list = []
            for varkey, vardata in pivdata.data.items():
                if vardata.ndim == 1:
                    ds = main.create_dataset(
                        name=varkey,
                        shape=vardata.shape,
                        maxshape=vardata.shape,
                        data=vardata,
                        compression=get_config('compression'),
                        compression_opts=get_config('compression_opts'),
                        attrs=pivdata.data_attrs.get(varkey, None))
                    ds.make_scale()
                elif vardata.ndim == 0:
                    main.create_dataset(name=varkey,
                                        data=vardata,
                                        dtype=vardata.dtype,
                                        attrs=pivdata.data_attrs[varkey])
                else:
                    ds = main.create_dataset(
                        name=varkey,
                        shape=vardata.shape,
                        chunks=vardata.shape,
                        maxshape=vardata.shape,
                        data=vardata,
                        dtype=vardata.dtype,
                        compression=get_config('compression'),
                        compression_opts=get_config('compression_opts'),
                        attrs=pivdata.data_attrs.get(varkey, None))
                    var_list.append(ds)
            for ds in var_list:
                ds.dims[0].attach_scale(main['y'])
                ds.dims[1].attach_scale(main['x'])
            for k, v in pivdata.data_attrs.items():
                if k in main:
                    for ak, av in v.items():
                        main[k].attrs[ak] = av
        return hdf_filename


def get_files(folder, suffix='.txt', parameter=None) -> List[OpenPIVFile]:
    """get OpenPIVFile instances from folder"""
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'{folder} is not a folder.')
    if not folder.exists():
        raise FileNotFoundError(f'{folder} does not exist.')
    openpiv_txt_filenames = sorted(folder.glob(f'*{suffix}'))
    folder = openpiv_txt_filenames[0].parent
    if parameter is None:
        parameter_filename_candiates = sorted(folder.glob('*.par'))
        if len(parameter_filename_candiates) > 1:
            warnings.warn(f'Multiple parameter files found in {folder}. Using {parameter_filename_candiates[0]}.')
        parameter = parameter_filename_candiates[0]
    if not isinstance(parameter, (str, pathlib.Path)):
        par = OpenPIVParameterFile.from_windef(parameter)
        par.save(folder / 'openpiv_parameters.par')
    return [OpenPIVFile(f) for f in openpiv_txt_filenames]
