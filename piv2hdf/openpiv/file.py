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
from ..interface import PIVFile
from ..time import create_recording_datetime_dataset
from ..utils import read_translation_yaml_file, get_uint_type, parse_z

__this_dir__ = pathlib.Path(__file__)

RESOURCES_DIR = __this_dir__.parent / '../resources'
TRANSLATION_EXT_DICT = read_translation_yaml_file(RESOURCES_DIR / 'openpiv/openpiv_ext_translation.yaml')

from ..piv_params import PIV_PeakFitMethod, PIV_METHOD

from ..interface import PIV_PARAMETER_GRP_NAME


def update_standard_names(h5: h5tbx.File) -> None:
    """openpiv post function"""
    for name, ds in h5.items():
        if name in TRANSLATION_EXT_DICT:
            ds.attrs['standard_name'] = TRANSLATION_EXT_DICT[name]

    def _update_fields(grp):
        peak_method = grp.attrs['subpixel_method']
        if peak_method == 'gaussian':
            grp.attrs['piv_peak_method'] = PIV_PeakFitMethod(0).name

        # piv_method
        warnings.warn('piv_method is assumed to be multi grid but not determined!')
        grp.attrs['piv_method'] = PIV_METHOD(2).name

    if PIV_PARAMETER_GRP_NAME not in h5:
        for param_grp in h5.find({'$basename': PIV_PARAMETER_GRP_NAME}, recursive=True):
            _update_fields(param_grp)
        return

    return _update_fields(h5[PIV_PARAMETER_GRP_NAME])


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

    def read(self, recording_time: float):
        pass

    def to_hdf(self, hdf_filename: pathlib.Path,
               config: Dict, recording_time: float) -> pathlib.Path:
        pass


class OpenPIVFile(PIVFile):
    """Open PIV File interface class

    Note, that .txt file from openPIV will not have all datasets, when ou use the openPIV.tools.save()
    method. Instead use the save method provided here:
    >>> import piv2hdf
    >>> piv2hf.openpiv.save('my_file.txt', 'my_file.h5')
    """
    suffix: str = '.txt'
    __parameter_cls__ = OpenPIVParameterFile

    def __init__(self, *args, **kwargs):
        post_func = kwargs.get('post_func', None)
        if post_func is None:
            kwargs['post_func'] = update_standard_names
        super().__init__(*args, **kwargs, )

    def read(self, relative_time: float):
        """Read data from file."""
        px_mm_scale = float(self._parameter.param_dict['scaling_factor'])  # pixel/mm

        data = pd.read_csv(self.filename, delimiter='\t', na_values='     nan')

        _ix = data["# x"].to_numpy()
        _iy = data["y"].to_numpy()

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

        return data_dict, {'software': json.dumps(OPENPIV_SOFTWARE)}, variable_attributes

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: Optional[Union[datetime, List[datetime]]] = None,
               z: Union[float, None] = None) -> pathlib.Path:
        """converts the snapshot into an HDF file"""
        data, root_attr, variable_attr = self.read(relative_time)
        if z is not None:
            data['z'], variable_attr['z']['units'] = parse_z(z)
        # building HDF file
        if hdf_filename is None:
            _hdf_filename = pathlib.Path.joinpath(self.filename.parent, f'{self.filename.stem}.hdf')
        else:
            _hdf_filename = hdf_filename

        with h5tbx.File(_hdf_filename, "w") as main:
            main.attrs['title'] = 'piv snapshot data'
            for ak, av in root_attr.items():
                main.attrs[ak] = av
            main.attrs['plane_directory'] = str(self.filename.parent.resolve())

            # # process piv_parameters. there must be a parameter file at the parent location
            # piv_param_grp = main.create_group(PIV_PARAMETER_GRP_NAME)
            #
            # self.write_parameters(piv_param_grp)

            if recording_dtime is not None:
                ds_rec_dtime = create_recording_datetime_dataset(main, recording_dtime, name='time')
            var_list = []
            for varkey, vardata in data.items():
                if vardata.ndim == 1:
                    ds = main.create_dataset(
                        name=varkey,
                        shape=vardata.shape,
                        maxshape=vardata.shape,
                        data=vardata,
                        compression=get_config('compression'),
                        compression_opts=get_config('compression_opts'),
                        attrs=variable_attr.get(varkey, None))
                    ds.make_scale()
                elif vardata.ndim == 0:
                    main.create_dataset(name=varkey,
                                        data=vardata,
                                        dtype=vardata.dtype,
                                        attrs=variable_attr[varkey])
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
                        attrs=variable_attr.get(varkey, None))
                    var_list.append(ds)
            for ds in var_list:
                ds.dims[0].attach_scale(main['y'])
                ds.dims[1].attach_scale(main['x'])
            for k, v in variable_attr.items():
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
