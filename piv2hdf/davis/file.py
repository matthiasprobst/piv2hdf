"""Davis interface"""

import json
import pathlib
from datetime import datetime
from typing import Optional, List
from typing import Union

import h5py
import numpy as np

from piv2hdf import UserDefinedHDF5Operation
from piv2hdf.interface import PIVParameterInterface
from . import parameter as davis_parameter
from .const import DAVIS_SOFTWARE, DEFAULT_DATASET_LONG_NAMES
from .._logger import logger
from ..config import get_config
from ..interface import PIVFile
from ..time import create_recording_datetime_dataset

try:
    import lvpyio as lv
except ImportError:
    raise ImportError('Package "lvpyio" not installed which is needed to read Davis files')
try:
    import h5rdmtoolbox as h5tbx
except ImportError:
    raise ImportError('Package "h5rdmtoolbox" not installed which is needed to write HDF files from Davis files')


def _compute_physical_quantity(data, offset, slope):
    return data * slope + offset


class VC7File(PIVFile):
    """Davis vc7 aka buffer file interface class"""
    suffix = ".vc7"
    __parameter_cls__ = davis_parameter.DavisParameterFile

    def __init__(self,
                 filename: Union[str, pathlib.Path],
                 parameter: Union[None, PIVParameterInterface, pathlib.Path] = None,
                 user_defined_hdf5_operations: Optional[
                     Union[UserDefinedHDF5Operation, List[UserDefinedHDF5Operation]]] = None,
                 **kwargs
                 ):
        self.filename = pathlib.Path(filename)
        assert self.filename.exists(), FileNotFoundError(f'File {self.filename} not found!')
        self._frame = None
        self._attrs = None
        self._variables = None
        self._root_attributes = None
        super().__init__(filename, parameter, user_defined_hdf5_operations=user_defined_hdf5_operations, **kwargs)

    def read(self, relative_time, **kwargs):
        """Read data from file."""
        recording_time = kwargs.get("recording_time", None)
        if not recording_time:
            raise ValueError('recording_time must be provided')
        build_coord_datasets = kwargs.get("build_coord_datasets", False)
        if not build_coord_datasets:
            raise ValueError('build_coord_datasets must be provided')
        is_mset = lv.is_multiset(self.filename)
        logger.debug(f'Reading davis file. is multiset. {is_mset}')
        raise NotImplementedError(f'{self.__class__.__name__}.read() not implemented')

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: Union[datetime, List[datetime]],
               z: Union[str, float, None] = None,
               **kwargs) -> pathlib.Path:
        """Convert file to HDF5 format."""
        if hdf_filename is None:
            _hdf_filename = pathlib.Path.joinpath(self.filename.parent, f'{self.filename.stem}.hdf')
        else:
            _hdf_filename = hdf_filename

        buffer = lv.read_buffer(str(self.filename))

        frame = buffer[0]

        ny, nx = frame.shape
        scales = frame.scales
        x = np.array([scales.x.slope * i + scales.x.offset for i in range(nx)])
        y = np.array([scales.y.slope * j + scales.y.offset for j in range(ny)])
        z = np.array([scales.z.slope * i + scales.z.offset for i in range(1)])
        ix = np.arange(0, nx)
        iy = np.arange(0, ny)

        with h5py.File(_hdf_filename, "w") as main:
            main.attrs['software'] = json.dumps(DAVIS_SOFTWARE)
            main.attrs['title'] = 'piv snapshot data'

            if recording_dtime is not None:
                ds_rec_dtime = create_recording_datetime_dataset(main, recording_dtime, name='time')

            ds_ix = main.create_dataset(
                name="ix",
                shape=ix.shape,
                maxshape=ix.shape,
                chunks=ix.shape,
                data=ix, dtype=ix.dtype,
                compression=get_config('compression'),
                compression_opts=get_config('compression_opts')
            )
            ds_ix.attrs["unit"] = scales.x.unit
            ds_ix.attrs["description"] = scales.x.description

            ds_iy = main.create_dataset(
                name="iy",
                shape=iy.shape,
                maxshape=iy.shape,
                chunks=iy.shape,
                data=iy, dtype=iy.dtype,
                compression=get_config('compression'),
                compression_opts=get_config('compression_opts')
            )
            ds_iy.attrs["unit"] = scales.x.unit
            ds_iy.attrs["description"] = scales.x.description
            ds_x = main.create_dataset(
                name="x",
                shape=x.shape,
                maxshape=x.shape,
                chunks=x.shape,
                data=x, dtype=x.dtype,
                compression=get_config('compression'),
                compression_opts=get_config('compression_opts'))
            ds_x.attrs["unit"] = "m"
            ds_y = main.create_dataset(
                name="y",
                shape=y.shape,
                maxshape=y.shape,
                chunks=y.shape,
                data=y, dtype=y.dtype,
                compression=get_config('compression'),
                compression_opts=get_config('compression_opts'))
            ds_y.attrs["unit"] = "m"

            ds_z = main.create_dataset(
                name="z",
                data=float(z),
            )

            ds_x.make_scale(DEFAULT_DATASET_LONG_NAMES["x"])
            ds_y.make_scale(DEFAULT_DATASET_LONG_NAMES["y"])
            ds_ix.make_scale(DEFAULT_DATASET_LONG_NAMES["ix"])
            ds_iy.make_scale(DEFAULT_DATASET_LONG_NAMES["iy"])
            ds_z.make_scale(DEFAULT_DATASET_LONG_NAMES["z"])

            ds_reltime = main.create_dataset(
                name="reltime",
                data=relative_time
            )
            ds_reltime.attrs["long_name"] = "Recording time since start"
            ds_reltime.attrs["units"] = "s"
            ds_reltime.make_scale(DEFAULT_DATASET_LONG_NAMES["reltime"])

            for name, comp in frame.components.items():
                ds = main.create_dataset(
                    name=name,
                    shape=comp.shape,
                    maxshape=comp.shape,
                    data=comp.planes[0],
                    compression=get_config('compression'),
                    compression_opts=get_config('compression_opts')
                )
                ds.attrs["description"] = comp.scale.description
                ds.attrs["unit"] = comp.scale.unit
                ds.attrs["slope"] = comp.scale.slope
                ds.attrs["offset"] = comp.scale.offset

                for ic, c in enumerate((('y', 'iy'), ('x', 'ix'))):
                    ds.dims[ic].attach_scale(main[c[0]])
                    ds.dims[ic].attach_scale(main[c[1]])

                if recording_dtime is None:
                    ds.attrs['COORDINATES'] = ['reltime', 'z']
                else:
                    ds.attrs['COORDINATES'] = ['reltime', 'z', ds_rec_dtime.name]
        return _hdf_filename
