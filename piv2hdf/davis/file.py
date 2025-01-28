"""Davis interface"""

import json
import pathlib
from datetime import datetime
from typing import Optional, List, Union

import h5py
import numpy as np

from piv2hdf import UserDefinedHDF5Operation
from piv2hdf.interface import PIVParameterInterface, PIVData
from . import parameter as davis_parameter
from .const import DAVIS_SOFTWARE, DEFAULT_DATASET_LONG_NAMES
from ..config import get_config
from ..interface import PIVFile
from ..pivview.const import IGNORE_ATTRS, DIM_NAMES
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

    def read(self, relative_time, **kwargs) -> PIVData:
        """Read data from file."""
        build_coord_datasets = kwargs.get("build_coord_datasets", True)

        buffer = lv.read_buffer(str(self.filename))

        frame = buffer[0]

        ny, nx = frame.shape
        scales = frame.scales
        x = np.array([scales.x.slope * i + scales.x.offset for i in range(nx)])
        y = np.array([scales.y.slope * j + scales.y.offset for j in range(ny)])
        z = np.array([scales.z.slope * i + scales.z.offset for i in range(1)])
        ix = np.arange(0, nx)
        iy = np.arange(0, ny)

        data_attrs = {
            "x": dict(unit="m", description=scales.x.description),
            "y": dict(unit="m", description=scales.y.description),
            "ix": dict(unit="", description="Index in x-direction"),
            "iy": dict(unit="", description="Index in y-direction"),
            "z": dict(unit="m", description=scales.z.description),
            "reltime": dict(unit="s", description="Recording time since start"),
        }

        data = {
            "ix": ix,
            "iy": iy,
            "x": x,
            "y": y,
            "z": float(z),
            "reltime": relative_time
        }
        for name, comp in frame.components.items():
            if name not in ("x", "y", "ix", "iy", "reltime"):
                data_attrs[name] = dict(
                    unit=comp.scale.unit,
                    description=comp.scale.description,
                    slope=comp.scale.slope,
                    offset=comp.scale.offset
                )
                data[name] = comp.planes[0]
        return PIVData(data, data_attrs, {})

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

        pivdata = self.read(relative_time, **kwargs)
        ny, nx = pivdata.data['y'].size, pivdata.data['x'].size
        with h5py.File(_hdf_filename, "w") as main:
            main.attrs['software'] = json.dumps(DAVIS_SOFTWARE)
            main.attrs['title'] = 'piv snapshot data'

            if recording_dtime is not None:
                ds_rec_dtime = create_recording_datetime_dataset(main, recording_dtime, name='time')

            for i, cname in enumerate(('x', 'y', 'ix', 'iy')):
                ds = main.create_dataset(
                    name=cname,
                    shape=pivdata.data[cname].shape,
                    maxshape=pivdata.data[cname].shape,
                    chunks=pivdata.data[cname].shape,
                    data=pivdata.data[cname],
                    dtype=pivdata.data[cname].dtype,
                    compression=get_config('compression'),
                    compression_opts=get_config('compression_opts'))
                for k, v in pivdata.data_attrs[cname].items():
                    ds.attrs[k] = v
                ds.make_scale(DEFAULT_DATASET_LONG_NAMES[cname])

            for i, cname in enumerate(('z', 'reltime')):
                ds = main.create_dataset(cname, data=pivdata.data[cname])
                for k, v in pivdata.data_attrs[cname].items():
                    ds.attrs[k] = v
                ds.make_scale(DEFAULT_DATASET_LONG_NAMES[cname])

            # Data Arrays
            _shape = (ny, nx)
            for k, v in pivdata.data.items():
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

                    if k in pivdata.data_attrs:
                        for attr_key, attr_val in pivdata.data_attrs[k].items():
                            if attr_key not in IGNORE_ATTRS:
                                ds.attrs[attr_key] = attr_val

            # for name, comp in frame.components.items():
            #     ds = main.create_dataset(
            #         name=name,
            #         shape=comp.shape,
            #         maxshape=comp.shape,
            #         data=comp.planes[0],
            #         compression=get_config('compression'),
            #         compression_opts=get_config('compression_opts')
            #     )
            #     ds.attrs["description"] = comp.scale.description
            #     ds.attrs["unit"] = comp.scale.unit
            #     ds.attrs["slope"] = comp.scale.slope
            #     ds.attrs["offset"] = comp.scale.offset
            #
            #     for ic, c in enumerate((('y', 'iy'), ('x', 'ix'))):
            #         ds.dims[ic].attach_scale(main[c[0]])
            #         ds.dims[ic].attach_scale(main[c[1]])
            #
            #     if recording_dtime is None:
            #         ds.attrs['COORDINATES'] = ['reltime', 'z']
            #     else:
            #         ds.attrs['COORDINATES'] = ['reltime', 'z', ds_rec_dtime.name]
        return _hdf_filename
