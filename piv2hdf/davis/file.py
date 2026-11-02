"""Davis interface"""

import json
import pathlib
from datetime import datetime
from typing import Optional, List, Union

import numpy as np

from piv2hdf import UserDefinedHDF5Operation
from piv2hdf.interface import PIVParameterInterface, PIVData
from . import parameter as davis_parameter
from .const import DAVIS_SOFTWARE
from ..interface import PIVFile

try:
    import lvpyio as lv
except ImportError:
    raise ImportError('Package "lvpyio" not installed which is needed to read Davis files')


def _compute_physical_quantity(data, offset, slope):
    return data * slope + offset


class VC7File(PIVFile):
    """Read a DaVis .vc7 file through optional lvpyio (Windows/Linux).

    Parameter information defaults to the same file. Coordinate units and vendor
    component names are preserved. Component slope/offset is applied once; raw
    values and calibration metadata are retained. Pixel displacement is not
    inferred to be a physical velocity.
    """
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
        if not self.filename.is_file():
            raise FileNotFoundError(f'File {self.filename} not found!')
        if parameter is None:
            parameter = davis_parameter.DavisParameterFile(self.filename)
        self._frame = None
        self._attrs = None
        self._variables = None
        self._root_attributes = None
        super().__init__(filename, parameter, user_defined_hdf5_operations=user_defined_hdf5_operations, **kwargs)

    def read(self, relative_time, **kwargs) -> PIVData:
        """Read data from file."""
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
            "x": dict(units=scales.x.unit, long_name=scales.x.description or 'x coordinate'),
            "y": dict(units=scales.y.unit, long_name=scales.y.description or 'y coordinate'),
            "ix": dict(units="", long_name="Index in x-direction"),
            "iy": dict(units="", long_name="Index in y-direction"),
            "z": dict(units=scales.z.unit, long_name=scales.z.description or 'z coordinate'),
            "reltime": dict(units="s", long_name="Recording time since start"),
        }

        data = {
            "ix": ix,
            "iy": iy,
            "x": x,
            "y": y,
            "z": float(z) if z.ndim == 0 else z,  # z can be a scalar or an array
            "reltime": relative_time
        }
        for name, comp in frame.components.items():
            if name not in ("x", "y", "ix", "iy", "reltime"):
                data_attrs[name] = dict(
                    units=comp.scale.unit,
                    long_name=comp.scale.description or name,
                    source_slope=comp.scale.slope,
                    source_offset=comp.scale.offset,
                    calibration_applied=True
                )
                data[name] = _compute_physical_quantity(comp.planes[0], comp.scale.offset, comp.scale.slope)
                data[f'raw_{name}'] = comp.planes[0]
                data_attrs[f'raw_{name}'] = dict(units='', long_name=f'Raw source values for {name}')
        return PIVData(data, data_attrs, {'software': json.dumps(DAVIS_SOFTWARE)})

    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               recording_dtime: Union[datetime, List[datetime]],
               z: Union[str, float, None] = None,
               **kwargs) -> pathlib.Path:
        """Serialize this adapter through the shared snapshot writer."""
        from ..writer import write_snapshot
        return write_snapshot(self.read(relative_time, **kwargs), hdf_filename,
                              self.filename, recording_dtime, z)
