"""Common snapshot serialization for all source formats."""
from pathlib import Path

import h5py
import h5rdmtoolbox as h5tbx
import numpy as np

from .config import get_config
from .time import create_recording_datetime_dataset


def write_snapshot(pivdata, filename, source, recording_dtime=None, z=None):
    """Write values, units, dimension scales and provenance by one policy."""
    source = Path(source)
    filename = Path(filename) if filename is not None else source.with_suffix('.hdf')
    filename.parent.mkdir(parents=True, exist_ok=True)
    pivdata = pivdata.with_z(z)
    ny, nx = len(pivdata.data['y']), len(pivdata.data['x'])
    coordinates = ('x', 'y', 'ix', 'iy', 'z', 'reltime')
    with h5tbx.use(None), h5py.File(filename, 'w') as output:
        output.attrs.update(pivdata.root_attrs)
        output.attrs.update(title='piv snapshot data', source_filename=str(source.resolve()),
                            plane_directory=str(source.parent.resolve()), piv2hdf_schema_version='2.0')
        if recording_dtime is not None:
            create_recording_datetime_dataset(output, recording_dtime, name='time')
        for name, values in pivdata.data.items():
            values = np.asanyarray(values)
            if name not in coordinates and values.shape != (ny, nx):
                raise ValueError(f'Unexpected shape for {name}: {values.shape}, expected {(ny, nx)}')
            options = {}
            if values.ndim:
                options.update(compression=get_config('compression'), compression_opts=get_config('compression_opts'))
            dataset = output.create_dataset(name, data=values, **options)
            dataset.attrs.update(pivdata.data_attrs.get(name, {}))
            if 'units' not in dataset.attrs:
                raise ValueError(f'Missing units for {name}')
            if 'long_name' not in dataset.attrs:
                dataset.attrs['long_name'] = name.replace('_', ' ')
            if name in coordinates:
                dataset.make_scale(name)
        for name in pivdata.data:
            if name not in coordinates:
                dataset = output[name]
                for axis, scales in enumerate((('y', 'iy'), ('x', 'ix'))):
                    for scale in scales:
                        dataset.dims[axis].attach_scale(output[scale])
                dataset.attrs['COORDINATES'] = [key for key in ('z', 'reltime', 'time') if key in output]
    return filename
