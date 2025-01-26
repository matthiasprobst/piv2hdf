"""Davis interface"""

import pathlib
from typing import Dict

import numpy as np

from .._logger import logger
from ..interface import PIVFile

try:
    import lvpyio as lv
except ImportError:
    raise ImportError('Package "lvpyio" not installed which is needed to read Davis files')
try:
    import h5rdmtoolbox as h5tbx
except ImportError:
    raise ImportError('Package "h5rdmtoolbox" not installed which is needed to write HDF files from Davis files')


class File:
    """Davis file interface class"""

    def __init__(self, filename: pathlib.Path):
        self.filename = filename
        self._lvset = None

    def __enter__(self):
        self._lvset = lv.read_set(self.filename)
        return self._lvset

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lvset.close()


def set_to_hdf(filename: pathlib.Path,
               z_m: float = 0,
               use_standard_names: bool = True) -> pathlib.Path:
    """Convert a davis set file to hdf"""
    filename = pathlib.Path(filename)
    if not filename.suffix == '.set':
        raise ValueError(f'File {filename} is not a Davis set file')

    if use_standard_names:
        from . import standard_name_translation

        snt_piv = h5tbx.conventions.standard_name.StandardNameTable.from_gitlab(url='https://git.scc.kit.edu',
                                                                                file_path='particle_image_velocimetry-v1.yaml',
                                                                                project_id='35942',
                                                                                ref_name='main')

    with lv.read_set(filename) as lvset:

        names = list(lvset[0].frames[0].components.keys())
        piv_shape = lvset[0].frames[0].components['U0'].shape
        ny, nx = piv_shape
        nt = len(lvset)

        frame = lvset[0].frames[0]
        iwin_size = int(frame.attributes['InterrogationWindowSize'])
        ix = np.arange(0, nx * iwin_size, iwin_size)
        if ix[-1] < 2 ** 8:
            ix_dtype = np.uint8
        elif ix[-1] < 2 ** 16:
            ix_dtype = np.uint16
        else:
            ix_dtype = np.uint32
        iy = np.arange(0, ny * iwin_size, iwin_size)
        if iy[-1] < 2 ** 8:
            iy_dtype = np.uint8
        elif iy[-1] < 2 ** 16:
            iy_dtype = np.uint16
        else:
            iy_dtype = np.uint32

        hdf_filename = pathlib.Path(filename.parent, filename.stem + '.hdf')
        with h5tbx.File(hdf_filename, 'w') as h5:

            if use_standard_names:
                h5.standard_name_table = snt_piv

            src = h5tbx.conventions.source.Software('Davis',
                                                    version=lvset[0].attributes['_DaVisVersion'],
                                                    author='LaVision GmbH',
                                                    url='https://www.lavision.de')
            h5.source = src

            piv_params = h5.create_group('piv_parameters')

            piv_params.create_dataset('InterrogationWindowSize',
                                      data=iwin_size,
                                      attrs={'units': 'pixel',
                                             'standard_name': 'final_interrogation_window_size'},
                                      dtype=np.uint8)

            n = len(str(len(lvset)))
            fmt = f'0{n}d'
            for iset, s in enumerate(lvset):
                g = piv_params.create_group(f'frame_{iset:{fmt}}')
                for k, v in s.frames[0].attributes.items():
                    g.attrs[k] = v

            # get attrs for x:
            attrs = {k: v for k, v in lvset[0].frames[0].scales.x.__dict__.items() if k not in ('slope', 'offset')}

            if use_standard_names:
                attrs['units'] = attrs['unit']
                attrs.pop('unit')
                attrs['standard_name'] = 'x_coordinate'
            h5.create_dataset('ix', data=ix,
                              dtype=ix_dtype,
                              attrs={'standard_name': 'x_pixel_coordinate',
                                     'units': 'pixel', },
                              make_scale=True)
            h5.create_dataset('x', data=[frame.scales.x.slope * i + frame.scales.x.offset for i in range(nx)],
                              attrs=attrs,
                              make_scale=True)

            # get attrs for y:
            attrs = {k: v for k, v in lvset[0].frames[0].scales.y.__dict__.items() if k not in ('slope', 'offset')}

            if use_standard_names:
                attrs['units'] = attrs['unit']
                attrs['standard_name'] = 'y_coordinate'
                attrs.pop('unit')
            h5.create_dataset('iy', data=iy,
                              dtype=iy_dtype,
                              attrs={'standard_name': 'y_pixel_coordinate',
                                     'units': 'pixel', },
                              make_scale=True)
            h5.create_dataset('y', data=[frame.scales.y.slope * j + frame.scales.y.offset for j in range(ny)],
                              attrs=attrs,
                              make_scale=True)

            h5.create_dataset('z', data=z_m, attrs={'standard_name': 'z_coordinate',
                                                    'units': 'm', }, make_scale=True)
            h5.create_dataset('reltime', shape=(nt,), attrs={'standard_name': 'reltime',
                                                             'units': 's', }, make_scale=True)
            t = np.array([float(s.frames[0].attributes['AcqTimeSeries']) / 10 ** 6 for s in lvset], dtype=np.float32)
            h5.time.values[:] = t

            for name in names:
                attrs = lvset[0].frames[0].components[name].scale.__dict__
                if use_standard_names:
                    attrs['units'] = attrs['unit']
                    attrs['standard_name'] = standard_name_translation.get(name, None)
                    attrs.pop('unit')
                    attrs['long_name'] = attrs['description']
                    attrs.pop('description')
                    if attrs['long_name'] == '':
                        attrs['long_name'] = name
                    if attrs['units'] == 'Valid':
                        attrs['units'] = ''
                ds = h5.create_dataset(name, shape=(nt, 1, *piv_shape), attrs=attrs,
                                       attach_scales=('reltime', 'z', ('y', 'iy'), ('x', 'ix')))
                for it, _set in enumerate(lvset):
                    data = lvset[0].frames[0].components[name][0]
                    h5[name][it, 0, :, :] = data
    return h5.hdf_filename


class DavisIm7(PIVFile):
    """Davis Imager 7 PIV File interface class"""

    def read(self, recording_time: float, build_coord_datasets):
        """Read data from file."""
        is_mset = lv.is_multiset(self.filename)
        logger.debug(f'Reading davis file. is multiset. {is_mset}')
        raise NotImplementedError('DavisIm7.read() not implemented')

    def to_hdf(self, hdf_filename: pathlib.Path, config: Dict, recording_time: float) -> pathlib.Path:
        """Convert file to HDF5 format."""
        raise NotImplementedError('DavisIm7.to_hdf() not implemented')
