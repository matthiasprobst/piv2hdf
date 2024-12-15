import abc
import datetime
import json
import os
import pathlib
import warnings
from functools import wraps
from typing import List, Union, Dict, Tuple, Callable

import h5py
import h5rdmtoolbox as h5tbx
import numpy as np
from tqdm import tqdm

from piv2hdf.config import get_config, get_pivattrs
from . import postproc
from ._logger import logger
from .time import TimeVectorWarning
from .time import create_recording_datetime_dataset
from .utils import generate_temporary_filename, get_uint_type, parse_z

PIV_PARAMETER_GRP_NAME = 'piv_parameters'
PIV_PARAMETER_ATTRS_NAME = 'piv_parameter_dict'

DEFAULT_MPLANE_TITLE = 'piv snapshot data'
DEFAULT_PLANE_TITLE = 'piv plane data'

IMAGE_INDEX = 'image_index'  # name of the image index coordinate


class LayoutValidationWarning(Warning):
    """Warning for layout validation"""
    pass


def userpost(func: Callable) -> Callable:
    """decorates to_hdf method of Snapshot, Plane and Multi-Plane, that runs user code to
    update meta data after the data has been written to the HDF5 file and before layout validation"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """call __to_hdf__ and then perform the layout validation. Raise an error on failure"""
        hdf_filename = func(*args, **kwargs)
        post_caller = args[0].__get_post_func__()
        if post_caller:
            with h5tbx.File(hdf_filename, 'r+') as h5:
                for pc in post_caller:
                    logger.debug(f'Calling postprocessing function {pc.__name__}...')
                    pc(h5)

        for pp_func_name in get_config()['postproc']:
            if callable(pp_func_name):
                pp_func = pp_func_name
            elif isinstance(pp_func_name, str):
                if pp_func_name not in dir(postproc):
                    raise NotImplementedError(f'Postprocessing function {pp_func_name} not implemented.')
                pp_func = getattr(postproc, pp_func_name)
            else:
                raise TypeError(f'Expected a string or callable for key "postproc", got {type(pp_func_name)}')
            with h5tbx.File(hdf_filename, 'r+') as h5:
                pp_func(h5)
        return hdf_filename

    return wrapper


def layoutvalidation(func: Callable) -> Callable:
    """decorates __to_hdf__ method of Snapshot, Plane and Multi-Plane classes to validate the created
    HDF5 content"""

    @wraps(func)
    def validation_wrapper(*args, **kwargs):
        """call __to_hdf__ and then perform the layout validation. Raise an error on failure"""
        hdf_filename = func(*args, **kwargs)

        config = get_config()

        # checking standard attributes:

        with h5tbx.use(get_config('convention')) as cv:
            logger.debug(
                f'Checking the PIV attributes based on convention currently set ({cv.name} @ {cv.filename})...'
            )
            with h5tbx.File(hdf_filename, 'r+') as h5:

                for sa_name, std_attr in h5.standard_attributes.items():
                    if sa_name not in h5.attrs.raw:
                        if std_attr.default_value == std_attr.__class__.NONE:
                            logger.debug(f'Attribute {sa_name} not set but is also only optional.')
                            continue
                        else:
                            logger.error(f'Attribute not set: {sa_name} but is required.')
                            continue

                    std_attr.validate(h5.attrs.raw[sa_name], parent=h5)

                if 'standard_name_table' in h5.standard_attributes:
                    snt = h5.standard_name_table
                    snt.check_hdf_group(h5)

        return hdf_filename

    return validation_wrapper


def scan_for_timeseries_nc_files(directory: pathlib.Path, suffix: str, prefix_pattern: None) -> List[pathlib.Path]:
    """
    Scans for files in `directory` according to the `suffix` provided. With `prefix_pattern` the pattern can be
    specified. `prefix_pattern` and `suffix` are concatenated to form the final pattern, which is passed to
    `pathlib.Path.glob()`. The result is a list of files sorted by name.
    """
    if prefix_pattern is None:
        prefix_pattern = '*'
    logger.debug(f'Scanning for files with pattern {prefix_pattern}{suffix} in {directory}')
    list_of_files = sorted(directory.glob(f'{prefix_pattern}{suffix}'))
    return list_of_files


def copy_piv_parameter_group(src: h5py.Group, trg: h5py.Group) -> None:
    """copies the piv parameters to the new group"""

    def _to_grp(_src, _trg):
        for ak, av in _src.attrs.items():
            _trg.attrs[ak] = av
        for k, v in _src.items():
            if isinstance(v, h5py.Group):
                obj = _trg.create_group(k)
                _to_grp(v, obj)
            else:
                obj = _trg.create_dataset(k, data=v[()])
            for ak, av in v.attrs.items():
                obj.attrs[ak] = av

    _to_grp(src, trg)


class PIVParameterInterface:
    """Abstract PIV Parameter Interface"""
    __slots__ = ('param_dict', '_piv_param_datasets', 'filename')
    __suffix__ = None

    def __init__(self, filename):
        if filename is not None:
            self.filename = pathlib.Path(filename)
            if not self.filename.exists():
                raise FileNotFoundError(f'File {filename} does not exist.')
        else:
            self.filename = filename
        self.param_dict = {}
        self._piv_param_datasets = dict(x_final_iw_size=dict(data=None,
                                                             units='pixel',
                                                             standard_name='x_final_interrogation_window_size',
                                                             long_name='Final interrogation window size in x-direction'),
                                        y_final_iw_size=dict(data=None,
                                                             units='pixel',
                                                             standard_name='y_final_interrogation_window_size',
                                                             long_name='Final interrogation window size in y-direction'),
                                        x_final_iw_overlap_size=dict(data=None,
                                                                     units='pixel',
                                                                     standard_name='x_final_interrogation_window_overlap_size',
                                                                     long_name='Final overlap of interrogation windows in x-direction'),
                                        y_final_iw_overlap_size=dict(data=None,
                                                                     units='pixel',
                                                                     standard_name='y_final_interrogation_window_overlap_size',
                                                                     long_name='Final overlap of interrogation windows in y-direction'),
                                        laser_pulse_delay=dict(data=None,
                                                               units='s',
                                                               standard_name='laser_pulse_delay',
                                                               long_name='Pulse delay'),
                                        piv_scaling_factor=dict(data=None,
                                                                units='pixel/m',
                                                                standard_name='piv_scaling_factor',
                                                                )
                                        )

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.filename}": {self.param_dict})'

    @property
    def piv_param_datasets(self):
        return self._piv_param_datasets

    @property
    def suffix(self):
        """Return the suffix of the PIV Parameter File"""
        return self.__suffix__

    @classmethod
    def from_dir(cls, directory_path):
        """reads parameter from `directory_path` and returns instance of this class

        Parameters
        ----------
        directory_path: pathlib.Path
            Path to (plane) directory containing the .nc-files

        Returns
        -------
        Instance of this class
        """
        pars = list(directory_path.glob(f'*{cls.__suffix__}'))
        if len(pars) > 1:
            raise FileExistsError(f'More than one parameter file with suffix "{cls.__suffix__}" '
                                  f'found here: "{directory_path}".')
        elif len(pars) == 0:
            raise FileExistsError(f'Cannot find the parameter file with suffix "{cls.__suffix__}" '
                                  f'here: "{directory_path}".')
        return cls(pars[0])

    @abc.abstractmethod
    def save(self, filename: pathlib.Path):
        """Save to original file format"""

    def to_dict(self) -> Dict:
        """Convert to a dictionary"""
        return self.param_dict

    def to_hdf(self, grp: h5py.Group) -> h5py.Group:
        """Write parameter content to HDF group"""
        grp.attrs[PIV_PARAMETER_ATTRS_NAME] = json.dumps(self.param_dict)
        for k, v in self.piv_param_datasets.items():
            if v['data']:
                ds = grp.create_dataset(name=k, data=v['data'])
                for ak, av in v.items():
                    if ak != 'data':
                        ds.attrs[ak] = av
        return grp


class PIVFile(abc.ABC):
    """Abstract Interface class for PIV. This class must be inherited for a each PIV file type.
    Required methods are:
        - read(recording_dtime: float, build_coord_datasets: bool = True)
        - to_hdf(hdf_filename: pathlib.Path, recording_dtime: float)
    """
    suffix: str = None
    __parameter_cls__ = PIVParameterInterface

    def __init__(self,
                 filename: pathlib.Path,
                 parameter: Union[None, PIVParameterInterface, pathlib.Path] = None,
                 post_func: Callable = None,
                 **kwargs):
        filename = pathlib.Path(filename)
        if not filename.is_file():
            raise TypeError(f'Snapshot file path is not a file: {filename}.')
        if self.suffix is not None:
            if self.suffix != '':
                if filename.suffix != self.suffix:
                    raise NameError(f'Expecting suffix {self.suffix}, not {filename.suffix}')
        self.filename = filename
        if isinstance(parameter, PIVParameterInterface):
            self._parameter = parameter
        elif parameter is None:
            logger.debug(
                f'Auto detecting parameter file {self.__parameter_cls__.__name__} from directory {filename.parent}')
            self._parameter = self.__parameter_cls__.from_dir(filename.parent)
        else:
            logger.debug(f'Initializing parameter file from filename using class {self.__parameter_cls__.__name__}')
            self._parameter = self.__parameter_cls__(parameter)

        if callable(post_func):
            self.post_func = [post_func, ]
        elif isinstance(post_func, (list, tuple)):
            self.post_func = post_func
        elif post_func is None:
            self.post_func = []
        else:
            raise TypeError('post_func must be callable or list/tuple of callables or None')

    @property
    def parameter(self):
        """Return the parameter class"""
        return self._parameter

    def __repr__(self):
        return f'<{self.__class__.__name__} ({self.filename})>'

    @abc.abstractmethod
    def read(self, relative_time: float, build_coord_datasets=True) -> Tuple[Dict, Dict, Dict]:
        """Read data from file.
        Except data, root_attr, variable_attr"""

    def write_parameters(self, param_grp: h5py.Group):
        """Write piv parameters to an opened and existing param_grp"""
        return self._parameter.to_hdf(param_grp)

    @abc.abstractmethod
    def to_hdf(self,
               hdf_filename: pathlib.Path,
               relative_time: float,
               z: Union[str, float, None] = None) -> pathlib.Path:
        """converts the snapshot into an HDF file"""


class PIVConverter(abc.ABC):
    """Abstract converter class"""

    def __init__(self):
        self.hdf_filename = None

    def __repr__(self):
        return f'<{self.__class__.__name__}>'

    @abc.abstractmethod
    def __get_post_func__(self):
        """return the post processing function"""

    @abc.abstractmethod
    def __to_hdf__(self, hdf_filename: pathlib.Path, **kwargs) -> pathlib.Path:
        """conversion method"""

    @layoutvalidation
    @userpost
    def to_hdf(self,
               piv_attributes: Dict = None,  # e.g. piv_medium, contact
               hdf_filename: pathlib.Path = None,
               z: float = None,
               **kwargs) -> pathlib.Path:
        """conversion method

        Parameters
        ----------
        piv_attributes: Dict
            Attributes to be added to the root group of the HDF file
        hdf_filename: pathlib.Path
            Path to the HDF file to be written. None will generate an appropriate filename.
            If the path is a directory, the filename will be generated from the directory name.
        z: float
            z-coordinate of the plane. If None, the z-coordinate is read from the parameter file
        kwargs: Dict
            Additional keyword arguments for the conversion method
        """
        if hdf_filename is not None:
            _hdf_filename = pathlib.Path(hdf_filename)
            if _hdf_filename.is_dir():
                hdf_filename = _hdf_filename / f'{_hdf_filename.stem}.hdf'
        hdf_filename = self.__to_hdf__(hdf_filename=hdf_filename,
                                       z=z,
                                       **kwargs)
        self.hdf_filename = hdf_filename

        _piv_attrs = get_pivattrs()

        if piv_attributes:
            _piv_attrs.update(piv_attributes)

        with h5tbx.File(hdf_filename, 'r+') as h5:
            for k, v in _piv_attrs.items():
                h5.attrs[k] = v

        return hdf_filename


class PIVSnapshot(PIVConverter):
    """Interface class"""

    def __init__(self,
                 piv_file: PIVFile,
                 recording_dtime: Union[datetime.datetime, None]):
        super().__init__()
        if not isinstance(piv_file, PIVFile):
            raise TypeError(f'Expecting type {PIVFile.__class__} fro parameter piv_file, not type {type(piv_file)}')

        # process recording_dtime:
        if recording_dtime is None:
            logger.debug('No datetime provided!')
            self.recording_dtime = None
        else:
            if not isinstance(recording_dtime, datetime.datetime):
                raise TypeError('Expecting type int, float or datetime for recording time, '
                                f'not type {type(recording_dtime)}')

            self.recording_dtime = recording_dtime  # .isoformat()
        self.relative_time = 0
        self.piv_file = piv_file

    def __repr__(self):
        return f'<{self.__class__.__name__} of {self.piv_file.__repr__()} >'

    def __get_post_func__(self):
        return self.piv_file.post_func

    def __to_hdf__(self,
                   hdf_filename: pathlib.Path = None,
                   z: Union[str, float, None] = None,
                   **kwargs) -> pathlib.Path:
        """converts the snapshot into an HDF file
        """

        if hdf_filename is None:
            hdf_filename = self.piv_file.filename.parent / f'{self.piv_file.filename.stem}.hdf'
        else:
            hdf_filename = pathlib.Path(hdf_filename)
            hdf_filename.parent.mkdir(parents=True, exist_ok=True)
        hdf_filename = self.piv_file.to_hdf(hdf_filename=hdf_filename,
                                            relative_time=self.relative_time,
                                            recording_dtime=self.recording_dtime,
                                            z=z)
        with h5py.File(hdf_filename, 'r+') as h5:
            self.piv_file.write_parameters(h5.create_group(PIV_PARAMETER_GRP_NAME))
        return hdf_filename


class LayoutValidationError(Exception):
    """Exception raised when Layout validation failed"""


class PIVPlane(PIVConverter):
    """Interface class"""

    __slots__ = 'list_of_piv_files', 'rel_time_vector'
    plane_coord_order = ('reltime', 'y', 'x')

    def __init__(self, list_of_piv_files: List[PIVFile],
                 time_info: Union[Tuple[datetime.datetime, float], List[datetime.datetime]]):
        """

        Parameters
        ----------
        list_of_piv_files: List[PIVFile]
            List of PIVFile objects to be combined
        time_info: Union[Tuple[datetime.datetime, float], List[datetime.datetime]]
            Time information based on which the relative and absolute time vectors are generated.
            The input can be two-fold:
            1. A Tuple of datetime object specifying the start of the PIV recording and a float specifying the
            measurement frequency
            2. A list of datetime objects corresponding to the recording datetime of each image
        """
        super().__init__()
        if not isinstance(list_of_piv_files, (tuple, list)):
            raise TypeError(f'Expecting a list of {PIVFile.__class__} objects but got {type(list_of_piv_files)}')
        if not all(isinstance(piv_file, PIVFile) for piv_file in list_of_piv_files):
            for piv_file in list_of_piv_files:
                if not isinstance(piv_file, PIVFile):
                    raise TypeError(
                        f'Expecting type {PIVFile.__class__} for each entry, but one entry is of type {type(piv_file)}')
        self.list_of_piv_files = list_of_piv_files
        n = len(list_of_piv_files)
        if n == 0:
            raise ValueError('List of piv files is empty!')
        if n == 1:
            raise ValueError('Only one piv file is provided. Use PIVSnapshot instead!')

        # process recording_dtime:
        if not isinstance(time_info, (tuple, list)):
            raise TypeError(f'Expecting type tuple or list for parameter time_info, '
                            f'not type {type(time_info)}')
        if len(time_info) == 2:
            if isinstance(time_info[1], (float, int)):
                # build time vector based on recording start datetime and frequency
                recording_start_dtime, meas_frequency = time_info
                if not isinstance(recording_start_dtime, datetime.datetime):
                    raise TypeError(f'Expecting type datetime for parameter time_info[0], '
                                    f'not type {type(time_info[0])}')
                # build time vector (rel and abs)
                rel_time = np.linspace(0, (n - 1) / meas_frequency, n)
                abs_time = [recording_start_dtime + datetime.timedelta(seconds=rel_time[i]) for i in range(n)]
            else:
                if not all(isinstance(r, datetime.datetime) for r in time_info):
                    raise TypeError('Expecting type datetime for all items of parameter time_info[0]')
                abs_time = time_info
                rel_time = [(abs_time[i] - abs_time[0]).total_seconds() for i in range(n)]
        else:
            # expecting a list of datetime objects
            if not all(isinstance(r, datetime.datetime) for r in time_info):
                raise TypeError('Expecting type datetime for all items of parameter time_info[0]')
            if n != len(time_info):
                warnings.warn(f'Length of time_info ({len(time_info)}) is inconsistent with number of piv files ({n}). '
                              f'Assuming that time_info is a correct time vector, thus taking only the first {n} '
                              'entries.', TimeVectorWarning)
            abs_time = time_info[0:n]
            rel_time = [(abs_time[i] - abs_time[0]).total_seconds() for i in range(n)]

        self.rel_time_vector = np.asarray(rel_time)
        self.abs_time_vector = np.asarray(abs_time)
        self.IMAGE_INDEX = np.arange(1, n + 1, 1)

        if len(self.rel_time_vector) != len(list_of_piv_files):
            raise ValueError(f'Length of time vector ({len(self.rel_time_vector)}) is inconsistent with number of piv '
                             f'files: ({len(list_of_piv_files)})!')

    def __get_post_func__(self):
        return self.list_of_piv_files[0].post_func

    def __getitem__(self, item):
        return self.list_of_piv_files[item]

    def __len__(self):
        return len(self.list_of_piv_files)

    def __repr__(self):
        if len(self.list_of_piv_files) > 0:
            return f'<{self.__class__.__name__} containing {len(self.list_of_piv_files)} ' \
                   f'<{self.list_of_piv_files[0].__class__.__name__}> files>'
        return f'<Empty {self.__class__.__name__}>'

    @staticmethod
    def from_folder(plane_directory: Union[str, pathlib.Path],
                    time_info: Union[Tuple[datetime.datetime, float], List[datetime.datetime]],
                    pivfile: PIVFile,
                    parameter: Union[PIVParameterInterface, str, pathlib.Path] = None,
                    n: int = -1,
                    prefix_pattern='*[0-9]'):
        """PIV Plane initialized from a piv plane folder.

        Parameters
        ----------
        plane_directory: Union[str, pathlib.Path]
            Path to folder with snapshots
        time_info: Union[Tuple[datetime.datetime, float], List[datetime.datetime]]
            Time information based on which the relative and absolute time vectors are generated.
            The input can be two-fold:
            1. A Tuple of datetime object specifying the start of the PIV recording and a float specifying the
            measurement frequency
            2. A list of datetime objects corresponding to the recording datetime of each image
        pivfile: PIVFile
            The piv file object associated to a software.
        parameter: Union[PIVParameterInterface, str, pathlib.Path], optional=None
            If provided, this parameter interface will be used for all files.
            Default is None and searches for the parameter file associated to
            `pivfile`. However, in case the auto-detection finds multiple
            parameter files in the plane folder the process stops. This can
            be avoided by passing the parameter file explicitly here.
        n: int, default=-1
            Number of snapshot to take. Default value -1 means taking all found snapshots.
        prefix_pattern: str, default=*[0-9]?
            Pattern to match the snapshot files. Default is *[0-9]. The complete pattern is
            constructed as follows: prefix_pattern + pivfile.suffix

        Returns
        -------
        The initialized `PIVPlane` object
        """

        # if parameter is not None:
        #     if isinstance(parameter, (str, pathlib.Path)):
        #         parameter = PIVParameterInterface(parameter)
        #     elif not isinstance(parameter, PIVParameterInterface):
        #         raise TypeError('Argument "parameter" must be of type PIVParameterInterface, not '
        #                         f'{type(parameter)}.')

        plane_directory = pathlib.Path(plane_directory)
        # TODO rename: scan_for_timeseries_nc_files. There should be no "nc" in it
        found_snapshot_files = scan_for_timeseries_nc_files(plane_directory,
                                                            pivfile.suffix,
                                                            prefix_pattern=prefix_pattern)

        n_files = len(found_snapshot_files)
        logger.debug(f'Found {n_files} snapshot files here: {plane_directory.absolute()}')

        if n == -1:
            n = n_files
        found_snapshot_files = found_snapshot_files[0:n]
        logger.debug(f'Taking {len(found_snapshot_files)}/{n_files} snapshot')
        n_files = len(found_snapshot_files)
        if n_files == 0:
            raise ValueError(f'No snapshot files found in {plane_directory.absolute()}')

        pbar = tqdm(found_snapshot_files, desc=f'Init {n_files} {pivfile.__name__} object')
        if parameter:
            return PIVPlane([pivfile(nc, parameter=parameter) for nc in pbar], time_info)

        return PIVPlane([pivfile(nc) for nc in pbar], time_info, )

    def __to_hdf__(self,
                   hdf_filename: Union[str, pathlib.Path] = None,
                   z: Union[float, str] = None,
                   **kwargs) -> pathlib.Path:
        """converts the snapshot into an HDF file

        Parameters
        ---------
        hdf_filename: Union[str, pathlib.Path], default=None
            The target hdf file. If None, a name is selected based on the plane folder name
        z: float, str, default=None
            the z-coordinate in [m] of the plane. Should be given if not present in the snapshot file or
            to be overwritten by this method. Instead of a float a string can be given, which is then
            interpreted as the z value with its unit, e.g. '1.2 mm'
        """

        iplane = kwargs.pop('iplane', None)
        nplanes = kwargs.pop('nplanes', None)
        title = kwargs.pop('title', DEFAULT_PLANE_TITLE)

        if hdf_filename is None:
            _parent = self.list_of_piv_files[0].filename.parent
            hdf_filename = _parent / f'{_parent.stem}.hdf'
        else:
            hdf_filename = pathlib.Path(hdf_filename)
            hdf_filename.parent.mkdir(parents=True, exist_ok=True)

        # get data from first snapshot to prepare the HDF5 file
        logger.debug('Reading first PIV file: '
                     f'{self.list_of_piv_files[0].__class__.__name__}({self.list_of_piv_files[0].filename}')
        data, root_attr, variable_attr = self.list_of_piv_files[0].read(self.rel_time_vector[0])

        if z is not None:
            data['z'], z_units = parse_z(z)
        else:
            z_units = None
        mandatory_keys = ('x', 'y', 'z', 'ix', 'iy', 'u', 'v', 'reltime')
        for mkey in mandatory_keys:
            if mkey not in data.keys():
                raise KeyError(f'Mandatory key {mkey} not provided.')
        nt, ny, nx = self.rel_time_vector.size, data['y'].size, data['x'].size
        _shape = dict(reltime=nt, y=ny, x=nx)
        dataset_shape = tuple([_shape[k] for k in self.plane_coord_order])
        _chunk = [_shape[n] for n in self.plane_coord_order]
        _chunk[self.plane_coord_order.index('reltime')] = 1
        dataset_chunk = tuple(_chunk)
        iy_idim = self.plane_coord_order.index('y')
        ix_idim = self.plane_coord_order.index('x')
        compression = get_config()['compression']
        compression_opts = get_config()['compression_opts']

        logger.debug(f'Writing plane data to (plane-) file {pathlib.Path(hdf_filename).absolute()}')

        if iplane is not None and nplanes is not None:
            pbar = tqdm(total=self.__len__(), desc=f'Writing plane HDF5 file {iplane}/{nplanes}')
        else:
            pbar = tqdm(total=self.__len__(), desc=f'Writing plane HDF5 file')

        with h5tbx.File(hdf_filename, 'w') as h5main:
            pbar.update()
            h5main.attrs['title'] = title
            for ak, av in root_attr.items():
                h5main.attrs[ak] = av
            # self.list_of_piv_files[0].write_parameters(h5main.create_group(PIV_PARAMETER_GRP_NAME))
            if PIV_PARAMETER_GRP_NAME not in h5main:
                h5main.create_group(PIV_PARAMETER_GRP_NAME)
            self.list_of_piv_files[0].write_parameters(h5main[PIV_PARAMETER_GRP_NAME])
            logger.debug('Creating datasets for coordinates (x, y, z, time, ix, iy)')

            if data['ix'].min() < 0:
                raise ValueError('Data of ix must not be smaller than zero!')
            if data['iy'].min() < 0:
                raise ValueError('Data of iy must not be smaller than zero!')

            for ds_name, maxshape in zip(('x', 'ix', 'y', 'iy', 'z'), (nx, nx, ny, ny, None)):
                h5main.create_dataset(name=ds_name,
                                      data=data[ds_name],
                                      maxshape=maxshape,
                                      dtype=get_config()['dtypes'].get(ds_name, None))

            h5main.create_dataset('reltime', data=self.rel_time_vector,
                                  dtype=get_config()['dtypes'].get('reltime', None))

            for varkey in ('x', 'y', 'ix', 'iy', 'reltime'):
                h5main[varkey].make_scale()
                for ak, av in variable_attr[varkey].items():
                    h5main[varkey].attrs[ak] = av

            if 'z' not in variable_attr:
                h5main['z'].attrs['units'] = z_units
            else:
                for ak, av in variable_attr['z'].items():
                    h5main['z'].attrs[ak] = av

            # write all other dataset to file:
            logger.debug('Writing abs time vector')
            ds_rec_dtime = create_recording_datetime_dataset(h5main, list(self.abs_time_vector), name='time')
            ds_rec_dtime.make_scale()
            ds_imgidx = h5main.create_dataset(IMAGE_INDEX, data=self.IMAGE_INDEX,
                                              attrs=dict(units='',
                                                         standard_name='piv_image_index'),
                                              dtype=get_config()['dtypes'].get(IMAGE_INDEX, None))
            ds_imgidx.make_scale()

            dataset_keys = []  # fill with dataset names that are not coordinates.

            for varkey, vardata in data.items():
                if varkey not in ('x', 'y', 'ix', 'iy', 'z', 'reltime'):
                    dataset_keys.append(varkey)
                    ds = h5main.create_dataset(name=varkey,
                                               shape=dataset_shape,
                                               dtype=vardata.dtype,
                                               maxshape=dataset_shape,
                                               chunks=dataset_chunk,
                                               compression=compression,
                                               compression_opts=compression_opts)
                    for idim, coordname in enumerate(self.plane_coord_order):
                        ds.dims[idim].attach_scale(h5main[coordname])
                    ds.dims[iy_idim].attach_scale(h5main['iy'])
                    ds.dims[ix_idim].attach_scale(h5main['ix'])
                    ds.dims[0].attach_scale(ds_rec_dtime)
                    ds.dims[0].attach_scale(ds_imgidx)
                    ds.attrs['COORDINATES'] = ['z', ]
                    ds[0, ...] = vardata[...]

                    # write attributes to datasets
                    if varkey in variable_attr:
                        for ak, av in variable_attr[varkey].items():
                            ds.attrs[ak] = av

            logger.debug(f'Writing all datasets to file. Number of source files: {len(self.list_of_piv_files[1:])}')
            for (ifile, piv_file), t in zip(enumerate(self.list_of_piv_files[1:]), self.rel_time_vector[1:]):
                pbar.update(1)
                logger.debug(f'Processing {piv_file.filename}')
                data, _, variable_attr = piv_file.read(t)
                for varkey in dataset_keys:
                    h5main[varkey][ifile + 1, ...] = data[varkey][...]

                # pivview hack: flag meaning might be different for each plane because not all planes
                # have the same flag values and only those that appear are added to the dict
                if 'piv_flags' in variable_attr.keys():
                    try:
                        flag_meaning_plane = variable_attr['piv_flags']['flag_meaning']
                        if not isinstance(flag_meaning_plane, dict):
                            flag_meaning_plane_dict = json.loads(flag_meaning_plane)

                        try:
                            flag_meaning_main_dict = h5main['piv_flags'].attrs['flag_meaning']
                            if not isinstance(flag_meaning_main_dict, dict):
                                flag_meaning_main_dict = json.loads(flag_meaning_main_dict)
                            flag_meaning_main_dict.update(flag_meaning_plane_dict)
                            h5main['piv_flags'].attrs['flag_meaning'] = json.dumps(flag_meaning_main_dict)
                        except KeyError:
                            pass
                    except KeyError:
                        pass
        self.hdf_filename = hdf_filename
        return pathlib.Path(hdf_filename)


class PIVMultiPlane(PIVConverter):
    """Interface class"""
    __slots__ = 'list_of_piv_folder'
    plane_coord_order = ('z', 'reltime', 'y', 'x')

    def __getitem__(self, item):
        return self.list_of_piv_folder[item]

    def __init__(self, list_of_piv_planes: List[PIVPlane], ):
        super().__init__()
        if not isinstance(list_of_piv_planes, (tuple, list)):
            raise TypeError(
                f'Expecting a lists of list of {PIVPlane.__class__} objects but got {type(list_of_piv_planes)}')
        for i, piv_folder in enumerate(list_of_piv_planes):
            if len(piv_folder) == 0:
                raise ValueError(f'Plane number {i} is empty. Cannot proceed.')

        self.list_of_piv_planes = list_of_piv_planes
        pivfile_types = []
        ref_type = type(list_of_piv_planes[0].list_of_piv_files[0])
        for folder in list_of_piv_planes:
            pivfile_types.append(all(isinstance(f, ref_type) for f in folder.list_of_piv_files))
        if not all(pivfile_types):
            warnings.warn('The provided PIVFile objects are not all from the same class which may cause problems '
                          'in the process of converting them to HDF file(s).', UserWarning)

    def __len__(self):
        return len(self.list_of_piv_planes)

    def __repr__(self):
        out = f'<{self.__class__.__name__} of {self.__len__()} planes:\n\t'
        out += '\n\t'.join([p.__repr__() for p in self.list_of_piv_planes])
        out += '\n>'
        return out

    def __get_post_func__(self):
        return self.list_of_piv_planes[0].__get_post_func__()

    @staticmethod
    def merge_planes(*,
                     hdf_filenames: List[pathlib.Path],
                     target_hdf_filename: Union[str, pathlib.Path],
                     title: str):
        """merges multiple piv plane hdf files together

        Parameters
        ----------
        hdf_filenames: List[pathlib.Path]
            List of hdf files to merge
        target_hdf_filename: Union[str, pathlib.Path]
            Name of the target hdf file
        title: str
            Title of the target hdf file
        """
        n_files = len(hdf_filenames)
        print(f'Merging {n_files} HDF files into {target_hdf_filename}.')
        software_list = []
        nt_list = []
        t_list = []
        x_data = []
        y_data = []
        z_data = []
        dim_names = []
        for hdf_file in hdf_filenames:
            with h5py.File(hdf_file) as h5plane:
                try:
                    software_list.append(h5plane.attrs['software'])
                except KeyError:
                    pass
                nt_list.append(h5plane['reltime'].size)
                t_list.append(h5plane['reltime'][()])
                x_data.append(h5plane['x'][:])
                y_data.append(h5plane['y'][:])
                z_data.append(h5plane['z'][()])
                dim_names.append([d[0].name for d in h5plane['u'].dims])
        if any(dim_names[0] != d for d in dim_names):
            raise RuntimeError('Inconsistent dimension names. All planes must have same dimension names'
                               'for all datasets.')
        # check compliance of planes:
        if len(software_list) > 1:
            if not np.all([software_list[0] == s for s in software_list[1:]]):
                raise ValueError('Plane have different software names registered')

        # they only can be merged if x and y data is equal but z is different:
        if not np.all([np.array_equal(x_data[0], x) for x in x_data[1:]]):
            raise ValueError('x coordinates of planes are different. Cannot merge.')
        if not np.all([np.array_equal(y_data[0], y) for y in y_data[1:]]):
            raise ValueError('y coordinates of planes are different. Cannot merge.')
        if np.any([z_data[0] == z for z in z_data[1:]]):
            raise ValueError(f'z coordinates must be different in order to merge the planes: {z_data}')

        with h5py.File(target_hdf_filename, 'w') as h5main:
            if software_list:
                h5main.attrs['software'] = software_list[0]
            # check if time vectors are close has been done by this time.
            nt_min = min(nt_list)
            dt_list = [np.diff(t[0:nt_min]) for t in t_list]
            abs_diff = np.abs(dt_list[0] - dt_list[1:])
            if np.all(abs_diff == 0.):
                print('Relative time vectors are identical. Merging into one group...')
                PIVMultiPlane._merge_planes_equal_time_vectors(h5main=h5main,
                                                               hdf_filenames=hdf_filenames,
                                                               title=title,
                                                               rel_time=np.asarray(t_list[0][0:nt_min]))
                return target_hdf_filename
            print('Time vectors are close enough but not identical. Merging into one group by using one '
                  'time vector. The time vectors of all planes are written in a separate group to keep them ...')
            av_time = np.asarray([t[:nt_min] for t in t_list]).mean(axis=0)
            PIVMultiPlane._merge_planes_equal_time_vectors(h5main=h5main,
                                                           hdf_filenames=hdf_filenames,
                                                           title=title,
                                                           rel_time=av_time)
            return target_hdf_filename

    @staticmethod
    def _merge_planes_equal_time_vectors(*,
                                         h5main: h5py.File,
                                         hdf_filenames: List[pathlib.Path],
                                         title: str,
                                         rel_time: [np.ndarray, List, None] = None) -> h5py.File:
        """

        If time is given, this time is used instead the one from the planes. Thus it is assumed
        that all plane times are close but not identical. Then time of each plane is
        put in a separate group.

        Parameters
        ----------
        h5main: h5py.File
            HDF file to write the merged data to
        hdf_filenames: List[pathlib.Path]
            List of hdf files to merge
        title: str
            Title of the target hdf file
        rel_time: [np.ndarray, List, None]=None
            Relative time vector to use.
        """
        config = get_config()
        nz = len(hdf_filenames)
        if rel_time is None:
            raise TypeError('rel_time must be given')

        if not isinstance(rel_time, np.ndarray):
            rel_time = np.asarray(rel_time)
        assert rel_time.ndim == 1
        nt = rel_time.size

        with h5py.File(hdf_filenames[0]) as h5plane:

            # write the absolute time of each plan into a dataset called "time", which will not be a dimension scale
            # for the PIV data!.
            ds_time = h5main.create_dataset('time', shape=(nz, nt), dtype=h5plane['time'].dtype)
            ds_time.attrs.update(h5plane['time'].attrs.items())
            ds_time[0, :] = h5plane['time'][:nt]
            for _iz, plane_hdf_filename in enumerate(hdf_filenames[1:]):
                with h5py.File(plane_hdf_filename) as _h5plane:
                    ds_time[_iz + 1, :] = _h5plane['time'][:nt]

            ny = h5plane['y'].size
            nx = h5plane['x'].size
            dim_names = [os.path.basename(d[0].name) for d in h5plane['u'].dims]
            plane_coord_order = PIVMultiPlane.plane_coord_order
            shape_dict = {'reltime': nt, 'x': nx, 'y': ny}
            _ds_shape_list = [shape_dict[n] for n in dim_names]

            ix = plane_coord_order.index('x')
            iy = plane_coord_order.index('y')
            if abs(ix - iy) > 1:
                raise ValueError('Invalid plane coord position. x and y must be next to each other.')
            zidx = plane_coord_order.index('z')
            it = plane_coord_order.index('reltime')
            if zidx == 0 and it == 1:
                ds_shape = (nz, nt, ny, nx)
                ds_chunk = (1, 1, ny, nx)
            elif zidx == 1 and it == 0:
                ds_shape = (nt, nz, ny, nx)
                ds_chunk = (1, 1, ny, nx)
            else:
                raise NotImplementedError('Cannot work with that shape...')

            coord_names = ('x', 'y', 'reltime', 'z', 'ix', 'iy', 'iz', 'time')
            dataset_properties = {k: {'dtype': v.dtype} for k, v in h5plane.items() if
                                  isinstance(v, h5py.Dataset) and k not in coord_names and v.ndim > 1}

            compression = h5plane['u'].compression
            compression_opts = h5plane['u'].compression_opts

            x = h5plane['x'][:]
            y = h5plane['y'][:]
            ix = h5plane['ix'][:]
            iy = h5plane['iy'][:]

        iz = np.arange(1, nz + 1, 1, dtype=int)

        h5main.attrs['title'] = title
        ds_x = h5main.create_dataset('x', data=x, dtype=config['dtypes'].get('x', None))
        ds_x.make_scale()
        ds_y = h5main.create_dataset('y', data=y, dtype=config['dtypes'].get('y', None))
        ds_y.make_scale()

        ds_ix = h5main.create_dataset('ix', data=ix, dtype=config['dtypes'].get('ix', get_uint_type(ix)))
        ds_ix.make_scale()
        ds_iy = h5main.create_dataset('iy', data=iy, dtype=config['dtypes'].get('iy', get_uint_type(iy)))
        ds_iy.make_scale()
        ds_iz = h5main.create_dataset('iz', data=iz, dtype=config['dtypes'].get('iz', get_uint_type(iz)))
        ds_iz.make_scale()

        ds_z = h5main.create_dataset('z', shape=(nz,), dtype=config['dtypes'].get('z', None))
        ds_z.make_scale()
        ds_relt = h5main.create_dataset('reltime', shape=(nt,), dtype=config['dtypes'].get('reltime', None))
        ds_relt[:] = rel_time
        ds_relt.make_scale()

        ds_imgidx = h5main.create_dataset(IMAGE_INDEX, shape=(nt,), dtype=config['dtypes'].get(IMAGE_INDEX, None))
        ds_imgidx[:] = np.arange(1, nt + 1, 1, dtype=int)
        ds_imgidx.make_scale()

        # ds_time.dims[0].attach_scale(h5main['z'])
        # ds_time.dims[1].attach_scale('reltime')
        for ds_name, ds_props in dataset_properties.items():
            ds = h5main.create_dataset(ds_name,
                                       shape=ds_shape,
                                       chunks=ds_chunk,
                                       dtype=ds_props['dtype'],
                                       compression=compression,
                                       compression_opts=compression_opts)
            for i, n in enumerate(plane_coord_order):
                ds.dims[i].attach_scale(h5main[n])
                if n == 'z':
                    ds.dims[i].attach_scale(h5main['iz'])
                if n == 'reltime':
                    # ds.dims[i].attach_scale(h5main['time'])  # don't attach, because it is 2D! and anyhow a common dimension cannot be found
                    ds.dims[i].attach_scale(h5main[IMAGE_INDEX])
                if n == 'x':
                    ds.dims[i].attach_scale(h5main['ix'])
                if n == 'y':
                    ds.dims[i].attach_scale(h5main['iy'])

        with h5py.File(hdf_filenames[0]) as h5plane:

            for ds_name in dataset_properties.keys():
                for ak, av in h5plane[ds_name].attrs.items():
                    if not ak.isupper():
                        h5main[ds_name].attrs[ak] = av

            # if time is not None:
            #     # time has been set already. now put all plane times in a separate group
            #     plane_times_group = h5main.create_group(PLANE_TIME_GROUP_NAME)

            for n in ('x', 'y', 'ix', 'iy', 'z', 'reltime'):
                for ak, av in h5plane[n].attrs.items():
                    if not ak.isupper():
                        h5main[n].attrs[ak] = av

        z_before_t = zidx < it

        for iplane, plane_hdf_filename in enumerate(hdf_filenames):
            with h5py.File(plane_hdf_filename) as h5plane:
                # if time is not None:
                # ds_plane_time = plane_times_group.create_dataset(f'planetime_{iplane:0{len(str(nt))}d}',
                #                                                  data=h5plane['reltime'][:],
                #                                                  dtype=config['dtypes'].get('reltime', None))
                # for ak, av in h5plane['reltime'].attrs.items():
                #     if not ak.isupper():
                #         ds_plane_time.attrs[ak] = av

                h5main['z'][iplane] = h5plane['z'][()]
                for k in dataset_properties.keys():
                    ds = h5main[k]
                    if z_before_t:
                        ds[iplane, :] = h5plane[k][0:nt, ...]
                    else:
                        for _it in range(nt):
                            ds[_it, iplane, ...] = h5plane[k][_it, :, :]
                # write piv parameter group
                plane_grp = h5main.create_group(f'plane{iplane:0{len(str(nz))}}')
                if PIV_PARAMETER_GRP_NAME not in plane_grp:
                    trg_grp = plane_grp.create_group(PIV_PARAMETER_GRP_NAME)
                else:
                    trg_grp = plane_grp[PIV_PARAMETER_GRP_NAME]
                copy_piv_parameter_group(h5plane[PIV_PARAMETER_GRP_NAME], trg_grp)

        return h5main

    @staticmethod
    def _merge_planes_unequal_time_vectors(h5main: h5py.File, hdf_filenames: List[pathlib.Path]) -> None:
        """merges multiple hdf files that have different time vectors lengths or entries"""
        nt_list = []
        x_data = []
        y_data = []
        z_data = []
        nz = len(hdf_filenames)
        dim_names = []
        for hdf_file in hdf_filenames:
            with h5py.File(hdf_file) as h5plane:
                nt_list.append(h5plane['reltime'].size)
                x_data.append(h5plane['x'][:])
                y_data.append(h5plane['y'][:])
                z_data.append(h5plane['z'][()])
                dim_names.append([d[0].name for d in h5plane['u'].dims])
        # different time steps --> put data in individual groups
        plane_grps = []

        h5main.attrs['title'] = DEFAULT_MPLANE_TITLE
        for iz, hdf_file in tqdm(enumerate(hdf_filenames), total=nz, desc='Copying groups to target file'):
            plane_grp = h5main.create_group(f'plane{iz:0{len(str(nz))}}')
            plane_grps.append(plane_grp)
            with h5py.File(hdf_file) as h5plane:
                for objname in h5plane:
                    if objname != PIV_PARAMETER_GRP_NAME:  # treat separately
                        h5main.copy(h5plane[objname], plane_grp)
                        # delete dimension scale attrs:
                        for ak in ('DIMENSION_LIST', 'REFERENCE_LIST', 'CLASS', 'NAME'):
                            try:
                                del plane_grp[objname].attrs[ak]
                            except KeyError:
                                pass
                    else:
                        copy_piv_parameter_group(h5plane[PIV_PARAMETER_GRP_NAME],
                                                 plane_grp.create_group(PIV_PARAMETER_GRP_NAME))

        for varkey in ('x', 'y', 'ix', 'iy'):
            h5main.move(plane_grps[0][varkey].name, varkey)
            h5main[varkey].make_scale()
            for plane_grp in plane_grps[1:]:
                del plane_grp[varkey]

        for plane_grp in plane_grps[1:]:
            _time_ds = plane_grp['reltime']
            for ak in ('DIMENSION_LIST', 'REFERENCE_LIST', 'CLASS', 'NAME'):
                try:
                    del _time_ds.attrs[ak]
                except KeyError:
                    pass
            _time_ds.make_scale()

        piv_variables = []
        for plane_grp in plane_grps[:]:
            for k, v in plane_grp.items():
                if isinstance(v, h5py.Dataset):
                    if v.ndim > 1:
                        piv_variables.append(v)
                        for i, d in enumerate(dim_names[0]):
                            if 'reltime' in d and 'reltime' in plane_grp:
                                # time is stored in each group
                                v.dims[i].attach_scale(plane_grp['reltime'])
                            else:
                                # coordinates one group above (in root)
                                v.dims[i].attach_scale(h5main[d])

        # sanity check:
        for piv_variable in piv_variables:
            assert piv_variable.dims[0][0] == piv_variable.parent['reltime']
            assert piv_variable.dims[1][0] == h5main['y']
            assert piv_variable.dims[2][0] == h5main['x']

    @staticmethod
    def from_folders(plane_directories: Union[List[str], List[pathlib.Path]],
                     time_infos: List[Union[Tuple[datetime.datetime, float], List[datetime.datetime]]],
                     pivfile: "PIVFile",
                     n: int = -1) -> "PIVMultiPlane":
        """init PIVPlane from multiple folders

        Parameter
        ----------
        plane_directories : list of str or pathlib.Path
            list of directories containing the plane data
        time_infos : List[Union[Tuple[datetime.datetime, float], List[datetime.datetime]]]+
            List of time_info. See PIVPlane docstring for more information about time_info
        pivfile : PIVFile
            PIVFile object
        n : int, default=-1
            number of snapshots to load. If n=-1, all snapshots are loaded
        """
        if len(time_infos) != len(plane_directories):
            raise ValueError("Number of planes don't match the number of recording (time) information")
        plane_objs = [PIVPlane.from_folder(pathlib.Path(d), time_info=time_info, pivfile=pivfile, n=n) for d, time_info
                      in
                      zip(plane_directories, time_infos)]
        return PIVMultiPlane(plane_objs)

    def __to_hdf__(self,
                   hdf_filename: Union[str, pathlib.Path] = None,
                   rtol: float = 1.e-5,
                   atol: float = 1.e-8,
                   trim_time_vectors: bool = True,
                   z: Union[List[float], np.ndarray] = None,
                   **kwargs) -> pathlib.Path:
        """converts the snapshot into an HDF file"""
        title = kwargs.pop('title', DEFAULT_MPLANE_TITLE)

        # if the time vectors are known, check if planes can be merged!
        t_list = [plane.rel_time_vector for plane in self.list_of_piv_planes]
        nt_list = [len(t) for t in t_list]
        equal_length = np.all([nt_list[0] == nt for nt in nt_list[1:]])
        # if not equal_length:
        #     # cannot merge into one dataset. put each plane into a separate group or trim the vectors to the common minimal length
        if not equal_length and not trim_time_vectors:
            raise ValueError(
                'Time vectors are not equal in length! Stopping here because "trim_time_vectors" is False. '
                'Set "trim_time_vectors" to True to trim the time vectors to the shortest one.'
            )

        # check if time vectors are close:
        nt_min = min(nt_list)

        # compute time deltas:
        dt_list = [np.diff(t[:nt_min]) for t in t_list]

        if atol is not None:
            abs_diff = [np.abs(dt_list[0] - dt) for dt in dt_list[1:]]
            atol_result = np.all([a < atol for a in abs_diff])
        else:
            atol_result = True
        if rtol is not None:
            rel_diff = [np.abs((dt_list[0] - dt) / dt_list[0]) for dt in dt_list[1:]]
            rtol_result = np.all([r < rtol for r in rel_diff])
        else:
            rtol_result = True

        if not rtol_result or not atol_result:
            raise ValueError('Time vectors are not close enough to be merged! Process each plane separately.')

        if hdf_filename is None:
            snapshot0_filename = self.list_of_piv_planes[0].list_of_piv_files[0].filename
            name = f'{snapshot0_filename.parent.parent}.hdf'
            hdf_filename = snapshot0_filename.parent.parent / name

        # get data from first snapshot to prepare the HDF5 file
        nplanes = len(self.list_of_piv_planes)
        if z is None:
            z = [None, ] * nplanes
        plane_hdf_files = [plane.__to_hdf__(generate_temporary_filename(suffix='_plane.hdf'),
                                            z=_z,
                                            iplane=iplane + 1,
                                            nplanes=nplanes) for (iplane, plane), _z
                           in zip(enumerate(self.list_of_piv_planes), z)]
        hdf_filename = self.merge_planes(hdf_filenames=plane_hdf_files,
                                         target_hdf_filename=hdf_filename,
                                         title=title)
        print('Done.')
        return hdf_filename

    def to_hdf(self,
               piv_attributes: Dict,  # e.g. piv_medium, contact
               hdf_filename: pathlib.Path = None,
               z: float = None,
               atol: float = 1e-6,
               rtol: float = 1e-3) -> pathlib.Path:
        return super().to_hdf(piv_attributes=piv_attributes,
                              hdf_filename=hdf_filename,
                              z=z,
                              atol=atol,
                              rtol=rtol)
