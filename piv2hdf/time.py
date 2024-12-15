import datetime
from typing import Union, List

import h5py
import h5rdmtoolbox as h5tbx
import numpy as np


class TimeVectorWarning(Warning):
    pass


def create_recording_datetime_dataset(
        h5,
        recording_dtime: Union[datetime.datetime, List[datetime.datetime]],
        name: str) -> h5py.Dataset:
    """creates a recording datetime dataset in the hdf5 file"""
    if isinstance(recording_dtime, np.ndarray):
        ds_rec_dtime = h5tbx.Group(h5).create_time_dataset(
            name,
            data=list(recording_dtime),
            time_format="iso"
        )
    else:
        ds_rec_dtime = h5tbx.Group(h5).create_time_dataset(
            name,
            data=recording_dtime,
            time_format="iso"
        )
    ds_rec_dtime.attrs['comment'] = 'Recording datetime in ISO 8601 format'
    ds_rec_dtime.attrs['long_name'] = 'Datetime vector of snapshots'
    ds_rec_dtime.attrs['units'] = ''
    return ds_rec_dtime
