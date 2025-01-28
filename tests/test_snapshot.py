"""tests for snapshot conversion"""
import datetime
import json
import logging
import pathlib
import unittest

import h5rdmtoolbox as h5tbx
import numpy as np

import piv2hdf
from piv2hdf import openpiv, pivview, tutorial, PIVSnapshot
from piv2hdf import set_loglevel, logger
from piv2hdf.interface import PIVFile
from piv2hdf.openpiv.user_operations import add_standard_name_operation as openpiv_add_standard_name_operation
from piv2hdf.pivview.user_operations import add_standard_name_operation as pivview_add_standard_name_operation

loglevel = logging.ERROR

set_loglevel(loglevel)
assert logger.level == loglevel
for handler in logger.handlers:
    assert handler.level == loglevel

__this_dir__ = pathlib.Path(__file__).parent


def read_meta():
    with open(__this_dir__ / "resources/meta.json") as f:
        meta = json.load(f)
    return meta


try:
    import lvpyio as lv

    lvpyio_installed = True
except ImportError:
    lvpyio_installed = False


class TestSnapshot(unittest.TestCase):
    if lvpyio_installed:
        def test_davis_snapshot(self):
            from piv2hdf.davis import VC7File
            from piv2hdf.davis.parameter import DavisParameterFile
            vc7_filename = tutorial.Davis.get_vc7_files()[1]
            self.assertTrue(vc7_filename.exists())
            param = DavisParameterFile(vc7_filename)
            vc7file = VC7File(vc7_filename, parameter=param)
            self.assertIsInstance(vc7file, PIVFile)
            self.assertEqual(vc7file.filename, vc7_filename)
            snapshot = PIVSnapshot(
                piv_file=vc7file,
                recording_dtime=datetime.datetime(2023, 1, 15, 13, 42, 2, 3)
            )
            snapshot.to_hdf()

    def test_pivview_snapshot(self):
        meta = read_meta()

        pivview_nc_file = tutorial.PIVview.get_snapshot_nc_files()[0]
        pivfile = pivview.PIVViewNcFile(
            filename=pivview_nc_file,
            parameter_filename=None,
            user_defined_hdf5_operations=pivview_add_standard_name_operation
        )
        snapshot = PIVSnapshot(piv_file=pivfile,
                               recording_dtime=datetime.datetime(2023, 1, 15, 13, 42, 2, 3))
        self.assertIsInstance(pivfile, PIVFile)

        piv2hdf.reset_pivattrs()

        # with self.assertRaises(h5tbx.errors.StandardAttributeError):
        #     snapshot.to_hdf()

        with self.assertRaises(TypeError):
            PIVSnapshot(piv_file=pivfile, recording_dtime=[0., 1.])

        snapshot = PIVSnapshot(piv_file=pivfile,
                               recording_dtime=datetime.datetime(2023, 1, 15, 13, 42, 2, 3))
        # with self.assertRaises(h5tbx.errors.StandardAttributeError):
        #     _ = snapshot.to_hdf(piv_attributes={'piv_medium': meta["PIV_MEDIUM"]})

        hdf_filename = snapshot.to_hdf(piv_attributes={'creator': meta["CREATOR"],
                                                       'piv_medium': meta["PIV_MEDIUM"],
                                                       'camera': meta["CAMERA"]})
        self.assertEqual(hdf_filename, snapshot.hdf_filename)
        nc_data = snapshot.piv_file.read(0).data

        with h5tbx.File(hdf_filename) as h5:
            self.assertEqual(h5.attrs['creator'], meta["CREATOR"])
            self.assertEqual(h5.attrs['piv_medium'], meta["PIV_MEDIUM"])
            self.assertEqual(h5.attrs['camera'], meta["CAMERA"])
            for k in nc_data.keys():
                if k not in ('z', 'reltime'):
                    self.assertIn(k, h5)
                    self.assertEqual(h5[k].shape, nc_data[k].shape)
                    self.assertEqual(h5[k].dtype, nc_data[k].dtype)
                    np.testing.assert_array_equal(h5[k][()], nc_data[k][()])
            self.assertIn('time', h5)
            self.assertIn('piv_medium', h5.attrs.raw)
            self.assertIn('piv_method', h5.piv_parameters.attrs.raw)
            self.assertIn('piv_peak_method', h5.piv_parameters.attrs.raw)
            self.assertIn('creator', h5.attrs.raw)
            self.assertIn('software', h5.attrs)
            # self.assertIn('recording_datetime', h5)  # datetime of plane (or in this case snapshot)
            self.assertEqual(h5['x'].size, 31)
            self.assertEqual(h5['y'].size, 15)
            self.assertEqual(h5['z'].size, 1)
            self.assertEqual(str(h5['u'][()].time.values, encoding='utf-8'),
                             datetime.datetime(2023, 1, 15, 13, 42, 2, 3).isoformat())
            self.assertIn('time', h5['u'][()].coords)
            self.assertEqual(h5['u'].shape, (15, 31))
            self.assertEqual(h5['v'].shape, (15, 31))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flags?'}}))
            self.assertEqual('uint8',
                             h5.find_one({'standard_name': {'$regex': 'piv_flags?'}}).dtype)

    def test_openpiv_snapshot(self):
        # get the test file:
        meta = read_meta()
        openpiv_txt_file = tutorial.OpenPIV.get_snapshot_txt_file()

        # init a openpiv-file instance:
        # None -> auto search
        openpiv_file = openpiv.OpenPIVFile(
            openpiv_txt_file,
            user_defined_hdf5_operations=openpiv_add_standard_name_operation,
            parameter_filename=None)

        openpiv_snapshot = PIVSnapshot(openpiv_file, recording_dtime=None)

        hdf_filename = openpiv_snapshot.to_hdf(piv_attributes={'creator': meta["CREATOR"],
                                                               'piv_medium': meta["PIV_MEDIUM"]})

        with h5tbx.File(hdf_filename) as h5:
            self.assertIn('piv_medium', h5.attrs.raw)
            # self.assertIn('piv_method', h5.piv_parameters.attrs.raw)
            self.assertIn('piv_peak_method', h5.piv_parameters.attrs.raw)
            self.assertIn('software', h5.attrs)
            self.assertIn('reltime', h5)
            self.assertEqual(h5['x'].size, 63)
            self.assertEqual(h5['y'].size, 59)
            self.assertEqual(h5['u'].shape, (59, 63))
            self.assertEqual(h5['v'].shape, (59, 63))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype, 'uint8')
