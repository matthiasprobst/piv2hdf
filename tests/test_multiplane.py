import datetime
import json
import pathlib
import unittest

import h5py
import h5rdmtoolbox as h5tbx
import numpy as np

from piv2hdf import pivview, openpiv, tutorial
from piv2hdf.interface import PIVPlane, PIVMultiPlane

NOW = datetime.datetime.now()

__this_dir__ = pathlib.Path(__file__).parent


def read_meta():
    with open(__this_dir__ / "resources/meta.json") as f:
        meta = json.load(f)
    return meta


class TestMPlane(unittest.TestCase):

    def setUp(self):
        self.meta = read_meta()

    def test_multiplane_different_pivfile(self):
        plane_dirs_pivview = tutorial.PIVview.get_multiplane_directories()[0:2]
        plane_dirs_openpiv = tutorial.OpenPIV.get_multiplane_directories()[0:2]

        # pass wrong folders which will raise an error in the process:
        with self.assertRaises(ValueError):
            _ = PIVPlane.from_folder(plane_dirs_pivview[0], (NOW, 5), openpiv.OpenPIVFile)
        with self.assertRaises(ValueError):
            _ = PIVPlane.from_folder(plane_dirs_openpiv[1], (NOW, 5), pivview.PIVViewNcFile)

        plane0 = PIVPlane.from_folder(plane_dirs_openpiv[0], (NOW, 5), openpiv.OpenPIVFile)
        plane1 = PIVPlane.from_folder(plane_dirs_pivview[1], (NOW, 5), pivview.PIVViewNcFile)
        with self.assertWarns(UserWarning):
            _ = PIVMultiPlane([plane0, plane1])

    def test_multi_piv_identical_nt_pivview(self):
        plane_dirs = tutorial.PIVview.get_multiplane_directories()[0:2]

        # init from folder:
        dt_start1 = datetime.datetime(2018, 1, 1)
        dt_start2 = datetime.datetime(2018, 1, 1) + datetime.timedelta(minutes=10)
        dt_start3 = datetime.datetime(2018, 1, 1) + datetime.timedelta(minutes=20)

        time_vector2 = [dt_start2 + datetime.timedelta(seconds=i / 10) for i in range(100)]
        time_infos = [(dt_start1, 10), time_vector2, (dt_start3, 10)]

        plane_objs = [PIVPlane.from_folder(d, pivfile=pivview.PIVViewNcFile,
                                           time_info=time_info) for d, time_info in
                      zip(plane_dirs, time_infos)]
        mplane = PIVMultiPlane(plane_objs)
        hdf_filename = mplane.to_hdf(
            piv_attributes=dict(piv_medium=self.meta["PIV_MEDIUM"], creator=self.meta["CREATOR"]))
        with h5tbx.File(hdf_filename) as h5:

            nt = h5['u'].shape[1]
            for k in ('reltime', 'image_index'):
                self.assertEqual(nt, h5[k].shape[0])

            self.assertEqual(nt, h5['time'].shape[1])

            self.assertEqual(np.datetime64(dt_start1), h5.time[0, 0].values)
            self.assertEqual(np.datetime64(dt_start2), h5.time[1, 0].values)
            for i in range(1, nt):
                self.assertEqual(np.datetime64(dt_start1 + datetime.timedelta(seconds=i / 10)),
                                 h5.time[0, i].values)
                self.assertEqual(np.datetime64(dt_start2 + datetime.timedelta(seconds=i / 10)),
                                 h5.time[1, i].values)

            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    self.assertEqual('/z', v.dims[0][0].name)
                    # self.assertEqual('/iz', v.dims[0][1].name)
                    self.assertEqual('/reltime', v.dims[1][0].name)
                    # self.assertEqual('/time', v.dims[1][1].name)
                    self.assertEqual('/image_index', v.dims[1][1].name)
                    self.assertEqual('/y', v.dims[2][0].name)
                    self.assertEqual('/iy', v.dims[2][1].name)
                    self.assertEqual('/x', v.dims[3][0].name)
                    self.assertEqual('/ix', v.dims[3][1].name)

            self.assertIn('software', h5.attrs)
            self.assertEqual(h5['x'].size, 31)
            self.assertEqual(h5['y'].size, 15)
            self.assertEqual(h5['z'].size, 2)
            self.assertEqual(list(h5['z'][()]), [-5., 0.])
            self.assertEqual(h5['u'].shape, (2, 3, 15, 31))
            self.assertEqual(h5['v'].shape, (2, 3, 15, 31))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual('uint8', h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype)
            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    assert v.dims[0][0] == h5['z']
                    assert v.dims[1][0] == h5['reltime']
                    assert v.dims[2][0] == h5['y']
                    assert v.dims[3][0] == h5['x']

    def test_multi_piv_similar_nt_pivview(self):
        plane_dirs = tutorial.PIVview.get_multiplane_directories()[0:2]

        # init from folder:
        dt_start1 = datetime.datetime(2018, 1, 1)
        dt_start2 = datetime.datetime(2018, 1, 1) + datetime.timedelta(minutes=10)
        dt_start3 = datetime.datetime(2018, 1, 1) + datetime.timedelta(minutes=20)
        time_infos = [(dt_start1, 10), (dt_start2, 10), (dt_start3, 10)]

        plane_objs = [PIVPlane.from_folder(d,
                                           pivfile=pivview.PIVViewNcFile,
                                           time_info=time_info) for d, time_info in
                      zip(plane_dirs, time_infos)]
        mplane = PIVMultiPlane(plane_objs)
        hdf_filename = mplane.to_hdf(
            piv_attributes=dict(piv_medium=self.meta["PIV_MEDIUM"], creator=self.meta["CREATOR"]),
            atol=1e-3, rtol=1e-3)
        self.assertEqual(hdf_filename, mplane.hdf_filename)
        with h5tbx.File(hdf_filename) as h5:
            self.assertIn('software', h5.attrs)
            # self.assertTrue(
            #     np.array_equal(h5['reltime'][()], np.asarray(rtfs).mean(axis=0).astype(h5['reltime'].dtype)))

            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    self.assertEqual('/z', v.dims[0][0].name)
                    # self.assertEqual('/iz', v.dims[0][1].name)
                    self.assertEqual('/reltime', v.dims[1][0].name)
                    # self.assertEqual('/time', v.dims[1][1].name)
                    self.assertEqual('/image_index', v.dims[1][1].name)
                    self.assertEqual(1, v.dims[1][1][0])
                    self.assertEqual('/y', v.dims[2][0].name)
                    self.assertEqual('/iy', v.dims[2][1].name)
                    self.assertEqual('/x', v.dims[3][0].name)
                    self.assertEqual('/ix', v.dims[3][1].name)

            self.assertEqual(h5['x'].size, 31)
            self.assertEqual(h5['y'].size, 15)
            self.assertEqual(h5['z'].size, 2)
            self.assertEqual(list(h5['z'][()]), [-5., 0.])
            self.assertEqual(h5['u'].shape, (2, 3, 15, 31))
            self.assertEqual(h5['v'].shape, (2, 3, 15, 31))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual('uint8', h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype)
            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    assert v.dims[0][0] == h5['z']
                    assert v.dims[1][0] == h5['reltime']
                    assert v.dims[2][0] == h5['y']
                    assert v.dims[3][0] == h5['x']

    def test_multi_piv_similar_time_but_different_nt_pivview(self):
        plane_dirs = tutorial.PIVview.get_multiplane_directories()[0:3]

        # init from folder:
        dt_start1 = datetime.datetime(2018, 1, 1)
        dt_start2 = datetime.datetime(2018, 1, 1, 1, 0, 0)
        dt_start3 = datetime.datetime(2018, 1, 1, 2, 0, 0)
        time_infos = [(dt_start1,
                       dt_start1 + datetime.timedelta(seconds=0.1),
                       dt_start1 + datetime.timedelta(seconds=0.3)),
                      (dt_start2,
                       dt_start2 + datetime.timedelta(seconds=0.1),
                       dt_start2 + datetime.timedelta(seconds=0.3)),
                      (dt_start3,
                       dt_start3 + datetime.timedelta(seconds=0.1)), ]

        plane_objs = [PIVPlane.from_folder(d,
                                           pivfile=pivview.PIVViewNcFile,
                                           time_info=time_info) for d, time_info in
                      zip(plane_dirs, time_infos)]
        mplane = PIVMultiPlane(plane_objs)
        hdf_filename = mplane.to_hdf(
            piv_attributes=dict(piv_medium=self.meta["PIV_MEDIUM"], creator=self.meta["CREATOR"]),
            atol=1e-1, rtol=1e-1)
        with h5tbx.File(hdf_filename) as h5:
            self.assertIn('software', h5.attrs)
            # self.assertTrue(np.array_equal(h5['reltime'][()],
            #                                np.asarray([t[:2] for t in rtfs]).mean(axis=0).astype(h5['reltime'].dtype)))

            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    self.assertEqual('/z', v.dims[0][0].name)
                    # self.assertEqual('/iz', v.dims[0][1].name)
                    self.assertEqual('/reltime', v.dims[1][0].name)
                    # self.assertEqual('/time', v.dims[1][1].name)
                    self.assertEqual('/image_index', v.dims[1][1].name)
                    self.assertEqual(1, v.dims[1][1][0])
                    self.assertEqual('/y', v.dims[2][0].name)
                    self.assertEqual('/iy', v.dims[2][1].name)
                    self.assertEqual('/x', v.dims[3][0].name)
                    self.assertEqual('/ix', v.dims[3][1].name)

            self.assertEqual(h5['x'].size, 31)
            self.assertEqual(h5['y'].size, 15)
            self.assertEqual(h5['z'].size, 3)
            self.assertEqual(list(h5['z'][()]), [-5., 0., 10.0])
            self.assertEqual(h5['u'].shape, (3, 2, 15, 31))
            self.assertEqual(h5['v'].shape, (3, 2, 15, 31))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual('uint8', h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype)
            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    assert v.dims[0][0] == h5['z']
                    assert v.dims[1][0] == h5['reltime']
                    assert v.dims[2][0] == h5['y']
                    assert v.dims[3][0] == h5['x']

    def test_unequal_time_vectors_pivview(self):
        plane_dirs = tutorial.PIVview.get_multiplane_directories()[0:3]

        dt_start1 = datetime.datetime(2018, 1, 1)
        dt_start2 = datetime.datetime(2018, 1, 1) + datetime.timedelta(minutes=5)
        dt_start3 = datetime.datetime(2018, 1, 1) + datetime.timedelta(minutes=20)
        rtfs = [(dt_start1, 10), (dt_start2, 8), (dt_start3, 10)]  # different freq!
        plane_objs = [PIVPlane.from_folder(d,
                                           pivfile=pivview.PIVViewNcFile,
                                           time_info=time_info) for d, time_info in
                      zip(plane_dirs, rtfs)]
        mplane = PIVMultiPlane(plane_objs)

        with self.assertRaises(ValueError):
            # time vectors are not close enough
            mplane.to_hdf(piv_attributes=dict(piv_medium=self.meta["PIV_MEDIUM"],
                                              creator=self.meta["CREATOR"]),
                          atol=1e-3, rtol=1e-3)

    def test_multi_piv_equal_nt_openpiv(self):
        plane_dirs = tutorial.OpenPIV.get_multiplane_directories()[0:2]
        plane_objs = [PIVPlane.from_folder(d, (NOW, 5), openpiv.OpenPIVFile) for d in
                      plane_dirs]
        mplane = PIVMultiPlane(plane_objs)
        hdf_filename = mplane.to_hdf(piv_attributes=dict(piv_medium=self.meta["PIV_MEDIUM"],
                                                         creator=self.meta["CREATOR"]),
                                     z=[-5., 0.])
        with h5tbx.File(hdf_filename) as h5:

            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    self.assertEqual('/z', v.dims[0][0].name)
                    # self.assertEqual('/iz', v.dims[0][1].name)
                    self.assertEqual('/reltime', v.dims[1][0].name)
                    # self.assertEqual('/time', v.dims[1][1].name)
                    self.assertEqual('/image_index', v.dims[1][1].name)
                    self.assertEqual(1, v.dims[1][1][0])
                    self.assertEqual('/y', v.dims[2][0].name)
                    self.assertEqual('/iy', v.dims[2][1].name)
                    self.assertEqual('/x', v.dims[3][0].name)
                    self.assertEqual('/ix', v.dims[3][1].name)
            self.assertIn('software', h5.attrs)
            self.assertEqual(h5['x'].size, 63)
            self.assertEqual(h5['y'].size, 59)
            self.assertEqual(h5['z'].size, 2)
            self.assertEqual(list(h5['z'][()]), [-5., 0.])
            self.assertEqual(h5['u'].shape, (2, 3, 59, 63))
            self.assertEqual(h5['v'].shape, (2, 3, 59, 63))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual('uint8', h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype)

    def test_multi_piv_unequal_nt_pivview(self):
        from piv2hdf import PIVMultiPlane
        from piv2hdf.pivview import PIVViewNcFile
        from piv2hdf import tutorial

        multi_plane_dirs = tutorial.PIVview.get_multiplane_directories()
        mplane = PIVMultiPlane.from_folders(plane_directories=multi_plane_dirs,
                                            time_infos=[(NOW, 5), (NOW, 10), (NOW, 3)],
                                            pivfile=PIVViewNcFile)
        with self.assertRaises(ValueError):
            mplane.to_hdf(
                piv_attributes=dict(piv_medium=self.meta["PIV_MEDIUM"], creator=self.meta["CREATOR"]),
            )

        multi_plane_dirs = tutorial.PIVview.get_multiplane_directories()
        mplane = PIVMultiPlane.from_folders(plane_directories=multi_plane_dirs,
                                            time_infos=[(NOW, 5), (NOW, 5.001), (NOW, 4.9999)],
                                            pivfile=PIVViewNcFile)
        hdf_filename = mplane.to_hdf(
            piv_attributes=dict(piv_medium=self.meta["PIV_MEDIUM"], creator=self.meta["CREATOR"]),
            atol=0.1, rtol=0.1
        )
        with h5tbx.File(hdf_filename) as h5:
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual('uint8', h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype)
            for plane_name in ('plane0', 'plane1', 'plane2'):
                for k, v in h5[plane_name].items():
                    if isinstance(v, h5py.Dataset) and v.ndim == 3:
                        assert v.dims[0][0] == v.parent['reltime']
                        assert v.dims[1][0] == h5['y']
                        assert v.dims[2][0] == h5['x']

            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 4:
                    self.assertEqual('/z', v.dims[0][0].name)
                    # self.assertEqual('/iz', v.dims[0][1].name)
                    self.assertEqual('/reltime', v.dims[1][0].name)
                    # self.assertEqual('/time', v.dims[1][1].name)
                    self.assertEqual('/image_index', v.dims[1][1].name)
                    self.assertEqual(1, v.dims[1][1][0])
                    self.assertEqual('/y', v.dims[2][0].name)
                    self.assertEqual('/iy', v.dims[2][1].name)
                    self.assertEqual('/x', v.dims[3][0].name)
                    self.assertEqual('/ix', v.dims[3][1].name)
