import datetime
import pathlib
import unittest

import h5py
import h5rdmtoolbox as h5tbx
import numpy as np

import piv2hdf
from piv2hdf import openpiv, pivview, tutorial
from piv2hdf.config import get_config, set_config
from piv2hdf.openpiv.user_operations import add_standard_name_operation as openpiv_add_standard_name_operation
from piv2hdf.pivview.user_operations import add_standard_name_operation as pivview_add_standard_name_operation
from piv2hdf.time import TimeVectorWarning

__this_dir__ = pathlib.Path(__file__).parent
CREATOR = {
    'name': 'Matthias Probst',
    'id': 'https://orcid.org/0000-0001-8729-0482',
    'role': 'Researcher'
}

try:
    import lvpyio as lv

    lvpyio_installed = True
except ImportError:
    lvpyio_installed = False


def validate_planar_plane_pivview(test, hdf_filename, nc_data, plane):
    """Test the hdf5 file created by pivview"""
    with h5tbx.File(hdf_filename) as h5:

        nt = h5['u'].shape[0]
        for k in ('time', 'reltime', 'image_index'):
            test.assertEqual(nt, h5[k].shape[0])

        test.assertEqual(np.datetime64(datetime.datetime(2023, 10, 1, 12)), h5.time[0].values)
        for i in range(1, nt):
            test.assertEqual(np.datetime64(datetime.datetime(2023, 10, 1, 12) + datetime.timedelta(seconds=i / 10)),
                             h5.time[i].values)

        for k in nc_data.keys():
            test.assertIn(k, h5)

        for i in range(h5['u'].shape[0]):
            nc_data = plane.list_of_piv_files[i].read(0).data
            for k in nc_data.keys():
                if k not in ('z', 'reltime', 'x', 'y', 'ix', 'iy', 'time', 'image_index'):
                    np.testing.assert_array_equal(h5[k][i, ...], nc_data[k][()])

        for k, v in h5.items():
            if isinstance(v, h5py.Dataset) and v.ndim == 3:
                test.assertEqual('/reltime', v.dims[0][0].name)
                test.assertEqual('/time', v.dims[0][1].name)
                test.assertEqual('/image_index', v.dims[0][2].name)
                test.assertEqual(1, v.dims[0][2][0])
                test.assertEqual('/y', v.dims[1][0].name)
                test.assertEqual('/iy', v.dims[1][1].name)
                test.assertEqual('/x', v.dims[2][0].name)
                test.assertEqual('/ix', v.dims[2][1].name)

        test.assertIn('software', h5.attrs)
        test.assertEqual(h5['x'].size, 31)
        test.assertEqual(h5['y'].size, 15)
        test.assertEqual(h5['z'].size, 1)
        test.assertAlmostEqual(h5['z'].values[()], 0.51, 6)
        test.assertEqual(h5['u'].shape, (8, 15, 31))
        test.assertEqual(h5['v'].shape, (8, 15, 31))
        test.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
        test.assertEqual('uint8', h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype)

        # test.assertEqual(h5['dudx'].attrs['long_name'], 'm/s')


class TestPlane(unittest.TestCase):

    def test_singleplane_stereo_pivview(self):
        from piv2hdf.pivview import PIVViewStereoNcFile

        stereo_plane_dir = __this_dir__ / "resources/pivview/stereo/plane_01"
        if stereo_plane_dir.exists():
            PIVViewStereoNcFile.__parameter_cls__(stereo_plane_dir / 'plane.cfg')

            plane_stereo = piv2hdf.PIVPlane.from_folder(
                plane_directory=stereo_plane_dir,
                time_info=(datetime.datetime(2023, 10, 1, 12, 0, 0), 10),
                pivfile=PIVViewStereoNcFile,
                user_defined_hdf5_operations=pivview_add_standard_name_operation,
                parameter=stereo_plane_dir / "plane.cfg"
            )

            spiv_filename = plane_stereo.to_hdf(
                piv_attributes={'creator': CREATOR, 'piv_medium': 'air'})

            with h5tbx.File(spiv_filename) as h5:
                self.assertTrue('flag_meaning' in h5['piv_flags'].attrs)
                for gradient_name in ('dudx', 'dudy', 'dvdx', 'dvdy', 'dwdx', 'dwdy'):
                    self.assertTrue(gradient_name in h5)
                    self.assertEqual(f'velocity gradient {gradient_name[0:2]}/{gradient_name[2:]}',
                                     h5[gradient_name].attrs['long_name'])
                    self.assertEqual('1/s', h5[gradient_name].attrs['units'])
                    self.assertTrue('flag_meaning' not in h5[gradient_name].attrs)  # checking issue #4

    def test_singleplane_pivivew(self):
        set_config(postproc=[])
        self.assertEqual(get_config()['postproc'], [])

        plane_dir = tutorial.PIVview.get_plane_directory()
        plane = piv2hdf.PIVPlane.from_folder(plane_directory=plane_dir,
                                             time_info=(datetime.datetime(2023, 10, 1, 12), 10),  # dtime and frequency
                                             pivfile=pivview.PIVViewNcFile,
                                             user_defined_hdf5_operations=pivview_add_standard_name_operation,
                                             prefix_pattern='*[0-9]')
        hdf_filename = plane.to_hdf(piv_attributes=dict(piv_medium='air',
                                                        creator=CREATOR),
                                    z=0.51)
        pivdata = plane.list_of_piv_files[0].read(0)
        validate_planar_plane_pivview(self, hdf_filename, pivdata.data, plane)
        with h5tbx.File(hdf_filename) as h5:
            self.assertTrue('mean_u' not in h5)

        set_config(postproc=['compute_time_averages'])
        self.assertEqual(get_config()['postproc'], ['compute_time_averages'])
        hdf_filename = plane.to_hdf(piv_attributes=dict(piv_medium='air',
                                                        creator=CREATOR),
                                    z=0.51)
        with h5tbx.File(hdf_filename) as h5:
            self.assertTrue('mean_u' in h5)

        with self.assertRaises(ValueError):
            set_config(postproc=['invalid'])

        time_vector = [datetime.datetime(2023, 10, 1, 12) + datetime.timedelta(seconds=i / 10) for i in range(100)]
        # now have a longer time vector:
        plane = piv2hdf.PIVPlane.from_folder(plane_directory=plane_dir,
                                             time_info=time_vector[0:len(plane.list_of_piv_files)],
                                             pivfile=pivview.PIVViewNcFile,
                                             user_defined_hdf5_operations=pivview_add_standard_name_operation,
                                             prefix_pattern='*[0-9]')
        hdf_filename = plane.to_hdf(piv_attributes=dict(piv_medium='air',
                                                        creator=CREATOR),
                                    z=0.51)
        validate_planar_plane_pivview(self, hdf_filename, pivdata.data, plane)

        # now have a longer time vector that is too long!
        with self.assertWarns(TimeVectorWarning):
            plane = piv2hdf.PIVPlane.from_folder(plane_directory=plane_dir,
                                                 time_info=time_vector[:],
                                                 pivfile=pivview.PIVViewNcFile,
                                                 user_defined_hdf5_operations=pivview_add_standard_name_operation,
                                                 prefix_pattern='*[0-9]')
        hdf_filename = plane.to_hdf(piv_attributes=dict(piv_medium='air',
                                                        creator=CREATOR),
                                    z=0.51)
        validate_planar_plane_pivview(self, hdf_filename, pivdata.data, plane)

    if lvpyio_installed:
        def test_singleplane_davis(self):
            from piv2hdf.davis import VC7File
            from piv2hdf.davis.parameter import DavisParameterFile

            vc7_filenames = tutorial.Davis.get_vc7_files()
            param = DavisParameterFile(vc7_filenames[0])
            plane_dir = vc7_filenames[0].parent
            plane = piv2hdf.PIVPlane.from_folder(
                plane_directory=plane_dir,
                time_info=(datetime.datetime(2023, 10, 1, 12), 10),  # dtime and frequency
                pivfile=VC7File,
                parameter=param,
                # user_defined_hdf5_operations=pivview_add_standard_name_operation,
                prefix_pattern='*[0-9]')
            hdf_filename = plane.to_hdf(piv_attributes=dict(piv_medium='air',
                                                            creator=CREATOR),
                                        z=0.51)

    def test_singleplane_pivivew_explicit_parameter(self):
        # explicitly pass parameter
        plane_dir = tutorial.PIVview.get_plane_directory()
        par_filename = plane_dir / 'piv_parameters.par'
        par = pivview.PIVviewParamFile(par_filename)
        plane_dir = tutorial.PIVview.get_plane_directory()
        plane = piv2hdf.PIVPlane.from_folder(plane_dir,
                                             time_info=(datetime.datetime(2023, 10, 1, 12, 0, 0), 10),
                                             pivfile=pivview.PIVViewNcFile,
                                             user_defined_hdf5_operations=pivview_add_standard_name_operation,
                                             parameter=par)

        hdf_filename = plane.to_hdf(piv_attributes=dict(piv_medium='air',
                                                        creator=CREATOR),
                                    z=0.51)
        with h5tbx.File(hdf_filename) as h5:
            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 3:
                    self.assertEqual('/reltime', v.dims[0][0].name)
                    self.assertEqual('/time', v.dims[0][1].name)
                    self.assertEqual('/image_index', v.dims[0][2].name)
                    self.assertEqual(1, v.dims[0][2][0])
                    self.assertEqual('/y', v.dims[1][0].name)
                    self.assertEqual('/iy', v.dims[1][1].name)
                    self.assertEqual('/x', v.dims[2][0].name)
                    self.assertEqual('/ix', v.dims[2][1].name)
            self.assertIn('software', h5.attrs)
            self.assertEqual(h5['ix'].dtype, np.uint8)
            self.assertEqual(h5['iy'].dtype, np.uint8)
            self.assertEqual(h5['x'].size, 31)
            self.assertEqual(h5['x'].dtype, get_config()['dtypes']['x'])
            self.assertEqual(h5['y'].size, 15)
            self.assertEqual(h5['z'].size, 1)
            self.assertAlmostEqual(h5['z'].values[()], 0.51, 6)
            self.assertEqual(h5['u'].shape, (8, 15, 31))
            self.assertEqual(h5['v'].shape, (8, 15, 31))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype, 'uint8')

        plane_dir = tutorial.PIVview.get_plane_directory()
        par_filename = plane_dir / 'piv_parameters.par'
        par = pivview.PIVviewParamFile(par_filename)
        plane_dir = tutorial.PIVview.get_plane_directory()
        plane = piv2hdf.PIVPlane.from_folder(plane_dir,
                                             time_info=(datetime.datetime(2023, 10, 1, 12, 0, 0), 10),
                                             pivfile=pivview.PIVViewNcFile,
                                             user_defined_hdf5_operations=pivview_add_standard_name_operation,
                                             parameter=par)

        hdf_filename = plane.to_hdf(
            piv_attributes=dict(piv_medium='air',
                                creator=CREATOR),
            z=0.51)
        self.assertEqual(hdf_filename, plane.hdf_filename)

        with h5tbx.File(hdf_filename) as h5:

            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 3:
                    self.assertEqual('/reltime', v.dims[0][0].name)
                    self.assertEqual('/time', v.dims[0][1].name)
                    self.assertEqual('/image_index', v.dims[0][2].name)
                    self.assertEqual(1, v.dims[0][2][0])
                    self.assertEqual('/y', v.dims[1][0].name)
                    self.assertEqual('/iy', v.dims[1][1].name)
                    self.assertEqual('/x', v.dims[2][0].name)
                    self.assertEqual('/ix', v.dims[2][1].name)

            self.assertEqual(get_config()['dtypes']['x'], 'float32')
            self.assertEqual(h5['x'].dtype.type, np.float32)
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype, 'uint8')

    def test_singleplane_openpiv(self):
        plane_dir = tutorial.OpenPIV.get_plane_directory()
        plane = piv2hdf.PIVPlane.from_folder(plane_dir,
                                             time_info=(datetime.datetime(2023, 10, 1, 12, 0, 0), 10),
                                             pivfile=openpiv.OpenPIVFile,
                                             user_defined_hdf5_operations=openpiv_add_standard_name_operation,
                                             )
        hdf_filename = plane.to_hdf(
            piv_attributes=dict(piv_medium='air', creator=CREATOR), z=0.51)
        with h5tbx.File(hdf_filename) as h5:

            for k, v in h5.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 3:
                    self.assertEqual('/reltime', v.dims[0][0].name)
                    self.assertEqual('/time', v.dims[0][1].name)
                    self.assertEqual('/image_index', v.dims[0][2].name)
                    self.assertEqual(1, v.dims[0][2][0])
                    self.assertEqual('/y', v.dims[1][0].name)
                    self.assertEqual('/iy', v.dims[1][1].name)
                    self.assertEqual('/x', v.dims[2][0].name)
                    self.assertEqual('/ix', v.dims[2][1].name)

            self.assertIn('software', h5.attrs)
            self.assertEqual(h5['x'].size, 63)
            self.assertEqual(h5['y'].size, 59)
            self.assertEqual(h5['z'].size, 1)
            self.assertAlmostEqual(h5['z'].values[()], 0.51, 6)
            for k, ds in h5.items():
                if isinstance(ds, h5py.Dataset):
                    if k not in ('ix', 'iy', 'x', 'y', 'z', 'reltime', 'time', 'image_index'):
                        self.assertEqual(ds.shape, (4, 59, 63))
            self.assertIsNotNone(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}))
            self.assertEqual(h5.find_one({'standard_name': {'$regex': 'piv_flag'}}).dtype, 'uint8')
