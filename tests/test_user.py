import datetime
import pathlib
import shutil
import unittest

from piv2hdf import pivview, tutorial
from piv2hdf.interface import PIVPlane, PIVMultiPlane

__this_dir__ = pathlib.Path(__file__).parent

from piv2hdf.pivview.user_operations import add_standard_name_operation

CREATOR = {
    'name': 'Matthias Probst',
    'id': 'https://orcid.org/0000-0001-8729-0482',
    'role': 'Researcher'}


class TestUser(unittest.TestCase):

    def test_tmpdir_change(self):
        from piv2hdf import user
        default_user_tmpdir = user.TEMPORARY_USER_DIRECTORY
        plane_dirs = tutorial.PIVview.get_multiplane_directories()[0:2]

        # init from folder:
        plane_objs = [PIVPlane.from_folder(
            d,
            time_info=(datetime.datetime.now(), 5),
            pivfile=pivview.PIVViewNcFile,
            user_defined_hdf5_operations=add_standard_name_operation) for d in
            plane_dirs]
        mplane = PIVMultiPlane(plane_objs)
        hdf_filename = mplane.to_hdf(piv_attributes=dict(piv_medium='air', creator=CREATOR))
        self.assertTrue(default_user_tmpdir in hdf_filename.parents)

        new_tmp = __this_dir__ / 'tmp'
        user.TEMPORARY_USER_DIRECTORY = new_tmp
        plane_dirs = tutorial.PIVview.get_multiplane_directories()[0:2]

        # init from folder:
        plane_objs = [PIVPlane.from_folder(d,
                                           time_info=(datetime.datetime.now(), 5),
                                           pivfile=pivview.PIVViewNcFile,
                                           user_defined_hdf5_operations=add_standard_name_operation) for d in
                      plane_dirs]
        mplane = PIVMultiPlane(plane_objs)
        hdf_filename = mplane.to_hdf(piv_attributes=dict(piv_medium='air',
                                                         creator=CREATOR))
        self.assertFalse(default_user_tmpdir in hdf_filename.parents)
        self.assertTrue(new_tmp in hdf_filename.parents)

    def tearDown(self):
        new_tmp = __this_dir__ / 'tmp'
        shutil.rmtree(new_tmp, ignore_errors=True)
