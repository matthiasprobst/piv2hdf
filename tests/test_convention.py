import pathlib
import unittest

import h5rdmtoolbox as h5tbx

from piv2hdf import cv


class TestConvention(unittest.TestCase):

    def test_snt_filename_exists(self):
        self.assertTrue(
            pathlib.Path(pathlib.Path(cv.properties[h5tbx.File]['standard_name_table'].default_value.value)).exists())
