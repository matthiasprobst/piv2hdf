import unittest

from piv2hdf import tutorial
from piv2hdf.openpiv import OpenPIVParameterFile
from piv2hdf.pivview import PIVviewParamFile


class TestParameter(unittest.TestCase):

    def test_pivview_parameter(self):
        with self.assertRaises(FileNotFoundError):
            PIVviewParamFile('nonexistent.par')
        pivview_parameter_file = tutorial.PIVview.get_parameter_file()
        par = PIVviewParamFile(pivview_parameter_file)
        self.assertEqual(par.suffix, '.par')
        pivview_parameter_dict = par.to_dict()
        self.assertIsInstance(pivview_parameter_dict, dict)

    def test_openpiv_parameter(self):
        openpiv_parameter_file = tutorial.OpenPIV.get_parameter_file()
        par = OpenPIVParameterFile(openpiv_parameter_file)
        self.assertEqual(par.suffix, '.par')
        openpiv_parameter_dict = par.to_dict()
        self.assertIsInstance(openpiv_parameter_dict, dict)
