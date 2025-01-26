import json
import pathlib
import unittest

import piv2hdf

__this_dir__ = pathlib.Path(__file__).parent


class TestVersion(unittest.TestCase):

    def test_version(self):
        this_version = 'x.x.x'
        setupcfg_filename = __this_dir__ / '../setup.cfg'
        with open(setupcfg_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'version =' in line:
                    this_version = line.split(' = ')[-1].strip()
        self.assertEqual(piv2hdf.__version__, this_version)

    def test_codemeta(self):
        """checking if the version in codemeta.json is the same as the one of the piv2hdf"""

        with open(__this_dir__ / '../codemeta.json', 'r') as f:
            codemeta = json.loads(f.read())

        self.assertEqual(codemeta['version'], piv2hdf.__version__)

    def test_citation_cff(self):
        """checking if the version in CITATION.cff is the same as the one of the piv2hdf"""
        this_version = 'x.x.x'
        setupcfg_filename = __this_dir__ / '../CITATION.cff'
        with open(setupcfg_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'version: ' in line:
                    this_version = line.split(':')[-1].strip()
        self.assertEqual(piv2hdf.__version__, this_version)
