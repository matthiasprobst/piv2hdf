import unittest

import h5rdmtoolbox as h5tbx


class TestConvention(unittest.TestCase):

    def test_using_convention(self):
        with h5tbx.use("planar_piv"):
            with self.assertRaises(h5tbx.errors.StandardAttributeError):
                with h5tbx.File():
                    pass
            with h5tbx.File(
                creator=dict(name="Matthias Probst"),
                camera=dict(name="My Camera", model="Model 1", manufacturer="My Manufacturer",
                            sensor_type="CCD"),
            ) as h5:
                creator = h5.creator
                camera = h5.camera
        self.assertEqual(creator.name, "Matthias Probst")
        self.assertEqual(camera.sensor_type, "CCD")