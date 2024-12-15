import json
import pathlib
import unittest

import h5rdmtoolbox as h5tbx
from h5rdmtoolbox import convention

import piv2hdf

__this_dir__ = pathlib.Path(__file__).parent


class TestConfig(unittest.TestCase):

    def read_meta(self):
        with open(__this_dir__ / "meta.json") as f:
            meta = json.load(f)
        return meta

    def test_contacts(self):
        contacts_bak = piv2hdf.contacts
        contacts = piv2hdf.contacts
        contacts.clear()
        self.assertEqual(len(contacts), 0)
        self.assertTrue(contacts.__repr__().startswith('Contacts('))
        self.assertEqual(len(contacts.registered_contacts), len(contacts))
        contacts.register(mp='https://orcid.org/0000-0001-8729-0482')
        self.assertDictEqual({'mp': 'https://orcid.org/0000-0001-8729-0482', },
                             contacts.registered_contacts)
        contacts.register(mp2='https://orcid.org/0000-0001-8729-0482')
        self.assertDictEqual({'mp': 'https://orcid.org/0000-0001-8729-0482',
                              'mp2': 'https://orcid.org/0000-0001-8729-0482'}, contacts.registered_contacts)

        # print(contacts.contacts_filename)
        self.assertEqual(len(contacts.registered_contacts), len(contacts))
        contacts.contacts_filename.unlink()

        _contacts = piv2hdf.Contacts()
        self.assertDictEqual({}, _contacts.registered_contacts)

        contacts.registered_contacts = {}
        contacts.register(mp='https://orcid.org/0000-0001-8729-0482')
        self.assertDictEqual({'mp': 'https://orcid.org/0000-0001-8729-0482'}, contacts.registered_contacts)

        contacts.clear()
        contacts.register(mp='https://orcid.org/0000-0001-8729-0482')
        self.assertDictEqual({'mp': 'https://orcid.org/0000-0001-8729-0482'}, contacts.registered_contacts)

        contacts_bak.save()

    def test_set_config(self):
        package_convention = piv2hdf.CONVENTION_FILENAME
        cv = h5tbx.Convention.from_yaml(package_convention)
        cv.register()

        current_cv_name = piv2hdf.get_config('convention')
        self.assertEqual('planar_piv', current_cv_name)

        piv2hdf.set_config(convention=None)
        current_cv_name = convention.get_current_convention().name
        self.assertEqual('h5py', current_cv_name)

        piv2hdf.set_config(convention='h5tbx')
        current_cv_name = convention.get_current_convention().name
        self.assertEqual('h5py', current_cv_name)

        piv2hdf.set_config(convention=cv)
        current_cv_name = convention.get_current_convention().name
        self.assertEqual('h5py', current_cv_name)

        with piv2hdf.set_config(convention='h5py'):
            current_cv_name = convention.get_current_convention().name
            self.assertEqual('h5py', current_cv_name)
            self.assertEqual('h5py', piv2hdf.get_config('convention'))

        current_cv_name = convention.get_current_convention().name
        self.assertEqual('h5py', current_cv_name)
        self.assertEqual(cv, piv2hdf.get_config('convention'))

    def test_set_pivattrs(self):
        meta = self.read_meta()
        _ = piv2hdf.set_pivattrs(creator=meta["CREATOR"],
                                 piv_medium='air',
                                 camera_type='My Camera',
                                 fstop=1.0,
                                 laser={'name': 'Big Sky laser', 'energy per pulse:': '130 mJ'})
