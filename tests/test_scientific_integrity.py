"""Independent physical reference cases and regressions for conversion contracts."""
import datetime as dt
from copy import deepcopy

import h5py
import h5rdmtoolbox as h5tbx
import numpy as np
import pytest

import piv2hdf
from piv2hdf import config
from piv2hdf.data import PIVData, validate_snapshot
from piv2hdf.flags import valid_vectors
from piv2hdf.interface import PIVFile, PIVParameterInterface, PIVPlane, PIVMultiPlane, PIVSnapshot
from piv2hdf.processing import mask_vectors
from piv2hdf.postproc import compute_time_averages
from piv2hdf.standard_names import TABLE_URL, load_table
from piv2hdf.validation import validate_layout, LayoutValidationError, LayoutValidationWarning


class ReferenceFile(PIVFile):
    suffix = '.dat'

    def __init__(self, path, *, units='m', z=0., shift=0.):
        path.write_text('synthetic reference field')
        super().__init__(path, PIVParameterInterface(None))
        self.units, self.z, self.shift = units, z, shift

    def read(self, relative_time, **kwargs):
        coordinates = np.array([0., 1.]) + self.shift
        data = dict(x=coordinates, y=np.array([0., 1.]), z=np.array(self.z),
                    ix=np.array([0, 1]), iy=np.array([0, 1]), reltime=np.array(relative_time),
                    u=np.full((2, 2), 3.), v=np.full((2, 2), 4.))
        attrs = {name: {'units': self.units if name in ('x', 'y', 'z') else '',
                         'long_name': name} for name in data}
        attrs['reltime']['units'] = 's'
        for name in ('u', 'v'):
            attrs[name]['units'] = 'm/s'
            attrs[name]['standard_name'] = f'{"x" if name == "u" else "y"}_velocity'
        return PIVData(data, attrs, {})

    def to_hdf(self, hdf_filename, relative_time, recording_dtime, z=None, **kwargs):
        from piv2hdf.writer import write_snapshot
        return write_snapshot(self.read(relative_time), hdf_filename, self.filename, recording_dtime, z)


def reference_plane(tmp_path, *, units='m', z=0., shift=0., times=None):
    files = [ReferenceFile(tmp_path / f'source-{z}-{i}.dat', units=units, z=z, shift=shift) for i in range(2)]
    return PIVPlane(files, times or (dt.datetime(2020, 1, 1), 2.))


@pytest.mark.parametrize('z, units, value', [('2 m', 'm', 2.), ('12 mm', 'millimeter', 12.), (3., 'm', 3.)])
def test_z_override_keeps_quantity(tmp_path, z, units, value):
    plane = reference_plane(tmp_path, units='mm')
    filename = plane.to_hdf(hdf_filename=tmp_path / 'plane.hdf', z=z)
    with h5py.File(filename) as output:
        quantity = h5tbx.get_ureg().Quantity(output['z'][()], output['z'].attrs['units'])
        assert quantity.to(units).magnitude == value
        import json
        assert len(json.loads(output.attrs['source_files'])) == 2
        assert output.attrs['piv2hdf_version'] == piv2hdf.__version__
        assert output.attrs['h5rdmtoolbox_version'] == h5tbx.__version__


def test_metadata_is_local_and_contexts_restore(tmp_path):
    with config.set_pivattrs(experiment='default'):
        with config.set_pivattrs(extra='temporary'):
            assert config.get_pivattrs()['extra'] == 'temporary'
        assert config.get_pivattrs() == {'experiment': 'default'}
        plane = reference_plane(tmp_path)
        plane.to_hdf({'experiment': 'A'}, tmp_path / 'a.hdf')
        filename = plane.to_hdf(hdf_filename=tmp_path / 'b.hdf')
        with h5py.File(filename) as output:
            assert output.attrs['experiment'] == 'default'
    assert config.get_pivattrs() == {}


def test_options_do_not_expose_mutable_defaults():
    original = config.get_config('dtypes')
    exposed = config.get_config()
    exposed['dtypes']['x'] = 'int8'
    assert config.get_config('dtypes') == original
    with pytest.raises(RuntimeError):
        with config.set_config(postproc=[]):
            config.CONFIG['dtypes']['x'] = 'int8'
            raise RuntimeError('callback failed')
    assert config.get_config('dtypes') == original


def test_callable_postprocessor_and_frozen_options(tmp_path):
    calls = []
    def callback(output):
        calls.append(output.filename)
        config.set_config(postproc=[])
    with config.set_config(postproc=[callback]):
        plane = reference_plane(tmp_path)
        plane.to_hdf(hdf_filename=tmp_path / 'callbacks.hdf')
    assert len(calls) == 1


@pytest.mark.parametrize('frequency', [0., -1., float('nan'), float('inf'), True])
def test_bad_frequency_rejected(tmp_path, frequency):
    with pytest.raises(ValueError, match='frequency'):
        reference_plane(tmp_path, times=(dt.datetime(2020, 1, 1), frequency))


@pytest.mark.parametrize('seconds', [(0, 0), (1, 0)])
def test_nonincreasing_timestamps_rejected(tmp_path, seconds):
    start = dt.datetime(2020, 1, 1)
    with pytest.raises(ValueError, match='increasing'):
        reference_plane(tmp_path, times=[start + dt.timedelta(seconds=s) for s in seconds])


def test_grid_drift_is_rejected(tmp_path):
    plane = reference_plane(tmp_path)
    plane.list_of_piv_files[1].shift = 10.
    path = tmp_path / 'drift.hdf'
    path.write_bytes(b'previous verified output')
    with pytest.raises(ValueError, match='x coordinates'):
        plane.to_hdf(hdf_filename=path)
    assert path.read_bytes() == b'previous verified output'
    assert not list(tmp_path.glob('.drift-*.hdf'))
    assert plane.hdf_filename is None


@pytest.mark.parametrize('mode, expected', [
    ('strict', [False, True, False, False, False, False, False]),
    ('sepeaks', [False, True, False, False, False, True, False]),
    ('slack', [False, True, False, False, True, True, False])])
def test_flag_combinations(mode, expected):
    flags = np.array([0, 1, 1 | 8, 1 | 2, 1 | 32, 1 | 64, 256])
    np.testing.assert_array_equal(valid_vectors(flags, mode), expected)


def test_linear_interpolation_and_raw_provenance():
    y, x = np.mgrid[:3, :3]
    values = 2.*x + 3.*y
    flags = np.ones((3, 3), dtype='uint8')
    flags[1, 1] = 8
    values[1, 1] = 999.
    data = {'u': values, 'piv_flags': flags}
    attrs = {'u': {'units': 'm/s'}, 'piv_flags': {'units': ''}}
    mask_vectors(data, attrs, 'strict', interpolate=True, coordinates=(x, y))
    assert data['u'][1, 1] == 5.
    assert data['raw_u'][1, 1] == 999.
    assert data['raw_piv_flags'][1, 1] == 8
    assert data['piv_flags'][1, 1] == 33
    assert data['valid'][1, 1] == 1


def test_interpolation_never_extrapolates_or_invents_all_invalid_data():
    y, x = np.mgrid[:3, :3]
    for flags in [np.zeros((3, 3), dtype='uint8'), np.ones((3, 3), dtype='uint8')]:
        flags[0, 0] = 8
        data = {'u': np.ones((3, 3)), 'piv_flags': flags}
        mask_vectors(data, {'u': {'units': 'm/s'}, 'piv_flags': {'units': ''}},
                     'strict', interpolate=True, coordinates=(x, y))
        assert np.isnan(data['u'][0, 0])
        assert data['valid'][0, 0] == 0


@pytest.mark.parametrize('method, expected', [('sample', 10./3), ('duration', 4.5)])
def test_irregular_sampling_estimators(tmp_path, method, expected):
    filename = tmp_path / 'statistics.hdf'
    with h5tbx.File(filename, 'w') as output:
        t = output.create_dataset('t', data=[0., 1., 10.], make_scale=True,
                                  attrs={'units': 's', 'standard_name': 'relative_time'})
        u = output.create_dataset('u', data=np.array([[0.], [0.], [10.]]),
                                  attrs={'units': 'm/s', 'standard_name': 'x_velocity'})
        u.dims[0].attach_scale(t)
        compute_time_averages(output, averaging=method)
        assert output['mean_u'].values[0] == pytest.approx(expected)
        assert output['mean_u_valid_count'].values[0] == 3


def test_means_exclude_flagged_values_and_report_empty_samples(tmp_path):
    with h5tbx.File(tmp_path / 'invalid.hdf', 'w') as output:
        t = output.create_dataset('t', data=[0., 1.], make_scale=True,
                                  attrs={'standard_name': 'relative_time', 'units': 's'})
        u = output.create_dataset('u', data=[[1., np.nan], [999., np.nan]],
                                  attrs={'standard_name': 'x_velocity', 'units': 'm/s'})
        u.dims[0].attach_scale(t)
        output.create_dataset('piv_flags', data=[[1, 1], [8, 1]])
        compute_time_averages(output)
        np.testing.assert_allclose(output['mean_u'].values[:], [1., np.nan], equal_nan=True)
        np.testing.assert_array_equal(output['mean_u_valid_count'].values[:], [1, 0])


def test_table_is_local_and_unknown_names_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr('piv2hdf.standard_names.urlopen', lambda *a, **k: pytest.fail('default table went online'))
    load_table.cache_clear()
    assert 'x_velocity' in load_table(TABLE_URL)
    from piv2hdf.standard_names import validate_standard_name
    with h5tbx.File(tmp_path / 'names.hdf', 'w', attrs={'standard_name_table': TABLE_URL}) as output:
        ds = output.create_dataset('u', data=[1.], attrs={'units': 'm/s'})
        assert validate_standard_name('x_velocity', ds) == 'x_velocity'
        with pytest.raises(ValueError):
            validate_standard_name('invented_velocity', ds)


def test_standard_name_tables_are_scoped_to_each_file(tmp_path):
    import json
    from piv2hdf.standard_names import validate_standard_name
    def table(name):
        return json.dumps({'@context': {'ssno': 'https://matthiasprobst.github.io/ssno#'},
                           'ssno:standardNames': [{'ssno:standardName': name,
                                                  'ssno:unit': 'http://qudt.org/vocab/unit/M'}]})
    with h5py.File(tmp_path/'a.hdf', 'w') as first, h5py.File(tmp_path/'b.hdf', 'w') as second:
        datasets = []
        for output, name in [(first, 'first_coordinate'), (second, 'second_coordinate')]:
            output.attrs['standard_name_table'] = table(name)
            ds = output.create_dataset('x', data=[1.])
            ds.attrs['units'] = 'm'
            datasets.append(ds)
        assert validate_standard_name('first_coordinate', datasets[0]) == 'first_coordinate'
        assert validate_standard_name('second_coordinate', datasets[1]) == 'second_coordinate'
        assert validate_standard_name('first_coordinate', datasets[0]) == 'first_coordinate'
        with pytest.raises(ValueError):
            validate_standard_name('second_coordinate', datasets[0])


def test_recursive_validation_and_warning_policy(tmp_path):
    filename = tmp_path / 'validation.hdf'
    convention = h5tbx.Convention(name='test_required_units', contact='https://example.org/contact')
    from h5rdmtoolbox.convention import StandardAttribute
    convention.add_standard_attribute(StandardAttribute(name='units', validator='$units', target_method='create_dataset',
                                     description='Physical units'))
    convention.register()
    with h5py.File(filename, 'w') as output:
        output.create_group('nested').create_dataset('missing_units', data=[1.])
    with config.set_config(convention=convention):
        with pytest.raises(LayoutValidationError, match='missing_units'):
            validate_layout(filename)
        with config.set_config(raise_layout_validation_error=False):
            with pytest.warns(LayoutValidationWarning, match='missing_units'):
                assert validate_layout(filename)


def plane_file(tmp_path, name, *, units='m', z=0., spacing=1., nt=2):
    path = tmp_path / name
    with h5py.File(path, 'w') as output:
        for key, values in [('x', [0., spacing]), ('y', [0., spacing]),
                            ('ix', [0, 1]), ('iy', [0, 1]), ('reltime', np.arange(nt) * .5)]:
            ds = output.create_dataset(key, data=values)
            ds.make_scale()
            ds.attrs['units'] = units if key in ('x', 'y') else 's' if key == 'reltime' else ''
        output.create_dataset('z', data=z).attrs['units'] = units
        output.create_dataset('time', data=np.array(['2020-01-01'] * nt, dtype='S10'))
        ds = output.create_dataset('u', data=np.ones((nt, 2, 2)))
        ds.attrs['units'] = 'm/s'
        for axis, name in enumerate(['reltime', 'y', 'x']):
            ds.dims[axis].attach_scale(output[name])
        output.create_group('piv_parameters')
    return path


def test_merge_compares_physical_coordinates(tmp_path):
    a = plane_file(tmp_path, 'a.hdf')
    b = plane_file(tmp_path, 'b.hdf', units='mm', z=1000., spacing=1000.)
    output = tmp_path / 'merged.hdf'
    PIVMultiPlane.merge_planes(hdf_filenames=[a, b], target_hdf_filename=output, title='test')
    with h5py.File(output) as data:
        np.testing.assert_array_equal(data['z'][:], [0., 1.])
        assert data['z'].attrs['units'] == 'm'
    b = plane_file(tmp_path, 'b.hdf', units='mm', z=1000., spacing=1.)
    with pytest.raises(ValueError, match='x coordinates'):
        PIVMultiPlane.merge_planes(hdf_filenames=[a, b], target_hdf_filename=output, title='test')


def test_merge_rejects_duplicate_later_z(tmp_path):
    files = [plane_file(tmp_path, f'{i}.hdf', z=z) for i, z in enumerate([0., 1., 1.])]
    with pytest.raises(ValueError, match='z coordinates'):
        PIVMultiPlane.merge_planes(hdf_filenames=files, target_hdf_filename=tmp_path/'out.hdf', title='test')


def test_trim_requires_consent_and_records_losses(tmp_path):
    files = [plane_file(tmp_path, 'short.hdf', z=0., nt=2), plane_file(tmp_path, 'long.hdf', z=1., nt=3)]
    kwargs = dict(hdf_filenames=files, target_hdf_filename=tmp_path/'out.hdf', title='test')
    with pytest.raises(ValueError, match='trim_time_vectors'):
        PIVMultiPlane.merge_planes(**kwargs)
    from piv2hdf.time import TimeVectorWarning
    with pytest.warns(TimeVectorWarning):
        PIVMultiPlane.merge_planes(**kwargs, trim_time_vectors=True)
    with h5py.File(kwargs['target_hdf_filename']) as output:
        np.testing.assert_array_equal(output.attrs['dropped_snapshot_count'], [0, 1])
        np.testing.assert_array_equal(output['relative_time_per_plane'][:], [[0., .5], [0., .5]])


def test_davis_calibration_is_applied_once_without_inventing_units(tmp_path, monkeypatch):
    pytest.importorskip('lvpyio')
    from types import SimpleNamespace
    from piv2hdf.davis.file import VC7File
    scale = SimpleNamespace(slope=2., offset=10., unit='mm', description='coordinate')
    comp_scale = SimpleNamespace(slope=3., offset=1., unit='pixel', description='displacement')
    frame = SimpleNamespace(shape=(2, 2), scales=SimpleNamespace(x=scale, y=scale, z=scale),
                            components={'U0': SimpleNamespace(scale=comp_scale, planes=[np.ones((2, 2))])})
    monkeypatch.setattr('piv2hdf.davis.file.lv.read_buffer', lambda filename: [frame])
    path = tmp_path/'test.vc7'
    path.write_bytes(b'stub')
    adapter = VC7File(path, parameter=PIVParameterInterface(None))
    data = adapter.read(0.)
    np.testing.assert_array_equal(data.data['x'], [10., 12.])
    assert data.data_attrs['x']['units'] == 'mm'
    np.testing.assert_array_equal(data.data['U0'], np.full((2, 2), 4.))
    np.testing.assert_array_equal(data.data['raw_U0'], np.ones((2, 2)))
    assert data.data_attrs['U0']['units'] == 'pixel'


@pytest.mark.parametrize('length_units, expected', [('m', (2., 3.)), ('mm', (2000., 3000.))])
def test_stereo_gradient_axes_and_units(tmp_path, length_units, expected):
    from netCDF4 import Dataset
    from piv2hdf.pivview.nc import process_pivview_nc_data
    path = tmp_path/'stereo.nc'
    with Dataset(path, 'w') as nc:
        dims = ('data_array_depth', 'data_array_height', 'data_array_width', 'vec_dims')
        for name, size in zip(dims, (1, 3, 4, 3)):
            nc.createDimension(name, size)
        nc.setncatts(dict(file_content='synthetic', creation_date='2020-01-01', software='test',
                         outlier_interpolate=0, outlier_try_other_peak=0, length_conversion_units=length_units))
        for name in ('velocity', 'piv_data'):
            var = nc.createVariable(name, 'f8', dims)
            var.setncatts(dict(coord_min=np.array([0., 0., 0.]), coord_max=np.array([3., 2., 0.]),
                               units='m/s' if name == 'velocity' else 'pixel', long_name=name))
            var[:] = 0.
        nc['velocity'][0, :, :, 2] = 2*np.arange(4)[None, :] + 3*np.arange(3)[:, None]
        nc.createVariable('piv_flags', 'i1', dims[:-1])[:] = 1
    data, _, attrs = process_pivview_nc_data(path, False, 'strict', 0.)
    np.testing.assert_allclose(data['dwdx'], expected[0])
    np.testing.assert_allclose(data['dwdy'], expected[1])
    assert attrs['dwdx']['units'] == '1/s'
    with Dataset(path, 'r+') as nc:
        for name, value in [('dudx', 2.), ('dvdy', -3.)]:
            ds = nc.createVariable(name, 'f8', dims[:-1])
            ds.setncatts(dict(units='1/s', long_name=f'velocity gradient {name}'))
            ds[:] = value
    data, _, attrs = process_pivview_nc_data(path, False, 'strict', 0., compute_dwdz=True)
    np.testing.assert_allclose(data['dwdz'], 1.)
    assert attrs['dwdz']['assumption'] == 'incompressible flow'


def test_openpiv_nonunit_calibration_and_fractional_pixel_locations(tmp_path):
    from piv2hdf.openpiv.file import OpenPIVFile
    param = PIVParameterInterface(None)
    param.param_dict = {'scaling_factor': 2.}
    path = tmp_path/'vectors.txt'
    path.write_text('# x\ty\tu\tv\tflags\n0.5\t0.5\t3\t4\t0\n2.5\t0.5\t3\t4\t0\n'
                    '0.5\t2.5\t3\t4\t0\n2.5\t2.5\t3\t4\t0\n')
    data = OpenPIVFile(path, parameter=param, scaling_factor_units='pixel/mm').read(0.)
    np.testing.assert_allclose(data.data['x'], [.00025, .00125])
    np.testing.assert_allclose(data.data['ix'], [.5, 2.5])
    np.testing.assert_allclose(data.data['u'], 3.)
    assert data.data_attrs['x']['units'] == 'm'


def test_real_pivview_conversion_with_full_convention(tmp_path):
    from piv2hdf import tutorial, pivview
    from piv2hdf.pivview.user_operations import add_standard_name_operation
    adapter = pivview.PIVViewNcFile(tutorial.PIVview.get_snapshot_nc_files()[0],
                                    udo=add_standard_name_operation)
    with config.set_config(convention='planar_piv'):
        path = PIVSnapshot(adapter).to_hdf({'creator': {'name': 'Test scientist'}}, tmp_path/'validated.hdf')
        assert validate_layout(path) == []


def test_import_preserves_caller_convention_and_does_not_create_contacts(tmp_path):
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    user_dir = tmp_path/'new-user'
    env['PIV2HDF_USER_DIR'] = str(user_dir)
    env.pop('PIV2HDF_LOG_DIR', None)
    script = """
import h5rdmtoolbox as h
from pathlib import Path
from h5rdmtoolbox.convention import get_current_convention
cv = h.Convention(name='caller_convention', contact='https://example.org/contact')
cv.register()
with h.use(cv):
    import piv2hdf
    assert get_current_convention().name == 'caller_convention'
    from piv2hdf.user import USER_DIRECTORY
    assert not USER_DIRECTORY.exists()
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert not user_dir.exists()


def test_unimplemented_memory_adapter_does_not_touch_existing_files(tmp_path, monkeypatch):
    from piv2hdf.openpiv.file import OpenPIVResultData
    monkeypatch.chdir(tmp_path)
    path = tmp_path/'__tmp__.txt'
    path.write_text('keep this file')
    with pytest.raises(NotImplementedError):
        OpenPIVResultData({})
    assert path.read_text() == 'keep this file'


def test_temporary_reservations_support_nested_prefixes():
    from piv2hdf import user
    from piv2hdf.utils import generate_temporary_directory, generate_temporary_filename
    directory = generate_temporary_directory(prefix='nested/')
    assert directory.parent == user.TEMPORARY_USER_DIRECTORY / 'nested'
    path = generate_temporary_filename(prefix='nested/file-', suffix='.hdf')
    assert path.is_file()
    assert path.parent == directory.parent
    with pytest.raises(ValueError):
        generate_temporary_directory(prefix='../escape')
