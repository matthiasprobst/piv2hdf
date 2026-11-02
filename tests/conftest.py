"""Keep conversion tests independent of user state and source fixtures."""
import os
import logging
import shutil
import tempfile
from pathlib import Path

import pytest

_STATE = tempfile.TemporaryDirectory(prefix='piv2hdf-tests-')
_ROOT = Path(_STATE.name)
for key, folder in [('PIV2HDF_USER_DIR', 'piv'), ('PIV2HDF_LOG_DIR', 'logs'),
                    ('H5RDMTOOLBOX_USER_DATA_DIR', 'h5'),
                    ('H5RDMTOOLBOX_LOG_DIR', 'h5-logs'),
                    ('H5RDMTOOLBOX_CACHE_DIR', 'h5-cache'), ('MPLCONFIGDIR', 'mpl')]:
    os.environ[key] = str(_ROOT / folder)
os.environ['MPLBACKEND'] = 'Agg'


def pytest_sessionfinish(session, exitstatus):
    # Close Windows file handlers before the temporary directory finalizer runs.
    logging.shutdown()


@pytest.fixture(autouse=True)
def independent_conversion_state(tmp_path):
    import h5rdmtoolbox as h5tbx
    import piv2hdf
    from piv2hdf import config, user
    from piv2hdf.convention import init_cv
    init_cv()
    fixtures = tmp_path / 'resources'
    shutil.copytree(Path(__file__).parent / 'resources', fixtures)
    old_resources, old_tmp = user.TEST_DATA_DIRECTORY, user.TEMPORARY_USER_DIRECTORY
    old_attrs = config.get_pivattrs()
    config.reset_pivattrs()
    user.TEST_DATA_DIRECTORY = fixtures
    user.TEMPORARY_USER_DIRECTORY = tmp_path / 'conversions'
    user.TEMPORARY_USER_DIRECTORY.mkdir()
    options = dict(convention=None, postproc=['compute_time_averages'])
    with config.set_config(**options), h5tbx.use(None):
        try:
            yield
        finally:
            config.PIV_ATTRS.clear()
            config.PIV_ATTRS.update(old_attrs)
            user.TEST_DATA_DIRECTORY, user.TEMPORARY_USER_DIRECTORY = old_resources, old_tmp
