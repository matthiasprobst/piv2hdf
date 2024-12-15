import warnings

from . import tools
from .file import OpenPIVFile, get_files
from .parameter import OpenPIVParameterFile

try:
    import openpiv

    try:
        openpiv_version = openpiv.__version__
    except AttributeError:
        import pkg_resources

        openpiv_version = pkg_resources.get_distribution('openpiv').version
    from packaging import version as version_pkg

    if version_pkg.Version(openpiv_version) < version_pkg.Version('0.25.0'):
        warnings.warn(f'OpenPIV version is too old ({openpiv_version}). Please update to version 0.25.0 or newer.',
                      UserWarning)
    # else:
    #     print('OpenPIV version is ok.')
except ImportError:
    pass
