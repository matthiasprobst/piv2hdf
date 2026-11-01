"""Record source paths, software versions and declared conversion policies."""
import json

import h5rdmtoolbox as h5tbx

from ._version import __version__
from .config import get_config


def record_conversion(output, sources):
    options = get_config()
    convention = options['convention']
    options['convention'] = getattr(convention, 'name', convention)
    options['postproc'] = [processor if isinstance(processor, str)
                          else f'{getattr(processor, "__module__", type(processor).__module__)}.'
                               f'{getattr(processor, "__qualname__", type(processor).__qualname__)}'
                          for processor in options['postproc']]
    output.attrs.update(
        piv2hdf_version=__version__, h5rdmtoolbox_version=h5tbx.__version__,
        source_files=json.dumps([str(source.resolve()) for source in sources]),
        conversion_options=json.dumps(options, default=str, sort_keys=True),
        piv2hdf_schema_version='2.0')
