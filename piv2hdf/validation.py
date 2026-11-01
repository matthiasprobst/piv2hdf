"""Validate completed HDF5 files through h5rdmtoolbox's convention API."""
import warnings

import h5rdmtoolbox as h5tbx
from h5rdmtoolbox.convention.core import MissingAttribute

from .config import get_config


class LayoutValidationWarning(Warning):
    """A completed conversion does not meet the requested convention."""


class LayoutValidationError(ValueError):
    """A completed conversion does not meet the requested convention."""


def validate_layout(filename):
    convention = get_config('convention')
    if convention is None:
        return []
    # Opening before activating the convention avoids injecting defaults.
    with h5tbx.use(None), h5tbx.File(filename, 'r') as output:
        with h5tbx.use(convention) as active:
            failures = active.validate(output)
            result = []
            for failure in failures:
                if isinstance(failure, MissingAttribute):
                    node = output[failure.object_name]
                    attribute = node.standard_attributes.get(failure.attribute_name)
                    alternative = getattr(attribute, 'alternative_standard_attribute', None)
                    if alternative and alternative in node.attrs.raw:
                        continue
                result.append(failure)
    if result:
        message = '\n'.join(str(failure) for failure in result)
        if get_config('raise_layout_validation_error'):
            raise LayoutValidationError(message)
        warnings.warn(message, LayoutValidationWarning)
    return result
