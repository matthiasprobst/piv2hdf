"""Validity and interpolation policies for vector fields."""
import warnings

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, QhullError

from .flags import Flags, valid_vectors


def mask_vectors(data, attrs, mode, *, interpolate=False, coordinates=None):
    """Preserve raw fields and flags, then mask or interpolate invalid vectors.

    Linear interpolation is confined to the donors' convex hull. Unsupported
    boundary holes remain NaN; no nearest-neighbor extrapolation is implicit.
    """
    fields = [name for name, values in data.items()
              if np.ndim(values) == 2 and np.issubdtype(np.asarray(values).dtype, np.floating)]
    if not fields:
        return
    shape = np.shape(data[fields[0]])
    flags = data.get('piv_flags')
    valid = valid_vectors(flags, mode) if flags is not None else np.ones(shape, dtype=bool)
    for name in fields:
        raw = np.asarray(np.ma.filled(data[name], np.nan), dtype=float)
        data[f'raw_{name}'] = raw.copy()
        attrs[f'raw_{name}'] = {'units': attrs[name].get('units', ''),
                               'long_name': f'Original source values of {name}'}
        data[name] = raw
        if name in ('u', 'v', 'w'):
            valid &= np.isfinite(raw)
    if flags is not None:
        data['raw_piv_flags'] = np.array(flags, copy=True)
        attrs['raw_piv_flags'] = dict(attrs.get('piv_flags', {}))
        attrs['raw_piv_flags']['long_name'] = 'Original source vector flags'
    for name in fields:
        data[name] = np.where(valid, data[name], np.nan)

    if interpolate and np.any(~valid) and np.any(valid):
        x, y = coordinates
        points = np.column_stack((x[valid], y[valid]))
        targets = np.column_stack((x[~valid], y[~valid]))
        try:
            tri = Delaunay(points)
        except (QhullError, ValueError):
            warnings.warn('Too few non-collinear valid vectors for interpolation; holes remain NaN', RuntimeWarning)
        else:
            for name in fields:
                donor_values = data[name][valid]
                data[name][~valid] = LinearNDInterpolator(tri, donor_values)(targets)
            filled = ~valid
            for name in ('u', 'v', 'w'):
                if name in data:
                    filled &= np.isfinite(data[name])
            valid |= filled
            if flags is not None:
                data['piv_flags'] = np.array(flags, copy=True)
                data['piv_flags'][filled] = Flags.ACTIVE.value | Flags.INTERPOLATED.value
    data['valid'] = valid.astype(np.uint8)
    attrs['valid'] = {'units': '', 'long_name': f'Vector validity using {mode} masking',
                      'flag_meaning': '{"0": "invalid", "1": "valid"}'}
