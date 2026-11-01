"""Physical-data invariants shared by adapters and HDF5 writers."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np
from h5rdmtoolbox import get_ureg


@dataclass
class PIVData:
    data: Dict
    data_attrs: Dict
    root_attrs: Dict

    def with_z(self, z):
        """Return independent data/metadata mappings with a quantity-safe z override."""
        from .utils import parse_z
        result = PIVData(dict(self.data), deepcopy(self.data_attrs), dict(self.root_attrs))
        if z is not None:
            result.data['z'], units = parse_z(z)
            result.data_attrs.setdefault('z', {})['units'] = units
        return result


def convert_values(values, source_units, target_units):
    """Convert values without silently assuming missing units or calibration."""
    if source_units is None or target_units is None:
        raise ValueError('Both source and target units must be specified')
    return get_ureg().Quantity(values, str(source_units)).to(str(target_units)).magnitude


def validate_snapshot(reference, candidate, source):
    """Reject slices that would change the meaning of a plane's shared coordinates."""
    if set(reference.data) != set(candidate.data):
        raise ValueError(f'{source}: snapshot variables differ from the first snapshot')
    for name in reference.data:
        expected = np.asanyarray(reference.data[name])
        actual = np.asanyarray(candidate.data[name])
        if actual.shape != expected.shape:
            raise ValueError(f'{source}: inconsistent shape for {name}')
        if name == 'reltime':
            continue
        expected_units = reference.data_attrs.get(name, {}).get('units')
        actual_units = candidate.data_attrs.get(name, {}).get('units')
        if expected_units != actual_units:
            raise ValueError(f'{source}: inconsistent units for {name}; normalize before combining')
        if name in ('x', 'y', 'z', 'ix', 'iy'):
            if not np.array_equal(expected, actual):
                raise ValueError(f'{source}: inconsistent {name} coordinates')


def validate_relative_time(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError('Relative time must be a nonempty finite vector')
    if np.any(np.diff(values) <= 0):
        raise ValueError('Recording timestamps must be strictly increasing')
    return values
