"""Explicit, bounded-memory statistics of valid PIV measurements."""
import h5py
import h5rdmtoolbox as h5tbx
import numpy as np


def compute_time_averages(h5, *, averaging='sample'):
    """Compute sample means or piecewise-linear duration-weighted means.

    Exclude nonfinite and invalid values. Duration weighting uses only intervals
    with two valid endpoints, without bridging gaps. Sample means are the default.

    Pass a filename or an open h5rdmtoolbox File. Each selected velocity requires
    its component standard name and one attached ``relative_time`` scale. Writes
    ``mean_<component>`` and ``mean_<component>_valid_count`` with remaining scales.
    A same-method existing mean is retained; different/unknown methods raise.
    """
    if averaging not in ('sample', 'duration'):
        raise ValueError('averaging must be sample or duration')
    if not isinstance(h5, h5py.File):
        with h5tbx.File(h5, 'r+') as opened:
            return compute_time_averages(opened, averaging=averaging)
    from .flags import valid_vectors
    from .data import validate_relative_time

    for sn in ('x_velocity', 'y_velocity', 'z_velocity'):
        found = h5.find_one({'standard_name': sn})
        if found is None:
            continue
        ds = h5[found.name]
        axes = [(i, scale) for i, dim in enumerate(ds.dims) for scale in dim.values()
                if scale.attrs.get('standard_name') == 'relative_time']
        if len(axes) != 1:
            continue
        axis, time = axes[0]
        t = validate_relative_time(h5py.Dataset.__getitem__(time, slice(None)))
        mean_name = f'mean_{ds.basename}'
        if mean_name in ds.parent:
            existing = ds.parent[mean_name]
            if existing.attrs.get('averaging_method') != averaging:
                raise ValueError(f'{existing.name} already exists with a different or unknown averaging method')
            continue
        shape = ds.shape[:axis] + ds.shape[axis + 1:]
        total = np.zeros(shape, dtype=np.float64)
        weight = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.uint64)
        flags = ds.parent.get('piv_flags')
        if flags is not None and flags.shape != ds.shape:
            raise ValueError('Velocity and flag shapes differ')
        previous = previous_valid = None
        for i in range(len(t)):
            index = (slice(None),) * axis + (i,)
            values = np.asarray(h5py.Dataset.__getitem__(ds, index), dtype=float)
            valid = np.isfinite(values)
            if flags is not None:
                valid &= valid_vectors(h5py.Dataset.__getitem__(flags, index), 'slack')
            count += valid
            if averaging == 'sample':
                total += np.where(valid, values, 0.)
                weight += valid
            elif i:
                interval_valid = valid & previous_valid
                duration = t[i] - t[i - 1]
                total += np.where(interval_valid, (previous + values) / 2 * duration, 0.)
                weight += interval_valid * duration
            previous, previous_valid = values, valid
        mean = np.divide(total, weight, out=np.full(shape, np.nan), where=weight > 0)
        attrs = {'units': ds.attrs['units'], 'averaging_method': averaging,
                 'missing_data_policy': 'exclude nonfinite and invalid; do not bridge gaps',
                 'source_dataset': ds.name}
        if averaging == 'sample':
            attrs['standard_name'] = f'arithmetic_mean_of_{sn}'
        else:
            attrs['long_name'] = f'Duration-weighted mean of {sn}'
        result = ds.parent.create_dataset(mean_name, data=mean, attrs=attrs)
        counts = ds.parent.create_dataset(f'{mean_name}_valid_count', data=count,
                                          attrs={'units': '', 'long_name': 'Number of valid samples'})
        for out_axis, in_axis in enumerate(i for i in range(ds.ndim) if i != axis):
            for scale in ds.dims[in_axis].values():
                result.dims[out_axis].attach_scale(scale)
                counts.dims[out_axis].attach_scale(scale)


PROCESSORS = {'compute_time_averages': compute_time_averages}
