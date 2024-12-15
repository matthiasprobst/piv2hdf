import h5py
import h5rdmtoolbox as h5tbx


def compute_time_averages(h5, **kwargs):
    """Compute time averages of the velocity field.
    This is only reasonable if one dimension scale is relative time!
    """
    # print('computing time averages is called!')
    if not isinstance(h5, h5py.File):
        with h5tbx.File(h5, 'r+') as h5:
            return compute_time_averages(h5, **kwargs)

    for sn in ('x_velocity', 'y_velocity', 'z_velocity'):
        lazy_u_ds = h5.find_one({'standard_name': sn})
        if lazy_u_ds is None:
            # print(f'Could not find dataset with standard_name "{sn}"')
            continue

        u_ds = h5[lazy_u_ds.name]
        dim_ds_standard_names = [dim[0].attrs.get('standard_name', '') for dim in u_ds.dims]
        if 'relative_time' not in dim_ds_standard_names:
            return  # skip this post-processing function, there is no relative_time dimension attached!

        u_mean_ds = u_ds[()].mean(dim='reltime')

        mean_name = f'mean_{u_ds.basename}'
        if mean_name in h5:
            # print(f'Dataset with name "{mean_name}" already exists. Skipping...')
            continue
        u_ds.parent.create_dataset(mean_name,
                                   data=u_mean_ds,
                                   attrs={'units': u_ds.attrs['units'],
                                          'standard_name': f'arithmetic_mean_of_{sn}'})
