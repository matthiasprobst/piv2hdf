from .nc import PIVViewNcFile, PIVViewStereoNcFile
from .parameter import PIVviewParamFile

config = {
    'interpolation': False,
    'masking': 'slack',  # all disabled and masked data points are set to np.nan
    'z_source': 'coord_min',
    'datetime_str': '%Y-%m-%dT%H:%M:%SZ%z',
    'attrs_unit_name': 'units',
    'compression': 'gzip',
    'compression_opts': 5,
    'take_min_nt': True,  # False will fill datasets up with np.NA
    'nc_as_time_averages_source': False,
}
