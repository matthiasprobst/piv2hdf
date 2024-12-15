from typing import Dict

import h5rdmtoolbox as h5tbx

from piv2hdf import postproc

CONFIG = dict(
    time_unit='s',
    datetime_str='%Y-%m-%dT%H:%M:%SZ%z',
    attrs_unit_name='units',
    compression='gzip',
    compression_opts=5,
    convention='planar_piv',
    take_min_nt=True,  # False will fill datasets up with np.NA
    raise_layout_validation_error=True,
    validate_layout=True,
    dtypes={'x': 'float32',
            'y': 'float32',
            'z': 'float32',
            'reltime': 'float32',
            # ix, iy, iz, image_index are identified automatically (see utils.get_uint_type())!
            },
    postproc=['compute_time_averages',
              ],
    # post={'compute_significance': False,  # computes running mean, std and relative std
    #       'compute_velocity_magnitude': False,
    #       'compute_time_average': False,  # computes mean_u, mean_v, mean_vel_magnitude
    #       'compute_velocity_gradients': False,  # computes du/dx, du/dy, dv/dx, dv/dy
    #       'compute_vorticity': False},
)

PIV_ATTRS = dict()
_ORIG_PIV_ATTRS_KEYS = list(PIV_ATTRS.keys())


class _Setter:
    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._update(self.old)


class PIVAttrsSetter(_Setter):
    """Set the piv attributes. PIV attributes are written to the
    root group of the hdf file.

    Mandatory piv attributes are:
    - contact: str
        The orcid must be used, e.g. 'https://orcid.org/0000-0001-8729-0482'
    - piv_medium: str
        E.g. air, water, etc.
    """

    def __init__(self):
        self.old = {}

    def __call__(self, **kwargs):
        self.old = {}
        with h5tbx.use(get_config('convention')) as cv:
            print(f'Checking the PIV attributes based on convention currently set ({cv.name})...')
            root_standard_attributes = cv.properties.get(h5tbx.File, None)
            if root_standard_attributes is not None:
                invalids = {}
                for k, v in kwargs.items():
                    sa = root_standard_attributes.get(k, None)
                    if sa is not None:
                        try:
                            sa.validate(v, None)
                            # sa.validator.model_validate(dict(value=v))
                        except Exception as e:
                            if sa.default_value == sa.__class__.EMPTY:
                                # collect invalids
                                invalids[k] = (v, e)

                if invalids:
                    invalids_str = '\n - '.join([f'"{k}": {v[0]} due to "{v[1]}"' for k, v in invalids.items()])
                    raise ValueError(f'Based on the set convention "{cv.name}",'
                                     f' the following PIV parameters are invalid: {invalids_str}')

        self._update(kwargs)

    def _update(self, options_dict: Dict):
        """Update piv attributes."""
        PIV_ATTRS.update(options_dict)


set_pivattrs = PIVAttrsSetter()


def get_pivattrs() -> Dict:
    """Return the piv attributes."""
    return PIV_ATTRS


def reset_pivattrs():
    PIV_ATTRS.clear()
    return PIV_ATTRS


class set_config(_Setter):
    """Set the configuration parameters."""

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in CONFIG:
                raise KeyError(f'Not a configuration key: {k}')
            if k == 'postproc':
                if not isinstance(v, list):
                    raise TypeError(f'Expected a list for key "postproc", got {type(v)}')

                for pp_name in v:
                    if isinstance(pp_name, str):
                        if pp_name not in dir(postproc):
                            raise ValueError(f'Unknown postproc name: {pp_name}')
                    elif isinstance(pp_name, callable):
                        pass
                    else:
                        raise TypeError(f'Expected a string or callable for key "postproc", got {type(pp_name)}')

            self.old[k] = CONFIG[k]
        self._update(kwargs)

    def _update(self, options_dict: Dict):
        CONFIG.update(options_dict)


def get_config(key=None):
    """Return the configuration parameters."""
    if key is None:
        return CONFIG
    return CONFIG[key]
