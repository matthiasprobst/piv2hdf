import pathlib
from functools import lru_cache

from h5rdmtoolbox import Convention

__this_dir__ = pathlib.Path(__file__).parent
RESOURCES_DIR = __this_dir__.parent / 'resources'
CONVENTION_FILENAME = RESOURCES_DIR / 'convention/planar_piv_convention.yaml'


@lru_cache(maxsize=1)
def init_cv() -> Convention:
    """Initialize the convention from the YAML file."""
    if not CONVENTION_FILENAME.exists():
        raise ValueError(f'{CONVENTION_FILENAME} does not exist!')
    return Convention.from_yaml(CONVENTION_FILENAME, overwrite=True)


def __getattr__(name):
    if name == 'cv':
        return init_cv()
    raise AttributeError(name)
