import pathlib

from h5rdmtoolbox import Convention

__this_dir__ = pathlib.Path(__file__).parent
RESOURCES_DIR = __this_dir__.parent / 'resources'
CONVENTION_FILENAME = RESOURCES_DIR / 'convention/standard_attributes.yaml'


def init_cv() -> Convention:
    """Initialize the convention from the YAML file."""
    if not CONVENTION_FILENAME.exists():
        raise ValueError(f'{CONVENTION_FILENAME} does not exist!')
    return Convention.from_yaml(CONVENTION_FILENAME, overwrite=True)


cv = init_cv()

if __name__ == '__main__':
    # force rewriting convention file
    cv = init_cv()
    import shutil

    shutil.rmtree(cv.filename.parent)
    cv = init_cv()
