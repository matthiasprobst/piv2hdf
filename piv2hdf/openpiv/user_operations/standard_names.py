import pathlib

import h5rdmtoolbox as h5tbx

from piv2hdf.interface import PIV_PARAMETER_GRP_NAME
from piv2hdf.interface import UserDefinedHDF5Operation
from piv2hdf.piv_params import PIV_PeakFitMethod
from piv2hdf.utils import read_translation_yaml_file

__this_dir__ = pathlib.Path(__file__).parent
RESOURCES_DIR = (__this_dir__.parent / '../resources').resolve()
TRANSLATION_EXT_DICT = read_translation_yaml_file(RESOURCES_DIR / 'openpiv/openpiv_ext_translation.yaml')


class AddStandardNameOperation(UserDefinedHDF5Operation):

    def __call__(self, h5: h5tbx.File) -> None:
        """openpiv post function"""
        for name, ds in h5.items():
            if name in TRANSLATION_EXT_DICT:
                ds.attrs['standard_name'] = TRANSLATION_EXT_DICT[name]

        def _update_fields(grp):
            peak_method = grp.attrs['subpixel_method']
            if peak_method == 'gaussian':
                grp.attrs['piv_peak_method'] = PIV_PeakFitMethod(0).name

            # # piv_method
            # warnings.warn('piv_method is assumed to be multi grid but not determined!')
            # grp.attrs['piv_method'] = PIV_METHOD(2).name

        if PIV_PARAMETER_GRP_NAME not in h5:
            for param_grp in h5.find({'$basename': PIV_PARAMETER_GRP_NAME}, recursive=True):
                _update_fields(param_grp)
            return

        return _update_fields(h5[PIV_PARAMETER_GRP_NAME])


add_standard_name_operation = AddStandardNameOperation()
