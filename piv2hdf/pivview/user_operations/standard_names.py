import pathlib

import h5rdmtoolbox as h5tbx
from ssnolib.namespace import SSNO

from piv2hdf.interface import PIV_PARAMETER_GRP_NAME, PIV_PARAMETER_ATTRS_NAME
from piv2hdf.piv_params import PIV_PeakFitMethod, PIV_METHOD
from piv2hdf.utils import read_translation_yaml_file

__this_file__ = pathlib.Path(__file__).resolve()

_RESOURCES = __this_file__.parent / '../../resources'
PIVVIEW_TRANSLATION = read_translation_yaml_file(_RESOURCES / 'pivview/pivview_translation.yaml')


class AddStandardNameOperation:

    def __call__(self, h5: h5tbx.File):
        """pivview post function. this is specific for a convention"""
        for name, ds in h5.items():
            if name in PIVVIEW_TRANSLATION:
                ds.attrs['standard_name', SSNO.standardName] = PIVVIEW_TRANSLATION[name]

        def _update_fields(grp: h5tbx.Group):
            piv_params = grp.attrs.get(PIV_PARAMETER_ATTRS_NAME, None)
            if piv_params is None:
                raise ValueError(f'No {PIV_PARAMETER_ATTRS_NAME} found in group {grp.name}')
            grp.attrs['piv_method'] = PIV_METHOD(
                piv_params['PIV processing parameters']['view0_piv_eval_method']).name
            grp.attrs['piv_peak_method'] = PIV_PeakFitMethod(
                piv_params['PIV processing parameters']['view0_piv_eval_peakfit_type']).name

            # for ds in grp.find({'$basename': {'$regex': 'planetime_[0-9]+'}}):
            #     ds.attrs['standard_name'] = pivview_translation['reltime']

        if PIV_PARAMETER_GRP_NAME not in h5:
            for param_grp in h5.find({'$basename': PIV_PARAMETER_GRP_NAME}, recursive=True, objfilter='group'):
                _update_fields(h5[param_grp.name])
            return
        return _update_fields(h5[PIV_PARAMETER_GRP_NAME])

        # # add `final_interrogation_window_size` to group "piv_parameters"
        # piv_parameter_groups = h5.find({'$basename': 'piv_parameters'})
        # if len(piv_parameter_groups) == 0:
        #     piv_parameter_groups = [h5.create_group('piv_parameters'),]
        #     # raise ValueError('No group named "piv_parameters" found in h5 file')
        # is_root_group = piv_parameter_groups[0].parent.name == '/'
        # if not is_root_group:
        #     piv_params = [_g.attrs.piv_parameters for _g in piv_parameter_groups]
        #     common_piv_params = _find_common_entries(piv_params)
        #     h5.create_group(PIV_PARAMETER_GRP_NAME)
        #     h5[PIV_PARAMETER_GRP_NAME].attrs['piv_parameters'] = common_piv_params
        #     for _g in piv_parameter_groups:
        #         _g.attrs['piv_parameters'] = {k: v for k, v in _g.attrs.piv_parameters.items() if k not in common_piv_params}
        #
        # piv_param_grp = h5[PIV_PARAMETER_GRP_NAME]
        #
        # piv_parameters = h5.attrs.piv_parameters
        # piv_proc_param = piv_parameters['PIV processing parameters']
        #
        # h5.attrs['piv_method'] = PIV_METHOD(int(piv_proc_param['view0_piv_eval_method'])).name
        #
        # if piv_proc_param['view0_piv_eval_method'] in (0, 1, 2):  # SinglePass and MultiPass
        #     x_final_iw_size, y_final_iw_size, _ = piv_proc_param['view0_piv_eval_samplesize']
        #     x_final_iw_overlap_size, y_final_iw_overlap_size, _ = piv_proc_param['view0_piv_eval_samplestep']
        # else:
        #     raise ValueError(f'Unknown PIV evaluation method: {piv_proc_param["view0_piv_eval_method"]}')
        # write_final_interrogation_window_size_to_h5_group(
        #     piv_param_grp, (x_final_iw_size, y_final_iw_size)
        # )
        # write_final_interrogation_overlap_size_to_h5_group(
        #     piv_param_grp,
        #     (x_final_iw_overlap_size, y_final_iw_overlap_size)
        # )


add_standard_name_operation = AddStandardNameOperation()
