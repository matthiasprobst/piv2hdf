# """providing the save method that is not properly done in openPIV, at least until now (version 0.25.0)"""
#
# import numpy as np
# import pathlib
# from openpiv import filters, \
#     validation
# from typing import Dict, Union
#
#
# def outlier_detection(u, v,
#                       sig2noise,
#                       s2n_threshold=1.0,
#                       ulim=None,
#                       vlim=None,
#                       replace_outliers_kwargs: Union[Dict, bool] = None):
#     mask_s2n = validation.sig2noise_val(sig2noise,
#                                         threshold=s2n_threshold)  # only masks points where s2n is higher than threshold
#     if ulim is not None:
#         assert isinstance(vlim, (list, tuple))
#         assert len(ulim) == 2
#     if vlim is not None:
#         assert isinstance(vlim, (list, tuple))
#         assert len(ulim) == 2
#     if ulim is not None:
#         mask_vel_th_threshold = validation.global_val(u, v, ulim, vlim)
#         mask_outlier = mask_s2n | mask_vel_th_threshold
#     else:
#         mask_outlier = mask_s2n
#
#     # replace outliers in velocity field:
#     if replace_outliers_kwargs is not False:
#         _replace_outliers_kwargs = {'method': 'localmean', 'max_iter': 10, 'kernel_size': 5}
#         if replace_outliers_kwargs is not None:
#             _replace_outliers_kwargs.update(replace_outliers_kwargs)
#         u, v = filters.replace_outliers(u.copy(), v.copy(),
#                                         mask_outlier,
#                                         **_replace_outliers_kwargs)
#         interpolated = mask_outlier
#         disabled = None
#     else:
#         disabled = mask_outlier
#         interpolated = None
#
#     # flags =
#     # ACTIVE = 1  # not manipulated
#     # DISABLED = 2
#     # FILTERED = 4
#     # INTERPOLATED = 8
#     # REPLACED = 16
#     # MANUALEDIT = 32
#     # MASKED = 64
#
#     FLAGS = {'ACTIVE': 1,  # not manipulated
#              'DISABLED': 2,
#              'FILTERED': 4,
#              'INTERPOLATED': 8,
#              'REPLACED': 16,
#              'MANUALEDIT': 32,
#              'MASKED': 64}
#
#     mask = np.ones_like(u, dtype=int)
#     mask_dict = {  # 'ACTIVE': np.ones_like(x),
#         'DISABLED': disabled,
#         'FILTERED': None,
#         'INTERPOLATED': interpolated,
#         'REPLACED': None,
#         'MANUALEDIT': None,
#         'MASKED': None}
#
#     for k, m in mask_dict.items():
#         if m is not None:
#             mask[m] |= FLAGS[k]
#
#     return mask, u, v
#
#
# def save(
#         filename: Union[pathlib.Path, str],
#         x: np.ndarray,
#         y: np.ndarray,
#         u: np.ndarray,
#         v: np.ndarray,
#         *,
#         additional_data_fields: Dict,
#         fmt: str = "%.4e",
#         delimiter: str = "\t",
# ) -> None:
#     """Save flow field to an ascii file. This method in openpiv version 0.25.0
#     is not able to store other fields than x, y, u, v. This method is able to
#     store additional fields in the same file by accepting `additional_data_fields` (dict).
#
#     Parameters
#     ----------
#     filename : string
#         the path of the file where to save the flow field
#
#     x : 2d np.ndarray
#         a two dimensional array containing the x coordinates of the
#         interrogation window centers, in pixels.
#
#     y : 2d np.ndarray
#         a two dimensional array containing the y coordinates of the
#         interrogation window centers, in pixels.
#
#     u : 2d np.ndarray
#         a two dimensional array containing the u velocity components,
#         in pixels/seconds.
#
#     v : 2d np.ndarray
#         a two dimensional array containing the v velocity components,
#         in pixels/seconds.
#
#     additional_data_fields: Dict
#         Additional data arrays to save wiht same shape as x, y, u, v
#
#
#     fmt : string
#         a format string. See documentation of numpy.savetxt
#         for more details.
#
#     delimiter : string
#         character separating columns
#
#     Examples
#     --------
#
#     >>> piv2hdf.tools.save('field_001.txt', x, y, u, v, *,
#     >>>                    additional_data_fields={'mask': mask, 'flag': flag},  fmt='%6.3f',
#     >>>                    delimiter='\t')
#
#     """
#     base_shape = x.shape
#     for da in additional_data_fields.values():
#         if da.shape != base_shape:
#             raise ValueError(f"Shape of data array {da} is not equal to base shape {base_shape}")
#
#     if isinstance(u, np.ma.MaskedArray):
#         u = u.filled(0.)
#         v = v.filled(0.)
#
#     # build output array
#     out = np.vstack([m.flatten() for m in [x, y, u, v, *additional_data_fields.values()]])
#
#     header = f'{delimiter}'.join(['x', 'y', 'u', 'v', *additional_data_fields.keys()])
#
#     # header = "x"+ delimiter+ "y"+ delimiter+ "u"+ delimiter+ "v"+ delimiter+ "flags"+ delimiter+ "mask"
#     # save data to file.
#     np.savetxt(
#         filename,
#         out.T,
#         fmt=fmt,
#         delimiter=delimiter,
#         header=header,
#     )
