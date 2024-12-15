from ast import literal_eval
from configparser import ConfigParser

import h5py

from .. import utils
from ..interface import PIVParameterInterface


class OpenPIVParameterFile(PIVParameterInterface):
    """open PIV parameter interface class"""

    __suffix__ = '.par'

    def __init__(self, filename):
        super().__init__(filename)
        if filename is not None:
            _cfg = ConfigParser()
            _cfg.read(filename)
            self.param_dict = {}
            if len(list(_cfg.sections())) == 1:
                for s in _cfg.sections():
                    self.param_dict = dict(_cfg[s])
            else:
                for s in _cfg.sections():
                    self.param_dict[s.strip('-').strip(' ')] = dict(_cfg[s])
            fwin = self._get_final_winsize()
            self.piv_param_datasets['x_final_iw_size']['data'] = fwin[0]
            self.piv_param_datasets['y_final_iw_size']['data'] = fwin[1]
            fov = self._get_overlap()
            self.piv_param_datasets['x_final_iw_overlap_size']['data'] = fov[0]
            self.piv_param_datasets['y_final_iw_overlap_size']['data'] = fov[1]
            self.piv_param_datasets['laser_pulse_delay']['data'] = float(self.param_dict['dt'])
            self.piv_param_datasets['piv_scaling_factor']['data'] = float(self.param_dict['scaling_factor'])

    def _get_final_winsize(self):
        """figure out the final interrogation window size"""
        winsize_data = self.param_dict['windowsizes']

        if isinstance(winsize_data, str):
            fwin = literal_eval(winsize_data)[-1]  # the last is the final interrogation window size
        else:
            fwin = winsize_data[-1]
        if isinstance(fwin, (tuple, list)):
            if len(fwin) == 2:
                fwin = [fwin[0], fwin[1], 1]
            elif len(fwin) == 3:
                fwin = [fwin[0], fwin[1], fwin[2]]
        else:
            fwin = [fwin, fwin, 1]
        return fwin

    def _get_overlap(self):
        overlap_size = self.param_dict['overlap']
        if isinstance(overlap_size, str):
            fov = literal_eval(overlap_size)[-1]
        else:
            fov = overlap_size[-1]
        if isinstance(fov, int):
            return [fov, fov, 1]
        # if isinstance(fov, (tuple, list)):
        if len(fov) == 2:
            return [fov[0], fov[1], 1]
        # elif len(fov) == 3:
        return [fov[0], fov[1], fov[2]]

    def save(self, filename):
        """Save parameter dictionary to file"""
        with open(filename, 'w') as f:
            f.write(f'[openpiv parameter]')
            for k, v in self.param_dict.items():
                line = f'\n{k}={v}'
                f.write(line.replace('%', '%%'))

    @staticmethod
    def from_windef(settings: "OpenPIVSetting"):
        """Initialize OpenPIVParameterFile from openpiv instance of class windef.Settings"""
        _param_dict = settings.__dict__.copy()
        try:
            _param_dict.pop('_FrozenClass__isfrozen')
        except KeyError:
            pass
        o = OpenPIVParameterFile(None)
        o.param_dict = _param_dict
        return o

    if True:
        def to_hdf(self, grp: h5py.Group):
            """Convert to HDF group"""

            def _to_grp(_dict, _grp):
                for k, v in _dict.items():
                    if isinstance(v, dict):
                        _grp = _to_grp(v, _grp.create_group(k))
                    else:
                        if k == 'windowsizes':
                            if isinstance(v, str):
                                fwin = literal_eval(v)[-1]  # the last is the final interrogation window size
                            else:
                                fwin = v[-1]
                            if isinstance(fwin, (tuple, list)):
                                if len(fwin) == 2:
                                    fwin = [fwin[0], fwin[1], 1]
                                elif len(fwin) == 3:
                                    fwin = [fwin[0], fwin[1], fwin[2]]
                            else:
                                fwin = [fwin, fwin, 1]

                            utils.write_final_interrogation_window_size_to_h5_group(_grp, fwin)
                        elif k == 'overlap':
                            if isinstance(v, str):
                                fov = literal_eval(v)[-1]
                            else:
                                fov = v[-1]
                            if isinstance(fov, (tuple, list)):
                                if len(fov) == 2:
                                    fov = [fov[0], fov[1], 1]
                                elif len(fov) == 3:
                                    fov = [fov[0], fov[1], fov[2]]
                            utils.write_final_interrogation_overlap_size_to_h5_group(_grp, fov)
                        elif k == 'dt':
                            ds = _grp.create_dataset(k, data=float(v))
                            ds.attrs.update({'units': 's', 'standard_name': 'laser_pulse_delay'})
                        elif k == 'scaling_factor':
                            ds = _grp.create_dataset(k, data=float(v))
                            ds.attrs.update({'units': 'pixel/m', 'standard_name': 'piv_scaling_factor'})
                        elif v is None:
                            _grp.attrs[k] = 'None'
                        else:
                            _grp.attrs[k] = v
                return _grp

            return _to_grp(self.param_dict, grp)
