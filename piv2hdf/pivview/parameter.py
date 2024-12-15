import ast
import configparser
import pathlib
from ast import literal_eval

import h5py
import pint
from netCDF4 import Dataset as ncDataset

from ..interface import PIVParameterInterface


def is_nc_file(nc_filename: pathlib.Path):
    """try to open a nc file and returns success"""
    try:
        with ncDataset(nc_filename, 'r') as nc_rootgrp:
            _ = nc_rootgrp.dimensions
            return True
    except OSError:
        return False


def eval_piv_param(value):
    try:
        return literal_eval(value)
    except Exception:
        # print(f'Could not evaluate value: {value}. Returning as string.')
        return value


class PIVviewParamFile(PIVParameterInterface):
    """Parameter file interface for PIVview"""

    __suffix__ = '.par'

    def __init__(self, filename):
        super().__init__(filename)
        self._read_file()

    def _read_file(self):
        _cfg = configparser.ConfigParser()
        _cfg.read(self.filename)
        self.param_dict = {}
        for s in _cfg.sections():
            self.param_dict[s.strip('-').strip(' ')] = {k: eval_piv_param(v) for k, v in dict(_cfg[s]).items()}
        # fill the rest of the parameters
        piv_proc_param = self.param_dict['PIV processing parameters']
        x_final_iw_size, y_final_iw_size, _ = piv_proc_param['view0_piv_eval_samplesize']
        self.piv_param_datasets['x_final_iw_size']['data'] = x_final_iw_size
        self.piv_param_datasets['y_final_iw_size']['data'] = y_final_iw_size
        x_final_iw_overlap_size, y_final_iw_overlap_size, _ = piv_proc_param['view0_piv_eval_samplestep']
        self.piv_param_datasets['x_final_iw_overlap_size']['data'] = x_final_iw_overlap_size
        self.piv_param_datasets['y_final_iw_overlap_size']['data'] = y_final_iw_overlap_size

        dt = self.param_dict['PIV conversion parameters']['view0_piv_conv_pulsedelay']
        dt_unit = self.param_dict['PIV conversion parameters']['view0_piv_conv_pulsedelayunits']
        self.piv_param_datasets['laser_pulse_delay']['data'] = dt
        self.piv_param_datasets['laser_pulse_delay']['units'] = dt_unit

        cf = self.param_dict['PIV conversion parameters']['view0_piv_conv_lengthconversion']
        cf_units = f"pixel/{self.param_dict['PIV conversion parameters']['view0_piv_conv_lengthconversionunits']}"
        cf_q = pint.Quantity(f'{cf} {cf_units}').to('pixel/m')
        self.piv_param_datasets['piv_scaling_factor']['data'] = cf_q.magnitude

    def save(self, filename: pathlib.Path):
        """Save to original file format"""
        _cfg = configparser.ConfigParser()
        _cfg.read_dict(self.param_dict)
        with open(filename, 'w') as f:
            self._cfg.write(f)

    def from_hdf(self, grp: h5py.Group) -> None:
        """Read from HDF group"""
        pass


class PivViewConfigFile(PIVviewParamFile):
    """Parameter/Config file for stereo PIVview files"""
    __suffix__ = '.cfg'

    def _read_file(self):
        _cfg = configparser.ConfigParser()
        _cfg.read(self.filename)
        self.param_dict = {}
        for s in _cfg.sections():
            dict_holder = {}
            for k, v in dict(_cfg[s]).items():
                try:
                    dict_holder[k] = ast.literal_eval(v)
                except SyntaxError:
                    dict_holder[k] = v
            self.param_dict[s.strip('-').strip(' ')] = dict_holder

        piv_proc_param = self.param_dict['PIV processing parameters']
        x_final_iw_size, y_final_iw_size, _ = piv_proc_param['view0_piv_eval_samplesize']
        self.piv_param_datasets['x_final_iw_size']['data'] = x_final_iw_size
        self.piv_param_datasets['y_final_iw_size']['data'] = y_final_iw_size
        x_final_iw_overlap_size, y_final_iw_overlap_size, _ = piv_proc_param['view0_piv_eval_samplestep']
        self.piv_param_datasets['x_final_iw_overlap_size']['data'] = x_final_iw_overlap_size
        self.piv_param_datasets['y_final_iw_overlap_size']['data'] = y_final_iw_overlap_size

        dt = self.param_dict['PIV conversion parameters']['view0_piv_conv_pulsedelay']
        dt_unit = self.param_dict['PIV conversion parameters']['view0_piv_conv_pulsedelayunits']
        self.piv_param_datasets['laser_pulse_delay']['data'] = dt
        self.piv_param_datasets['laser_pulse_delay']['units'] = dt_unit

        cf = self.param_dict['PIV conversion parameters']['view0_piv_conv_lengthconversion']
        cf_units = f"pixel/{self.param_dict['PIV conversion parameters']['view0_piv_conv_lengthconversionunits']}"
        cf_q = pint.Quantity(f'{cf} {cf_units}').to('pixel/m')
        self.piv_param_datasets['piv_scaling_factor']['data'] = cf_q.magnitude

        assert len(self.param_dict) != 0, f'Could not read config file: {self.filename}'
