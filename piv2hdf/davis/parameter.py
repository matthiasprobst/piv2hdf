import lvpyio as lv
import numpy as np

from piv2hdf.interface import PIVParameterInterface


class DavisParameterFile(PIVParameterInterface):
    __suffix__ = '.vc7'

    def __init__(self, filename):
        super().__init__(filename)
        self._read_file()

    def _read_file(self):
        self._set = lv.read_buffer(str(self.filename.resolve()))
        self._attributes = lv.read_buffer(str(self.filename.resolve()))[0].attributes
        self._piv_param_datasets["x_final_iw_size"]["data"] = self._attributes["InterrogationWindowSize"]
        self._piv_param_datasets["y_final_iw_size"]["data"] = self._attributes["InterrogationWindowSize"]
        for k, v in self._attributes.items():
            if isinstance(v, str):
                if "s" in v:
                    value, unit = v.split(" ", 1)
                    try:
                        self._piv_param_datasets[k] = dict(data=float(value), unit=unit)
                    except ValueError:
                        self._piv_param_datasets[k] = dict(data=value, unit=unit)
                else:
                    try:
                        value = float(v)
                    except ValueError:
                        value = v
                    self._piv_param_datasets[k] = dict(data=value)
            else:
                if isinstance(v, np.ndarray):
                    self._piv_param_datasets[k] = dict(data=v[0][0])
                else:
                    self._piv_param_datasets[k] = dict(data=v)

    @classmethod
    def from_dir(cls, directory_path):
        raise NotImplementedError("DavisParameterFile should be instantiated with a filename")
