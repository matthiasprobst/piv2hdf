import pathlib
import shutil
from typing import Tuple, List

from . import user
from .utils import generate_temporary_directory

__this_dir__ = pathlib.Path(__file__).parent


class PIVview:
    """PIVview tutorial class"""

    @staticmethod
    def get_parameter_file() -> pathlib.Path:
        """Return pivview parameter file"""
        return user.TEST_DATA_DIRECTORY / 'pivview/piv_challenge1_E/piv_parameters.par'

    @staticmethod
    def get_plane_directory() -> pathlib.Path:
        """Return the path to the respective example PIV plane"""
        return user.TEST_DATA_DIRECTORY / 'pivview/piv_challenge1_E'

    @staticmethod
    def get_multiplane_directories() -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        """Copies the piv_challenge1_E data to three directories in the tmp directory
        Two planes have three nc files, one plane has 2 nc files only"""
        try:
            from netCDF4 import Dataset as ncDataset
        except ImportError:
            raise ImportError('Package netCDF4 is not installed.')

        def _set_z_in_nc(nc_filename, z_val):
            with ncDataset(nc_filename, 'r+') as nc:
                nc.setncattr('origin_offset_z', z_val)
                for k, v in nc.variables.items():
                    if 'coord_min' in nc[k].ncattrs():
                        coord_min = nc[k].getncattr('coord_min')
                        coord_min[-1] = z_val
                        nc[k].setncattr('coord_min', coord_min)
                        coord_max = nc[k].getncattr('coord_max')
                        coord_max[-1] = z_val
                        nc[k].setncattr('coord_max', coord_max)

        src_dir = user.TEST_DATA_DIRECTORY / 'pivview/piv_challenge1_E'
        nc_files = sorted(src_dir.glob('*[0-9].nc'))
        if len(nc_files) == 0:
            raise FileNotFoundError('No nc files found in piv_challenge1_E test data directory. Contact the developer.')

        plane0 = generate_temporary_directory(prefix='mplane/')
        _ = shutil.copy2(src_dir / 'piv_parameters.par', plane0.joinpath('piv_parameter.par'))
        dst = shutil.copy2(nc_files[0], plane0.joinpath('f0.nc'))
        _set_z_in_nc(dst, -5.)
        dst = shutil.copy2(nc_files[1], plane0.joinpath('f1.nc'))
        _set_z_in_nc(dst, -5.)
        dst = shutil.copy2(nc_files[2], plane0.joinpath('f2.nc'))
        _set_z_in_nc(dst, -5.)

        plane1 = generate_temporary_directory(prefix='mplane/')
        _ = shutil.copy2(src_dir / 'piv_parameters.par', plane1.joinpath('piv_parameter.par'))
        dst = shutil.copy2(nc_files[3], plane1.joinpath('f0.nc'))
        _set_z_in_nc(dst, 0.)
        dst = shutil.copy2(nc_files[4], plane1.joinpath('f1.nc'))
        _set_z_in_nc(dst, 0.)
        dst = shutil.copy2(nc_files[5], plane1.joinpath('f2.nc'))
        _set_z_in_nc(dst, 0.)

        plane2 = generate_temporary_directory(prefix='mplane/')
        _ = shutil.copy2(src_dir / 'piv_parameters.par', plane2.joinpath('piv_parameter.par'))
        dst = shutil.copy2(nc_files[6], plane2.joinpath('f0.nc'))
        _set_z_in_nc(dst, 10.)
        dst = shutil.copy2(nc_files[7], plane2.joinpath('f1.nc'))
        _set_z_in_nc(dst, 10.)

        return plane0, plane1, plane2

    @staticmethod
    def get_snapshot_nc_files():
        """Return a list of sorted nc files"""
        files = sorted((user.TEST_DATA_DIRECTORY / "pivview/piv_challenge1_E").glob('E00A*.nc'))
        if len(files) == 0:
            raise FileNotFoundError('No nc files found in piv_challenge1_E test data directory. Contact the developer.')
        return files

    @staticmethod
    def get_avg_file() -> pathlib.Path:
        """Return the path to the avg.dat file"""
        file = user.TEST_DATA_DIRECTORY / "pivview/piv_challenge1_E/avg.dat"
        if not file.exists():
            raise FileNotFoundError(
                'No avg.dat file found in piv_challenge1_E test data directory. Contact the developer.')
        return file

    @staticmethod
    def get_rms_file() -> pathlib.Path:
        """Return the path to the rms.dat file"""
        file = user.TEST_DATA_DIRECTORY / "pivview/piv_challenge1_E/rms.dat"
        if not file.exists():
            raise FileNotFoundError(
                'No rms.dat file found in piv_challenge1_E test data directory. Contact the developer.')
        return file

    @staticmethod
    def get_reyn_file() -> pathlib.Path:
        """Return the path to the reyn.dat file"""
        file = user.TEST_DATA_DIRECTORY / "pivview/piv_challenge1_E/reyn.dat"
        if not file.exists():
            raise FileNotFoundError(
                'No reyn.dat file found in piv_challenge1_E test data directory. Contact the developer.')
        return file


class OpenPIV:
    """OpenPIV tutorial class"""

    @staticmethod
    def get_snapshot_txt_file() -> pathlib.Path:
        """Return snapshot piv result from ILA vortex"""
        return user.TEST_DATA_DIRECTORY / "openpiv" / "vortex.txt"

    @staticmethod
    def get_parameter_file() -> pathlib.Path:
        """Return openpiv parameters as file"""
        return user.TEST_DATA_DIRECTORY / "openpiv" / "openpiv.par"

    @staticmethod
    def get_plane_directory() -> pathlib.Path:
        """Return the path to the respective example PIV plane"""
        plane = generate_temporary_directory()
        snapshot_filename = OpenPIV.get_snapshot_txt_file()
        parameter_filename = OpenPIV.get_parameter_file()
        shutil.copy2(snapshot_filename, plane / 'vortex1.txt')
        shutil.copy2(snapshot_filename, plane / 'vortex2.txt')
        shutil.copy2(snapshot_filename, plane / 'vortex3.txt')
        shutil.copy2(snapshot_filename, plane / 'vortex4.txt')
        shutil.copy2(parameter_filename, plane / parameter_filename.name)
        return plane

    @staticmethod
    def get_multiplane_directories() -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        case = generate_temporary_directory()
        plane1 = case / 'p1'
        plane2 = case / 'p2'
        plane3 = case / 'p3'
        plane1.mkdir()
        plane2.mkdir()
        plane3.mkdir()
        snapshot_filename = OpenPIV.get_snapshot_txt_file()
        parameter_filename = OpenPIV.get_parameter_file()
        shutil.copy2(snapshot_filename, plane1 / 'vortex1.txt')
        shutil.copy2(snapshot_filename, plane1 / 'vortex2.txt')
        shutil.copy2(snapshot_filename, plane1 / 'vortex3.txt')
        shutil.copy2(snapshot_filename, plane2 / 'vortex1.txt')
        shutil.copy2(snapshot_filename, plane2 / 'vortex2.txt')
        shutil.copy2(snapshot_filename, plane2 / 'vortex3.txt')
        shutil.copy2(snapshot_filename, plane3 / 'vortex1.txt')
        shutil.copy2(snapshot_filename, plane3 / 'vortex2.txt')
        shutil.copy2(parameter_filename, plane1 / parameter_filename.name)
        shutil.copy2(parameter_filename, plane2 / parameter_filename.name)
        shutil.copy2(parameter_filename, plane3 / parameter_filename.name)
        return plane1, plane2, plane3


class Davis:
    """LaVision Davis tutorial data class"""

    @staticmethod
    def get_set_file() -> pathlib.Path:
        """Return the path to the davis set file"""
        return user.TEST_DATA_DIRECTORY / 'davis/dtau04.0.set'

    @staticmethod
    def get_vc7_files() -> List[pathlib.Path]:
        """Return the path to the davis set file"""
        return sorted((user.TEST_DATA_DIRECTORY / 'davis/dtau04.0/').glob('*.vc7'))
