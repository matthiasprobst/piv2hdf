"""PIV FLAGS"""
import enum
import numpy as np


# FLAGS = {'INACTIVE': 0,
#          'ACTIVE': 1,  # not manipulated
#          'DISABLED': 2,
#          'FILTERED': 4,
#          'INTERPOLATED': 8,
#          'REPLACED': 16,
#          'MANUALEDIT': 32,
#          'MASKED': 64}


class Flags(enum.Enum):
    """PIV validation flags. Adopted from PIVTec's PIVview."""
    INACTIVE = 0
    ACTIVE = 1
    MASKED = 2
    NORESULT = 4
    DISABLED = 8
    FILTERED = 16
    INTERPOLATED = 32
    REPLACED = 64
    MANUALEDIT = 128


# def get_flag_names(flag_value) -> List:
#     """Return a list of flag names for a given flag value"""
#     if flag_value == 0:
#         return ['INACTIVE']
#
#     flag_names = []
#
#     for flag in Flags:
#         if flag_value & flag.value:
#             flag_names.append(flag.name)
#
#     return flag_names


flag_translation_dict = {
    'openpiv': {
        0: Flags.ACTIVE,
        1: Flags.DISABLED,
        2: Flags.INTERPOLATED
    }
}


def valid_vectors(values, mode='slack'):
    """Select documented flag combinations; reject invalid and unknown bits.

    Strict accepts ACTIVE; sepeaks also accepts REPLACED; slack additionally
    accepts FILTERED, INTERPOLATED and MANUALEDIT. Inactive is always rejected.
    """
    accepted = {'strict': 1, 'sepeaks': 1 | 64, 'slack': 1 | 16 | 32 | 64 | 128}
    if mode not in accepted:
        raise ValueError(f'Unknown masking mode: {mode}')
    values = np.asarray(values, dtype=np.uint64)
    allowed = np.uint64(accepted[mode])
    return ((values & allowed) != 0) & ((values & ~allowed) == 0)
