"""PIV FLAGS"""
import enum


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
