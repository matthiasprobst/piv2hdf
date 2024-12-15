"""
U0, V0, U1, V1, U2, V2, U3, V3, ACTIVE_CHOICE, ENABLED, TS:Peak ratio, MASK, TS:Correlation value, TS:IntWin angle,
 TS:IntWin factor X, TS:IntWin factor Y, TS:IntWin type, TS:Particle size, TS:Uncertainty V, TS:Uncertainty Vx, TS:Uncertainty Vy
"""

from .file import set_to_hdf, File

standard_name_translation = {
    'x': 'x_coordinate',
    'y': 'y_coordinate',
    'z': 'z_coordinate',
    'U0': 'x_velocity',
    'V0': 'y_velocity',
    'W0': 'z_velocity', }
