import enum


class PIV_METHOD(enum.Enum):
    """Available PIV Methods"""
    single_pass = 0
    multi_pass = 1
    multi_grid = 2


class PIV_PeakFitMethod(enum.Enum):
    """Available PIV Peak Fit Methods"""
    Gauss3pt = 0  # openpiv has it, too
    Gauss3x3 = 1
    Parabolic = 2  # openpiv has it, too
    Centroid = 3  # openpiv has it, too
    Whittaker = 4
    NonLinGauss = 5
    FitBlob = 6
    FitRect = 7
    CentroidBinarized = 6
