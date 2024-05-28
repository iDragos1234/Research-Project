from enum import Enum
from functools import cached_property
import re


N_BONEFINDER_POINTS = 160


class ExtEnum(Enum):
    @classmethod
    def values(self):
        return list(item.value for item in self)
    
    @classmethod
    def items(self):
        return list((item.name, item.value) for item in self)


class HipSide(ExtEnum):
    RIGHT = 'right'
    LEFT  = 'left'


class Dataset(ExtEnum):
    CHECK = 'CHECK'
    OAI   = 'OAI'


class FilenamePattern(ExtEnum):
    CHECK = re.compile('(?P<subject_id>[0-9]+)_(?P<subject_visit>T[0-9]+)_APO.dcm')
    OAI   = re.compile('OAI-(?P<subject_id>[0-9]+)-(?P<subject_visit>V[0-9]+)-[0-9]+.dcm')


class HipSideOffset(ExtEnum):
    RIGHT = 0
    LEFT  = 80


class HipBoneCurve(ExtEnum):
    PROXIMAL_FEMUR     = list(range(0, 35))
    GREATER_TROCHANTER = [6] + list(range(35, 40))
    POSTERIOR_WALL     = list(range(40, 45))
    ISCHIUM_AND_PUBIS  = list(range(44, 60))
    FORAMEN            = list(range(60, 67))
    ACETABULAR_ROOF    = list(range(67, 75))
    TEARDROP           = list(range(75, 80))


class HipBoneSubCurve(ExtEnum):
    FEMORAL_HEAD       = list(range(18, 28))
    SOURCIL            = list(range(70, 75))


class MaskLabel(ExtEnum):
    IGNORE      = 0
    BACKGROUND  = 1
    ACETABULUM  = 2
    FEMUR       = 3
    JOINT_SPACE = 4


class MaskLabelColorRGB(ExtEnum):
    IGNORE      = [0, 0, 0]
    BACKGROUND  = [1, 0, 0]
    ACETABULUM  = [1, 1, 0]
    FEMUR       = [1, 0, 1]
    JOINT_SPACE = [0, 0, 1]


class DicomAttributes(ExtEnum):
    PIXEL_SPACING              = 'PixelSpacing'
    IMAGER_PIXEL_SPACING       = 'ImagerPixelSpacing'
    PHOTOMETRIC_INTERPRETATION = 'PhotometricInterpretation'
    VOILUT_FUNCTION            = 'VOILUTFunction'


class PhotometricInterpretation(ExtEnum):
    MONOCHROME1 = 'MONOCHROME1'
    MONOCHROME2 = 'MONOCHROME2'


class VoilutFunction(ExtEnum):
    LINEAR = 'LINEAR'
