import re


N_BONEFINDER_POINTS = 160


class MyEnum:
    def items():
        ...


class HipSide(MyEnum):
    RIGHT = 'right'
    LEFT  = 'left'

    def items():
        return [HipSide.RIGHT, HipSide.LEFT]


class Dataset(MyEnum):
    CHECK = 'CHECK'
    OAI   = 'OAI'

    def items():
        return [Dataset.CHECK, Dataset.OAI]


class FilenameRegex(MyEnum):
    CHECK = re.compile('(?P<subject_id>[0-9]+)_(?P<subject_visit>T[0-9]+)_APO.dcm')
    OAI   = re.compile('OAI-(?P<subject_id>[0-9]+)-(?P<subject_visit>V[0-9]+)-[0-9]+.dcm')


class HipSideOffset(MyEnum):
    RIGHT = (HipSide.RIGHT, 0)
    LEFT  = (HipSide.LEFT, 80)

    def items():
        return [HipSideOffset.RIGHT, HipSideOffset.LEFT]


class HipBoneCurve(MyEnum):
    PROXIMAL_FEMUR     = ('proximal femur',     list(range(0, 35)))
    GREATER_TROCHANTER = ('greater trochanter', [6] + list(range(35, 40)))
    POSTERIOR_WALL     = ('posterior wall',     list(range(40, 45)))
    ISCHIUM_AND_PUBIS  = ('ischium and pubis',  list(range(44, 60)))
    FORAMEN            = ('foramen',            list(range(60, 67)))
    ACETABULAR_ROOF    = ('acetabular roof',    list(range(67, 75)))
    TEARDROP           = ('teardrop',           list(range(75, 80)))

    def items():
        return [
            HipBoneCurve.PROXIMAL_FEMUR,
            HipBoneCurve.GREATER_TROCHANTER,
            HipBoneCurve.POSTERIOR_WALL,
            HipBoneCurve.ISCHIUM_AND_PUBIS,
            HipBoneCurve.FORAMEN,
            HipBoneCurve.ACETABULAR_ROOF,
            HipBoneCurve.TEARDROP,
        ]


class HipBoneSubCurve(MyEnum):
    FEMORAL_HEAD = ('femoral head', list(range(18, 28)))
    SOURCIL      = ('sourcil',      list(range(70, 75)))

    def items():
        return [
            HipBoneSubCurve.FEMORAL_HEAD,
            HipBoneSubCurve.SOURCIL,
        ]


class MaskLabel:
    IGNORE      = 0
    BACKGROUND  = 1
    ACETABULUM  = 2
    FEMUR       = 3
    JOINT_SPACE = 4


class DicomAttributes(MyEnum):
    PIXEL_SPACING              = 'PixelSpacing'
    IMAGER_PIXEL_SPACING       = 'ImagerPixelSpacing'
    PHOTOMETRIC_INTERPRETATION = 'PhotometricInterpretation'
    VOILUT_FUNCTION            = 'VOILUTFunction'


class PhotometricInterpretation(MyEnum):
    MONOCHROME1 = 'MONOCHROME1'
    MONOCHROME2 = 'MONOCHROME2'


class VoilutFunction(MyEnum):
    LINEAR = 'LINEAR'
