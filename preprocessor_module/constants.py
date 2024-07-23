'''
Constant values used in the preprocessing phase.
'''
import re


'''
Fixed number of points contained in each BoneFinder points file.
'''
N_BONEFINDER_POINTS = 160


class MyEnum:
    def items() -> list:
        ...


'''
Hip sides can be either 'right' or 'left'.
'''
class HipSide(MyEnum):
    RIGHT = 'right'
    LEFT  = 'left'

    def items():
        return [ HipSide.RIGHT, HipSide.LEFT ]


'''
Names of supported datasets of X-ray images: CHECK and OAI cohorts.
'''
class Dataset(MyEnum):
    CHECK = 'CHECK'
    OAI   = 'OAI'

    def items():
        return [ Dataset.CHECK, Dataset.OAI ]


'''
Regular expressions for the DICOM filenames, for each of the CHECK and OAI datasets.
'''
class FilenameRegex(MyEnum):
    CHECK = re.compile('(?P<subject_id>[0-9]+)_(?P<subject_visit>T[0-9]+)_APO.dcm')
    OAI   = re.compile('OAI-(?P<subject_id>[0-9]+)-(?P<subject_visit>V[0-9]+)-[0-9]+.dcm')


'''
Offsets added to the BoneFinder points for each side of the hip.
'''
class HipSideOffset(MyEnum):
    RIGHT = (HipSide.RIGHT, 0)
    LEFT  = (HipSide.LEFT, 80)

    def items():
        return [ HipSideOffset.RIGHT, HipSideOffset.LEFT ]


'''
Group indices of BoneFinder points by the pelvic curve they belong to. 
'''
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


'''
Select indices of the BoneFinder points for the femoral head and acetabular roof curves.
'''
class HipBoneSubCurve(MyEnum):
    FEMORAL_HEAD    = ('femoral head',    list(range(18, 28)))
    ACETABULAR_ROOF = ('acetabular roof', list(range(70, 75)))

    def items():
        return [
            HipBoneSubCurve.FEMORAL_HEAD,
            HipBoneSubCurve.ACETABULAR_ROOF,
        ]


'''
Class labels expressed by indices of the submasks for each region in the combined mask. 
'''
class MaskLabel(MyEnum):
    BACKGROUND  = 0
    FEMUR_HEAD  = 1
    ACETABULUM  = 2
    JOINT_SPACE = 3

    def items():
        return [
            MaskLabel.BACKGROUND,
            MaskLabel.FEMUR_HEAD,
            MaskLabel.ACETABULUM,
            MaskLabel.JOINT_SPACE,
        ]


'''
Names of the metadata attributes of interest in preprocessed DICOM files.
'''
class DicomAttributes(MyEnum):
    PIXEL_SPACING              = 'PixelSpacing'
    IMAGER_PIXEL_SPACING       = 'ImagerPixelSpacing'
    PHOTOMETRIC_INTERPRETATION = 'PhotometricInterpretation'
    VOILUT_FUNCTION            = 'VOILUTFunction'

    def items():
        return [
            DicomAttributes.PIXEL_SPACING,
            DicomAttributes.IMAGER_PIXEL_SPACING,
            DicomAttributes.PHOTOMETRIC_INTERPRETATION,
            DicomAttributes.VOILUT_FUNCTION,
        ]


'''
Supported photometric interpretations (i.e., whether the grayscale intensities are inverted).
NOTE: the MONOCHROME1 represents the inverted MONOCHROME2 intensities.
'''
class PhotometricInterpretation(MyEnum):
    MONOCHROME1 = 'MONOCHROME1'
    MONOCHROME2 = 'MONOCHROME2'

    def items():
        return [
            PhotometricInterpretation.MONOCHROME1,
            PhotometricInterpretation.MONOCHROME2,
        ]


'''
The supported values for the VOILUT function attribute of the DICOM file metadata.
'''
class VoilutFunction(MyEnum):
    LINEAR = 'LINEAR'

    def items():
        return [ VoilutFunction.LINEAR, ]
