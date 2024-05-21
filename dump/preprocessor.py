import matplotlib.pyplot as plt
import matplotlib.patches as patches
import highdicom as hd
import pydicom as pd
from pydicom.sr.codedict import codes
import numpy as np
import circle_fit
import skimage.draw
import imageio


# define the curves: right first, then left
SIDES = { 'right': 0, 'left': 80 }
CURVES = {
    'proximal femur':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
    'greater trochanter': [6, 35, 36, 37, 38, 39],
    'posterior wall':     [40, 41, 42, 43, 44],
    'ischium and pubis':  [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    'foramen':            [60, 61, 62, 63, 64, 65, 66],
    'acetabular roof':    [67, 68, 69, 70, 71, 72, 73, 74],
}
SUB_CURVES = {
    'femoral head':       [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    'sourcil':            [70, 71, 72, 73, 74],
}


class Preprocessor:

    image_file_base: str
    image_file_dicom: str
    image_file_points: str
    image_file_segmentation: str

    target_pixel_spacing: float

    def __init__(self, image_file_base, target_pixel_spacing=None):
        # Input file
        self.image_file_base = image_file_base
        self.image_file_dicom = f'{self.image_file_base}.dcm'
        self.image_file_points = f'{self.image_file_dicom}.pts'

        # Output file
        self.image_file_segmentation = f'{self.image_file_base}--segmentation.dcm'

        # set to mm/pixel, or None to disable resampling
        self.target_pixel_spacing = target_pixel_spacing
        return
    
    def load_data(self):
        self.img = pd.dcmread(self.image_file_dicom)
        return self
    
    def load_points(self):
        """
        Load points from BoneFinder.
        Coordinates are defined in mm.
        """
        with open(self.image_file_points, 'r') as f:
            assert f.readline().strip().startswith('version: ')
            n_points = int(f.readline().strip().split(' ')[1])
            assert f.readline().strip() == '{'
            self.points = [[float(x) for x in f.readline().strip().split(' ')] for _ in range(n_points)]
            self.points = np.array(self.points)
            assert f.readline().strip() == '}'
        return self
    
    def get_source_pixel_spacing(self):
        """
        Extract pixel spacing (mm/pixel) from the DICOM headers
        """
        self.source_pixel_spacing = self.img.get('PixelSpacing') or self.img.get('ImagerPixelSpacing')
        assert self.source_pixel_spacing is not None, 'No pixel spacing found'
        assert self.source_pixel_spacing[0] == self.source_pixel_spacing[1], 'Assymetric pixel spacing is untested'
        self.pixel_spacing = self.source_pixel_spacing
        return self
    
    def resample_to_target_resolution(self):
        """
        Resample image to the required resolution
        """
        if self.target_pixel_spacing is None:
            self.img_pixels = self.img.pixel_array
        else:
            scale_factor = self.pixel_spacing[0] / self.target_pixel_spacing
            self.img_pixels = skimage.transform.rescale(self.img.pixel_array, scale_factor)
            self.pixel_spacing = [self.target_pixel_spacing, self.target_pixel_spacing]
        return self
    
    def check_photometric_interpretation(self):
        """
        Are the intensities stored as MONOCHROME2 (white=max, black=min) or as MONOCHROME1 (white=min, black=max)?
        """
        self.photometric_interpretation = self.img.get('PhotometricInterpretation')
        if self.photometric_interpretation == 'MONOCHROME1':
            """
            If photometric interpretation is MONOCHROME1, invert intensities
            """
            self.img_pixels = np.max(self.img_pixels) - self.img_pixels
        else:
            assert self.photometric_interpretation == 'MONOCHROME2',\
                f'{self.photometric_interpretation} not supported'
        return self
    
    def check_VOILUT_function(self):
        assert self.img.get('VOILUTFunction', 'LINEAR') == 'LINEAR', \
            'only supporting VOILUTFunction LINEAR'
        return self
    
    def plot_image(self):
        plt.imshow(self.img_pixels, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_curves(self):
        # plot the image with superimposed curves
        plt.figure(figsize=(13, 8))
        plt.imshow(self.img_pixels, cmap='gray')
        plt.colorbar()

        # plot curves for right and left
        for side, offset in SIDES.items():
            for idx, (name, curve) in enumerate(CURVES.items()):
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][idx]
                plt.plot(*(self.points[np.array(curve) + offset] / self.pixel_spacing).transpose(),
                        marker='o', color=color,
                        label=f'{side} {name}',)
        plt.title(self.image_file_base)
        plt.legend(title='Curves')
        plt.tight_layout()
        plt.show()
        return self
        