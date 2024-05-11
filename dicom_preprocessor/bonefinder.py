"""
Code adapted from `example-preprocessing-code/bonefinder.py`.
"""
import circle_fit
import numpy as np
from functools import cached_property


# define the curves: right first, then left
SIDES = { 'right': 0, 'left': 80 }
NUM_POINTS = 160
CURVES = {
    'proximal femur':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 73, 10, 74, 11, 75,
                           12, 76, 77, 13, 78, 14, 15, 16, 17, 18, 19, 20, 21,
                           22, 23, 24, 25, 26, 27, 28],
    'greater trochanter': [6, 29, 30, 31, 32, 33],
    'posterior wall':     [34, 35, 36, 37, 38],
    'ischium and pubis':  [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                           51, 52, 53],
    'foramen':            [54, 55, 56, 57, 58, 59, 60],
    'acetabular roof':    [61, 62, 63, 79, 64, 65, 66, 67],
}
SUB_CURVES = {
    'femoral head':       [13, 78, 14, 15, 16, 17, 18, 19, 20, 21],
    'sourcil':            [79, 64, 65, 66, 67],
}


class BonefinderPoints:
    def __init__(self, filename):
        self._load_points(filename)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx]

    def __str__(self):
        s = ['BonefinderPoints:']
        minx, miny, maxx, maxy = self.bounding_box
        s.append(f'  Bounds: [{minx}, {miny}] to [{maxx}, {maxy}]')
        s.append('  Circles:')
        for name, circle in self.circles.items():
            s.append(f'  - {name}: [{"  ".join([f"{k}: {v:.2f}" for k, v in circle.items()])}]')
        return '\n'.join(s)

    def _load_points(self, filename):
        # load points from BoneFinder
        # coordinates are defined in mm
        points = []
        with open(filename, 'r') as f:
            # skip until start of points: line with {
            line = f.readline()
            while line and line.strip() != '{':
                line = f.readline()
            points = []
            # read points until end: line with }
            line = f.readline()
            while line and line.strip() != '}':
                points.append([float(i) for i in line.strip().split(' ')])
                line = f.readline()
        self.points = np.array(points)
        assert self.points.shape == (NUM_POINTS, 2), \
               f'expected ({NUM_POINTS}, 2) coordinates, found {self.points.shape}'

    @cached_property
    def bounding_box(self):
        # retursn [min x, min y, max x, max y]
        return [*self.points.min(axis=0), *self.points.max(axis=0)]

    @cached_property
    def circles(self):
        # fit circles to femoral head and sourcil
        # returns "left femoral head", "left sourcil", etc.
        circles = {}
        for side, offset in SIDES.items():
            for name, curve in SUB_CURVES.items():
                xc, yc, r, sigma = circle_fit.taubinSVD(self.points[np.array(curve) + offset])
                circles[f'{side} {name}'] = {'xc': xc, 'yc': yc, 'r': r, 'sigma': sigma}
        return circles

    @cached_property
    def curves(self):
        # return curve coordinates
        curves = {}
        for side, offset in SIDES.items():
            for name, curve in [*CURVES.items(), *SUB_CURVES.items()]:
                curves[f'{side} {name}'] = self.points[np.array(curve) + offset]
        return curves

    def circles_in_pixels(self, pixel_spacing):
        assert pixel_spacing[0] == pixel_spacing[1], 'expecting isotropic pixel spacing'
        return { name: { 'xc': circle['xc'] / pixel_spacing[0],
                         'yc': circle['yc'] / pixel_spacing[1],
                         'r': circle['r'] / pixel_spacing[0] }
                 for name, circle in self.circles.items() }

    def curves_in_pixels(self, pixel_spacing):
        return { name: curve / pixel_spacing
                 for name, curve in self.curves.items() }
