import numpy as np


N_POINTS = 160

SIDES = {
    'right': 0, 
    'left':  80,
}

CURVES = {
    'proximal femur':     list(range(0, 35)),
    'greater trochanter': [6] + list(range(35, 40)),
    'posterior wall':     list(range(40, 45)),
    'ischium and pubis':  list(range(44, 60)),
    'foramen':            list(range(60, 67)),
    'acetabular roof':    list(range(67, 75)),
    'something':          list(range(75, 80))
}

SUB_CURVES = {
    'femoral head':       list(range(18, 28)),
    'sourcil':            list(range(70, 75)),
}

LABELS = {
    'ignore':      0,
    'background':  1,
    'acetabulum':  2,
    'femur':       3,
    'joint space': 4,
}

COLORS = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
])
