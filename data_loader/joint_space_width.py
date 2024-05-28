import matplotlib.pyplot as plt
import numpy as np
import torch
import dtw


def joint_space_width(
    mask: torch.tensor, 
    pixel_spacing: float=None,
) -> tuple[float, np.array]:
    '''
    Compute the Joint Space Width (JSW) array and minJSW:
    1. identify the points corresponding to the borders of femur head and acetabulum,
    2. compute matrix of pairwise distances between the femur head and the acetabulum points,
    3. use the Dynamic Time Warping (DTW) algorithm to align the border points sequences,
    4. extract the JSW array from the DTW alignment,
    5. get the minJSW from the distance matrix.

    Parameters
    ----------
    mask (torch.tensor): The segmentation mask from which to extract the minJSW and JSW array

    pixel_spacing (float, optional): Spacing between pixels in mm; 
        used to scale pixel-wise distances to mm 
    
    Returns
    -------
    (tuple[float, numpy.array]): The minJSW and the JSW array
    
    Note: it may happen that the alignment algorithm does not include minJSW in JSW array.
    '''
    # Masks for the upper/lower margins of the borders
    femur_head_border_lower_margin = mask[0] * mask[2].roll(+1, 0)
    femur_head_border_upper_margin = mask[2] * mask[0].roll(-1, 0)
    acetabulum_border_upper_margin = mask[1] * mask[2].roll(-1, 0)
    acetabulum_border_lower_margin = mask[2] * mask[1].roll(+1, 0)

    # Points for the upper/lower margins of the borders
    femur_head_border_lower_margin_points = np.argwhere(femur_head_border_lower_margin)
    femur_head_border_upper_margin_points = np.argwhere(femur_head_border_upper_margin)
    acetabulum_border_upper_margin_points = np.argwhere(acetabulum_border_upper_margin)
    acetabulum_border_lower_margin_points = np.argwhere(acetabulum_border_lower_margin)

    # Interpolate upper/lower margins points to get a better approximation of the borders
    femur_head_border_points = 0.5 * (
        femur_head_border_lower_margin_points + femur_head_border_upper_margin_points
    )
    acetabulum_border_points = 0.5 * (
        acetabulum_border_lower_margin_points + acetabulum_border_upper_margin_points
    )

    # Initialize sequences of points that are to be aligned
    query, reference = femur_head_border_points, acetabulum_border_points

    # Swap sequences based on their lengths,
    # such that the DTW matches points from the longer sequence to points in the shorter one
    query, reference = (reference, query) if len(query) < len(reference) else (query, reference)

    # Sort points by their y-coordinates,
    # since the border curves are spanned along the y-axis
    query     = np.array(sorted(query,     key=lambda p: p[1]))
    reference = np.array(sorted(reference, key=lambda p: p[1]))

    # Compute pixel-wise distance matrix, using the Euclidean distance
    distance_matrix = np.array([
        np.sqrt(np.sum((p - reference) ** 2, axis=-1)) for p in query
    ])
    
    # Scale values in the distance matrix by the pixel spacing
    # to express distances in mm
    if pixel_spacing is not None:
        distance_matrix *= pixel_spacing

    # Compute minJSW:
    min_jsw = np.min(distance_matrix)

    # Align sequences of points using the DTW algorithm
    alignment = dtw.dtw(distance_matrix, step_pattern='asymmetric')

    # Compute JSW array 
    jsw_array = np.array([
        distance_matrix[i, j] for i, j in zip(alignment.index1, alignment.index2)
    ])

    return min_jsw, jsw_array


def _plot_dtw_results(query, reference, alignment, jsw_array):
    plt.figure('DTW in action')

    plt.subplot(1, 2, 1)
    plt.title('DTW alignments')
    for i, j in zip(alignment.index1, alignment.index2):
        plt.plot(
            [query[i, 1], reference[j, 1]],
            [query[i, 0], reference[j, 0]],
            'c',
        )

    plt.subplot(1, 2, 2)
    plt.title('JSW array generated from the DTW alignments')
    plt.plot(list(range(len(jsw_array))), jsw_array, 'o-')

    plt.show()

