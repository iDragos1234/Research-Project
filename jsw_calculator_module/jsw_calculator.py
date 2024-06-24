import os
import sys

import tqdm
sys.path.append('./../research-project')
sys.path.append('./../research-project/trainer_module')

import matplotlib.pyplot as plt
import numpy as np
import torch
import dtw

from monai.data import DataLoader
from monai.utils import set_determinism

from skimage.measure import find_contours
from scipy.spatial.distance import cdist


from trainer_module import (
    data_loader_builder,
    models,
)


class JSWCalculator:

    def __init__(self,
        hdf5_filepath: str,
        data_split_csv_filepath: str,
        model_id: str,
        model_state_filepath: str,
        output_dir: str,
        seed: int,
        verbose: bool,
    ) -> None:
        
        # Set seed for reproducibility purposes.
        set_determinism(seed)

        # Build datasets (training, validation and testing)
        # using the split specified in the data split CSV file.
        (
            train_data_loader,  # <--- Not used for testing
            valid_data_loader,  # <--- Not used for testing
            test_data_loader,  
        ) = data_loader_builder.DataLoaderBuilder(
            hdf5_filepath           = hdf5_filepath,
            data_split_csv_filepath = data_split_csv_filepath,
            batch_size              = 1,
            num_workers             = 0,
            verbose                 = verbose,
        ).build()

        # Fetch the selected model setting to be trained.
        model_setting = models.MODELS[model_id]

        # Extract model setting:
        self.model       = model_setting.model
        self.loss_func   = model_setting.loss_func
        self.metric_func = model_setting.metric_func
        self.optimizer   = model_setting.optimizer
        self.pre_transf  = model_setting.pre_transf
        self.post_transf = model_setting.post_transf

        self.test_data_loader     = test_data_loader
        self.model_state_filepath = model_state_filepath
        self.output_dir           = output_dir

        self.seed    = seed
        self.verbose = verbose


    def calculate_jsw(self):

        # Load model state from specified .pth filepath
        self.model.load_state_dict(torch.load(self.model_state_filepath))

        # Set model to evaluation mode
        self.model.eval()

        dataset = self.test_data_loader.dataset

        stats = []
        errors = []
        jsw_scores = []

        # Disable gradient calculation (improves performance)
        with torch.no_grad():
            # Testing loop
            loading_bar = range(len(dataset))
            if not self.verbose:
                loading_bar = tqdm.tqdm(loading_bar)
            
            for idx in loading_bar:

                if self.verbose:
                    print(f'{idx + 1}/{len(dataset)}')
                
                sample = dataset[idx]
                meta   = dataset.get_item_meta(idx)
                # Load batch inputs (images) and labels (masks) to the device
                input_image = sample['image'][None, :]
                label_mask  = sample['mask' ][None, :]

                # Predict labels
                pred_mask = self.model(input_image)

                # Apply the post-prediction transform, if any
                if self.post_transf is not None:
                    pred_mask = [self.post_transf(item) for item in pred_mask]

                label_mask = np.array(label_mask )
                pred_mask  = np.array(pred_mask)

                source_pixel_spacing = meta['group attributes']['source_pixel_spacing'][0]
                pixel_spacing        = meta['group attributes']['pixel_spacing'       ][0]


                # Compute minJSW for label and predicted masks
                try:
                    stats.append([
                        compute_min_jsw(label_mask[0], source_pixel_spacing, pixel_spacing),
                        compute_min_jsw(pred_mask [0], source_pixel_spacing, pixel_spacing),
                    ])
                    jsw_scores.append([stats[-1][0][0], stats[-1][1][0]])
                except:
                    errors.append(f'Unexpected error: {sys.exc_info()[0]}; for sample {idx}/{len(dataset)}.')
                    print(f'Unexpected error: {sys.exc_info()[0]}; for sample {idx}/{len(dataset)}.')

                plt.axis('off')
                plt.imshow(
                    sum((idx + 1) * mask for idx, mask in enumerate(pred_mask[0] - label_mask[0])),
                    'grey',
                    alpha=0.5
                )
                plt.colorbar()
                plt.show()

                if False and self.verbose:
                        # Plot minJSW for label mask
                        mJSW, femur, acetabulum, femur_head_border, acetabulum_border  = stats[-1][0]
                        print(mJSW, femur, acetabulum)

                        plt.subplot(1, 2, 1)
                        plt.axis('off')
                        plt.title('Label Mask minJSW')
                        # plt.scatter(femur_head_border[:, 1], femur_head_border[:, 0], marker='.')
                        # plt.scatter(acetabulum_border[:, 1], acetabulum_border[:, 0], marker='.')
                        plt.imshow(input_image[0][0], 'grey')
                        plt.plot([femur[1], acetabulum[1]], [femur[0], acetabulum[0]], 'g-')
                        plt.imshow(sum((i + 1) * xs for (i, xs) in enumerate(label_mask[0])), 'magma', alpha=0.5)

                        # Plot minJSW for predicted mask
                        mJSW, femur, acetabulum, femur_head_border, acetabulum_border = stats[-1][1]
                        print(mJSW, femur, acetabulum)

                        plt.subplot(1, 2, 2)
                        plt.axis('off')
                        plt.title('Predicted Mask minJSW')
                        # plt.scatter(femur_head_border[:, 1], femur_head_border[:, 0], marker='.')
                        # plt.scatter(acetabulum_border[:, 1], acetabulum_border[:, 0], marker='.')
                        plt.imshow(input_image[0][0], 'grey')
                        plt.plot([femur[1], acetabulum[1]], [femur[0], acetabulum[0]], 'g-')
                        plt.imshow(sum((i + 1) * xs for (i, xs) in enumerate(pred_mask[0])), 'magma', alpha=0.5)

                        plt.show()
                    

        jsw_scores = np.asarray(jsw_scores)
        jsw_diffs  = np.abs(jsw_scores[:, 0] - jsw_scores[:, 1])
        print('Errors:', errors)

        print(
            f'Number of succesfully computed JSW scores: {len(jsw_diffs)}/{len(dataset)};\n'
            f'Mean difference between real and predicted minJSW score: {np.mean(jsw_diffs)};\n'
            f'Standard deviation of difference between real and predicted minJSW score: {np.std(jsw_diffs)}.\n'
        )

        return


class JSWException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def compute_min_jsw(
    mask: torch.tensor,
    source_pixel_spacing: float = None,
    pixel_spacing: float = None,
) -> tuple[float, np.array]:
    # If background included in the mask, omit the background mask
    if mask.shape[0] == 4:
        mask = mask[1:]    
    elif mask.shape[0] != 3:
        raise JSWException(f'Unsupported mask shape. Was: {mask.shape}.')
    
    def get_pixel_neighbours(array, x, y):
        m, n = array.shape
        neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        neighbours = [(p, q) for (p, q) in neighbours if 0 <= p < m and 0 <= q < n]
        neighbours = [(p, q) for (p, q) in neighbours if array[p, q] == 1]
        return np.array(neighbours)
    
    def find_borders(femur_head, acetabulum, joint_space):
        femur_head_border = []
        acetabulum_border = []

        for x in range(len(femur_head)):
            for y in range(len(femur_head[0])):
                if femur_head[x, y] == 1:
                    neighbours = get_pixel_neighbours(joint_space, x, y)
                    if len(neighbours) != 0:
                        femur_head_border.append([
                            (x + sum(neighbours[:, 0])) / (1 + len(neighbours)),
                            (y + sum(neighbours[:, 1])) / (1 + len(neighbours)),
                        ])

        for x in range(len(acetabulum)):
            for y in range(len(acetabulum[0])):
                if acetabulum[x, y] == 1:
                    neighbours = get_pixel_neighbours(joint_space, x, y)
                    if len(neighbours) != 0:
                        acetabulum_border.append([
                            (x + sum(neighbours[:, 0])) / (1 + len(neighbours)),
                            (y + sum(neighbours[:, 1])) / (1 + len(neighbours)),
                        ])

        femur_head_border = np.array(femur_head_border)
        acetabulum_border = np.array(acetabulum_border)

        return (
            femur_head_border,
            acetabulum_border,
        )

    (
        femur_head_border,
        acetabulum_border,
    ) = find_borders(mask[0], mask[1], mask[2])

    # Compute distance matrix
    dist_matrix = np.array([
        np.sqrt(((p - acetabulum_border) ** 2).sum(axis=-1)) for p in femur_head_border
    ])

    # Scale distances to real (source) pixel spacing
    dist_matrix *= (source_pixel_spacing / pixel_spacing)

    # Get minJSW:
    m, n = dist_matrix.shape
    idx = np.argmin(dist_matrix)
    minX, minY = idx // n, idx % n
    mJSW = dist_matrix[minX, minY]
    return (
        mJSW,
        femur_head_border[minX],
        acetabulum_border[minY],
        femur_head_border,
        acetabulum_border,
    )



















def contour_mjsw(
    mask: torch.tensor,
    source_pixel_spacing: float = None,
    pixel_spacing: float = None,
) -> tuple[float, np.array]:

    femur_head  = mask[0 if mask.ndim == 3 else 1 if mask.ndim == 4 else None]
    acetabulum  = mask[1 if mask.ndim == 3 else 2 if mask.ndim == 4 else None]
    joint_space = mask[2 if mask.ndim == 3 else 3 if mask.ndim == 4 else None]

    femur_head  = find_contours(femur_head )[0]
    acetabulum  = find_contours(acetabulum )[0]
    joint_space = find_contours(joint_space)[0]

    mJSW = np.min(cdist(femur_head, acetabulum)) * (source_pixel_spacing / pixel_spacing)

    return mJSW


def euclidean_jsw(
    mask: torch.tensor,
    source_pixel_spacing: float = None,
    pixel_spacing: float = None,
) -> tuple[float, np.array]:
    
    # If background included in the mask, omit the background mask
    if mask.shape[0] == 4:
        mask = mask[1:]    
    elif mask.shape[0] != 3:
        raise JSWException(f'Unsupported mask shape. Was: {mask.shape}.')

    femur_head_lower = mask[0] * np.roll(mask[2], +1, 0)
    femur_head_upper = mask[2] * np.roll(mask[0], -1, 0)
    acetabulum_upper = mask[1] * np.roll(mask[2], -1, 0)
    acetabulum_lower = mask[2] * np.roll(mask[1], +1, 0)

    femur_head_lower_pts = np.argwhere(femur_head_lower)
    femur_head_upper_pts = np.argwhere(femur_head_upper)
    acetabulum_upper_pts = np.argwhere(acetabulum_upper)
    acetabulum_lower_pts = np.argwhere(acetabulum_lower)

    femur_head_pts = (femur_head_lower_pts + femur_head_upper_pts) / 2
    acetabulum_pts = (acetabulum_lower_pts + acetabulum_upper_pts) / 2

    # Create combined mask
    combined = 2 * (femur_head_lower + acetabulum_upper) + \
               3 * (femur_head_upper + acetabulum_lower)
    combined += mask[0] + mask[1] + mask[2]

    # Sequences
    seqA = femur_head_pts
    seqB = acetabulum_pts

    seqA = np.array(sorted(seqA, key=lambda p: p[1]))
    seqB = np.array(sorted(seqB, key=lambda p: p[1]))

    # Compute distance matrix
    dist_matrix = np.array([
        np.sqrt(((p - seqB) ** 2).sum(axis=-1)) for p in seqA
    ])

    # Scale distances to real (source) pixel spacing
    dist_matrix *= (source_pixel_spacing / pixel_spacing)

    # Get minJSW:
    m, n = dist_matrix.shape
    idx = np.argmin(dist_matrix)
    minX, minY = idx // n, idx % n
    mJSW = dist_matrix[minX, minY]
    return mJSW, seqA[minX], seqB[minY]


def dtw_jsw(
    mask: torch.tensor,
    source_pixel_spacing: float = None,
    pixel_spacing: float = None,
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
    femur_head_border_lower_margin = mask[0] * np.roll(mask[2], +1, 0)
    femur_head_border_upper_margin = mask[2] * np.roll(mask[0], -1, 0)
    acetabulum_border_upper_margin = mask[1] * np.roll(mask[2], -1, 0)
    acetabulum_border_lower_margin = mask[2] * np.roll(mask[1], +1, 0)

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
    distance_matrix *= (source_pixel_spacing / pixel_spacing)

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
