import sys
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
        model_setting: models.MyModel,
        test_data_loader: DataLoader,
        model_state_filepath: str,
        verbose: bool,
    ) -> None:
        # Extract model setting:
        self.model       = model_setting.model
        self.loss_func   = model_setting.loss_func
        self.metric_func = model_setting.metric_func
        self.optimizer   = model_setting.optimizer
        self.pre_transf  = model_setting.pre_transf
        self.post_transf = model_setting.post_transf

        self.test_data_loader = test_data_loader

        self.verbose     = verbose

        self.model_state_filepath = model_state_filepath


    def calculate_jsw(self):

        # Testing constants
        NUM_BATCHES = len(self.test_data_loader)

        # Load model state from specified .pth filepath
        self.model.load_state_dict(torch.load(self.model_state_filepath))

        # Set model to evaluation mode
        self.model.eval()

        dataset = self.test_data_loader.dataset

        # Disable gradient calculation (improves performance)
        with torch.no_grad():
            # Testing loop
            for test_step, test_batch_data in enumerate(self.test_data_loader):
                if self.verbose:
                    print(f'{test_step + 1}/{NUM_BATCHES}')
                # Load batch inputs (images) and labels (masks) to the device
                test_inputs = test_batch_data['image']
                test_labels = test_batch_data['mask' ]

                # Predict labels
                test_outputs = self.model(test_inputs)


                # Apply the post-prediction transform, if any
                if self.post_transf is not None:
                    test_outputs = [self.post_transf(item) for item in test_outputs]

                test_labels  = np.array(test_labels)
                test_outputs = np.array(test_outputs)

                print(type(test_labels), type(test_outputs))

                source_pixel_spacing = dataset.get_item_meta(test_step)['group attributes']['source_pixel_spacing'][0]
                pixel_spacing = dataset.get_item_meta(test_step)['group attributes']['pixel_spacing'][0]

                # print(contour_mjsw (test_outputs[0], source_pixel_spacing, pixel_spacing))
                # print(contour_mjsw (test_labels [0], source_pixel_spacing, pixel_spacing))
                
                # print(dtw_jsw      (test_outputs[0], source_pixel_spacing, pixel_spacing)[0])
                # print(dtw_jsw      (test_labels [0], source_pixel_spacing, pixel_spacing)[0])

                plt.subplot(1, 2, 1)
                mJSW, minX, minY = euclidean_jsw(test_labels[0], source_pixel_spacing, pixel_spacing)
                print(mJSW, minX, minY)
                plt.imshow(sum((i + 1) * xs for (i, xs) in enumerate(test_labels[0])), 'magma')

                plt.subplot(1, 2, 2)
                mJSW, minX, minY = euclidean_jsw(test_outputs[0], source_pixel_spacing, pixel_spacing)
                plt.imshow(sum((i + 1) * xs for (i, xs) in enumerate(test_outputs[0])), 'magma')
                plt.show()

        return


class JSWCalculatorBuilder:

    def __init__(self,
        hdf5_filepath: str,
        data_split_csv_filepath: str,
        model_id: str,
        model_state_filepath: str,

        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        num_workers: int,

        seed: int,
        verbose: bool,
    ) -> None:
        self.hdf5_filepath = hdf5_filepath
        self.data_split_csv_filepath = data_split_csv_filepath
        self.model_state_filepath = model_state_filepath
        self.model_id = model_id

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.seed = seed
        self.verbose = verbose

    def build(self) -> JSWCalculator:
        # Set seed for reproducibility purposes.
        set_determinism(self.seed)

        # Build datasets (training, validation and testing)
        # using the split specified in the data split CSV file.
        (
            train_data_loader,  # <--- Not used for testing
            valid_data_loader,  # <--- Not used for testing
            test_data_loader,  
        ) = data_loader_builder.DataLoaderBuilder(
            hdf5_filepath           = self.hdf5_filepath,
            data_split_csv_filepath = self.data_split_csv_filepath,
            batch_size              = self.batch_size,
            num_workers             = self.num_workers,
            verbose                 = self.verbose,
        ).build()

        # Fetch the selected model setting to be trained.
        model_setting = models.MODELS[self.model_id](
            learning_rate = self.learning_rate,
            weight_decay  = self.weight_decay,
        )

        # Initialize model evaluator with the selected model setting.
        return JSWCalculator(
            model_setting        = model_setting,
            test_data_loader     = test_data_loader,
            model_state_filepath = self.model_state_filepath,
            verbose              = self.verbose,
        )


class JSWException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)



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
    return mJSW, minX, minY



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




def main():

    jsw_calculator = JSWCalculatorBuilder(
        hdf5_filepath = './all_with_bg.h5',
        data_split_csv_filepath = './data_split.csv',
        model_id = '1',
        model_state_filepath = './my_runs/training_v2_08-06-2024_00-00/results/best_metric_model.pth', # './my_runs/training_v1_05-06-2024_20-00/best_metric_model.pth',
        learning_rate = 1e-3,
        weight_decay = 1e-5,
        batch_size = 1,
        num_workers = 0,
        seed = 42,
        verbose = True,
    ).build()

    jsw_calculator.calculate_jsw()

if __name__ == '__main__':
    main()