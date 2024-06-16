import time, tqdm, sys
sys.path.append('./../research-project')
sys.path.append('./../research-project/trainer_module')

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.data import decollate_batch


from trainer_module import (
    data_loader_builder,
    models,
)


class Evaluator:

    def __init__(self,
        hdf5_filepath: str,
        data_split_csv_filepath: str,
        model_state_filepath: str,
        output_stats_dir: str,

        device_name: str,
        model_id: str,
        batch_size: int,
        num_workers: int,
        seed: int,
        verbose: bool,
    ) -> None:
        # Build datasets (training, validation and testing)
        # using the split specified in the data split CSV file.
        (
            train_data_loader,  # <--- Not used for testing
            valid_data_loader,  # <--- Not used for testing
            test_data_loader,  
        ) = data_loader_builder.DataLoaderBuilder(
            hdf5_filepath           = hdf5_filepath,
            data_split_csv_filepath = data_split_csv_filepath,
            batch_size              = batch_size,
            num_workers             = num_workers,
            verbose                 = verbose,
        ).build()

        # Get the specified device (`'cpu'` or `'cuda'`).
        device = torch.device(device_name)

        # Fetch the selected model setting to be trained.
        model_setting = models.MODELS[model_id]

        # Extract model setting:
        self.model       = model_setting.model
        self.loss_func   = model_setting.loss_func
        self.metric_func = model_setting.metric_func
        # self.optimizer   = model_setting.optimizer
        self.pre_transf  = model_setting.pre_transf
        self.post_transf = model_setting.post_transf

        self.test_data_loader = test_data_loader

        self.device  = device
        self.seed    = seed
        self.verbose = verbose

        self.model_state_filepath = model_state_filepath
        self.output_stats_dir     = output_stats_dir

    def evaluate(self):

        if self.verbose:
            print('Starting evaluation...\n')

        # Set seed for reproducibility purposes.
        set_determinism(self.seed)

        # Keep track of relevant stats during testing
        stats = {
            'start time': time.time(),
            'ellapsed time': None,
            'test metric': None,
        }

        # Tensorboard writer to log testing metrics and predicted/actual labels.
        writer = SummaryWriter(self.output_stats_dir)

        # Testing constants
        NUM_BATCHES = len(self.test_data_loader)


        # Load model state from specified .pth filepath
        if self.verbose:
            print('Loading model state...')
        self.model.load_state_dict(torch.load(self.model_state_filepath))
        
        # Load model to device
        self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Disable gradient calculation (improves performance)
        with torch.no_grad():
            # Testing loop
            for test_step, test_batch_data in tqdm.tqdm(enumerate(self.test_data_loader)):
                if self.verbose:
                    print(f'{test_step + 1}/{NUM_BATCHES}')
                # Load batch inputs (images) and labels (masks) to the device
                test_inputs = test_batch_data['image'].to(self.device)
                test_labels = test_batch_data['mask' ].to(self.device)

                # Predict labels
                test_outputs = self.model(test_inputs)

                # Decollate inputs, labels and outputs batches
                test_inputs  = decollate_batch(test_inputs )
                test_labels  = decollate_batch(test_labels )
                test_outputs = decollate_batch(test_outputs)

                # Apply the post-prediction transform, if any
                if self.post_transf is not None:
                    test_outputs = [self.post_transf(item) for item in test_outputs]

                # Compute metric
                self.metric_func(y_pred = test_outputs, y = test_labels)

                plot_2d_or_3d_image(test_inputs,  test_step,  writer, tag = 'image',  max_channels = 4)
                plot_2d_or_3d_image(test_labels,  test_step,  writer, tag = 'label',  max_channels = 4)
                plot_2d_or_3d_image(test_outputs, test_step,  writer, tag = 'output', max_channels = 4)

            # Aggregate train metric for the current epoch
            test_metric = self.metric_func.aggregate().item()
        
            # Reset metric function
            self.metric_func.reset()

            # Log the mean test metric
            stats['test metric'] = test_metric
            writer.add_scalar(
                tag = 'test mean metric',
                scalar_value = test_metric,
            )

        # Close tensorboard writer
        writer.close()  
        
        stats['ellapsed time'] = time.time() - stats['start time']

        if self.verbose:
            print(
                f'Testing completed:\n'
                f'  - test metric: {stats["test metric"]};\n'
                f'  - ellapsed time: {stats["ellapsed time"]}.\n'
            )

        return
