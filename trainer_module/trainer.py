import os, time, tqdm
from typing import Union
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.data import decollate_batch


import data_loader_builder
import models


class Trainer:

    def __init__(self,
        hdf5_filepath: str,
        data_split_csv_filepath: str,
        input_model_state_filepath: Union[str, None],
        output_model_state_filepath: str,
        model_id: str,

        device_name: str,

        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        batch_size: int,
        num_workers: int,
        valid_interval: int,
        output_stats_dir: str,

        seed: int,
        verbose: bool,
    ) -> None:
        # Build datasets (training, validation and testing)
        # using the split specified in the data split CSV file.
        (
            train_data_loader,
            valid_data_loader,
            test_data_loader,  # <--- Not used for training
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
        model_setting = models.MODELS[model_id](
            learning_rate = learning_rate,
            weight_decay  = weight_decay,
        )

        # Extract model setting:
        self.model       = model_setting.model
        self.loss_func   = model_setting.loss_func
        self.metric_func = model_setting.metric_func
        self.optimizer   = model_setting.optimizer
        self.pre_transf  = model_setting.pre_transf
        self.post_transf = model_setting.post_transf

        self.input_model_state_filepath  = input_model_state_filepath
        self.output_model_state_filepath = output_model_state_filepath

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.device           = device
        self.max_epochs       = max_epochs
        self.valid_interval   = valid_interval
        self.output_stats_dir = output_stats_dir
        self.seed             = seed
        self.verbose          = verbose

    def train(self) -> None:
        if self.verbose:
            print('Starting training...\n')

        # Set seed for reproducibility purposes.
        set_determinism(self.seed)

        # Keep track of relevant stats during training
        stats = {
            'start time': time.time(),
            'ellapsed time': None,
            'train loss values per epoch': [],
            'train metric values per epoch': [],
            'validation metric values per epoch': [],
        }

        # Select the model with the best validation metric across all epochs
        best_valid_metric       = float('-inf')
        best_valid_metric_epoch = None

        # Tensorboard writer for logging loss and metric values
        # used to assess the model performance after training.
        writer = SummaryWriter(self.output_stats_dir)

        # Training constants
        NUM_BATCHES = len(self.train_data_loader)
        BATCH_SIZE  = self.train_data_loader.batch_size

        # Even if verbosity is disabled, display a nice tqdm loading bar
        loading_bar = range(self.max_epochs)
        if not self.verbose:
            loading_bar = tqdm.tqdm(loading_bar)

        # Load model state from specified .pth filepath
        if self.input_model_state_filepath is not None:
            if self.verbose:
                print('Loading model state...')
            self.model.load_state_dict(torch.load(self.input_model_state_filepath))

        # Load model to device
        self.model = self.model.to(self.device)

        # Training loop
        for epoch in loading_bar:
            if self.verbose:
                print(f'Epoch {epoch + 1}/{self.max_epochs}')

            # Set model to training mode
            self.model.train()

            # Aggregate loss for current epoch
            epoch_train_loss = 0.0

            # Train on each batch of data samples
            for train_step, train_batch_data in enumerate(self.train_data_loader):

                # Load batch inputs (images) and labels (masks) to the device
                train_inputs = train_batch_data['image'].to(self.device)
                train_labels = train_batch_data['mask' ].to(self.device)

                # Reset gradients
                self.optimizer.zero_grad()

                # Predict labels
                train_outputs = self.model(train_inputs)

                # Compute loss
                train_loss = self.loss_func(train_outputs, train_labels)

                # Backpropagate loss
                train_loss.backward()
                self.optimizer.step()

                # Decollate inputs, labels and outputs batches
                train_inputs  = decollate_batch(train_inputs )
                train_labels  = decollate_batch(train_labels )
                train_outputs = decollate_batch(train_outputs)

                # Apply the post-prediction transform, if any
                if self.post_transf is not None:
                    train_outputs = [self.post_transf(item) for item in train_outputs]

                # Compute metric
                self.metric_func(y_pred = train_outputs, y = train_labels)

                # Aggregate loss
                epoch_train_loss += train_loss.item()

                # Log current batch loss
                writer.add_scalar(
                    tag = 'train loss',
                    scalar_value = train_loss.item(),
                    global_step  = NUM_BATCHES * epoch + train_step,
                )

                if self.verbose:
                    print(
                        f'{train_step + 1}/{NUM_BATCHES}, '
                        f'train loss: {train_loss.item():.4f}'
                    )

            # Compute train mean loss for the current epoch
            epoch_train_loss /= BATCH_SIZE

            # Aggregate train metric for the current epoch
            epoch_train_metric = self.metric_func.aggregate().item()

            # Reset metric function
            self.metric_func.reset()

            # Log the mean loss and mean metric for the current train epoch
            stats['train loss values per epoch'  ].append(epoch_train_loss  )
            stats['train metric values per epoch'].append(epoch_train_metric)

            writer.add_scalar(
                tag = 'train mean loss per epoch',
                scalar_value = epoch_train_loss,
                global_step  = epoch,
            )

            writer.add_scalar(
                tag = 'train mean metric per epoch',
                scalar_value = epoch_train_metric,
                global_step  = epoch,
            )

            # Validation step:
            #   - occurs periodically, with periodicity specified by `valid_interval`;
            #   - stores the best performing model state;
            if (epoch + 1) % self.valid_interval == 0:

                # Set model to evaluation mode
                self.model.eval()

                # Disable gradient calculation (improves performance)
                with torch.no_grad():

                    # Validation loop
                    for valid_step, valid_batch_data in enumerate(self.valid_data_loader):

                        # Load batch inputs (images) and labels (masks) to the device
                        valid_inputs = valid_batch_data['image'].to(self.device)
                        valid_labels = valid_batch_data['mask' ].to(self.device)

                        # Predict labels
                        valid_outputs = self.model(valid_inputs)

                        # Decollate inputs, labels and outputs batches
                        valid_inputs  = decollate_batch(valid_inputs )
                        valid_labels  = decollate_batch(valid_labels )
                        valid_outputs = decollate_batch(valid_outputs)

                        # Apply the post-prediction transform, if any
                        if self.post_transf is not None:
                            valid_outputs = [self.post_transf(item) for item in valid_outputs]

                        # Compute metric
                        self.metric_func(y_pred = valid_outputs, y = valid_labels)

                    # Aggregate train metric for the current epoch
                    epoch_valid_metric = self.metric_func.aggregate().item()

                    # Reset metric function
                    self.metric_func.reset()

                    # Log the mean loss and mean metric for the current train epoch
                    stats['validation metric values per epoch'].append(epoch_valid_metric)

                    writer.add_scalar(
                        tag = 'validation mean metric per epoch',
                        scalar_value = epoch_valid_metric,
                        global_step  = epoch,
                    )

                    # If current validation metric is the best so far,
                    # save the current model state
                    if epoch_valid_metric >= best_valid_metric:
                        best_valid_metric       = epoch_valid_metric
                        best_valid_metric_epoch = epoch + 1
                        torch.save(
                            self.model.state_dict(),
                            self.output_model_state_filepath,
                        )
                        
                        if self.verbose:
                            print('Saved new best metric model ')

                    if self.verbose:
                        print(
                            f'Current epoch: {epoch + 1}, '
                            f'current validation metric: {epoch_valid_metric:.4f}, '
                            f'best validation metric: {best_valid_metric:.4f} '
                            f'at epoch {best_valid_metric_epoch}'
                        )

                    # Save the last validation inputs, labels and outputs to tensorboard
                    plot_2d_or_3d_image(valid_inputs,  epoch + 1,  writer, tag = 'image',  max_channels = 4)
                    plot_2d_or_3d_image(valid_labels,  epoch + 1,  writer, tag = 'label',  max_channels = 4)
                    plot_2d_or_3d_image(valid_outputs, epoch + 1,  writer, tag = 'output', max_channels = 4)

            # Save stats:
            np.save(
                os.path.join(self.output_stats_dir, 'train_loss_per_epoch.npy'),
                stats['train loss values per epoch'],
            )
            np.save(
                os.path.join(self.output_stats_dir, 'train_metric_per_epoch.npy'),
                stats['train metric values per epoch'],
            )
            np.save(
                os.path.join(self.output_stats_dir, 'validation_metric_per_epoch.npy'),
                stats['validation metric values per epoch'],
            )
        
        # Close tensorboard writer
        writer.close()

        # Calculate ellapsed time of training process
        stats['ellapsed time'] = time.time() - stats['start time']

        if self.verbose:
            print(
                f'Training completed:\n'
                f'  - best_metric: {best_valid_metric:.4f} at epoch: {best_valid_metric_epoch};\n'
                f'  - ellapsed time: {stats["ellapsed time"]:.4f}s.\n'
            )

        return
