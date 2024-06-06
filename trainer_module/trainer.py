import os, time, tqdm
import numpy as np


import torch
from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.data import (
    DataLoader,
    decollate_batch
)

import data_loader_builder
import models


class Trainer:

    def __init__(self,
        model_setting: models.MyModel,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        device: torch.device,
        max_epochs: int,
        model_dir_path: str,
        valid_interval: int,
        verbose: bool,
    ) -> None:
        # Extract model setting:
        self.model       = model_setting.model
        self.loss_func   = model_setting.loss_func
        self.metric_func = model_setting.metric_func
        self.optimizer   = model_setting.optimizer
        self.pre_transf  = model_setting.pre_transf
        self.post_transf = model_setting.post_transf

        self.device      = device
        self.max_epochs  = max_epochs
        self.verbose     = verbose

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.valid_interval = valid_interval
        self.model_dir_path = model_dir_path


    def train(self) -> None:
        if self.verbose:
            print('Starting training...\n')

        # Get training parameters
        model       = self.model
        loss_func   = self.loss_func
        metric_func = self.metric_func
        optimizer   = self.optimizer
        pre_transf  = self.pre_transf
        post_transf = self.post_transf

        device     = self.device
        max_epochs = self.max_epochs
        verbose    = self.verbose

        train_data_loader = self.train_data_loader
        valid_data_loader = self.valid_data_loader

        valid_interval = self.valid_interval
        model_dir_path = self.model_dir_path

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
        writer = SummaryWriter()

        # Training constants
        NUM_BATCHES = len(train_data_loader)
        BATCH_SIZE  = train_data_loader.batch_size

        # Even if verbosity is disabled, display a nice tqdm loading bar
        loading_bar = range(max_epochs)
        if not verbose:
            loading_bar = tqdm.tqdm(loading_bar)

        # Load model to device
        model = model.to(device) 

        # Training loop
        for epoch in loading_bar:
            if verbose:
                print(f'Epoch {epoch + 1}/{max_epochs}')

            # Set model to training mode
            model.train()

            # Aggregate loss for current epoch
            epoch_train_loss = 0.0

            # Train on each batch of data samples
            for train_step, train_batch_data in enumerate(train_data_loader):

                # Load batch inputs (images) and labels (masks) to the device
                train_inputs = train_batch_data['image'].to(device)
                train_labels = train_batch_data['mask' ].to(device)

                # Reset gradients
                optimizer.zero_grad()

                # Predict labels
                train_outputs = model(train_inputs)

                # Compute loss
                train_loss = loss_func(train_outputs, train_labels)

                # Backpropagate loss
                train_loss.backward()
                optimizer.step()

                # Decollate inputs, labels and outputs batches
                train_inputs  = decollate_batch(train_inputs )
                train_labels  = decollate_batch(train_labels )
                train_outputs = decollate_batch(train_outputs)

                # Apply the post-prediction transformation, if any
                if post_transf is not None:
                    train_outputs = [post_transf(item) for item in train_outputs]

                # Compute metric
                metric_func(y_pred = train_outputs, y = train_labels)

                # Aggregate loss
                epoch_train_loss += train_loss.item()

                # Log current batch loss
                writer.add_scalar(
                    tag = 'train loss',
                    scalar_value = train_loss.item(),
                    global_step  = NUM_BATCHES * epoch + train_step,
                )

                if verbose:
                    print(
                        f'{train_step + 1}/{NUM_BATCHES}, '
                        f'train loss: {train_loss.item():.4f}'
                    )

            # Compute train mean loss for the current epoch
            epoch_train_loss /= BATCH_SIZE

            # Aggregate train metric for the current epoch
            epoch_train_metric = metric_func.aggregate().item()

            # Reset metric function
            metric_func.reset()

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
            if (epoch + 1) % valid_interval == 0:

                # Set model to evaluation mode
                model.eval()

                # Disable gradient calculation (improves performance)
                with torch.no_grad():

                    # Validation loop
                    for valid_step, valid_batch_data in enumerate(valid_data_loader):

                        # Load batch inputs (images) and labels (masks) to the device
                        valid_inputs = valid_batch_data['image'].to(device)
                        valid_labels = valid_batch_data['mask' ].to(device)

                        # Predict labels
                        valid_outputs = model(valid_inputs)

                        # Decollate inputs, labels and outputs batches
                        valid_inputs  = decollate_batch(valid_inputs )
                        valid_labels  = decollate_batch(valid_labels )
                        valid_outputs = decollate_batch(valid_outputs)
                        
                        # Apply the post-prediction transformation, if any
                        if post_transf is not None:
                            valid_outputs = [post_transf(item) for item in valid_outputs]

                        # Compute metric
                        metric_func(y_pred = valid_outputs, y = valid_labels)

                    # Aggregate train metric for the current epoch
                    epoch_valid_metric = metric_func.aggregate().item()

                    # Reset metric function
                    metric_func.reset()

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
                            model.state_dict(),
                            os.path.join(model_dir_path, 'best_metric_model.pth'),
                        )
                        
                        if verbose:
                            print('Saved new best metric model ')

                    if verbose:
                        print(
                            f'Current epoch: {epoch + 1}, '
                            f'current validation metric: {epoch_valid_metric:.4f}, '
                            f'best validation metric: {best_valid_metric:.4f} '
                            f'at epoch {best_valid_metric_epoch}'
                        )

                    # Save the last validation inputs, labels and outputs to tensorboard
                    plot_2d_or_3d_image(valid_inputs,  epoch + 1,  writer, tag = 'image',  max_channels = 3)
                    plot_2d_or_3d_image(valid_labels,  epoch + 1,  writer, tag = 'label',  max_channels = 3)
                    plot_2d_or_3d_image(valid_outputs, epoch + 1,  writer, tag = 'output', max_channels = 3)

            # Save the last epoch model state
            torch.save(
                model.state_dict(),
                os.path.join(model_dir_path, 'last_epoch_model.pth'),
            )
            
            # Save stats:
            np.save(
                os.path.join(model_dir_path, 'train_loss_per_epoch.npy'),
                stats['train loss values per epoch'],
            )
            np.save(
                os.path.join(model_dir_path, 'train_metric_per_epoch.npy'),
                stats['train metric values per epoch'],
            )
            np.save(
                os.path.join(model_dir_path, 'validation_metric_per_epoch.npy'),
                stats['validation metric values per epoch'],
            )
        
        # Close tensorboard writer
        writer.close()

        # Calculate ellapsed time of training process
        stats['ellapsed time'] = time.time() - stats['start time']

        if verbose:
            print(
                f'Train completed:\n'
                f'  - best_metric: {best_valid_metric:.4f} '
                f'at epoch: {best_valid_metric_epoch};\n'
                f'  - ellapsed time: {stats["ellapsed time"]:.4f}s.'
            )

        return


class TrainerBuilder:

    def __init__(self,
        hdf5_filepath: str,
        data_split_csv_filepath: str,
        model_dir_path: str,
        model_id: str,

        device_name: str,

        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        batch_size: int,
        num_workers: int,
        validation_interval: int,

        seed: int,
        verbose: bool,
    ) -> None:
        self.hdf5_filepath = hdf5_filepath
        self.data_split_csv_filepath = data_split_csv_filepath
        self.model_dir_path = model_dir_path
        self.model_id = model_id

        self.device_name = device_name

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_interval = validation_interval

        self.seed = seed
        self.verbose = verbose

    def build(self) -> Trainer:
        # Set seed for reproducibility purposes.
        set_determinism(self.seed)

        # Build datasets (training, validation and testing)
        # using the split specified in the data split CSV file.
        (
            train_data_loader,
            valid_data_loader,
            test_data_loader,  # <--- Not used for training
        ) = data_loader_builder.DataLoaderBuilder(
            self.hdf5_filepath,
            self.data_split_csv_filepath,
            self.batch_size,
            self.num_workers,
            self.verbose,
        ).build()

        # Get the specified device (`'cpu'` or `'cuda'`).
        device = torch.device(self.device_name)

        # Fetch the selected model setting to be trained.
        model_setting = models.MODELS[self.model_id](
            self.learning_rate,
            self.weight_decay,
        )

        # Initialize model trainer with the selected model setting.
        return Trainer(
            model_setting,
            train_data_loader,
            valid_data_loader,
            device,
            self.max_epochs,
            self.model_dir_path,
            self.validation_interval,
            self.verbose,
        )
