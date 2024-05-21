import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image
from monai.data import decollate_batch


def train(
    model, 
    data_in, 
    loss_function, 
    metric_function, 
    optimizer, 
    max_epochs, 
    model_directory_path, 
    validation_interval, 
    device,
    post_transf,
    verbose,
):
    if verbose:
        print('Start training ...')
    #--------------------------------------------
    best_valid_metric           = float('-inf')
    best_valid_metric_epoch     = None

    train_loss_values_per_epoch = []
    valid_loss_values_per_epoch = []

    train_metric_values         = []
    valid_metric_values         = []

    writer = SummaryWriter()
    #--------------------------------------------

    #--------------------------------------------
    test_loader, valid_loader, train_loader = data_in
    #--------------------------------------------

    train_epoch_length = len(train_loader)

    '''
    Start training loop
    '''
    for epoch in range(max_epochs):

        if verbose:
            print('-' * 20)
            print(f'Epoch {epoch + 1}/{max_epochs}')

        '''
        Training step:
        '''
        model.train()
        epoch_train_loss = 0
        for train_step, train_batch_data in enumerate(train_loader):
            
            #--------------------------------------------
            train_inputs = train_batch_data['image'].to(device)
            train_labels = train_batch_data['mask'].to(device)
            #--------------------------------------------
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            train_loss = loss_function(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()
            #--------------------------------------------
            train_outputs = [post_transf(i) for i in decollate_batch(train_outputs)]
            metric_function(y_pred=train_outputs, y=train_labels)
            #--------------------------------------------

            #--------------------------------------------
            epoch_train_loss += train_loss.item()
            writer.add_scalar(
                'train_loss', train_loss.item(), 
                train_epoch_length * epoch + train_step,
            )

            if verbose:
                print(
                    f'{train_step + 1}/{train_epoch_length}, '
                    f'train_loss: {train_loss.item():.4f}'
                )
            #--------------------------------------------

            #--------------------------------------------
            # TODO: DELETE >>>
            if verbose and epoch == max_epochs - 1:
                plt.figure(f'label {train_step}', (18, 6))
                plt.subplot(1, 4, 1)
                plt.imshow(train_inputs.detach().cpu()[0][0])
                for i in range(3):
                    plt.subplot(1, 4, i + 2)
                    plt.title(f'label channel {i}')
                    plt.imshow(train_outputs[0][i])
                plt.show()
            # <<< TODO: DELETE
            #--------------------------------------------

        train_metric = metric_function.aggregate().item()
        train_metric_values.append(train_metric)
        metric_function.reset()
        
        if verbose:
            print(f'Train metric: {train_metric}')

        #--------------------------------------------
        epoch_train_loss /= train_epoch_length
        train_loss_values_per_epoch.append(epoch_train_loss)

        np.save(
            os.path.join(model_directory_path, 'loss_train.npy'), 
            train_loss_values_per_epoch,
        )
        #--------------------------------------------

        if verbose:
            print(f'Epoch {epoch + 1}, average loss: {epoch_train_loss:.4f}')
            print('-' *20)
        
        '''
        Validation step (frequency of this step is specified by `validation_interval`):
        '''
        if (epoch + 1) % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for valid_step, valid_batch_data in enumerate(valid_loader):

                    #--------------------------------------------
                    valid_inputs = valid_batch_data['image'].to(device)
                    valid_labels = valid_batch_data['mask'] .to(device)
                    #--------------------------------------------
                    valid_outputs = model(valid_inputs)
                    valid_outputs = [post_transf(i) for i in decollate_batch(valid_outputs)]
                    metric_function(y_pred=valid_outputs, y=valid_labels)
                    #--------------------------------------------

                    # TODO: DELETE >>>
                    if verbose and epoch == max_epochs - 1:

                        valid_inputs = valid_inputs.detach().cpu()
                        valid_labels = valid_labels.detach().cpu()

                        plt.figure(f'validation label {valid_step}')
                        plt.subplot(1, 7, 1)
                        plt.imshow(valid_inputs[0][0])
                        for i in range(3):
                            plt.subplot(1, 7, i + 2)
                            plt.title(f'y channel {i}')
                            plt.imshow(valid_labels[0][i])

                        for i in range(3):
                            plt.subplot(1, 7, i + 5)
                            plt.title(f'y_hat channel {i}')
                            plt.imshow(valid_outputs[0][i])
                        plt.show()
                    # <<< TODO: DELETE

                valid_metric = metric_function.aggregate().item()
                valid_metric_values.append(valid_metric)
                metric_function.reset()

                '''
                If current metric is the best so far, save the current model 
                '''
                if valid_metric > best_valid_metric:
                    best_valid_metric       = valid_metric
                    best_valid_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), 
                        os.path.join(model_directory_path, 'best_metric_model.pth'),
                    )
                    
                    if verbose:
                        print('Saved new best metric model.')

                if verbose:
                    print(
                        f'Current epoch: {epoch + 1}, '
                        f'current metric: {valid_metric:.4f}, '
                        f'best metric: {best_valid_metric:.4f} '
                        f'at epoch {best_valid_metric_epoch}'
                    )
                writer.add_scalar('val_mean_dice', valid_metric, epoch + 1)

                plot_2d_or_3d_image(valid_inputs, epoch + 1, writer, index=0, tag='image')
                plot_2d_or_3d_image(valid_labels, epoch + 1, writer, index=0, tag='label')
                plot_2d_or_3d_image(valid_outputs, epoch + 1, writer, index=0, tag='output')

    if verbose:
        print(
            'Train completed, '
            f'best_metric: {best_valid_metric:.4f} '
            f'at epoch: {best_valid_metric_epoch}'
        )
    writer.close()
