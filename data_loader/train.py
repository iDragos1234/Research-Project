from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter <<< TODO

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def calculate_weights(val1, val2):
    '''
    In this function we take the number of the background and the forgroud pixels to return the `weights` 
    for the cross entropy loss values.
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    weights = 1/weights
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)

def train(
    model, 
    data_in, 
    loss_function, 
    optimizer, 
    max_epochs, 
    model_directory_path, 
    validation_interval, 
    device,
):
    #--------------------------------------------
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_train_values = []
    epoch_loss_validion_values = []
    metric_train_values = []
    metric_validation_values = []
    # writer = SummaryWriter() <<< TODO
    #--------------------------------------------

    #--------------------------------------------
    test_loader, validation_loader, train_loader = data_in
    
    #--------------------------------------------

    for epoch in range(max_epochs):

        #--------------------------------------------
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        #--------------------------------------------

        for batch_data in train_loader:
            
            train_step += 1

            #--------------------------------------------
            inputs = batch_data['image'].to(device)
            labels = batch_data['mask'].to(device)
            #--------------------------------------------
            
            #-------------------------------------------
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = loss_function(outputs, labels)
            train_loss.backward()
            optimizer.step()
            #--------------------------------------------

            #--------------------------------------------
            train_epoch_loss += train_loss.item()
            train_epoch_len   = len(train_loader) // train_loader.batch_size
            print(
                f'{train_step}/{train_epoch_len}, '
                f'train_loss: {train_loss.item():.4f}'
            )
            # writer.add_scalar(
            #     'train_loss', train_loss.item(), 
            #     train_epoch_len * epoch + train_step,
            # ) <<< TODO
            #--------------------------------------------

            #--------------------------------------------
            if epoch == max_epochs - 1:
                plt.imshow(outputs.detach().numpy()[0][0])
                plt.show()
            #--------------------------------------------

        #--------------------------------------------
        train_epoch_loss /= train_step
        epoch_loss_train_values.append(train_epoch_loss)
        print(f'epoch {epoch + 1}, average loss: {train_epoch_loss:.4f}')

        np.save(
            os.path.join(model_directory_path, 'loss_train.npy'), 
            epoch_loss_train_values,
        )
        #--------------------------------------------


        print('-'*20)
        
        if (epoch + 1) % validation_interval == 0:
            test_epoch_loss = 0
            test_metric = 0
            epoch_metric_test = 0
            test_step = 0

            model.eval()

            with torch.no_grad():
                for val_data in validation_loader:
                    test_step += 1

        #             #--------------------------------------------
        #             val_data               = transforms(val_data)
        #             val_inputs, val_labels = val_data['image'], val_data['mask']
        #             val_inputs, val_labels = val_inputs[:, None], val_labels[:, None]
        #             val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    #--------------------------------------------
                    # roi_size = (96, 96)
                    # sw_batch_size = 4
                    # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    # val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    #--------------------------------------------
                    # TODO >>>
        #             test_outputs = model(test_image)
                    
        #             test_loss = loss(test_outputs, test_mask)
        #             test_epoch_loss += test_loss.item()
        #             test_metric = dice_metric(test_outputs, test_mask)
        #             epoch_metric_test += test_metric
                    
                
        #         test_epoch_loss /= test_step
        #         print(f'test_loss_epoch: {test_epoch_loss:.4f}')
        #         save_loss_test.append(test_epoch_loss)
        #         np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

        #         epoch_metric_test /= test_step
        #         print(f'test_dice_epoch: {epoch_metric_test:.4f}')
        #         save_metric_test.append(epoch_metric_test)
        #         np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

        #         if epoch_metric_test > best_metric:
        #             best_metric = epoch_metric_test
        #             best_metric_epoch = epoch + 1
        #             torch.save(model.state_dict(), os.path.join(
        #                 model_dir, "best_metric_model.pth"))
                
        #         print(
        #             f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
        #             f"\nbest mean dice: {best_metric:.4f} "
        #             f"at epoch: {best_metric_epoch}"
        #         )

    len_train = len(train_loader)

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )


def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    """
    This function is to show one patient from your datasets, so that you can see if the it is okay or you need 
    to change/delete something.

    `data`: this parameter should take the patients from the data loader, which means you need to can the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `test`: this parameter is to say that you want to display a patient from the testing patients.
    """

    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    
    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_test_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()


def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val