from monai.networks.nets.unet import UNet
import torch


device = torch.device("cuda:0")
model = UNet()
# model = DenseNet121(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=num_class
# ).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
epoch_num = 4
val_interval = 1

"""
* Train-Validation-Test split can be done by splitting the list obtained from ListDicomFilesMetadata

* This split can be used to make three instances of MyDataset: 
    * one for training, 
    * one for validation, 
    * one for testing.

* Still need to learn how to use the Monai Nets (in particular, the (Basic)UNet).

* And what loss function to use.
"""
