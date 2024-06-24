from typing import Union
from torch.nn import Module

from torch.nn import CrossEntropyLoss, BCELoss

from monai.networks.nets.unet import UNet
from torch.optim import (
    Optimizer,
    Adam,
    SGD,
)
from monai.losses import (
    DiceLoss,
    DiceCELoss,
    DiceFocalLoss,
    HausdorffDTLoss,
)
from monai.metrics import (
    Metric,
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
)
from monai.transforms import (
    Transform,
    Compose,
    Activations,
    AsDiscrete,
)


class MyModel:

    model: Module
    loss_func: Module
    metric_func: Metric
    optimizer: Optimizer
    pre_transf: Union[Transform, None]
    post_transf: Union[Transform, None]


class UNetModelV1(MyModel):

    def __init__(self) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.learning_rate = 1e-3
        self.weight_decay  = 1e-5

        self.loss_func = DiceLoss(sigmoid=True)

        self.metric_func = DiceMetric()

        self.optimizer = Adam(
            params       = self.model.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        self.pre_transf = None
        self.post_transf = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ])


class UNetModelV2(MyModel):

    def __init__(self) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 4,
            channels = (16, 32, 64, 128, 256),
            strides = (2, 2, 2, 2),
            num_res_units = 2,
        )

        self.learning_rate = 1e-3
        self.weight_decay  = 1e-5

        self.loss_func = DiceLoss(softmax = True)

        self.metric_func = DiceMetric(include_background = False)

        self.optimizer = Adam(
            params       = self.model.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        self.pre_transf = None
        self.post_transf = Compose([
            Activations(softmax = True),
            AsDiscrete (
                argmax = True,
                to_onehot = 4,
            ),
        ])


class UNetModelV3(MyModel):

    def __init__(self) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 4,
            channels = (16, 32, 64, 128, 256),
            strides = (2, 2, 2, 2),
            num_res_units = 2,
        )

        self.learning_rate = 1e-3
        self.weight_decay  = 1e-5

        self.loss_func = DiceCELoss(softmax = True)

        self.metric_func = DiceMetric(include_background = False)

        self.optimizer = Adam(
            params       = self.model.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        self.pre_transf = None
        self.post_transf = Compose([
            Activations(softmax = True),
            AsDiscrete (
                argmax = True,
                to_onehot = 4,
            ),
        ])


class UNetModelV4(MyModel):

    def __init__(self) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 4,
            channels = (16, 32, 64, 128, 256),
            strides = (2, 2, 2, 2),
            num_res_units = 2,
        )

        self.learning_rate = 1e-3
        self.weight_decay  = 1e-5

        self.loss_func = DiceCELoss(softmax = True, lambda_dice = 0)

        self.metric_func = HausdorffDistanceMetric(include_background = False)

        self.optimizer = Adam(
            params       = self.model.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        self.pre_transf = None
        self.post_transf = Compose([
            Activations(softmax = True),
            AsDiscrete (
                argmax = True,
                to_onehot = 4,
            ),
        ])

MODELS: dict[str, MyModel] = {
    '1': UNetModelV1(),  # U-Net, Dice   loss, sigmoid activation, Dice metric, learning rate = 1e-3, weight decay = 1e-5, Adam optimizer,
    '2': UNetModelV2(),  # U-Net, Dice   loss, softmax activation, Dice metric, learning rate = 1e-3, weight decay = 1e-5, Adam optimizer,
    '3': UNetModelV3(),  # U-Net, DiceCE loss, softmax activation, Dice metric, learning rate = 1e-3, weight decay = 1e-5, Adam optimizer,
    '4': UNetModelV4(),  # U-Net, CE     loss, softmax activation, Dice metric, learning rate = 1e-3, weight decay = 1e-5, Adam optimizer,
}
