from typing import Union
from torch.nn import Module
from monai.networks.nets.unet import UNet
from torch.optim import (
    Optimizer,
    Adam,
)
from monai.losses import (
    DiceLoss,
    DiceCELoss,
)
from monai.metrics import (
    Metric,
    DiceMetric,
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
    
    def __init__(self,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.loss_func = DiceLoss(sigmoid=True)

        self.metric_func = DiceMetric()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.pre_transf = None
        self.post_transf = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ])


class UNetModelV2(MyModel):
    
    def __init__(self,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.loss_func = DiceLoss(softmax=True)

        self.metric_func = DiceMetric()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.pre_transf = None
        self.post_transf = Compose([
            Activations(softmax = True),
            AsDiscrete (argmax  = True),
        ])


MODELS = {
    '1': UNetModelV1,  # U-Net, Dice loss, sigmoid activation, Dice metric

    '2': UNetModelV2,  # U-Net, Dice loss, softmax activation, Dice metric
}
