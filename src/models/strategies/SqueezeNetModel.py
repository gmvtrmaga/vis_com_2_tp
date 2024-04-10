import torch
import torchvision
import torchmetrics

class CustomSqueezeNet(torch.nn.Module):
    def __init__(self, n_freeze: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.squeezeNet = torchvision.models.squeezenet1_1(
            weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)
        
        # Las n_freeze últimos grupos del features se liberan para entrenamiento
        for param in list(self.squeezeNet.features.parameters())[:-n_freeze]:
            param.requires_grad = False

        self.squeezeNet.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
        self.squeezeNet.num_classes = 1

    def forward(self, x):
        x = torch.cat((x, x, x), axis=1)
        return self.squeezeNet.forward(x).flatten()


class SqueezeNetModelTrainConfig():
    def __init__(self, n_freeze: int) -> None:
        self.model = CustomSqueezeNet(n_freeze)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.metric = torchmetrics.classification.BinarySpecificity()
