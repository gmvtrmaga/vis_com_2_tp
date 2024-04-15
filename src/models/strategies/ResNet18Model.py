import torch
import torchmetrics
import torchvision

DROPOUT_P = 0.3


class CustomResNet18Net(torch.nn.Module):
    RESENT18_LAYERS = 4

    def __init__(self, n_freeze: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.resNet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )

        for param in self.resNet18.parameters():
            param.requires_grad = False

        for i in range(n_freeze):
            layer_name = "layer" + str(CustomResNet18Net.RESENT18_LAYERS - i)
            for param in getattr(self.resNet18, layer_name).parameters():
                param.requires_grad = True

        self.resNet18.fc = torch.nn.Linear(
            in_features=512, out_features=1, bias=True
        )
        self.act_fc = torch.nn.Sigmoid()

    def forward(self, x):
        return self.act_fc(self.resNet18.forward(x)).flatten()


class ResNet18ModelTrainConfig:
    def __init__(self, n_freeze: int, lr: float) -> None:
        self.model = CustomResNet18Net(n_freeze)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.BCELoss()
        self.metrics = {'F1': torchmetrics.F1Score(task='binary'),
                        'Accuracy': torchmetrics.Accuracy(task='binary'),
                        'Recall': torchmetrics.Recall(task='binary'),
                        'Specificity': torchmetrics.classification.BinarySpecificity()}
