import torch
import torchmetrics
import torchvision

DROPOUT_P = 0.5


class CustomResNet18Net(torch.nn.Module):
    def __init__(self, n_freeze: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.resNet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )

        CustomResNet18Net.freezeLayersBlocks(self.resNet18, n_freeze)

        self.resNet18.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=DROPOUT_P),
            torch.nn.Linear(in_features=512, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def freezeLayersBlocks(resNEt18Model, n_freeze):
        RESENT18_LAYERS = 4
        RESENT18_BLOCKS_PER_LAYER = 2

        for param in resNEt18Model.parameters():
            param.requires_grad = False

        for n_layer in reversed(range(RESENT18_LAYERS)):
            for n_block in reversed(range(RESENT18_BLOCKS_PER_LAYER)):
                n_freeze -= 1

                layer_name = "layer" + str(n_layer + 1)
                block = getattr(resNEt18Model, layer_name)[n_block]
                
                for param in block.parameters():
                    param.requires_grad = True

                if n_freeze <= 0:
                    break

            if n_freeze <= 0:
                break

    def forward(self, x):
        return self.resNet18.forward(x).flatten()


class ResNet18ModelTrainConfig:
    def __init__(self, n_freeze: int, lr: float) -> None:
        self.model = CustomResNet18Net(n_freeze)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss = torch.nn.BCELoss()
        self.metrics = {'F1': torchmetrics.F1Score(task='binary'),
                        'Accuracy': torchmetrics.Accuracy(task='binary'),
                        'Recall': torchmetrics.Recall(task='binary'),
                        'Specificity': torchmetrics.classification.BinarySpecificity()}
