import torch
import torchmetrics
import torchvision

N_LAYERS_TO_KEEP = 8

class CustomSqueezeNet(torch.nn.Module):
    def __init__(self, image_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.squeezeNet = torchvision.models.squeezenet1_1(
            weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT
        )

        for param in self.squeezeNet.parameters():
            param.requires_grad = False
        
        # Obtiene el tamaño de salida de la última capa que se toma
        # del modelo
        y = torch.zeros([1, 3, image_size, image_size])
        for i in range(N_LAYERS_TO_KEEP):
            y = self.squeezeNet.features[i].forward(y)
        out_size = torch.flatten(y, start_dim=1).shape[1]


        self.fc1 = torch.nn.Linear(in_features=out_size, out_features=512, bias=True)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=1, bias=True)

    def forward(self, x):
        for i in range(N_LAYERS_TO_KEEP):
            x = self.squeezeNet.features[i].forward(x)

        x = torch.relu(self.fc1(torch.flatten(x, start_dim=1)))
        return torch.sigmoid(self.fc2(x)).flatten()


class SqueezeNetModelTrainConfig:
    def __init__(self, image_size: int, lr: float) -> None:
        self.model = CustomSqueezeNet(image_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.BCELoss()
        self.metrics = {'F1': torchmetrics.F1Score(task='binary'),
                        'Accuracy': torchmetrics.Accuracy(task='binary'),
                        'Recall': torchmetrics.Recall(task='binary'),
                        'Specificity': torchmetrics.classification.BinarySpecificity()}
