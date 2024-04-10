import torch
import torchvision
import torchmetrics

class CustomResNet18Net(torch.nn.Module):
    RESENT18_LAYERS = 4

    def __init__(self, n_freeze: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.resNet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        for param in self.resNet18.parameters():
            param.requires_grad = False

        for i in range(n_freeze):
            layer_name = "layer" + str(CustomResNet18Net.RESENT18_LAYERS - i)
            for param in getattr(self.resNet18, layer_name).parameters():
                param.requires_grad = True

        self.resNet18.fc = torch.nn.Linear(in_features=512, out_features=4096, bias=True)
        self.fc1 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=1, bias=True)

    def forward(self, x):
        x = torch.cat((x, x, x), axis=1)
        x = torch.relu(self.resNet18.forward(x))
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).flatten()

class ResNet18ModelTrainConfig():
    def __init__(self, n_freeze: int) -> None:
        self.model = CustomResNet18Net(n_freeze)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.metric = torchmetrics.classification.BinarySpecificity()