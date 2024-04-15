import torch
import torchmetrics
from train_utils import get_linear_from_conv_block


class ConvModel(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same"
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding="same"
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=3, stride=1, padding="same"
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = get_linear_from_conv_block(
            conv_blocks=self, in_size=(
                image_size, image_size), out_features=512
        )

        self.fc2 = torch.nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(x).flatten())
        return y


class ConvModelTrainConfig:
    def __init__(self, image_size: int, lr: float) -> None:
        self.model = ConvModel(image_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.metrics = {'F1': torchmetrics.F1Score(task='binary'),
                        'Accuracy': torchmetrics.Accuracy(task='binary'),
                        'Recall': torchmetrics.Recall(task='binary'),
                        'Specificity': torchmetrics.classification.BinarySpecificity()}
