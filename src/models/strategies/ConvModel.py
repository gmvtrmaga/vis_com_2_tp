import torch
import torchmetrics

from train_utils import trainModel, get_linear_from_conv_block

class ConvModel(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv3 = torch.nn.Conv2d(
        #    in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        #self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv4 = torch.nn.Conv2d(
        #    in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        #self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = get_linear_from_conv_block(
            conv_blocks=[[self.conv1, self.pool1], [self.conv2, self.pool2]],
            h_in=image_size, w_in=image_size, out_features=512)

        self.fc2 = torch.nn.Linear(
            in_features=512, out_features=1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        #x = self.pool3(torch.relu(self.conv3(x)))
        #x = self.pool4(torch.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x).flatten()
        return torch.sigmoid(x)


def train_ConvModel(train_loader, valid_loader, image_size, train_epochs, log_output_filepath):
    conv_model = ConvModel(image_size)
    optimizer = torch.optim.Adam(conv_model.parameters(), lr=0.0001)
    loss = torch.nn.BCEWithLogitsLoss()
    metric = torchmetrics.classification.BinarySpecificity()

    return trainModel(conv_model, optimizer, loss, metric,
                      train_loader, valid_loader, train_epochs, tensorboard_log=True,
                      register_path=log_output_filepath, image_size=image_size)
