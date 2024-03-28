import torch
import torchvision
import os
import math

from collections.abc import Iterable

from torch.utils.tensorboard import SummaryWriter

TRAIN_FOLDER_NAME = 'train/'
VALID_FOLDER_NAME = 'test/'


def getTrainTestDataLoaders(input_filepath, image_size, batch_size):

    aug_data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(image_size, image_size)),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomResizedCrop(
            size=(image_size, image_size), scale=(0.5, 1.0)),
        torchvision.transforms.ColorJitter(saturation=0.1, hue=0.1),
        torchvision.transforms.ToTensor()
    ])

    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(image_size, image_size)),
        torchvision.transforms.ToTensor()
    ])

    base_path = os.path.join(input_filepath, str(image_size) + '/')

    train_set = torchvision.datasets.ImageFolder(
        root=os.path.join(base_path, TRAIN_FOLDER_NAME), transform=aug_data_transforms)
    valid_set = torchvision.datasets.ImageFolder(
        root=os.path.join(base_path, VALID_FOLDER_NAME), transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


def trainModel(model, optimizer, criterion, metric, train_loader, valid_loader,
               epochs, tensorboard_log, register_path, image_size):

    if tensorboard_log:
        train_writer = SummaryWriter(
            log_dir=os.path.join(register_path, "train/"))
        valid_writer = SummaryWriter(
            log_dir=os.path.join(register_path, "valid/"))

        train_writer.add_graph(model, torch.zeros(
            (1, 3, image_size, image_size)))
        valid_writer.add_graph(model, torch.zeros(
            (1, 3, image_size, image_size)))

    if torch.cuda.is_available():
        model.to("cuda")
        metric.to("cuda")

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    for epoch in range(epochs):

        # Pongo el modelo en modo entrenamiento
        model.train()

        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0

        for train_data, train_target in train_loader:

            if torch.cuda.is_available():
                train_data = train_data.to("cuda")
                train_target = train_target.to("cuda")

            optimizer.zero_grad()
            output = model(train_data.float())
            loss = criterion(output, train_target)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            accuracy = metric(output, train_target)
            epoch_train_accuracy += accuracy.item()

        epoch_train_loss = epoch_train_loss / len(train_loader)
        epoch_train_accuracy = epoch_train_accuracy / len(train_loader)

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_accuracy)

        # Pongo el modelo en modo testeo
        model.eval()

        epoch_valid_loss = 0.0
        epoch_valid_accuracy = 0.0

        for valid_data, valid_target in valid_loader:
            if torch.cuda.is_available():
                valid_data = valid_data.to("cuda")
                valid_target = valid_target.to("cuda")

            output = model(valid_data.float())
            epoch_valid_loss += criterion(output, valid_target).item()
            epoch_valid_accuracy += metric(output, valid_target).item()

        epoch_valid_loss = epoch_valid_loss / len(valid_loader)
        epoch_valid_accuracy = epoch_valid_accuracy / len(valid_loader)

        valid_loss.append(epoch_valid_loss)
        valid_acc.append(epoch_valid_accuracy)

        print("Epoch: {}/{} - Train loss {:.6f} - Train Accuracy {:.6f} - Valid Loss {:.6f} - Valid Accuracy {:.6f}".format(
            epoch+1, epochs, epoch_train_loss, epoch_train_accuracy, epoch_valid_loss, epoch_valid_accuracy))

        if tensorboard_log:
            train_writer.add_scalar("loss", epoch_train_loss, epoch)
            valid_writer.add_scalar("loss", epoch_valid_loss, epoch)
            train_writer.add_scalar("accuracy", epoch_train_accuracy, epoch)
            valid_writer.add_scalar("accuracy", epoch_valid_accuracy, epoch)
            train_writer.flush()
            valid_writer.flush()

    history = {"train_loss": train_loss,
               "train_acc": train_acc,
               "valid_loss": valid_loss,
               "valid_acc": valid_acc}

    return history, model


def get_linear_from_conv_block(conv_blocks: Iterable[Iterable], h_in: int, w_in: int, out_features: int):
    def get_output_size(layer: torch.nn.Conv2d | torch.nn.MaxPool2d, h_in: int, w_in: int):

        def to_list(value: int | Iterable[int]) -> list:
            return [value, value] if isinstance(value, int) else value

        def calculate_size(dim: int, padd: int, dilat: int, ks: int, stride: int):
            return math.floor(((dim+2*padd-dilat*(ks-1)-1)/stride)+1)

        stride = to_list(layer.stride)
        dilat = to_list(layer.dilation)

        if isinstance(layer.padding, str):
            padd = [0, 0]
            if layer.padding == 'same':
                dilat = [0, 0]
        else:
            padd = to_list(layer.padding)

        kernel_size = to_list(layer.kernel_size)

        h_out = calculate_size(
            h_in, padd[0], dilat[0], kernel_size[0], stride[0])
        w_out = calculate_size(
            w_in, padd[1], dilat[1], kernel_size[1], stride[1])

        return h_out, w_out

    for conv_block in conv_blocks:
        h_in, w_in = get_output_size(conv_block[0], h_in, w_in)
        h_in, w_in = get_output_size(conv_block[1], h_in, w_in)

    size = conv_blocks[-1][0].out_channels*h_in*w_in

    return torch.nn.Linear(in_features=size, out_features=out_features)
