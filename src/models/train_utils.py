import math
import os
import time
from collections.abc import Iterable

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from src.data.file_utils import TRAIN_DIRECTORY, VALID_DIRECTORY, clean_directory


def getDataTransoforms(image_size):

    # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]

    aug_data_transforms = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.7, 1.0)
            ),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )

    data_transforms = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )

    return aug_data_transforms, data_transforms


class TransformDataSet(torch.utils.data.Dataset):
    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return self.subset.__len__()

    def __getitem__(self, idx):
        sample, target = self.subset.__getitem__(idx)
        return self.transform(sample), target


def getKFoldDataLoaders(splits, input_filepath, image_size, batch_size):
    aug_data_transforms, data_transforms = getDataTransoforms(image_size)

    base_path = os.path.join(input_filepath, str(image_size) + "/")

    folds = KFold(n_splits=splits, shuffle=True)
    dataset = torchvision.datasets.ImageFolder(base_path)

    num_total_files = len(dataset)

    print("Loaded {} files.".format(num_total_files))

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(dataset)):
        train_set = TransformDataSet(
            torch.utils.data.Subset(dataset, train_idx), transform=aug_data_transforms
        )
        valid_set = TransformDataSet(
            torch.utils.data.Subset(dataset, valid_idx), transform=data_transforms
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, shuffle=True
        )

        yield i_fold, train_loader, valid_loader


def getTrainTestDataLoaders(input_filepath, image_size, batch_size):
    aug_data_transforms, data_transforms = getDataTransoforms(image_size)

    base_path = os.path.join(input_filepath, str(image_size) + "/")

    train_set = torchvision.datasets.ImageFolder(
        root=os.path.join(base_path, TRAIN_DIRECTORY), transform=aug_data_transforms
    )
    num_aug_files = len(train_set)

    valid_set = torchvision.datasets.ImageFolder(
        root=os.path.join(base_path, VALID_DIRECTORY), transform=data_transforms
    )
    num_valid_files = len(valid_set)

    print(
        "Loaded {} augmented files. Loaded {} validation files.".format(
            num_aug_files, num_valid_files
        )
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True
    )

    return train_loader, valid_loader


def trainModel(
    model,
    optimizer,
    criterion,
    metrics,
    train_loader,
    valid_loader,
    epochs,
    tensorboard_log,
    register_path,
    n_fold,
):

    if tensorboard_log:
        if n_fold is None or n_fold == 0:
            clean_directory(register_path)

        register_path_train = os.path.join(register_path, TRAIN_DIRECTORY)
        if not n_fold is None:
            register_path_train = os.path.join(
                register_path_train, "fold" + str(n_fold) + "/"
            )

        register_path_valid = os.path.join(register_path, VALID_DIRECTORY)
        if not n_fold is None:
            register_path_valid = os.path.join(
                register_path_valid, "fold" + str(n_fold) + "/"
            )

        train_writer = SummaryWriter(log_dir=register_path_train)
        valid_writer = SummaryWriter(log_dir=register_path_valid)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available! Using GPU.")
        print("GPU Device Name:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    model.to(device)
    for metric in metrics.values():
        metric.to(device)

    train_loss = []
    train_metrics = []

    valid_loss = []
    valid_metrics = []

    time_ini = time.time()

    for epoch in range(epochs):

        model.train()

        epoch_train_loss = 0.0
        epoch_train_metrics = dict([(name, 0) for name, _ in metrics.items()])

        for train_data, train_target in train_loader:
            train_data = train_data.to(device)
            train_target = train_target.to(device)

            optimizer.zero_grad()
            output = model(train_data.float())
            loss = criterion(output, train_target.float())
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            for m_name, m_function in metrics.items():
                m_value = m_function(output, train_target.float())
                epoch_train_metrics[m_name] += m_value.item()

        epoch_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(epoch_train_loss)

        for m_name, _ in metrics.items():
            epoch_train_metrics[m_name] = epoch_train_metrics[m_name] / len(
                train_loader
            )

        train_metrics.append(epoch_train_metrics)

        model.eval()

        epoch_valid_loss = 0.0
        epoch_valid_metrics = dict([(name, 0) for name, _ in metrics.items()])

        for valid_data, valid_target in valid_loader:
            valid_data = valid_data.to(device)
            valid_target = valid_target.to(device)

            output = model(valid_data.float())
            epoch_valid_loss += criterion(output, valid_target.float()).item()

            for m_name, m_function in metrics.items():
                m_value = m_function(output, valid_target.float())
                epoch_valid_metrics[m_name] += m_value.item()

        epoch_valid_loss = epoch_valid_loss / len(valid_loader)
        valid_loss.append(epoch_valid_loss)

        for m_name, _ in metrics.items():
            epoch_valid_metrics[m_name] = epoch_valid_metrics[m_name] / len(
                train_loader
            )

        valid_metrics.append(epoch_valid_metrics)

        time_now = time.time()
        time_iter = time_now - time_ini

        print(
            "Epoch: {}/{} (Time: {:.2f} s)\n\tTrain: Loss {:.6f} - Metrics {}\n\tValid: Loss {:.6f} - Metrics {}\n".format(
                epoch + 1,
                epochs,
                time_iter,
                epoch_train_loss,
                str(epoch_train_metrics),
                epoch_valid_loss,
                str(epoch_valid_metrics),
            )
        )

        if tensorboard_log:
            train_writer.add_scalar("training time", time_iter, epoch)

            train_writer.add_scalar("loss", epoch_train_loss, epoch)
            valid_writer.add_scalar("loss", epoch_valid_loss, epoch)

            for m_name, m_value in epoch_train_metrics.items():
                train_writer.add_scalar(m_name, m_value, epoch)

            for m_name, m_value in epoch_valid_metrics.items():
                valid_writer.add_scalar(m_name, m_value, epoch)

            valid_writer.flush()
            train_writer.flush()

    history = {
        "train_loss": train_loss,
        "train_metrics": train_metrics,
        "valid_loss": valid_loss,
        "valid_metrics": valid_metrics,
    }

    return history, model


def get_linear_from_conv_block(
    conv_blocks: Iterable[Iterable] | torch.nn.Module,
    in_size: tuple[int, int],
    out_features: int,
):
    def get_output_size(
        layer: torch.nn.Conv2d | torch.nn.MaxPool2d, h_in: int, w_in: int
    ):

        def to_list(value: int | Iterable[int]) -> list:
            return [value, value] if isinstance(value, int) else value

        def calculate_size(dim: int, padd: int, dilat: int, ks: int, stride: int):
            return math.floor(((dim + 2 * padd - dilat * (ks - 1) - 1) / stride) + 1)

        stride = to_list(layer.stride)
        dilat = to_list(layer.dilation)

        if isinstance(layer.padding, str):
            padd = [0, 0]
            if layer.padding == "same":
                dilat = [0, 0]
        else:
            padd = to_list(layer.padding)

        kernel_size = to_list(layer.kernel_size)

        h_out = calculate_size(h_in, padd[0], dilat[0], kernel_size[0], stride[0])
        w_out = calculate_size(w_in, padd[1], dilat[1], kernel_size[1], stride[1])

        return h_out, w_out

    CONV_ATTR_SUFIX = "conv"
    POOL_ATTR_SUFIX = "pool"

    w_in, h_in = in_size

    if isinstance(conv_blocks, torch.nn.Module):
        object_attrs = conv_blocks.__dict__["_modules"]
        index = 1

        while (CONV_ATTR_SUFIX + str(index)) in object_attrs.keys():
            h_in, w_in = get_output_size(
                object_attrs[CONV_ATTR_SUFIX + str(index)], h_in, w_in
            )

            pooling_attr_name = POOL_ATTR_SUFIX + str(index)
            if pooling_attr_name in object_attrs.keys():
                h_in, w_in = get_output_size(
                    object_attrs[pooling_attr_name], h_in, w_in
                )

            index += 1

        out_channels = object_attrs[CONV_ATTR_SUFIX + str(index - 1)].out_channels
    else:
        for conv_block in conv_blocks:
            h_in, w_in = get_output_size(conv_block[0], h_in, w_in)
            h_in, w_in = get_output_size(conv_block[1], h_in, w_in)

        out_channels = conv_blocks[-1][0].out_channels

    return torch.nn.Linear(
        in_features=out_channels * h_in * w_in, out_features=out_features
    )
