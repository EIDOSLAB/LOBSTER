import os

import numpy
import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms


def get_data_loaders(data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers, pin_memory,
                     random_seed=0):
    """
    Build and returns a torch.utils.data.DataLoader for the torchvision.datasets.ImageNet DataSet.
    :param data_dir: Location of the DataSet. Downloading not supported.
    :param train_batch_size: DataLoader batch size.
    :param valid_size: If greater than 0 defines a validation DataLoader composed of $valid_size * len(train DataLoader)$ elements.
    :param shuffle: If True, reshuffles data.
    :param num_workers: Number of DataLoader workers.
    :param pin_memory: If True, uses pinned memory for the DataLoader.
    :param random_seed: Value for generating random numbers.
    :return: (train DataLoader, validation DataLoader, test DataLoader) if $valid_size > 0$, else (train DataLoader, test DataLoader)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = os.path.join(data_dir)
    train_data_dir = os.path.join(data_dir, "train")

    parent_dataset = datasets.ImageFolder(
        root=train_data_dir, transform=transform_train,
    )

    total = len(parent_dataset)
    train_split = int(numpy.floor(total * 0.8))
    valid_split = int(numpy.floor(total * 0.1))
    test_split = total - train_split - valid_split

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(parent_dataset,
                                                                               [train_split, valid_split, test_split])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=train_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader
