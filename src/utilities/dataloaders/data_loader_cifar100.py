import os

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

from utilities.dataloaders.utils import MapDataset


def get_data_loaders(data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers, pin_memory,
                     random_seed=0):
    """
    Build and returns a torch.utils.data.DataLoader for the torchvision.datasets.CIFAR100 DataSet.
    :param data_dir: Location of the DataSet or where it will be downloaded if not existing.
    :param train_batch_size: DataLoader batch size.
    :param valid_size: If greater than 0 defines a validation DataLoader composed of $valid_size * len(train DataLoader)$ elements.
    :param shuffle: If True, reshuffles data.
    :param num_workers: Number of DataLoader workers.
    :param pin_memory: If True, uses pinned memory for the DataLoader.
    :param random_seed: Value for generating random numbers.
    :return: (train DataLoader, validation DataLoader, test DataLoader) if $valid_size > 0$, else (train DataLoader, test DataLoader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
    ])

    data_dir = os.path.join(data_dir)

    parent_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True,
    )

    if valid_size > 0:

        dataset_length = len(parent_dataset)
        valid_length = int(np.floor(valid_size * dataset_length))
        train_length = dataset_length - valid_length
        train_dataset, valid_dataset = random_split(parent_dataset,
                                                    [train_length, valid_length],
                                                    generator=torch.Generator().manual_seed(random_seed))

        train_dataset = MapDataset(train_dataset, transform_train)
        valid_dataset = MapDataset(valid_dataset, transform_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=test_batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory,
        )

    else:
        parent_dataset = MapDataset(parent_dataset, transform_train)
        train_loader = torch.utils.data.DataLoader(
            parent_dataset, batch_size=train_batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory,
        )

    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=transform_test,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory
    )

    return (train_loader, valid_loader, test_loader) if valid_size > 0 else (train_loader, test_loader)
