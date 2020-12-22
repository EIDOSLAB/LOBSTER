from .data_loader_cifar10 import get_data_loaders as data_loader_cifar10
from .data_loader_fashionmnist import get_data_loaders as data_loader_fashion_mnist
from .data_loader_imagenet import get_data_loaders as data_loader_imagenet
from .data_loader_mnist import get_data_loaders as data_loader_mnist


def get_dataloader(dataset, data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers, pin_memory,
                   random_seed=0):
    """
    Build and returns a torch.utils.data.DataLoader for the specified dataset.
    :param dataset: String that indicates the requested dataset.
    :param data_dir: Location of the DataSet or where it will be downloaded if not existing.
    :param batch_size: DataLoader batch size.
    :param valid_size: If greater than 0 defines a validation DataLoader composed of $valid_size * len(train DataLoader)$ elements.
    :param shuffle: If True, reshuffles data.
    :param num_workers: Number of DataLoader workers.
    :param pin_memory: If True, uses pinned memory for the DataLoader.
    :param random_seed: Value for generating random numbers.
    :return: (train DataLoader, validation DataLoader, test DataLoader) if $valid_size > 0$, else (train DataLoader, test DataLoader)
    """
    
    if dataset == "mnist":
        return data_loader_mnist(data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers,
                                 pin_memory, random_seed)
    elif dataset == "fashion-mnist":
        return data_loader_fashion_mnist(data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers,
                                         pin_memory,
                                         random_seed)
    elif dataset == "cifar10":
        return data_loader_cifar10(data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers,
                                   pin_memory, random_seed)
    elif dataset == "imagenet":
        return data_loader_imagenet(data_dir, train_batch_size, test_batch_size, valid_size, shuffle, num_workers,
                                    pin_memory, random_seed)
