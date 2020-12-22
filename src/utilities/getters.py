import os

import torch
from .dataloaders import get_dataloader
from .architectures import LeNet300, LeNet5, resnet32
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as torchvision_models

from config import LOGS_ROOT, LAYERS
from .pruning.sensitivity import LOBSTER


def get_tb_writer(args):
    return SummaryWriter(os.path.join(LOGS_ROOT, args.dataset, args.name, "tb_{}".format(args.name)))


def get_dataloaders(args):
    if args.valid_size > 0:
        train_loader, valid_loader, test_loader = get_dataloader(
            args.dataset, args.data_dir, args.train_batch_size, args.test_batch_size, args.valid_size, True, 4, True,
            args.seed
        )
    else:
        train_loader, test_loader = get_dataloader(
            args.dataset, args.data_dir, args.train_batch_size, args.test_batch_size, args.valid_size, True, 4, True,
            args.seed
        )
        valid_loader = None
    
    return train_loader, valid_loader, test_loader


def get_model(args):
    model = _load(args.model)
    if args.ckp_path is not None:
        print("Loading model dictionary from: {}".format(args.ckp_path))
        model.load_state_dict(torch.load(args.ckp_path, map_location="cpu"))
    
    device = torch.device(args.device)
    model.to(device)
    
    os.makedirs(os.path.join(LOGS_ROOT, args.dataset, args.name, "models"), exist_ok=True)
    
    return model


def get_optimizers(args, model):
    pytorch_optimizer = _get_pytorch_optimizer(args, model)
    sensitivity_optimizer = _get_sensitivity_optimizer(args, model) if args.sensitivity != "" else None
    
    return pytorch_optimizer, sensitivity_optimizer


def _load(model):
    if model == "lenet300":
        return LeNet300()
    if model == "lenet5":
        return LeNet5()
    if model == "resnet32":
        return resnet32("A")
    if model == "resnet18":
        return torchvision_models.resnet18(True)
    if model == "resnet101":
        return torchvision_models.resnet101(True)


def _get_pytorch_optimizer(args, model):
    pytorch_optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                            weight_decay=args.wd, nesterov=args.nesterov)
    
    return pytorch_optimizer


def _get_sensitivity_optimizer(args, model):
    if args.sensitivity == "lobster":
        return LOBSTER(model, args.lmbda, LAYERS)
