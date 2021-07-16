import os
import random
import zipfile

import numpy
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import LOGS_ROOT
from utilities import parse_args, get_model, get_dataloaders, get_optimizers, get_tb_writer
from utilities.evaluation import test_model
from utilities.losses import SoftJaccardBCEWithLogitsLoss


def save_and_zip_model(model, path):
    torch.save(model.state_dict(), path)
    
    try:
        zip_path = path.split(os.sep)
        zip_path[-1] = zip_path[-1].replace("pt", "zip")
        zip_path = os.path.join(*zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_LZMA) as zip_file:
            zip_file.write(path, path.split(os.sep)[-1])
        os.remove(path)
    except Exception as ex:
        print("Error zipping model {}\n"
              "{}".format(path, ex))


def select_device(device=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
    
    cuda = not cpu and torch.cuda.is_available()
    
    return torch.device('cuda:0' if cuda else 'cpu')


def simple_train(args, model, train_loader, valid_loader, test_loader, pytorch_optimizer, tb_writer, device):
    epochs = 1000
    steps = list(numpy.arange(150, epochs, 50))
    lr_scheduler = ReduceLROnPlateau(pytorch_optimizer, threshold=0)
    # loss_function = CrossEntropyLoss()
    loss_function = SoftJaccardBCEWithLogitsLoss(8)
    # task = "classification"
    task = "segmentation"
    top_acc = 0
    top_epoch = -1
    
    # Epochs
    for epoch in range(epochs):
        
        model.train()
        
        # Epoch progress bar
        print("")
        epoch_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training epoch {}".format(epoch))
        
        # Batches
        for batch, (data, target) in epoch_pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            pytorch_optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            pytorch_optimizer.step()
        
        # Test this epoch model
        test_performance, valid_performance = eval_epoch(epoch, model, loss_function, train_loader, test_loader,
                                                         valid_loader, device, task, tb_writer, pytorch_optimizer)
        
        if test_performance[1] > top_acc:
            top_acc = test_performance[1]
            top_epoch = epoch
            save_and_zip_model(model, os.path.join(LOGS_ROOT, args.dataset, args.name, "models", "best.pt"))
        
        lr_scheduler.step(valid_performance[2])
    
    print("Best model accuracy {} at epoch {}".format(top_acc, top_epoch))


def eval_epoch(epoch, model, loss_function, train_loader, test_loader, valid_loader, device, task, tb_writer, optim):
    train_performance = test_model(model, loss_function, train_loader, device, task,
                                   desc="Evaluating model on train set")
    test_performance = test_model(model, loss_function, test_loader, device, task, desc="Evaluating model on test set")
    valid_performance = test_model(model, loss_function, valid_loader, device, task,
                                   desc="Evaluating model on test set")
    
    tb_writer.add_scalar("Performance/Train/{}".format("Top-1" if task == "classification" else "Jaccard"),
                         train_performance[0], epoch)
    tb_writer.add_scalar("Performance/Train/{}".format("Top-5" if task == "classification" else "Dice"),
                         train_performance[1], epoch)
    tb_writer.add_scalar("Performance/Train/Loss", train_performance[2], epoch)
    tb_writer.add_scalar("Performance/Test/{}".format("Top-1" if task == "classification" else "Jaccard"),
                         test_performance[0], epoch)
    tb_writer.add_scalar("Performance/Test/{}".format("Top-5" if task == "classification" else "Dice"),
                         test_performance[1], epoch)
    tb_writer.add_scalar("Performance/Test/Loss", test_performance[2], epoch)
    tb_writer.add_scalar("Performance/Valid/{}".format("Top-1" if task == "classification" else "Jaccard"),
                         valid_performance[0], epoch)
    tb_writer.add_scalar("Performance/Valid/{}".format("Top-5" if task == "classification" else "Dice"),
                         valid_performance[1], epoch)
    tb_writer.add_scalar("Performance/Valid/Loss", valid_performance[2], epoch)

    lr = [p['lr'] for p in pytorch_optimizer.param_groups]
    for i, val in enumerate(lr):
        tb_writer.add_scalar("Params/Learning Rate {}".format(i), val, epoch)
    
    return valid_performance, test_performance


if __name__ == '__main__':
    args = parse_args()
    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Get Model and Dataset
    model = get_model(args)
    train_loader, valid_loader, test_loader = get_dataloaders(args)
    
    torch.save(model.state_dict(), os.path.join(LOGS_ROOT, args.dataset, args.name, "models", "init.pt"))
    
    # Optimizers
    pytorch_optimizer, sensitivity_optmizer = get_optimizers(args, model)
    
    # SummaryWriter
    tb_writer = get_tb_writer(args)
    
    simple_train(args, model, train_loader, valid_loader, test_loader, pytorch_optimizer, tb_writer,
                 torch.device(args.device))
