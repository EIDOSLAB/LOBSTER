import torch
from torch import nn
from tqdm import tqdm

from config import LAYERS
from utilities import get_dataloaders, log_statistics, print_data
from .evaluation import test_model, architecture_stat
from .pruning import get_mask_par
from .pruning.thresholding import threshold_scheduler


def train_model_epoch_pruning(args, model, train_loader, valid_loader, test_loader, pytorch_optmizer,
                              sensitivity_optmizer):
    device, loss_function, cross_valid, \
    top_cr, top_acc, cr_data, \
    epochs_count, high_lr, low_lr, current_lr, DLC = init_train(args, train_loader, valid_loader, test_loader)
    
    get_and_save_statistics(args, "INIT", model, loss_function,
                            train_loader, valid_loader, test_loader,
                            pytorch_optmizer, top_cr, top_acc,
                            cr_data, device)
    
    # Get threshold scheduler
    TS = threshold_scheduler(model, LAYERS, valid_loader, loss_function, args.twt, args.pwe, device)
    
    # Epochs
    for epoch in range(args.epochs):
        mask_params = get_masks(args, model)
        
        # Batches
        print("")
        for data, target in tqdm(train_loader, desc="Training epoch {}".format(epoch)):
            model.train()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Perform and update step
            optimizer_steps(args, model, data, target, loss_function, pytorch_optmizer, sensitivity_optmizer,
                            mask_params)
        
        # Get and save epoch statistics
        valid_performance, top_cr, top_acc, cr_data = get_and_save_statistics(args, epoch, model, loss_function,
                                                                              train_loader, valid_loader, test_loader,
                                                                              pytorch_optmizer, top_cr, top_acc,
                                                                              cr_data, device)
        
        # Perform pruning step
        if pruning_step(args, TS, valid_performance, cross_valid, DLC):
            train_loader, valid_loader, test_loader = DLC.get_dataloaders()
    
    print_data(args, cr_data)


def train_model_batch_pruning(args, model, train_loader, valid_loader, test_loader, pytorch_optmizer,
                              sensitivity_optmizer):
    device, loss_function, cross_valid, \
    top_cr, top_acc, cr_data, \
    epochs_count, high_lr, low_lr, current_lr, DLC = init_train(args, train_loader, valid_loader, test_loader)
    
    get_and_save_statistics(args, "INIT", model, loss_function,
                            train_loader, valid_loader, test_loader,
                            pytorch_optmizer, top_cr, top_acc,
                            cr_data, device)
    
    # Get threshold scheduler
    TS = threshold_scheduler(model, LAYERS, valid_loader, loss_function, args.twt, args.pwe, device)
    prune_iter = len(train_loader) // args.prune_iter
    test_iter = len(train_loader) // args.test_iter
    
    print("Batch pruning with pruning every {} batches and test every {} batches"
          .format(prune_iter, test_iter))
    
    # Epochs
    for epoch in range(args.epochs):
        
        # Batches
        print("")
        for batch, (data, target) in tqdm(enumerate(train_loader), desc="Training epoch {}".format(epoch)):
            ni = batch + len(train_loader) * epoch  # total batches since training start
            
            mask_params = get_masks(args, model)
            
            model.train()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer_steps(args, model, data, target, loss_function, pytorch_optmizer, sensitivity_optmizer,
                            mask_params)
            
            # Get and save epoch statistics
            if ((batch + 1) % test_iter) == 0:
                valid_performance, top_cr, top_acc, cr_data = get_and_save_statistics(args, ni, model, loss_function,
                                                                                      train_loader, valid_loader,
                                                                                      test_loader,
                                                                                      pytorch_optmizer, top_cr, top_acc,
                                                                                      cr_data, device)
            
            # Perform pruning step
            if ((batch + 1) % prune_iter) == 0:
                
                # Evaluate model performance for pruning purposes only if this batch is not already a 'test_iter'
                if ((batch + 1) % test_iter) != 0:
                    valid_performance = test_model(model, loss_function, valid_loader, device,
                                                   "Evaluating model on validation set")
                
                if pruning_step(args, TS, valid_performance, cross_valid, DLC):
                    train_loader, valid_loader, test_loader = DLC.get_dataloaders()
    
    print_data(args, cr_data)


def optimizer_steps(args, model, data, target, loss_function,
                    pytorch_optmizer, sensitivity_optimizer, mask_params):
    """
    Performs inference and parameters update using both the pytorch optimizer and the sensitivity optimizer
    :param args: Run arguments
    :param model: PyTorch model
    :param data: Model's input
    :param target: Inference target
    :param loss_function: Loss function used to compute the classification loss
    :param pytorch_optmizer: PyTorch optimizer (e.g. SGD)
    :param sensitivity_optimizer: Sensitivity optimizer
    :param mask_params: Dictionary of binary tensors, returned by `get_masks`
    """
    # Zero grad, inference, loss computation
    pytorch_optmizer.zero_grad()
    output = model(data)
    loss = loss_function(output, target)
    
    if sensitivity_optimizer is not None:
        loss.backward()
        pytorch_optmizer.step()
        sensitivity_optimizer.step([mask_params])
    else:
        loss.backward()
        pytorch_optmizer.step()
        apply_masks(model, [mask_params])
    
    del output, loss


def get_and_save_statistics(args, epoch, model, loss_function,
                            train_loader, valid_loader, test_loader, pytorch_optmizer,
                            top_cr, top_acc, cr_data, device):
    pruning_stat = architecture_stat(model)
    
    train_performance = test_model(model, loss_function, train_loader, device, "Evaluating model on training set")
    valid_performance = test_model(model, loss_function, valid_loader, device, "Evaluating model on validation set")
    test_performance = test_model(model, loss_function, test_loader, device, "Evaluating model on test set")
    
    top_cr, top_acc, cr_data = log_statistics(args, epoch, model, pruning_stat, train_performance,
                                              valid_performance,
                                              test_performance, pytorch_optmizer.param_groups[0]['lr'], top_cr,
                                              top_acc, cr_data)
    
    return valid_performance, top_cr, top_acc, cr_data


def pruning_step(args, TS, valid_performance, cross_valid, DLC):
    if TS.step(valid_performance[2], args.batch_pruning):
        if cross_valid:
            args.seed += 1
            train_loader, valid_loader, test_loader = get_dataloaders(args)
            
            TS.set_validation_loader(valid_loader)
            DLC.set_dataloaders(train_loader, valid_loader, test_loader)
        return True
    
    return False


def cycle_lr(epochs_count, cycle_up, cycle_down, current_lr, low_lr, high_lr, pytorch_optmizer):
    if epochs_count == cycle_up and current_lr == low_lr:
        for param_group in pytorch_optmizer.param_groups:
            param_group['lr'] = high_lr
        
        current_lr = high_lr
        epochs_count = 1
    
    elif epochs_count == cycle_down and current_lr == high_lr:
        for param_group in pytorch_optmizer.param_groups:
            param_group['lr'] = low_lr
        
        current_lr = low_lr
        epochs_count = 1
    
    else:
        epochs_count += 1
    
    return epochs_count, current_lr


def get_masks(args, model):
    mask_params = get_mask_par(model, LAYERS)
    return mask_params


def init_train(args, train_loader, valid_loader, test_loader):
    device = torch.device(args.device)
    loss_function = nn.CrossEntropyLoss().to(device)
    cross_valid = args.cross_valid
    top_cr = 1
    top_acc = 0
    cr_data = {}
    
    epochs_count = 1
    high_lr = args.lr
    low_lr = args.lr / 10
    current_lr = high_lr
    
    DLC = DataLoaderContainer()
    DLC.set_dataloaders(train_loader, valid_loader, test_loader)
    
    return device, loss_function, cross_valid, top_cr, top_acc, cr_data, epochs_count, high_lr, low_lr, current_lr, DLC


def apply_masks(model, masks):
    for mask in masks:
        if mask is not None:
            for n_m, mo in model.named_modules():
                if isinstance(mo, nn.modules.Conv2d) or isinstance(mo, nn.modules.Linear):
                    for n_p, p in mo.named_parameters():
                        p.data.mul_(mask["{}.{}".format(n_m, n_p)])


class DataLoaderContainer(object):
    def __init__(self):
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
    
    def set_dataloaders(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
    
    def get_dataloaders(self):
        return self.train_loader, self.valid_loader, self.test_loader
