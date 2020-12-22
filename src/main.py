import os
import random

import numpy
import torch

from config import LOGS_ROOT
from utilities import *


def main(args):
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
    pytorch_optmizer, sensitivity_optmizer = get_optimizers(args, model)
    
    # SummaryWriter
    get_tb_writer(args)
    
    # Train the model
    if args.batch_pruning:
        train_model_batch_pruning(args, model, train_loader, valid_loader, test_loader, pytorch_optmizer,
                                  sensitivity_optmizer)
    else:
        train_model_epoch_pruning(args, model, train_loader, valid_loader, test_loader, pytorch_optmizer,
                                  sensitivity_optmizer)
    
    torch.save(model.state_dict(), os.path.join(LOGS_ROOT, args.dataset, args.name, "models", "end.pt"))


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    main(args)
