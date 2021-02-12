import argparse
import csv
import os
import sys
from collections import OrderedDict

from config import AVAILABLE_MODELS, AVAILABLE_DATASETS, AVAILABLE_SENSITIVITIES, LOGS_ROOT


def get_arg_parser():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--epochs", default=1000, type=int,
                        help="Number of training epochs. Default = 1000.")
    parser.add_argument("--lr", default=1e-1, type=float,
                        help="PyTorch optimizer's learning rate. Default = 0.1.")
    parser.add_argument("--lmbda", default=1e-4, type=float,
                        help="Sensitivity lambda. Default = 0.0001.")
    parser.add_argument("--twt", default=0, type=float,
                        help="Threshold worsening tolerance. Default = 0.")
    parser.add_argument("--pwe", default=0, type=int,
                        help="Plateau waiting epochs. Default = 0.")
    parser.add_argument("--mom", type=float, default=0,
                        help="Momentum. Default = 0")
    parser.add_argument("--nesterov", default=False, action="store_true",
                        help="Use Nesterov momentum. Default = False.")
    parser.add_argument("--wd", type=float, default=0,
                        help="Weight decay. Default = 0.")
    
    # Modelt
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, required=True,
                        help="Neural network architecture.")
    parser.add_argument("--ckp_path", type=str,
                        help="Path to model state_dict.")
    
    # Dataset
    parser.add_argument("--dataset", type=str, choices=AVAILABLE_DATASETS, required=True,
                        help="Dataset")
    parser.add_argument("--valid_size", type=float, default=0.1,
                        help="Percentage of training dataset to use as validation. Default = 0.1.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Folder containing the dataset. Default = data.")
    parser.add_argument("--train_batch_sizte", type=int, default=100,
                        help="Batch size. Default = 100.")
    parser.add_argument("--test_batch_size", type=int, default=1000,
                        help="Batch size. Default = 100.")
    parser.add_argument("--cross_valid", default=False, action="store_true",
                        help="Perform cross validation. Default = False.")
    
    # Sensitivity optimizer
    parser.add_argument("--sensitivity", type=str, choices=AVAILABLE_SENSITIVITIES,
                        help="Sensitivty optimizer.")
    
    # Masks
    parser.add_argument("--mask_params", default=False, action="store_true",
                        help="Pruned parameters mask. Default = False.")
    parser.add_argument("--mask_neurons", default=False, action="store_true",
                        help="Pruned neurons mask. Default = False.")
    
    # Batch-wise pruning
    parser.add_argument("--batch_pruning", default=False, action="store_true",
                        help="Activate batch-wise pruning, requires -prune_batches and -test_batches. Default = False.")
    parser.add_argument("--prune_iter", type=int,
                        help="Defines how many batch iterations should pass before a pruning step"
                             "in dataset fractions e.g. 5 = 1/5 of the dataset.")
    parser.add_argument("--test_iter", type=int,
                        help="Defines how many batch iterations should pass before testing"
                             "in dataset fractions e.g. 5 = 1/5 of the dataset.")
    
    # Extra
    parser.add_argument("--device", default="cpu", type=str,
                        help="Device (cpu, cuda:0, cuda:1, ...). Default = cpu.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sets the seed for generating random numbers. Default = 0.")
    parser.add_argument("--name", default="test", type=str,
                        help="Run name. Default = test.")
    parser.add_argument("--dev", default=False, action="store_true",
                        help="Development mode. Default = False.")
    
    return parser


def parse_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Save arguments to csv
    save_args_csv(args)
    
    return args


def save_args_csv(args):
    try:
        os.makedirs(os.path.join(LOGS_ROOT, args.dataset, args.name))
    except OSError:
        if not args.dev:
            sys.exit("Run named -- {} -- already exists.\n"
                     "Execution interrupted to avoid accidental overwriting.".format(args.name))
        
    args_keys = ["lr", "lmbda", "twt", "pwe", "mom", "nesterov", "wd",
                 "model", "ckp_path",
                 "dataset", "valid_size", "data_dir", "train_batch_size", "test_batch_size", "cross_valid",
                 "sensitivity",
                 "mask_params", "mask_neurons",
                 "batch_pruning", "prune_iter", "test_iter",
                 "device", "seed", "name"]
    args_dict = OrderedDict([(k, getattr(args, k)) for k in args_keys])
    print(args_dict)
    
    csv_file_name = "hyp.csv"
    cvs_path = os.path.join(LOGS_ROOT, args.dataset, args.name, csv_file_name)
    
    if not os.path.exists(cvs_path):
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(list(args_dict.keys()))
            writer.writerow(list(args_dict.values()))
    else:
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(list(args_dict.values()))
