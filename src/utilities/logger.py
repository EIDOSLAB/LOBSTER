import os
from copy import deepcopy

import torch

from config import LOGS_ROOT


@torch.no_grad()
def log_statistics(args, epoch, model, pruning_stat, train_performance, valid_performance, test_performance, lr,
                   top_cr, top_acc, cr_data):
    print_epoch_stat(epoch, pruning_stat, train_performance, valid_performance, test_performance)
    
    if pruning_stat["network_param_ratio"] > top_cr:
        top_cr = pruning_stat["network_param_ratio"]
        top_acc = 0
        
        # Print data of previous CR
        print_data(args, cr_data)
    
    if valid_performance[0] > top_acc:
        top_acc = valid_performance[0]
        
        with open(os.path.join(LOGS_ROOT, args.dataset, args.name, "progression_log.txt"), "a") as prog_file:
            prog_file.write("CR: {:<20} Top-1 Valid: {:<20} Top-1 Test: {:<20} epoch: {:<20}\n"
                            .format(top_cr, 100 - valid_performance[0], 100 - test_performance[0], epoch))
        cr_data = {
            "epoch":             epoch,
            "valid_performance": valid_performance,
            "test_performance":  test_performance,
            "pruning_stat":      pruning_stat,
            "lr":                lr,
            "model":             deepcopy(model)
        }
    
    return top_cr, top_acc, cr_data


def print_epoch_stat(epoch, pruning_stat, train_performance, valid_performance, test_performance):
    print("###########" + "#" * len(str(epoch)))
    print("# EPOCH: {} #".format(epoch))
    print("###########" + "#" * len(str(epoch)))
    print("\n")
    
    print("-- Performance --")
    print("Train: Top-1 {:.2f}, Top-5 {:.2f}, Loss {:.4f}".format(
        train_performance[0], train_performance[1], train_performance[2]))
    print("Validation: Top-1 {:.2f}, Top-5 {:.2f}, Loss {:.4f}".format(
        valid_performance[0], valid_performance[1], valid_performance[2]))
    print("Test: Top-1 {:.2f}, Top-5 {:.2f}, Loss {:.4f}".format(
        test_performance[0], test_performance[1], test_performance[2]))
    print("\n")
    
    print("-- Architecture --")
    print("Remaining neurons (%): {:.2f}".format(pruning_stat["network_neuron_non_zero_perc"]))
    print("Neurons CR: {:.2f}".format(pruning_stat["network_neuron_ratio"]))
    print("Remaining parameters (%): {:.2f}".format(pruning_stat["network_param_non_zero_perc"]))
    print("Parameters CR: {:.2f}".format(pruning_stat["network_param_ratio"]))
    print("\n")


def print_data(args, cr_data):
    with open(os.path.join(LOGS_ROOT, args.dataset, args.name, "log.txt"), "a") as cr_file:
        try:
            cr_file.write("Epoch: {}\n".format(cr_data["epoch"]))
            cr_file.write("Validation Error @1 (%): {:.2f}\n".format(100 - cr_data["valid_performance"][0]))
            cr_file.write("Test Error @1 (%): {:.2f}\n".format(100 - cr_data["test_performance"][0]))
            cr_file.write("Test Error @5 (%): {:.2f}\n".format(100 - cr_data["test_performance"][1]))
            cr_file.write("Neurons CR: {:.2f}\n".format(cr_data["pruning_stat"]["network_neuron_ratio"]))
            cr_file.write(
                "Remaining neurons (%): {:.2f}\n".format(cr_data["pruning_stat"]["network_neuron_non_zero_perc"]))
            cr_file.write("Parameters CR: {:.2f}\n".format(cr_data["pruning_stat"]["network_param_ratio"]))
            cr_file.write(
                "Remaining parameters (%): {:.2f}\n".format(cr_data["pruning_stat"]["network_param_non_zero_perc"]))
            cr_file.write("Learning Rate: {}\n".format(cr_data["lr"]))
            cr_file.write("=" * 20 + "\n\n")
            
            torch.save(cr_data["model"].state_dict(),
                       os.path.join(LOGS_ROOT, args.dataset, args.name, "models", "{}.pt".format(cr_data["epoch"])))
        except:
            pass
