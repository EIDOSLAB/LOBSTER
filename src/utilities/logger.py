import csv
import os
from copy import deepcopy

import torch

from config import LOGS_ROOT


@torch.no_grad()
def log_statistics(args, epoch, model, pruning_stat, train_performance, valid_performance, test_performance, lr,
                   top_cr, top_acc, cr_data, tb_writer, task):
    print_epoch_stat(args, epoch, pruning_stat, train_performance, valid_performance, test_performance, tb_writer, task,
                     lr)
    
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


def print_epoch_stat(args, epoch, pruning_stat, train_performance, valid_performance, test_performance, tb_writer, task,
                     lr):
    csv_file_name = "training_performance.csv"
    cvs_path = os.path.join(LOGS_ROOT, args.dataset, args.name, csv_file_name)
    
    vals = [epoch,
            train_performance[0], train_performance[1], train_performance[2],
            valid_performance[0], valid_performance[1], valid_performance[2],
            test_performance[0], test_performance[1], test_performance[2],
            pruning_stat["network_neuron_non_zero_perc"], pruning_stat["network_neuron_ratio"],
            pruning_stat["network_param_non_zero_perc"], pruning_stat["network_param_ratio"]]
    
    if not os.path.exists(cvs_path):
        titles = ["iteration",
                  "tr_t1", "tr_t5", "tr_lo",
                  "va_t1", "va_t5", "va_lo",
                  "te_t1", "te_t5", "te_lo",
                  "ne_perc", "ne_cr",
                  "pa_perc", "pa_cr"]
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(titles)
            writer.writerow(vals)
    else:
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(vals)
    
    epoch = -1 if epoch == "INIT" else epoch
    save_tb_and_wandb(epoch, tb_writer, task,
                      train_performance, valid_performance, test_performance, pruning_stat, lr)


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


def save_tb_and_wandb(epoch, tb_writer, task,
                      train_performance, valid_performance, test_performance, pruning_stat, lr):
    tb_writer.add_scalar("Performance/Train/{}".format("Top-1" if task == "classification" else "Jaccard"),
                         train_performance[0], epoch)
    tb_writer.add_scalar("Performance/Train/{}".format("Top-5" if task == "classification" else "Dice"),
                         train_performance[1], epoch)
    tb_writer.add_scalar("Performance/Train/Loss", train_performance[2], epoch)
    
    tb_writer.add_scalar("Performance/Validation/{}".format("Top-1" if task == "classification" else "Jaccard"),
                         valid_performance[0], epoch)
    tb_writer.add_scalar("Performance/Validation/{}".format("Top-5" if task == "classification" else "Dice"),
                         valid_performance[1], epoch)
    tb_writer.add_scalar("Performance/Validation/Loss", valid_performance[2], epoch)
    
    tb_writer.add_scalar("Performance/Test/{}".format("Top-1" if task == "classification" else "Jaccard"),
                         test_performance[0], epoch)
    tb_writer.add_scalar("Performance/Test/{}".format("Top-5" if task == "classification" else "Dice"),
                         test_performance[1], epoch)
    tb_writer.add_scalar("Performance/Test/Loss", test_performance[2], epoch)
    
    tb_writer.add_scalar("Architecture/Neurons Percentage", pruning_stat["network_neuron_non_zero_perc"], epoch)
    tb_writer.add_scalar("Architecture/Neurons CR", pruning_stat["network_neuron_ratio"], epoch)
    tb_writer.add_scalar("Architecture/Parameters Percentage", pruning_stat["network_param_non_zero_perc"], epoch)
    tb_writer.add_scalar("Architecture/Parameters CR", pruning_stat["network_param_ratio"], epoch)
    
    tb_writer.add_scalar("Params/Learning Rate", lr, epoch)
