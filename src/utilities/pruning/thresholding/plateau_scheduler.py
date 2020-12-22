from copy import deepcopy
from math import inf

import torch


class Scheduler(object):
    """
    Identifies a plateau in the classificaiton loss, based on `torch.optim.lr_scheduler.ReduceLROnPlateau`.
    """

    def __init__(self, model, pwe):
        """
        :param model: PyTorch model to evaluate, the state_dict of model corresponding to the best loss metric is
        memorized and restored when a plateau is found.
        :param pwe: Plateau Wating Epochs, the scheduler patience, i.e. how many adjacent bad epoch should pass before
        a plateau is identified.
        """
        self.model = model
        self.patience = pwe
        self.tolerance = 0.01

        self.best = inf
        self.best_dict = None

        self.num_bad_epochs = 0
        self.last_epoch = -1

    @torch.no_grad()
    def step(self, metrics, epoch=None):
        """
        Performs a scheduling step:
        - Evaluate if the current metric is better than the best found previously
        - If the current set it as the new best and reset the number of bad epochs
        - Else increase the number of bad epochs by one.
        if the number of bad epochs exceeds the scheduler patients it identities a perfomance plateau.
        :param metrics: Metric value, i.e. the classification loss.
        :param epoch: Iteration in which the metric values is computed.
        :return: True if a performance plateau is found, False otherwise.
        """
        current = float(metrics)

        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        if self._compare_with_best(current):
            self.num_bad_epochs = 0
            if current < self.best:
                self.best = current
                self.best_dict = deepcopy(self.model.state_dict())
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.model.load_state_dict(self.best_dict)
            self._reset()

            return True

        return False

    def _compare_with_best(self, current):
        """
        Check if the give value is less than the current lowest, given a relative tolerance
        :param current: Value compared to the current lowest
        :return: True if current is less then best given the tolerance, False otherwise
        """
        return current < (self.best + (self.best * self.tolerance))

    def _reset(self):
        """
        Reset the state of the scheduler to the initialization.
        """
        self.best = inf
        self.num_bad_epochs = 0
        self.best_dict = None
