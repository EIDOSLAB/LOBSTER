from cmath import inf
from copy import deepcopy

import torch

from .plateau_scheduler import Scheduler as plateau_scheduler
from .. import magnitude_threshold
from ...evaluation import test_model


class Scheduler(object):
    """
    Identifies the highest threshold value that lead to a worsening in performance at most of a relative $twt$ amount.
    """

    def __init__(self, model, layers, valid_loader, loss_function, twt, pwe, device, task):
        """
        :param model: PyTorch model
        :param layers: Tuple of layers considered for the pruning procedure
        :param valid_loader: Validation DataLoader
        :param loss_function: Loss function
        :param twt: Threshold Worsening Tolerance, relative amount that define the maximum tolerated worsening in performance (classification loss)
        :param pwe: Plateau Waiting Epochs, amount of epochs needed to define a performance plateau
        """
        self.model = model
        self.layers = layers
        self.starting_state = None

        self.valid_loader = valid_loader
        self.loss_function = loss_function

        self.twt = twt

        self.device = device
        
        self.task = task

        self.a = -inf
        self.b = inf
        self.center = 0

        self.loss_a = 0
        self.loss_b = 0
        self.bound = 0

        self.plateau = plateau_scheduler(model, pwe)

    @torch.no_grad()
    def step(self, loss, epoch=None):
        """
        Perform a single scheduler step, first check if the current epoch correspond to a performance plateau,
        then look for the highest threshold value that lead to a worsening in performance at most by `twt`.
        :param loss: Model classification loss used to identify the eventual plateau.
        :param epoch: Iteration in which the metric values is computed.
        :return: True if a threshold value is found, False otherwise.
        """

        if self.plateau.step(loss, epoch):
            print("Pruning")
            self.starting_state = deepcopy(self.model.state_dict())
            best_T = self._find_threshold_with_bisection()
            if best_T is not None:
                magnitude_threshold(self.model, self.layers, best_T)
                return True
            else:
                return False
        else:
            return False

    def set_validation_loader(self, valid_loader):
        """
        Change the validation DataLoader used to compute the model's loss.
        :param valid_loader: Validation DataLoader
        """
        self.valid_loader = valid_loader

    def _find_min_T(self):
        """
        Get the lowest parameters value in the network.
        :return: Lowest parameter value
        """
        min_w = inf
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    if "weight" in n_p:
                        data = torch.abs(p).clone().detach()
                        p_min = data[data != 0].min()
                        if float(p_min) < min_w:
                            min_w = float(p_min)

        return min_w

    def _find_max_T(self):
        """
        Get the highest parameters value in the network.
        :return: Highest parameter value
        """
        max_w = 0
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    if "weight" in n_p:
                        data = torch.abs(p).clone().detach()
                        p_max = data[data != 0].max()
                        if float(p_max) > max_w:
                            max_w = float(p_max)

        return max_w

    def _find_mean_T(self):
        """
        Get the average parameters value in the network.
        :return: Average parameter value
        """
        mean_w = 0
        layers = 0
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    if "weight" in n_p:
                        data = torch.abs(p.data).clone().detach()
                        mean_w += data[data != 0].mean()
                        layers += 1
                        del data

        return mean_w / layers

    def _find_threshold_with_bisection(self):
        """
        Employ a bisection approach to find the best threshold value.
        :return: Threshold value.
        """
        mean_T = self._find_mean_T()

        _, _, loss_base_model = test_model(self.model, self.loss_function, self.valid_loader, self.device, self.task)
        self.bound = loss_base_model + (loss_base_model * self.twt)

        magnitude_threshold(self.model, self.layers, mean_T)
        _, _, loss_threshold_model = test_model(self.model, self.loss_function, self.valid_loader, self.device, self.task)
        self.model.load_state_dict(self.starting_state)

        if loss_threshold_model < self.bound:
            self.a = mean_T
            self.loss_a = loss_threshold_model
            self.b = self._find_max_T()
            self.loss_b = inf
        else:
            self.a = self._find_min_T()
            self.loss_a = loss_base_model
            self.b = mean_T
            self.loss_b = loss_threshold_model

        previous_T = inf
        eps = 1e-10

        while not self._check():
            self.center = (self.a + self.b) / 2

            if abs(previous_T - self.center) <= eps:
                return self.a

            previous_T = self.center

            magnitude_threshold(self.model, self.layers, self.center)
            _, _, loss_threshold_model = test_model(self.model, self.loss_function, self.valid_loader, self.device, self.task)
            self.model.load_state_dict(self.starting_state)

            if loss_threshold_model <= self.bound:
                self.a = self.center
                self.loss_a = loss_threshold_model
            else:
                self.b = self.center
                self.loss_b = loss_threshold_model

        return (self.a + self.b) / 2

    def _check(self, delta=0.05):
        loss_diff = abs(self.loss_a - self.loss_b)
        return True if loss_diff < self.bound * delta else False
