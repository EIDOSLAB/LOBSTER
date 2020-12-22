import torch

from .. import utilities


class LOBSTER:
    def __init__(self, model, lmbda, layers):
        """
        Initialize the LOBSTER regularizer.
        :param model: PyTorch model.
        :param lmbda: Lambda hyperparameter.
        :param layers: Tuple of layer on which apply the regularization e.g. (nn.modules.Conv2d, nn.modules.Linear)
        """
        self.model = model
        self.lmbda = lmbda
        self.layers = layers

    @torch.no_grad()
    def step(self, masks):
        """
        Regularization step.
        :param masks: Dictionary of type `layer: tensor` containing, for each layer of the network a tensor
        with the same size of the layer that is element-wise multiplied to the layer.
        See `utilities.get_mask_neur` or `utilities.get_mask_par` for an example of mask construction.
        """
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    name = "{}.{}".format(n_m, n_p)

                    insensitivity = torch.nn.functional.relu(1 - torch.abs(p.grad))
                    regu = p.mul(insensitivity)
                    p.add_(regu, alpha=-self.lmbda)

                    utilities.apply_masks(p, masks, name)
