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

        self.hooks = []
        self.eps = torch.tensor([1e-10])

    @torch.no_grad()
    def step(self, mask_params):
        """
        Regularization step.
        :param masks: Dictionary of type `layer: tensor` containing, for each layer of the network a tensor
        with the same size of the layer that is element-wise multiplied to the layer.
        See `utilities.get_mask_neur` or `utilities.get_mask_par` for an example of mask construction.
        :param rescaled: If True rescale the sensitivity in [0, 1] as sensitivty /= max(sensitivity)
        """
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    name = "{}.{}".format(n_m, n_p)

                    sensitivity = torch.abs(p.grad)

                    # TODO assert sens in [0,1]

                    insensitivity = (1. - sensitivity).to(p.device)

                    insensitivity = torch.nn.functional.relu(insensitivity)

                    regu = p.mul(insensitivity)
                    p.add_(regu, alpha=-self.lmbda)

                    if mask_params is not None:
                        utilities.apply_mask_params(mask_params, p, name)

    @torch.enable_grad()
    def evaluate_sensitivity(self, dataloader, loss_function, device):

        sensitivity = {}

        for data, target in dataloader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            self.model.zero_grad()

            output = self.model(data)
            loss = loss_function(output, target)
            loss.backward()

            for param_name, p in self.model.named_parameters():
                if "weight" in param_name:

                    if param_name in sensitivity.keys():
                        sensitivity[param_name] = sensitivity[param_name] + torch.abs(p).detach().cpu()
                    else:
                        sensitivity[param_name] = torch.abs(p).detach().cpu()

        for k in sensitivity.keys():
            sensitivity[k] /= len(dataloader)
            sensitivity[k] /= torch.max(torch.max(sensitivity[k], self.eps))

        return sensitivity

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
