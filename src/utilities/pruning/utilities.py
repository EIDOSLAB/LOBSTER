import torch
from torch import nn


@torch.no_grad()
def get_activation(preact_dict, param_name, hook_type):
    """
    Hooks used for in sensitivity schedulers (LOBSTE, Neuron-LOBSTER, SERENE).
    :param preact_dict: Dictionary in which save the parameters information.
    :param param_name: Name of the layer, used a dictionary key.
    :param hook_type: Hook type.
    :return: Returns a forward_hook if $hook_type$ is forward, else a backward_hook.
    """

    def forward_hook(model, inp, output):
        preact_dict[param_name] = output

    def backward_hook(module, grad_input, grad_output):
        preact_dict[param_name] = None
        preact_dict[param_name] = grad_output[0].detach().cpu()

    return forward_hook if hook_type == "forward" else backward_hook


@torch.no_grad()
def apply_mask_params(mask, tensor, tensor_key):
    """
    Element-wise multiplication between a tensor and the corresponding mask.
    :param mask: Dictionary containing the tensor mask at the given key.
    :param tensor: Tensor on which apply the mask.
    :param tensor_key: Key at which the mask for the tensor is stored in the dictionary.
    """
    for m in mask:
        tensor.mul_(m[tensor_key])


@torch.no_grad()
def apply_mask_neurons(mask, tensor, tensor_key):
    """
    Element-wise multiplication between a tensor and the corresponding mask.
    :param mask: Dictionary containing the tensor mask at the given key.
    :param tensor: Tensor on which apply the mask.
    :param tensor_key: Key at which the mask for the tensor is stored in the dictionary.
    """
    if len(tensor.shape) == 1:
        tensor.mul_(mask[tensor_key])
    elif len(tensor.shape) == 2:
        tensor.copy_(torch.einsum(
            'ij,i->ij',
            tensor,
            mask[tensor_key]
        ))
    elif len(tensor.shape) == 4:
        tensor.copy_(torch.einsum(
            'ijnm,i->ijnm',
            tensor,
            mask[tensor_key]
        ))


@torch.no_grad()
def substitute_module(model, new_module, sub_module_names):
    """
    Substitute a nn.module in a given PyTorch model with another.
    :param model: PyTorch model on which the substitution occurs.
    :param new_module: New module to insert in the model.
    :param sub_module_names: List of string representing the old module name.
    i.e if the module name is layer1.0.conv1 `sub_module_names` should be ["layer1", "0", "conv1"]
    """
    if new_module is not None:
        attr = model
        for idx, sub in enumerate(sub_module_names):
            if idx < len(sub_module_names) - 1:
                attr = getattr(attr, sub)
            else:
                setattr(attr, sub, new_module)


@torch.no_grad()
def find_module(model, name):
    """
    Find a module in the given model by name.
    :param model: PyTorch model.
    :param name: Module name
    :return: The searched module and the following.
    """
    found_module = False
    current_module = None
    next_module = None
    for module_name, module in model.named_modules():
        if len(list(module.children())) == 0:
            dict_name = "{}.weight".format(module_name)
            if name in dict_name:
                current_module = module
                found_module = True
                continue
            if found_module and not isinstance(module, nn.Identity):
                next_module = module
                break

    return current_module, next_module


@torch.no_grad()
def get_mask_neur(model, layers):
    """
    Defines a dictionary of type {layer: tensor} containing for each layer of a model, the binary mask representing
    which neurons have a value of zero (all of its parameters are zero).
    :param model: PyTorch model.
    :param layers: Tuple of layers on which apply the threshold procedure. e.g. (nn.modules.Conv2d, nn.modules.Linear)
    :return: Mask dictionary.
    """
    mask = {}
    for n_m, mo in model.named_modules():
        if isinstance(mo, layers):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)

                if "weight" in n_p:
                    if isinstance(mo, nn.modules.Linear):
                        sum = torch.abs(p).sum(dim=1)
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    elif isinstance(mo, nn.modules.Conv2d):
                        sum = torch.abs(p).sum(dim=(1, 2, 3))
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    else:
                        mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))
                else:
                    mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))

    return mask


@torch.no_grad()
def get_mask_par(model, layers):
    """
    Defines a dictionary of type {layer: tensor} containing for each layer of a model, the binary mask representing
    which parameters have a value of zero.
    :param model: PyTorch model.
    :param layers: Tuple of layers on which apply the threshold procedure. e.g. (nn.modules.Conv2d, nn.modules.Linear)
    :return: Mask dictionary.
    """
    mask = {}
    for n_m, mo in model.named_modules():
        if isinstance(mo, layers):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)
                mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))

    return mask


@torch.no_grad()
def magnitude_threshold(model, layers, T):
    """
    Performs magnitude thresholding on a network, all the elements of the tensor below a threshold are zeroed.
    :param model: PyTorch model on which apply the thresholding, layer by layer.
    :param layers: Tuple of layers on which apply the threshold procedure. e.g. (nn.modules.Conv2d, nn.modules.Linear)
    :param T: Threhsold value.
    """
    for n_m, mo in model.named_modules():
        if isinstance(mo, layers):
            for n_p, p in mo.named_parameters():
                p.copy_(torch.where(torch.abs(p) < T, torch.zeros_like(p), p))


@torch.no_grad()
def sensitivity_threshold(model, layers, T, sensitivity, layer_name):
    """
    Performs magnitude thresholding on a network, all the elements of the tensor below a threshold are zeroed.
    :param model: PyTorch model on which apply the thresholding, layer by layer.
    :param layers: Tuple of layers on which apply the threshold procedure. e.g. (nn.modules.Conv2d, nn.modules.Linear)
    :param T: Threhsold value.
    """
    for n_m, mo in model.named_modules():
        if n_m == layer_name:
            if isinstance(mo, layers):
                s = sensitivity[n_m]
                prune_mask = torch.where(s < T, torch.zeros_like(s), torch.ones_like(s))
                for n_p, p in mo.named_parameters():

                    if prune_mask.device != p.device:
                        prune_mask = prune_mask.to(p.device)

                    if "weight" in n_p:
                        if isinstance(mo, nn.modules.Linear):
                            p.copy_(torch.einsum(
                                'ij,i->ij',
                                p,
                                prune_mask
                            ))
                        elif isinstance(mo, nn.modules.Conv2d):
                            p.copy_(torch.einsum(
                                'ijnm,i->ijnm',
                                p,
                                prune_mask
                            ))
                        else:
                            p.copy_(torch.mul(p, prune_mask))

                        # Bias
                    else:
                        p.copy_(torch.mul(p, prune_mask))

                return prune_mask
