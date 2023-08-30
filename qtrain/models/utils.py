import inspect
import torch


def freeze_layers(layer_names, model):
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = False


def unfreeze_layers(layer_names, model):
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = True


def is_from_torchmetrics(func):
    if not callable(func):
        return False
    module_name = inspect.getmodule(func).__name__
    return module_name.startswith("torchmetrics")


def put_torchmetric_to_device(func, device):
    if is_from_torchmetrics(func):
        func.to(device)
    else:
        if hasattr(func, "weight"):
            if func.weight is not None:
                func.weight = func.weight.to(device)
    return func
