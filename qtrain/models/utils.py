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
