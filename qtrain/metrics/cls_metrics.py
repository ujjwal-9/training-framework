import torch
import numpy as np


def get_sens_spec_youden(confusion_matrix):
    sens = (confusion_matrix["tp"] + 1e-6) / (
        confusion_matrix["tp"] + confusion_matrix["fn"] + 1e-6
    )
    spec = (confusion_matrix["tn"] + 1e-6) / (
        confusion_matrix["tn"] + confusion_matrix["fp"] + 1e-6
    )
    youden = sens + spec - 1
    return sens, spec, youden
