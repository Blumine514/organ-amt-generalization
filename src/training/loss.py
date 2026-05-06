"""Loss functions."""

import torch.nn as nn


def build_bce_loss():
    return nn.BCEWithLogitsLoss()
