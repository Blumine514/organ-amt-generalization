"""Optimizer builders."""

import torch.optim as optim


def build_adamw(model, lr=3e-4, weight_decay=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
