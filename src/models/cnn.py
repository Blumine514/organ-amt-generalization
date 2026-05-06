"""Simple CNN baseline."""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, n_bins=229, n_pitches=88):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64 * n_bins, n_pitches)

    def forward(self, x):
        # x: [batch, time, freq]
        x = x.unsqueeze(1)          # [batch, 1, time, freq]
        x = self.net(x)             # [batch, channels, time, freq]
        x = x.permute(0, 2, 1, 3)   # [batch, time, channels, freq]
        x = x.flatten(2)            # [batch, time, channels * freq]
        return self.classifier(x)   # [batch, time, pitches]
