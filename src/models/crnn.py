"""CRNN baseline for AMT."""

import torch.nn as nn


class CRNNBaseline(nn.Module):
    def __init__(self, n_bins=229, n_pitches=88, hidden_size=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=64 * n_bins,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.frame_head = nn.Linear(hidden_size * 2, n_pitches)
        self.onset_head = nn.Linear(hidden_size * 2, n_pitches)

    def forward(self, x):
        # x: [batch, time, freq]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3).flatten(2)
        h, _ = self.rnn(x)
        return {
            "frame_logits": self.frame_head(h),
            "onset_logits": self.onset_head(h),
        }
