from torch import nn
import torch


class OverfitRawAudio(nn.Module):
    def __init__(self, shape, std=1):
        super().__init__()
        self.audio = nn.Parameter(torch.zeros(*shape).normal_(0, std))

    def forward(self, _):
        return self.audio
