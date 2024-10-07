from torch import nn
import torch

from modules.normalization import max_norm


class OverfitRawAudio(nn.Module):
    def __init__(self, shape, std=1, normalize=False):
        super().__init__()
        self.audio = nn.Parameter(torch.zeros(*shape).normal_(0, std))
        self.normalize = normalize

    @property
    def as_numpy_array(self):
        return self.audio.data.cpu().numpy()

    def forward(self, _):
        output = self.audio
        if self.normalize:
            output = max_norm(output, dim=-1)
        return output
