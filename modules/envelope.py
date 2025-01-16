import torch
from torch.nn import functional as F

def amplitude_envelope(audio: torch.Tensor, n_frames: int) -> torch.Tensor:
    batch, channels, time = audio.shape
    step_size = time // n_frames
    window_size = step_size * 2
    amp = F.avg_pool1d(torch.abs(audio), window_size, step_size, padding=step_size)
    return amp