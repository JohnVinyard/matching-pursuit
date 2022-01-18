import torch


def stft(x, ws=512, step=256):
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x)
    x = torch.abs(x)
    return x