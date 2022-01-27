import torch


def stft(x, ws=512, step=256):
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')
    x = torch.abs(x)
    return x


def stft_relative_phase(x, ws=512, step=256):
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')

    # get the magnitude
    mag = torch.log(1e-4 + torch.abs(x))

    # compute the angle in radians
    phase = torch.angle(x)

    # get instantaneous frequency.  We should now be phase
    # agnostic, while still emphasizing periodicity
    padding = torch.zeros(phase.shape[1]).to(x.device)[None, :, None]
    phase = torch.diff(phase, axis=-1, prepend=padding)

    return mag, phase


def log_stft(x, ws=512, step=256, a=0.001):
    x = stft(x, ws, step)
    return torch.log(a + x)
