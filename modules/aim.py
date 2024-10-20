import torch
from torch.nn import functional as F
from modules.transfer import fft_convolve


def rectified_filter_bank(
    signal: torch.Tensor,
    filters: torch.Tensor,
):
    n_samples = signal.shape[-1]

    n_filters, n_taps = filters.shape

    filters = filters.view(1, n_filters, n_taps)
    padded_filters = F.pad(filters, (0, n_samples - n_taps))
    spec = fft_convolve(signal, padded_filters)

    # half-wave rectification
    spec = torch.relu(spec)
    return spec


def auditory_image_model(
        signal: torch.Tensor,
        filters: torch.Tensor,
        aim_window_size:int,
        aim_step_size: int) -> torch.Tensor:

    # n_samples = signal.shape[-1]
    #
    # n_filters, n_taps = filters.shape
    #
    # filters = filters.view(1, n_filters, n_taps)
    # padded_filters = F.pad(filters, (0, n_samples - n_taps))
    # spec = fft_convolve(signal, padded_filters)
    #
    # # half-wave rectification
    # spec = torch.relu(spec)

    spec = rectified_filter_bank(signal, filters)
    spec = spec.unfold(-1, aim_window_size, aim_step_size)
    aim = torch.abs(torch.fft.rfft(spec, dim=-1))
    return aim