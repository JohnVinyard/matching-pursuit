import torch
from scipy.signal import tukey

def fft_frequency_decompose(x, min_size):
    # coeffs = torch.rfft(input=x, signal_ndim=1, normalized=True)
    coeffs = torch.fft.rfft(x, norm='ortho')

    def make_mask(size, start, stop):
        mask = torch.zeros(size).to(x.device)
        mask[start:stop] = 1
        return mask[None, None, :]

    output = {}

    current_size = min_size

    while current_size <= x.shape[-1]:
        sl = coeffs[:, :, :current_size // 2 + 1]
        if current_size > min_size:
            mask = make_mask(
                size=sl.shape[2],
                start=current_size // 4,
                stop=current_size // 2 + 1)
            sl = sl * mask

        # recon = torch.irfft(
        #     input=sl,
        #     signal_ndim=1,
        #     normalized=True,
        #     signal_sizes=(current_size,))

        recon = torch.fft.irfft(sl, n=current_size, norm='ortho')

        # if recon.shape[-1] != x.shape[-1]:
        #     recon = torch.zeros_like(recon)

        output[recon.shape[-1]] = recon
        current_size *= 2
    
    return output


def fft_resample(x, desired_size, is_lowest_band):
    batch, channels, time = x.shape
    coeffs = torch.fft.rfft(x, norm='ortho')
    # (batch, channels, coeffs, 2)
    n_coeffs = coeffs.shape[2]

    new_coeffs_size = desired_size // 2 + 1
    new_coeffs = torch.zeros(batch, channels, new_coeffs_size, dtype=torch.complex64).to(x.device)
    if is_lowest_band:
        new_coeffs[:, :, :n_coeffs] = coeffs
    else:
        new_coeffs[:, :, n_coeffs // 2:n_coeffs] = coeffs[:, :, n_coeffs // 2:]


    samples = torch.fft.irfft(
        new_coeffs,
        n=desired_size,
        norm='ortho')
    return samples


def fft_frequency_recompose(d, desired_size):
    bands = []
    first_band = min(d.keys())
    for size, band in d.items():
        resampled = fft_resample(band, desired_size, size == first_band)
        # if size != desired_size:
        #     resampled = torch.zeros_like(resampled)
        bands.append(resampled)
    return sum(bands)