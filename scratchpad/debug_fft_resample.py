from matplotlib import pyplot as plt
import numpy as np
import torch


def make_window(size, fade_in, device):
    window = torch.ones(size, device=device)
    window[:fade_in] = torch.linspace(0, 1, fade_in, device=device)
    window[-fade_in:] = torch.linspace(1, 0, fade_in, device=device)
    return window


def window_signal(signal, fade_in):
    size = signal.shape[-1]
    win = make_window(size, fade_in, device=signal.device)
    new_sig = signal * win
    return new_sig

def fft_frequency_decompose(x, min_size):
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

        recon = torch.fft.irfft(sl, n=current_size, norm='ortho')

        output[recon.shape[-1]] = recon
        current_size *= 2
    
    return output


def fft_resample(x, desired_size, is_lowest_band):
    batch, channels, time = x.shape

    coeffs = torch.fft.rfft(x, norm='ortho')
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
        bands.append(resampled)
    return sum(bands)

"""
1. take the fft of the entire signal
2. take the inverse fft of windows of the whole fft to produce bands
3. upsample each band to the full sample rate
4. add all upsampled bands together
"""

def is_fft_resample_part_of_the_problem(tick_size, final_size):
    """
    TODO: listen to each
    """
    signal = torch.zeros(1, 1, tick_size)
    start = tick_size // 2
    end = start + 10
    signal[:, :, start: end] = torch.zeros(10).uniform_(-1, 1)

    orig_spec = torch.abs(torch.fft.rfft(signal, dim=-1, norm='ortho'))
    print('orig_spec')
    plt.plot(orig_spec.data.cpu().numpy().squeeze())
    plt.show()

    resampled = fft_resample(signal, final_size, is_lowest_band=True)
    new_spec = torch.abs(torch.fft.rfft(resampled, dim=-1, norm='ortho'))
    print('new spec')
    plt.plot(new_spec.data.cpu().numpy().squeeze()[:orig_spec.shape[-1] + 10])
    plt.show()


if __name__ == '__main__':
    n_samples = 32768

    sizes = [512, 1024, 2048, 4096, 8192]

    for size in sizes:
        is_fft_resample_part_of_the_problem(size, n_samples)
    

    # create a test signal with white noise
    # signal = torch.zeros(1, 1, n_samples)
    # noise = torch.zeros(10).uniform_(-1, 1)
    # signal[:, :, 8192: 8192 + 10] = noise

    # print(f'Original signal')
    # plt.plot(signal.data.cpu().numpy().reshape((-1,)))
    # plt.show()

    # orig_spec = torch.abs(torch.fft.rfft(signal, dim=-1, norm='ortho'))
    # print(f'Original spec')
    # plt.plot(orig_spec.data.cpu().numpy().reshape((-1,)))
    # plt.show()


    # # decompose
    # bands = fft_frequency_decompose(signal, 512)

    # # recon
    # recon = fft_frequency_recompose(bands, n_samples)
    # print(f'Recon signal')
    # plt.plot(signal.data.cpu().numpy().reshape((-1,)))
    # plt.show()

    # recon_spec = torch.abs(torch.fft.rfft(recon, dim=-1, norm='ortho'))
    # print(recon_spec.shape)
    # indices = torch.argsort(recon_spec.view(-1), descending=True)
    # print(indices[:16])

    # print(f'Recon spec')
    # plt.plot(recon_spec.data.cpu().numpy().reshape((-1,)))
    # plt.show()


    # # check the spectrum of each band
    # # for size, band in bands.items():
    # #     print('=========================================')
    # #     print(size)

    # #     print(f'band norm is {torch.norm(band).item()}')

    # #     spec = torch.fft.rfft(band, dim=-1, norm='ortho')

    # #     disp = spec.data.cpu().numpy().reshape((-1,))
    # #     disp = np.abs(disp)

    # #     print(f'band size {size} spectrum')
    # #     plt.plot(disp)
    # #     plt.show()

    #     # up = fft_resample(band, n_samples, size == 512)
    #     # spec = torch.fft.rfft(up, dim=-1, norm='ortho')
    #     # disp = spec.data.cpu().numpy().reshape((-1,))
    #     # disp = np.abs(disp)

    #     # print(f'band size upsampled {size} spectrum')
    #     # plt.plot(disp)
    #     # plt.show()