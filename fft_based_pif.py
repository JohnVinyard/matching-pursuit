import torch


def fft_based_pif(audio: torch.Tensor, freq_window_size: int, time_window_size: int):
    batch_size = audio.shape[0]
    spec: torch.Tensor = torch.fft.rfft(audio, dim=-1)
    
    freq_step = freq_window_size // 2
    
    windowed = spec.unfold(-1, freq_window_size, freq_step)
    windowed = windowed * torch.hamming_window(freq_window_size, device=windowed.device)[None, None, None, :]
    channels = torch.fft.irfft(windowed, dim=-1)
    n_channels = channels.shape[2]
    
    time_domain_window_size = time_window_size
    time_domain_step_size = time_domain_window_size // 2
    
    channels = channels.view(batch_size, n_channels, -1)
    channels = channels.unfold(-1, time_domain_window_size, time_domain_step_size)
    channels = channels * torch.hamming_window(channels.shape[-1])[None, None, None, :]
    
    # we want to capture fine-scale information in a phase-invariant way
    # just keep the magnitudes
    channels = torch.abs(torch.fft.rfft(channels, dim=-1))
    
    return channels

if __name__ == '__main__':
    n_samples = 2**15
    n_coeffs = n_samples // 2 + 1
    
    batch_size = 4
    audio = torch.zeros(batch_size, 1, n_samples).uniform_(-1, 1)
    
    # factor this out
    
    batch_size = audio.shape[0]
    spec: torch.Tensor = torch.fft.rfft(audio, dim=-1)
    
    freq_window_size = 256
    freq_step = freq_window_size // 2
    
    windowed = spec.unfold(-1, freq_window_size, freq_step)
    windowed = windowed * torch.hamming_window(freq_window_size, device=windowed.device)[None, None, None, :]
    channels = torch.fft.irfft(windowed, dim=-1)
    n_channels = channels.shape[2]
    
    time_domain_window_size = 64
    time_domain_step_size = time_domain_window_size // 2
    
    channels = channels.view(batch_size, n_channels, -1)
    channels = channels.unfold(-1, time_domain_window_size, time_domain_step_size)
    channels = channels * torch.hamming_window(channels.shape[-1])[None, None, None, :]
    channels = torch.abs(torch.fft.rfft(channels, dim=-1))
    print(channels.shape)