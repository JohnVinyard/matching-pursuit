from base64 import b64encode
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt



def create_data_url(b: bytes, content_type: str):
    return  f'data:{content_type};base64,{b64encode(b).decode()}'

def spectrogram(audio: torch.Tensor, window_size: int = 2048, step_size: int = 256):
    audio = audio.view(1, 1, n_samples)
    spec = stft(audio, window_size, step_size, pad=True)
    n_coeffs = window_size // 2 + 1
    spec = max_norm(spec.view(-1)).view(-1, n_coeffs)
    spec = spec.data.cpu().numpy()
    spec = np.rot90(spec)
    
    img_data = np.zeros((spec.shape[0], spec.shape[1], 4), dtype=np.uint8)
    
    img_data[:, :, 3:] = np.clip((spec[:, :, None] * 255).astype(np.uint8), 0, 255)
    img_data[:, :, :3] = 0
    
    
    img = Image.fromarray(img_data, mode='RGBA')
    img.save('spec.png', format='png')
    

# TODO: try matrix rotation instead: https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
def to_polar(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    return mag, phase

def to_complex(mag, phase):
    return mag * torch.exp(1j * phase)

def advance_one_frame(x):
    mag, phase = to_polar(x)
    phase = phase + torch.linspace(0, np.pi, x.shape[-1])[None, None, :]
    x = to_complex(mag, phase)
    return x


def test():

    n_samples = 2 ** 15
    window_size = 1024
    step_size = window_size // 2
    n_coeffs = window_size // 2 + 1
    
    impulse = torch.zeros(1, 1, 2048).uniform_(-1, 1)
    impulse = F.pad(impulse, (0, n_samples - 2048))
    windowed = windowed_audio(impulse, window_size, step_size)
    
    n_frames = windowed.shape[-2]
    
    transfer_func = torch.zeros(1, n_coeffs).uniform_(0, 0.99)
    print(torch.norm(transfer_func).item())
    transfer_warp = torch.eye(n_coeffs)
    transfer_warp = torch.roll(transfer_warp, (0, 4), dims=(0, 1))
    
    
    frames = []
    
    for i in range(n_frames):
        
        transfer_func = transfer_func @ transfer_warp
        print(torch.norm(transfer_func).item())
        
        if i == 0:
            spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
            spec = spec * transfer_func
            audio = torch.fft.irfft(spec, dim=-1)
            frames.append(audio)
        else:
            prev = frames[i - 1]
            prev_spec = torch.fft.rfft(prev, dim=-1)
            prev_spec = advance_one_frame(prev_spec)
            
            current_spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
            spec = current_spec + prev_spec
            spec = spec * transfer_func
            audio = torch.fft.irfft(spec, dim=-1)
            frames.append(audio)
    
    
    frames = torch.cat([f[:, :, None, :] for f in frames], dim=2)
    audio = overlap_add(frames, apply_window=True)[..., :n_samples]
    
    return audio


def dumb_shifted_time_matrix(samples: int, frames: int) -> np.ndarray:
    t = np.linspace(0, 1, num=samples)
    accum = []
    
    
    # TODO: This just happens to work because frames and samples
    # are equal here
    
    # frame_rate = samples // frames
    # print(samples, frames, frame_rate)
    
    for i in range(frames):
        shifted = np.roll(t, shift=i)
        shifted[:i] = 0
        accum.append(shifted[None, :])
    
    # TODO: How do I make a rectangular mask?
    accum = np.concatenate(accum, axis=0)
    return accum



def another_shifted_time_matrix(samples: int, frames: int) -> np.ndarray:
    t1 = np.linspace(0, 1, num=samples)
    shifts = np.linspace(0, 1, num=frames)
    
    shifted = fft_shift(
        torch.from_numpy(t1[None, :]), 
        torch.from_numpy(shifts[:, None])
    ).data.cpu().numpy()
    
    # TODO: figure out how to mask this and/or pad/window to avoid
    # FFT artifacts
    
    return shifted


def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    n_samples = a.shape[-1]
    orig_coeffs = n_samples // 2 + 1
    
    shift_samples = (shift * n_samples) * (1/3)
    
    a = F.pad(a, (0, n_samples * 2))
    
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
    # shift = torch.linspace(0, 1, steps=n_coeffs) * 2j * np.pi
    
    shift = torch.exp(-shift * shift_samples)

    spec = spec * shift
    
    diff = spec.shape[-1] - orig_coeffs
    # spec[..., orig_coeffs:] *= np.linspace(1, 0, diff) ** 2
    
    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = samples[..., :n_samples]
    return samples

# def fft_shift(a: torch.Tensor, shift: torch.Tensor):
#     n_samples = a.shape[-1]
#     shift_samples = shift * n_samples
#     spec = torch.fft.rfft(a, dim=-1, norm='ortho')
#     n_coeffs = spec.shape[-1]
#     shift = (torch.arange(0, n_coeffs) * 2j * np.pi).to(a.device) / n_coeffs
#     shift = torch.exp(-shift * shift_samples)
#     spec = spec * shift
#     samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
#     samples = samples[..., :n_samples]
#     return samples


def damped(
        n_t: int, 
        n_frames: int,
        amplitude: np.ndarray, 
        friction: float, 
        mass: float,
        use_fft: bool = False):
    
    """Implementation of a damped harmonic oscillator
    """
    
    time_matrix1 = dumb_shifted_time_matrix(n_t, n_frames)
    time_matrix2 = another_shifted_time_matrix(n_t, n_frames)
    
    print(time_matrix1.shape)
    print(time_matrix2.shape)
    
    plt.matshow(time_matrix1)
    plt.show()
    plt.matshow(time_matrix2)
    plt.show()
    
    
    if use_fft:
        time_matrix = time_matrix2
        mask = np.zeros((n_frames, time_matrix.shape[-1]))
        row, col = torch.triu_indices(*mask.shape, offset=2).data.cpu().numpy()
        mask[row, col] = 1
    else:
        time_matrix = time_matrix1
        mask = time_matrix > 0
    
    
    x = amplitude[:, None] * (np.e **((-friction / (2 * mass) * time_matrix)))
    x = x * mask
    
    plt.matshow(x)
    plt.show()
    
    x = np.sum(x, axis=0)
    return x


# Note, eventually, we'll be adding batch and n_events dimensions


def instrument(
        t: torch.Tensor, 
        shift: torch.Tensor, 
        energy: torch.Tensor, 
        shape: torch.Tensor,
        properties: torch.Tensor):
    
    
    batch, n_events, time = t.shape
    
    print(t.shape, shift.shape)
    # right away, we apply the time shifts to each
    # positional encoding
    t = fft_shift(t, shift)
    plt.matshow(t[0])
    plt.show()
    raise NotImplementedError()
    
    
    _, _, cp, n_frames = energy.shape
    assert energy.shape == shape.shape
    
    assert properties.shape == (batch, n_events, cp, 2)
    
    mass = properties[..., :1]
    friction = properties[..., 1:]
    
    # TODO: This is currently unused, but would be used by
    # fft_shift
    frame_shifts = torch.linspace(0, 1, steps=n_frames)
    
    frame_rate = time // n_frames
    
    expanded_t = t.view(batch, n_events, 1, time).repeat(1, 1, n_frames, 1)
    
    # TODO: This is where FFT shift comes in
    
    # This is the beginning, default positional encoding
    for i in range(n_frames):
        shift = frame_rate * i
        # shift and mask
        expanded_t[:, :, i, :] = torch.roll(expanded_t[:, :, i, :], shift, dims=-1)
        expanded_t[:, :, i, :shift] = 0
    
    
    print(expanded_t.shape)
    
    plt.matshow(expanded_t[0, 0, :, :].data.cpu().numpy())
    plt.show()



def test_shift():
    n_samples = 128
    
    signal = torch.zeros(n_samples)
    signal[0] = 1
    plt.plot(signal)
    plt.show()
    
    
    shifted = fft_shift(signal, torch.zeros(1).fill_(0.5))
    plt.plot(shifted)
    plt.show()

    index = torch.argmax(shifted, dim=-1)
    assert index.item() == 64



if __name__ == '__main__':
    
    n_samples = 128
    n_frames = 64
    batch_size = 2
    n_events = 8
    control_plane = 32
    
    # time encoding, a linear ramp from 0 to 1 for
    # each event 
    t = torch.linspace(0, 1, steps=n_samples)\
        .view(1, 1, n_samples)\
        .repeat(batch_size, n_events, 1)
    
    # random time shifts for each event
    shifts = torch.zeros(batch_size, n_events, 1).uniform_(0, 0.5)
    
    energy = torch.zeros(
        batch_size, n_events, control_plane, n_frames).bernoulli_(p=0.01)
    
    shape = torch.zeros(
        batch_size, n_events, control_plane, n_frames).uniform_(-1, 1)
    
    properties = torch.zeros(
        batch_size, n_events, control_plane, 2).uniform_(0.01, 1)
    
    result = instrument(t, shifts, energy, shape, properties)
    
    
    # n_t = 128
    # n_frames = 128
    
    
    # amplitude = np.zeros(n_frames)
    # amplitude[0] = 8
    # amplitude[10] = 20
    # amplitude[73] = 2
    
    # friction = 1
    # mass = 0.1
    
    # x1 = damped(n_t, n_frames, amplitude, friction, mass, use_fft=False)        
    # x2 = damped(n_t, n_frames, amplitude, friction, mass, use_fft=True)        
    
    # plt.plot(x1)
    # plt.plot(x2)
    # plt.show()
    
    
    
    
    