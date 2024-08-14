from base64 import b64encode
from subprocess import PIPE, Popen
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
from io import BytesIO
from soundfile import SoundFile
from scipy.signal import stft

from modules.hypernetwork import HyperNetworkLayer
from modules.transfer import fft_convolve


"""
The NERF-like network is familiar, but still requires a scan, meaning
it would be impossible to render an individual sample of an event.

Path forward:
    - filterbank that doesn't use zounds so I can plot energy
    - view energy to understand static behavior of NERF network
    - finish `instrument()` implementation
    - unify NERF network and `instrument()`
    - fft_shift experiment - consider a straight-through-estimator
"""

def listen_to_sound(samples: np.ndarray, samplerate: int, wait_for_user_input: bool = True) -> None:
    proc = Popen(f'aplay', shell=True, stdin=PIPE)
    
    bio = BytesIO()
    with SoundFile(
            bio, 
            mode='w',
            channels=1,
            samplerate=samplerate, 
            format='WAV', 
            subtype='PCM_16') as sf:
        
        sf.write(samples)
    
    bio.seek(0)
    
    if proc.stdin is not None:
        proc.stdin.write(bio.read())
        proc.communicate()
    
    if wait_for_user_input:
        input('Next')


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
    
    shift_samples = (shift * n_samples) * (1/3)
    
    a = F.pad(a, (0, n_samples * 2))
    
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
    
    shift = torch.exp(-shift * shift_samples)

    spec = spec * shift
    
    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = samples[..., :n_samples]
    return samples



def damped(
        n_t: int, 
        n_frames: int,
        amplitude: np.ndarray, 
        friction: float, 
        mass: float,
        use_fft: bool = True):
    
    """Implementation of a damped harmonic oscillator
    """
    
    if use_fft:
        time_matrix2 = another_shifted_time_matrix(n_t, n_frames)
        time_matrix = time_matrix2
        mask = np.zeros((n_frames, time_matrix.shape[-1]))
        row, col = torch.triu_indices(*mask.shape, offset=2).data.cpu().numpy()
        mask[row, col] = 1
    else:
        time_matrix1 = dumb_shifted_time_matrix(n_t, n_frames)
        time_matrix = time_matrix1
        mask = time_matrix > 0
    
    
    x = amplitude[:, None] * (np.e **((-friction / (2 * mass) * time_matrix)))
    x = x * mask
    
    plt.matshow(x)
    plt.show()
    
    x = np.sum(x, axis=0)
    return x


def instrument(
        t: torch.Tensor, 
        shift: torch.Tensor, 
        energy: torch.Tensor, 
        properties: torch.Tensor):
    
    
    batch, n_events, time = t.shape
    
    # right away, we apply the time shifts to each
    # positional encoding
    t = fft_shift(t, shift)
    
    _, _, cp, n_frames = energy.shape
    # assert energy.shape == shape.shape
    
    assert properties.shape == (batch, n_events, cp, 2)
    
    frame_shifts = torch.linspace(0, 1, steps=n_frames)
    
    # now we expand the time encoding, and apply constant, monotonically-increasing
    # shifts for each frame
    expanded_t = t.view(batch, n_events, 1, time).repeat(1, 1, n_frames, 1)
    expanded_t = fft_shift(expanded_t, frame_shifts[None, None, :, None])
    
    # we introduce a new dimension for the control plane
    expanded_t = expanded_t.view(batch, n_events, 1, n_frames, time)
    
    # create a mask to apply to pre t0 elements
    mask = torch.zeros((n_frames, time))
    row, col = torch.triu_indices(*mask.shape, offset=2)
    mask[row, col] = 1
    
    # shift the mask
    mask = fft_shift(mask[None, None, :, :], shift[:, :, None, :])
    
    mass = properties[..., :1]
    friction = properties[..., 1:]

    x = energy[..., None] * (np.e **((-friction[..., None] / (2 * mass[..., None]) * expanded_t)))
    x = x * mask[:, :, None, :, :]
    
    x = torch.sum(x, dim=-2)
    
    return x
    


def exponential_decay(
        decay_values: torch.Tensor, 
        n_atoms: int, 
        n_frames: int, 
        base_resonance: float,
        n_samples: int):
    
    # decay_values = torch.sigmoid(decay_values.view(-1, n_atoms, 1).repeat(1, 1, n_frames))
    decay_values = decay_values.view(-1, n_atoms, 1).repeat(1, 1, n_frames)
    resonance_factor = (1 - base_resonance) * 0.99
    decay = base_resonance + (decay_values * resonance_factor)
    decay = torch.log(decay + 1e-12)
    decay = torch.cumsum(decay, dim=-1)
    decay = torch.exp(decay).view(-1, n_atoms, n_frames)
    
    if n_samples != n_frames:    
        decay = F.interpolate(decay, size=n_samples, mode='linear')
    
    return decay

class Instrument3(nn.Module):
    
    def __init__(
            self, 
            encoding_channels: int, 
            channels: int, 
            n_frames: int, 
            n_samples: int,
            shape_channels: int):
        
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.shape_channels = shape_channels
        
        self.hyper = HyperNetworkLayer(
            shape_channels, 64, channels, encoding_channels)
        
    def _pos_encoding(self, n_samples: int):
        """Returns a filterbank with periodic functions
        """
        freqs = torch.linspace(0.00001, 0.49, steps=self.encoding_channels) ** 2
        t = torch.linspace(0, n_samples, steps=n_samples)
        p = torch.sin(t[None, :] * freqs[:, None] * np.pi)
        p = p.view(1, 1, self.encoding_channels, self.n_samples)
        return p

    def forward(
            self, 
            energy: torch.Tensor,
            transforms: torch.Tensor,
            decays: torch.Tensor):
        
        batch, n_events, cp, frames = energy.shape
        
        pos = self._pos_encoding(self.n_samples)
        
        envelopes = exponential_decay(
            decay_values=decays,
            n_atoms=n_events,
            n_frames=frames,
            base_resonance=0.5,
            n_samples=frames
        )
        envelopes = envelopes.view(batch, n_events, cp, frames)
        
        energy = fft_convolve(energy, envelopes)
        energy = torch.tanh(energy)
        orig_energy = energy
        
        energy = energy.permute(0, 1, 3, 2)
        

        # the shape describes how the control plane translates into
        # a mixture of resonators        
        _, _, shape_shape, shape_frames = transforms.shape
        transforms = transforms.view(batch * n_events, shape_shape, shape_frames)
        transforms = F.interpolate(transforms, size=self.n_frames, mode='linear')
        transforms = transforms.view(batch, n_events, shape_shape, frames)
        
        transforms = transforms.permute(0, 1, 3, 2)
        w, fwd = self.hyper.forward(transforms)
        
        energy = energy.reshape(-1, self.channels)
        transformed = fwd(energy)
        transformed = transformed.view(batch, n_events, frames, self.encoding_channels)
        transformed = transformed.permute(0, 1, 3, 2).view(batch * n_events, self.encoding_channels, self.n_frames)
        transformed = F.interpolate(transformed, size=self.n_samples, mode='linear')
        transformed = transformed.view(batch, n_events, self.encoding_channels, self.n_samples)
        
        
        final = pos * transformed
        final = torch.sum(final, dim=2)
        
        return final, orig_energy
        
class InstrumentStack(nn.Module):
    def __init__(
            self, 
            encoding_channels: int, 
            channels: int, 
            n_frames: int, 
            n_samples: int,
            shape_channels: int,
            n_layers: int):
        
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.shape_channels = shape_channels
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            Instrument3(
                encoding_channels, 
                channels, 
                n_frames, 
                n_samples, 
                shape_channels) 
            for _ in range(self.n_layers)
        ])
    
    def forward(
            self, 
            energy: torch.Tensor, 
            transforms: List[torch.Tensor],
            decays: List[torch.Tensor],
            mix: torch.Tensor):
        
        batch, n_events, layers = mix.shape
        
        batch, n_events, channels, frames = energy.shape
        
        e = energy
        output = torch.zeros(batch, n_events, self.n_layers, self.n_samples)
        
        for i, layer in enumerate(self.layers):
            print(i, e.shape)
            audio, e = layer.forward(e, transforms[i], decays[i])
            output[:, :, i, :] = audio
        
        mx = torch.softmax(mix, dim=-1)
        
        output = output * mx[:, :, :, None]
        output = torch.sum(output, dim=2)
        return output


def tryout_instrument_stack():
    batch_size = 2
    n_events = 4
    control_plane = 32
    shape_channels = 16
    encoding_channels = 512
    n_frames = 128
    n_samples = 2**15
    n_shape_frames = 4
    layers = 4
    
    energy = torch.zeros(
        batch_size, n_events, control_plane, n_frames).bernoulli_(p=0.001)
    energy = energy * torch.zeros_like(energy).uniform_(0, 10)
    
    shapes = [
        torch.zeros(
            batch_size, n_events, shape_channels, n_shape_frames).bernoulli_(p=0.24) 
        for _ in range(layers)
    ]
    
    decays = [
        torch.zeros(batch_size, n_events, control_plane, 1).uniform_(0.5, 0.6) 
        for _ in range(layers)
    ]
    
    mix = torch.zeros(batch_size, n_events, layers).uniform_(-1, 1)
    
    instr = InstrumentStack(
        encoding_channels=encoding_channels,
        channels=control_plane,
        n_frames=n_frames,
        n_samples=n_samples,
        shape_channels=shape_channels,
        n_layers=layers
    )
    
    audio = instr.forward(
        energy=energy,
        transforms=shapes,
        decays=decays,
        mix=mix
    )
    
    audio = audio[0, 0].data.cpu().numpy()
    audio = audio / audio.max()
    
    listen_to_sound(audio, 22050, wait_for_user_input=True)
    
def tryout_instrument3():
    
    batch_size = 2
    n_events = 4
    control_plane = 32
    shape_channels = 16
    encoding_channels = 512
    n_frames = 128
    n_samples = 2**15
    n_shape_frames = 4
    
    energy = torch.zeros(
        batch_size, n_events, control_plane, n_frames).bernoulli_(p=0.001)
    energy = energy * torch.zeros_like(energy).uniform_(0, 10)
    
    shape = torch.zeros(
        batch_size, n_events, shape_channels, n_shape_frames).bernoulli_(p=0.24)
    
    decays = torch.zeros(batch_size, n_events, control_plane, 1).uniform_(0.5, 0.99)
    
    inst = Instrument3(
        encoding_channels, 
        control_plane, 
        n_frames=n_frames,
        n_samples=n_samples,
        shape_channels=shape_channels)
    
    result, energy = inst.forward(
        energy=energy, 
        transforms=shape, 
        decays=decays)
    
    
    audio = result[0, 0].data.cpu().numpy()
    audio = audio / audio.max()
    
    listen_to_sound(audio, 22050, wait_for_user_input=True)
    

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
    
    
    tryout_instrument_stack()
    
    # tryout_instrument3()
    
    # TODO: 
    # values = 1 - (np.linspace(0, 1) ** 4)
    # plt.plot(values)
    # plt.show()
    
    # decays = torch.zeros(1, 1, 1).fill_(0.5)
    # env = exponential_decay(decays, 1, 128, 0, 128)
    # plt.plot(env.data.cpu().numpy().squeeze())
    # plt.show()
    
    # energy = torch.zeros(128)
    # energy[10] = 1
    # energy[50] = 3
    # plt.plot(energy.data.cpu().numpy().squeeze())
    # plt.show()
    
    
    # print(energy.shape, env.shape)
    # final = fft_convolve(energy[None, None, :], env)
    # plt.plot(final.data.cpu().numpy().squeeze())
    # plt.show()
    
    