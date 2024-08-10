from base64 import b64encode
from subprocess import PIPE, Popen
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
from io import BytesIO
from soundfile import SoundFile

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
    # orig_coeffs = n_samples // 2 + 1
    
    shift_samples = (shift * n_samples) * (1/3)
    
    a = F.pad(a, (0, n_samples * 2))
    
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
    # shift = torch.linspace(0, 1, steps=n_coeffs) * 2j * np.pi
    
    shift = torch.exp(-shift * shift_samples)

    spec = spec * shift
    
    # diff = spec.shape[-1] - orig_coeffs
    # spec[..., orig_coeffs:] *= np.linspace(1, 0, diff) ** 2
    
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




# TODO: Is it actually possible to evaluate this for a single, isolated sample,
# or is some kind of scan operation stil required?
def instrument(
        t: torch.Tensor, 
        shift: torch.Tensor, 
        energy: torch.Tensor, 
        shape: torch.Tensor,
        properties: torch.Tensor):
    
    # TODO: look into using named tensors for this
    
    
    batch, n_events, time = t.shape
    
    # right away, we apply the time shifts to each
    # positional encoding
    t = fft_shift(t, shift)
    
    _, _, cp, n_frames = energy.shape
    assert energy.shape == shape.shape
    
    assert properties.shape == (batch, n_events, cp, 2)
    
    frame_shifts = torch.linspace(0, 1, steps=n_frames)
    
    # now we expand the time encoding, and apply constant, monotonically-increasing
    # shifts for each frame
    expanded_t = t.view(batch, n_events, 1, time).repeat(1, 1, n_frames, 1)
    expanded_t = fft_shift(expanded_t, frame_shifts[None, None, :, None])
    
    
    # we introduce a new dimension for the control plane
    expanded_t = expanded_t.view(batch, n_events, 1, n_frames, time)
    
    # next we need to spread the energy across frames
    # frame_rate = time // n_frames
    
    # create a mask to apply to pre t0 elements
    mask = torch.zeros((n_frames, time))
    row, col = torch.triu_indices(*mask.shape, offset=2)
    mask[row, col] = 1
    
    # shift the mask
    mask = fft_shift(mask[None, None, :, :], shift[:, :, None, :])
    
    mass = properties[..., :1]
    friction = properties[..., 1:]

    print('energy', energy.shape)
    print('friction', friction.shape)
    print('mass', mass.shape)
    print('time', expanded_t.shape)
    print('mask', mask.shape)

    # x = amplitude[:, None] * (np.e **((-friction / (2 * mass) * time_matrix)))
    
    x = energy[..., None] * (np.e **((-friction[..., None] / (2 * mass[..., None]) * expanded_t)))
    x = x * mask[:, :, None, :, :]
    
    x = torch.sum(x, dim=-2)
    
    # TODO: upsample to sampling rate
    return x
    
    


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.l1 = nn.Linear(in_channels, out_channels)
        self.l2 = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm([out_channels,])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = torch.relu(x)
        x = self.l2(x)
        x = x + skip
        # x = self.norm(x)
        return x

class NERF(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.channels = channels
        self.layers = layers
        
        self.layers = nn.ModuleList([Layer(channels, channels) for _ in range(layers)])
        
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Instrument(nn.Module):
    def __init__(self, encoding_channels: int, channels: int, n_layers: int, n_frames: int, n_samples: int):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.n_layers = n_layers
        self.n_frames = n_frames
        self.n_samples = n_samples
        
        self.embed_pos = nn.Linear(encoding_channels, channels)
        
        
        self.shape_network = NERF(channels, n_layers)
        self.decay_network = NERF(channels, n_layers)
        self.deformation = NERF(channels, n_layers)
        self.apply_shape = NERF(channels, n_layers)
        
        self.base_resonance = 0.02
    
    
    def _pos_encoding(self, n_samples: int):
        """Returns a filterbank with periodic functions
        """
        freqs = torch.linspace(0.00001, 0.5, steps=self.encoding_channels) ** 2
        t = torch.linspace(0, n_samples, steps=n_samples)
        p = torch.sin(t[None, :] * freqs[:, None] * np.pi)
        p = p.T
        plt.matshow(p[:512, :])
        plt.show()
        p = p.view(1, 1, n_samples, self.encoding_channels)
        return p

    def forward(self, energy: torch.Tensor, deformation: torch.Tensor):
        batch, n_events, cp, frames = energy.shape
        assert energy.shape == deformation.shape
        
        pos = self._pos_encoding(self.n_samples)
        pos = self.embed_pos(pos)
        
        df = deformation.permute(0, 1, 3, 2)
        df = self.deformation.forward(df)
        
        shape = self.shape_network.forward(df)
        shape = shape.permute(0, 1, 3, 2).view(-1, self.channels, self.n_frames)
        shape = F.interpolate(shape, size=self.n_samples, mode='linear')
        shape = shape.view(batch, n_events, self.channels, self.n_samples)
        
        shape = shape.permute(0, 1, 3, 2)
        shape = self.apply_shape.forward(pos * shape)
        shape = shape.permute(0, 1, 3, 2)        
        
        # compute the resonance values at frame rate and then upsample
        energy = energy.permute(0, 1, 3, 2)
        decay = self.decay_network.forward(energy)
        decay = torch.sigmoid(decay)
        resonance_factor = (1 - self.base_resonance) * 0.9999
        decay = self.base_resonance + (decay * resonance_factor)
        decay = torch.log(decay + 1e-12)
        decay = torch.cumsum(decay, dim=-1)
        decay = torch.exp(decay).view(-1, n_events, frames)
        amp = F.interpolate(decay, size=self.n_samples, mode='linear')
        amp = amp.view(batch, n_events, self.channels, self.n_samples)
        
        print(shape.shape, amp.shape)
        x = shape * amp
        x = torch.sum(x, dim=2)
        return x
        
        
        

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
    
    n_samples = 512
    n_frames = 128
    batch_size = 2
    n_events = 4
    control_plane = 16
    
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
    
    plt.plot(result[0, 0, 0, :])
    plt.show()
    
    
    # instr = Instrument(
    #     encoding_channels=32,
    #     channels=control_plane,
    #     n_layers=3,
    #     n_frames=n_frames,
    #     n_samples=n_samples
    # )
    
    # audio = instr.forward(energy, shape)
    # audio = audio / audio.max()
    
    
    # audio = playable(audio.view(-1, 1, n_samples), zounds.SR22050())
    # print(audio.shape)
    # listen_to_sound(audio.data.cpu().numpy()[0, 0], 22050, wait_for_user_input=True)
        
    
    
    