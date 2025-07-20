from typing import List
import torch
from torch import nn
from data.audioiter import AudioIterator
from modules import stft, sparsify
from modules.auditory import gammatone_filter_bank
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve, n_fft_coeffs
from modules.instrument import InstrumentStack
from conjure import audio_conjure, serve_conjure, LmdbCollection, bytes_conjure, SupportedContentType, numpy_conjure
from torch.optim import Adam
from modules.normalization import max_norm
from util import device
from io import BytesIO
from soundfile import SoundFile
from torch.nn import functional as F
from matplotlib import pyplot as plt


collection = LmdbCollection(path='instrumentmodel')

samplerate = 22050
n_samples = 2 ** 16
samples_per_frame = 256
n_frames = n_samples // samples_per_frame


def phase_invariant_feature(
        signal: torch.Tensor, 
        filters: torch.Tensor) -> torch.Tensor:
    
    
    n_samples = signal.shape[-1]
    
    signal = signal.reshape((-1, n_samples))
    
    n_filters, filter_size = filters.shape
    padded = F.pad(filters, (0, n_samples - filter_size))    
    
    spec = fft_convolve(signal ,padded)
    
    # half-wave rectification
    spec = torch.relu(spec)
    rectified = spec.view(1, n_filters, n_samples)
    aim_window_size = 128
    rectified = rectified.unfold(-1, aim_window_size, aim_window_size // 2)
    aim = torch.abs(torch.fft.rfft(rectified, dim=-1)) # (batch, channels, time, periodicity)
    aim = aim.view(n_filters, -1, n_fft_coeffs(aim_window_size)) # (channels, time, periodicity)
    return aim

fb = gammatone_filter_bank(
        n_filters=128, 
        size=256, 
        min_freq_hz=20, 
        max_freq_hz=samplerate // 2 - 10, 
        samplerate=samplerate, 
        freq_spacing_type='geometric')
fb = torch.from_numpy(fb).to(device)

exp_fb = torch.linspace(1, 0, steps=128)[:, None]
decays = torch.linspace(1, 100, steps=n_samples)[None, :]
exp_fb = (exp_fb ** decays).to(device).view(1, 128, n_samples)


def exponential_transform(audio: torch.Tensor) -> torch.Tensor:
    transform = fft_convolve(audio, exp_fb)
    transform = torch.relu(transform)
    print(transform.shape)
    transform = F.avg_pool1d(transform, kernel_size=512, stride=256, padding=256)
    return transform


class OverfitInstrument(nn.Module):
    
    def __init__(
            self, 
            osc_bank_size: int, 
            control_plane_dim: int, 
            shape_channels: int, 
            layers: int,
            n_shape_frames: int,
            n_frames: int,
            n_samples: int,
            n_events: int = 1,
            learnable_resonances: bool = False):
        
        super().__init__()
        
        self.osc_bank_size = osc_bank_size
        self.control_plane_dim = control_plane_dim
        self.shape_channels = shape_channels
        self.layers = layers
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.n_shape_frames = n_shape_frames
        self.learnable_resonances = learnable_resonances
        
        # time-varying input to the control plane
        energy = torch.zeros(
            1, n_events, control_plane_dim, n_frames).uniform_(0, 1)
        self.energy = nn.Parameter(energy)
        
        # time-varying shape deformations for each layers
        shapes = [
            torch.zeros(
                1, n_events, shape_channels, n_shape_frames).uniform_(-1, 1) 
            for _ in range(layers)
        ]
        self.shapes = nn.ParameterList(shapes)
        
        # decay, or resonance values for each dimension of the control
        # plane, for each layer
        decays = [
            torch.zeros(1, n_events, control_plane_dim, 1).uniform_(0.1, 0.5) 
            for _ in range(layers)
        ]
        self.decays = nn.ParameterList(decays)
    
        mix = torch.zeros(1, n_events, layers).uniform_(-1, 1)
        self.mix = nn.Parameter(mix)
        
        self.stack = InstrumentStack(
            encoding_channels=osc_bank_size,
            channels=control_plane_dim,
            n_frames=n_frames,
            n_samples=n_samples,
            shape_channels=shape_channels,
            n_layers=layers,
            learnable_resonances=self.learnable_resonances
        )

    @property
    def sparse_energy(self):
        return sparsify(self.energy, n_to_keep=32)
    
    def with_random_excitement(self, energy: torch.Tensor) -> torch.Tensor:
        assert energy.shape == self.energy.shape
        result = self.stack.forward(
            energy=energy, 
            transforms=self.shapes, 
            decays=self.decays, 
            mix=self.mix)
        result = torch.sum(result, dim=1, keepdim=True)
        result = max_norm(result)
        return result
    
    def forward(self) -> torch.Tensor:
        result = self.stack.forward(
            energy=self.sparse_energy,
            transforms=self.shapes, 
            decays=self.decays, 
            mix=self.mix)
        result = torch.sum(result, dim=1, keepdim=True)
        result = max_norm(result)
        return result

def audio(x: torch.Tensor):
    x = x.data.cpu().numpy().reshape((-1,))
    io = BytesIO()
    
    with SoundFile(
            file=io, 
            mode='w', 
            samplerate=samplerate, 
            channels=1, 
            format='WAV', 
            subtype='PCM_16') as sf:
        
        sf.write(x)
    
    io.seek(0)
    return io.read()
    
@audio_conjure(storage=collection)
def recon_audio(x: torch.Tensor):
    return audio(x)

@audio_conjure(storage=collection)
def orig_audio(x: torch.Tensor):
    return audio(x)

@audio_conjure(storage=collection)
def random_excitement(x: torch.Tensor):
    return audio(x)

@numpy_conjure(storage=collection, content_type=SupportedContentType.Spectrogram.value)
def random_excitement_energy(x: torch.Tensor):
    x = x[0, 0, :, :].data.cpu().numpy()
    return x

@numpy_conjure(storage=collection, content_type=SupportedContentType.Spectrogram.value)
def energy(x: torch.Tensor):
    x = max_norm(x)
    x = x[0, 0, :, :].data.cpu().numpy()
    return x

@numpy_conjure(storage=collection, content_type=SupportedContentType.Spectrogram.value)
def shape(x: torch.Tensor):
    x = x[0, 0, :, :].data.cpu().numpy()
    return x


def transform(x: torch.Tensor):
    batch_size, channels, _ = x.shape
    bands = multiband_transform(x)
    return torch.cat([b.reshape(batch_size, channels, -1) for b in bands.values()], dim=-1)
        
def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_xl': stft(v, 512, 64, pad=True) for k, v in bands.items()}
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
    normal = stft(x, 2048, 256, pad=True).reshape(-1, 128, 1025).permute(0, 2, 1)
    # return dict(normal=normal)
    return dict(**d1, **d3, **d4, normal=normal)

def multiband_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    real_spec = transform(real)
    fake_spec = transform(fake)
    return F.mse_loss(fake_spec, real_spec)

def pif_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    real_pif = phase_invariant_feature(real.view(1, 1, n_samples), fb)
    fake_pif = phase_invariant_feature(fake.view(1, 1, n_samples), fb)
    return F.mse_loss(fake_pif, real_pif)

def stft_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    real_spec = stft(real, 2048, 256)
    fake_spec = stft(fake, 2048, 256)
    return F.mse_loss(fake_spec, real_spec)

def exp_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    real_spec = exponential_transform(real)
    fake_spec = exponential_transform(fake)
    return F.mse_loss(fake_spec, real_spec)
    
def train(target: torch.Tensor):
    
    control_plane = 8
    layers = 3
    n_events = 1
    n_shape_frames = 1
    n_osc_bank_size = 128
    
    
    model = OverfitInstrument(
        osc_bank_size=n_osc_bank_size, 
        control_plane_dim=control_plane, 
        shape_channels=control_plane, 
        layers=layers, 
        n_frames=n_frames, 
        n_shape_frames=n_shape_frames,
        n_samples=n_samples,
        n_events=n_events,
        learnable_resonances=True
    ).to(device)
    
    optim = Adam(model.parameters(), lr=1e-3)
    
    while True:
        optim.zero_grad()
        recon = model.forward()
        recon_audio(max_norm(recon))
        
        energy(model.sparse_energy)
        shape(model.shapes[0])
        
        # energy_loss = torch.abs(model.energy).sum() * 1e-3
        
        loss = stft_loss(recon, target)
        # loss = pif_loss(recon, target)
        # loss = exp_loss(recon, target)
        # loss = multiband_loss(recon, target)
        
        # loss = loss + energy_loss
        
        loss.backward()
        optim.step()
        print(loss.item())
        
        re = torch.zeros_like(model.energy).bernoulli_(p=0.01)
        random_excitement_energy(re)
        rnd = model.with_random_excitement(re)
        random_excitement(max_norm(rnd))
    
    

if __name__ == '__main__':
    ai = AudioIterator(
        batch_size=1, 
        n_samples=n_samples, 
        samplerate=samplerate, 
        normalize=True, 
        overfit=True)
    example = next(iter(ai))
    example = example.view(1, 1, n_samples)
    orig_audio(example)
    
    serve_conjure(
        conjure_funcs=[
            recon_audio, 
            orig_audio, 
            energy,
            shape,
            random_excitement,
            random_excitement_energy
        ], 
        port=9999, 
        n_workers=1
    )
    
    train(example)