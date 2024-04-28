from base64 import b64encode
import numpy as np
import torch
import torch.nn.functional as F
from data.audioiter import AudioIterator
from modules.angle import windowed_audio
from modules.decompose import fft_frequency_decompose
from modules.multibanddict import BandSpec, MultibandDictionaryLearning
from modules.normal_pdf import pdf2
from modules.normalization import max_norm
from modules.overlap_add import overlap_add
from modules.stft import stft
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
from torch.optim import Adam
from PIL import Image
from matplotlib import pyplot as plt
from torch.distributions import Normal
from scipy.interpolate import interp1d
from scipy.stats import norm
from util import device
import pickle


from modules.transfer import gaussian_bandpass_filtered

n_samples = 2**15

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


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ln = weight_norm(nn.Linear(in_channels, out_channels))
    
    def forward(self, x):
        x = self.ln(x)
        x = torch.relu(x)
        return x


class Nerf(nn.Module):
    def __init__(self, n_layers = 8, channels=128, pos_encoding_channels=33):
        super().__init__()
        self.n_layers = n_layers
        self.pos_encoding_channels = pos_encoding_channels
        self.from_pos = weight_norm(nn.Linear(pos_encoding_channels, channels))
        self.stack = nn.Sequential(
            *[Layer(channels, channels) for _ in range(n_layers)],
        )
        self.to_samples = weight_norm(nn.Linear(channels, 1))
    
    def forward(self, times):
        x = self.from_pos(times)
        
        intermediates = []
        
        for layer in self.stack:   
            x = layer(x)
            intermediates.append(x)
        
        x = self.to_samples(x)
        intermediates = torch.cat(intermediates, dim=-1)
        
        return x, intermediates
        

model = Nerf()
optim = Adam(model.parameters(), lr=1e-3)

def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()
    forward = mask
    backward = x
    y = backward + (forward - backward).detach()
    return y.sum()


n_atoms = 512
n_samples = 2**15

model = MultibandDictionaryLearning([
    BandSpec(512,   n_atoms, 128, device=device, signal_samples=n_samples, is_lowest_band=True),
    BandSpec(1024,  n_atoms, 128, device=device, signal_samples=n_samples),
    BandSpec(2048,  n_atoms, 128, device=device, signal_samples=n_samples),
    BandSpec(4096,  n_atoms, 128, device=device, signal_samples=n_samples),
    BandSpec(8192,  n_atoms, 128, device=device, signal_samples=n_samples),
    BandSpec(16384, n_atoms, 128, device=device, signal_samples=n_samples),
    BandSpec(32768, n_atoms, 128, device=device, signal_samples=n_samples),
], n_samples=n_samples)


if __name__ == '__main__':
    p = pickle.dumps(model, pickle.HIGHEST_PROTOCOL)
    
    rehyrdrated: MultibandDictionaryLearning = pickle.loads(p)
    print(rehyrdrated.bands[512].d.shape)

    # embeddings = torch.zeros(128, 16).uniform_(-1, 1)
    # query = embeddings[10]
    
    # results = k_nearest(query, embeddings)
    # print(results)

# if __name__ == '__main__':
    
    # times = np.linspace(0, 1, num=100)
    # values = np.array([
    #     [0, 0.01],
    #     [0.02, 1],
    #     [0.04, 0.01],
    #     [1, 0],
    # ])
    
    # func = interp1d(values[:, 0], values[:, 1], kind='linear', assume_sorted=False)
    
    # values = func(times)
    # plt.plot(values.T)
    # plt.show()
    
    # dist = norm(0.7, 0.01)
    # pdf = dist.pdf(np.linspace(0, 1, 1000))
    # plt.plot(pdf)
    # plt.show()