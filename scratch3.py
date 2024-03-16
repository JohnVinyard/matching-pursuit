from base64 import b64encode
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
from data.audioiter import AudioIterator
from modules import pos_encode
from modules.angle import windowed_audio
from modules.normalization import max_norm
from modules.overlap_add import overlap_add
from modules.stft import stft
from modules.transfer import STFTResonanceGenerator
from util.playable import playable
import zounds
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
from torch.optim import Adam
from modules.pos_encode import pos_encoded
from PIL import Image

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

if __name__ == '__main__':
    z_dim = 16
    
    model = STFTResonanceGenerator(512, 2**15, z_dim, 128)
    
    batch_size = 8
    n_events = 16
    impulse_size = 8192
    
    z = torch.zeros(batch_size, n_events, z_dim)
    impulse = torch.zeros(batch_size, n_events, impulse_size)
    
    print(z.shape, impulse.shape)
    audio = model.forward(z, impulse)
    print(audio.shape)