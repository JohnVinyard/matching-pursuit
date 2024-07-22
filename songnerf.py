import os
from typing import Tuple, Union
import numpy as np
import torch
from librosa import load
from conjure import LmdbCollection, pickle_conjure, audio_conjure
from conjure.serve import serve_conjure
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import zounds

from modules import stft
from modules.decompose import fft_frequency_decompose


port = 9999
collection = LmdbCollection('/tmp/songnerf', port=port)

FileName = Union[str, bytes, os.PathLike]

@pickle_conjure(collection, read_hook=lambda x: print('reading from cache'))
def load_song(path: FileName) -> Tuple[int, int, np.ndarray]:
    samples, sr = load(path, sr=22050, mono=True)
    orig_samples = samples.shape[0]
    
    i = 0
    while 2**i < orig_samples:
        i += 1
    
    n_samples = 2**i
    diff = n_samples - orig_samples
    padded = np.pad(samples, [(0, diff)])
    
    return i, n_samples, padded

@audio_conjure(collection)
def listenable(recon: torch.Tensor):
    samples = recon[0, 0, :].data.cpu().numpy()
    samples = zounds.AudioSamples(samples, zounds.SR22050()).pad_with_silence()
    samples /= (samples.max() + 1e-8)
    b = samples.encode()
    b = b.read()
    return b


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.l1 = nn.Linear(in_channels, out_channels)
        self.l2 = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = torch.relu(x)
        x = self.l2(x)
        x = x + skip
        return x


class NERF(nn.Module):
    def __init__(self, encoding_channels, channels, layers):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.layers = layers
        
        self.up = nn.Linear(encoding_channels, channels)
        self.layers = nn.ModuleList([Layer(channels, channels) for _ in range(layers)])
        
        self.to_samples = nn.Linear(channels, 32)
        
        
        def init_weights(p):
            with torch.no_grad():
                try:
                    p.weight.uniform_(-0.05, 0.05)
                except AttributeError:
                    pass

                try:
                    p.bias.fill_(0)
                except AttributeError:
                    pass
        self.apply(lambda x: init_weights(x))
        
    
    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        batch, channels, time = encoding.shape
        encoding = encoding.permute(0, 2, 1)
        x = self.up(encoding)
        for layer in self.layers:
            x = layer(x)
        
        
        x = self.to_samples(x).permute(0, 2, 1)
        x = x.view(batch, -1, time)
        x = torch.sum(x, dim=1, keepdim=True)
        return x



def pos_encoding(start: int, end: int, full_length: int, device: torch.device):
    pow = int(np.log2(full_length))
    indices = torch.arange(start, end, step=1, device=device)
    resolutions = 2 ** torch.arange(1, pow + 1, step=1, device=device)
    enc = (indices[:, None] % resolutions[None, :]) / resolutions[None, :]
    return enc.permute(1, 0)

def batch_stream(path: FileName, batch_dim: Tuple[int, int]):
    pow, n_samples, samples = load_song(path)
    
    samples /= (samples.max() + 1e-8)
    
    batch_size, batch_len = batch_dim
    while True:
        # target
        batch = torch.zeros((batch_size, 1, batch_len), device='cuda')
        
        # input
        encoding = torch.zeros((batch_size, pow, batch_len), device='cuda')
        
        for i in range(batch_size):
            start = np.random.randint(0, n_samples - batch_len)
            stop = start + batch_len
            enc = pos_encoding(start, stop, n_samples, batch.device)
            segment = torch.from_numpy(samples[start: stop]).to(batch.device)
            batch[i, :, :] = segment
            encoding[i, :, :] = enc
        
        yield batch, encoding

def transform(x: torch.Tensor):
    batch_size, channels, _ = x.shape
    bands = multiband_transform(x)
    return torch.cat([b.reshape(batch_size, channels, -1) for b in bands.values()], dim=-1)
        
def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    # d1 = {f'{k}_xl': stft(v, 512, 64, pad=True) for k, v in bands.items()}
    # d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    # d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    # d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
    
    normal = stft(x, 2048, 256, pad=True).reshape(-1, 128, 1025).permute(0, 2, 1)
    return dict(normal=normal)

    # return dict(**d1, **d3, **d4, normal=normal)
    return bands

def train(
        path: FileName, 
        batch_dim: Tuple[int, int], 
        channels: int, 
        layers: int, 
        device: str = 'cuda'):
    
    model = NERF(22, channels, layers).to(device)
    optim = Adam(model.parameters(), lr=1e-3)    
    iteration = 0
    
    for batch, enc in batch_stream(path, batch_dim):
        optim.zero_grad()
        recon = model.forward(enc)
        listenable(recon.sum(dim=1, keepdim=True))
        
        recon_spec = transform(recon)
        real_spec = transform(batch)
        loss = torch.abs(recon_spec - real_spec).sum()
        
        loss.backward()
        optim.step()
        print(iteration, loss.item())
        iteration += 1

if __name__ == '__main__':
    path = '/home/john/workspace/audio-data/musicnet/train_data/2307.wav'
    
    serve_conjure([listenable], port=port, n_workers=1)
    train(path, batch_dim=(8, 2**15), channels=1024, layers=5, device='cuda')
    
    # for i, b in enumerate(batch_stream(path, batch_dim=(8, 2**15))):
    #     b, enc = b
    #     plt.matshow(enc.data.cpu().numpy().squeeze()[:, :128])
    #     plt.show()
    
    