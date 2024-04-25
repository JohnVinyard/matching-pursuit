from torch import nn
import torch
import glob
import pathlib
import numpy as np
from librosa import load, to_mono
from config.dotenv import Config

from modules.fft import simple_fft_convolve
from modules.linear import LinearOutputStack
from modules.softmax import sparse_softmax


class NeuralReverb(nn.Module):
    def __init__(self, size, n_rooms, impulses=None):
        super().__init__()
        self.size = size
        self.n_rooms = n_rooms

        if impulses is None:
            imp = torch.FloatTensor(self.n_rooms, self.size).uniform_(-0.01, 0.01)
            self.rooms = nn.Parameter(imp)
        else:
            imp = torch.from_numpy(impulses).float()
            if imp.shape != (self.n_rooms, self.size):
                raise ValueError(
                    f'impulses must have shape ({self.n_rooms}, {self.size}) but had shape {imp.shape}')
            self.register_buffer('rooms', imp)
    
    @staticmethod
    def from_directory(path, samplerate, n_samples):
        root = pathlib.Path(path)
        g = root.joinpath('*.wav')

        audio = []
        to_process = list(glob.iglob(str(g)))
        
        for p in to_process:
            a, sr = load(p)
            a = to_mono(a)
            if len(a) < n_samples:
                a = np.pad(a, [(0, n_samples - len(a))])
            else:
                a = a[:n_samples]
            audio.append(a[None, ...])
        
        print(f'Processed {len(to_process)} impulse responses')
        
        audio = np.concatenate(audio, axis=0)
        return NeuralReverb(n_samples, audio.shape[0], audio)

        

    def forward(self, x, reverb_mix):

        # choose a linear mixture of "rooms"
        mix = (reverb_mix[:, None, :] @ self.rooms)
        

        orig_shape = x.shape
        x = x.view(mix.shape)
        x = simple_fft_convolve(mix, x)
        x = x.view(*orig_shape)


        return x


class ReverbGenerator(nn.Module):
    def __init__(self, channels, layers, samplerate, n_samples, norm=None, hard_choice=False):
        super().__init__()
        self.channels = channels
        self.layeres = layers
        self.samplerate = samplerate
        self.n_samples = n_samples
        self.hard_choice = hard_choice

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate, n_samples)
        
        self.n_rooms = self.verb.n_rooms

        self.to_mix = LinearOutputStack(channels, layers, out_channels=2, shortcut=True, norm=norm)
        self.to_room = LinearOutputStack(
            channels, layers, out_channels=self.n_rooms, shortcut=True, norm=norm)
    
    def precomputed(self, dry, mx, rm):
        wet = self.verb.forward(dry, rm)
        stacked = torch.stack([dry, wet], dim=-1)

        mixed = stacked * mx.view(-1, 1, 1, 2)
        mixed = torch.sum(mixed, dim=-1)
        # mixed = (dry * mx) + (wet * (1 - mx))
        return mixed
        
    
    def forward(
            self, 
            context: torch.Tensor, 
            dry: torch.Tensor, 
            return_parameters: bool = False):
        
        
        if self.hard_choice:
            rm = sparse_softmax(self.to_room(context).view(-1, self.n_rooms), dim=-1, normalize=True)
        else:
            rm = torch.softmax(self.to_room(context).view(-1, self.n_rooms), dim=-1)
        
        # mx = torch.sigmoid(self.to_mix(context).view(-1, 1, 1))

        mx = torch.softmax(self.to_mix(context), dim=-1).view(-1, 1, 1, 2)

        wet = self.verb.forward(dry, rm)
        
        stacked = torch.stack([dry, wet], dim=-1)
        
        mx = mx.view(stacked.shape[0], stacked.shape[1], 1, 2)

        mixed = stacked * mx
        mixed = torch.sum(mixed, dim=-1)
        # mixed = (dry * mx) + (wet * (1 - mx))
        
        if return_parameters:
            return mixed, rm, mx
        else:
            return mixed


if __name__ == '__main__':

    n_samples = 2**14
    n_rooms = 1
    batch_size = 4

    signal = torch.FloatTensor(batch_size, 1, n_samples).uniform_(-1, 1)
    mix = torch.FloatTensor(batch_size, n_rooms).uniform_(-1, 1)

    reverb = NeuralReverb(n_samples, n_rooms)

    z = reverb.forward(signal, mix)

    print(z.shape)
