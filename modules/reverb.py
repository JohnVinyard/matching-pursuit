from torch import nn
import torch
import glob
import pathlib
import zounds
import numpy as np
from librosa import load, to_mono

from modules.fft import fft_convolve


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
        for p in glob.iglob(str(g)):
            a, sr = load(p)
            a = to_mono(a)
            if len(a) < n_samples:
                a = np.pad(a, [(0, n_samples - len(a))])
            else:
                a = a[:n_samples]
            audio.append(a[None, ...])
            print('Processed', p)
        
        audio = np.concatenate(audio, axis=0)
        return NeuralReverb(n_samples, audio.shape[0], audio)

        

    def forward(self, x, reverb_mix):

        # mx, _ = torch.max(self.rooms, dim=-1, keepdim=True)
        # rooms = self.rooms / (mx + 1e-12)

        # choose a linear mixture of "rooms"
        mix = (reverb_mix[:, None, :] @ self.rooms)

        x = fft_convolve(mix, x)

        # reverb_spec = torch.fft.rfft(mix, dim=-1, norm='ortho')
        # signal_spec = torch.fft.rfft(x, dim=-1, norm='ortho')

        # # convolution in the frequency domain
        # x = reverb_spec * signal_spec

        # x = torch.fft.irfft(x, dim=-1, n=self.size, norm='ortho')

        return x


if __name__ == '__main__':

    n_samples = 2**14
    n_rooms = 1
    batch_size = 4

    signal = torch.FloatTensor(batch_size, 1, n_samples).uniform_(-1, 1)
    mix = torch.FloatTensor(batch_size, n_rooms).uniform_(-1, 1)

    reverb = NeuralReverb(n_samples, n_rooms)

    z = reverb.forward(signal, mix)

    print(z.shape)
