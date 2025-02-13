import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from data import get_one_audio_segment
from modules import pos_encoded, stft
from matplotlib import pyplot as plt

from modules.fft import fft_convolve
from modules.softmax import sparse_softmax

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


class BinaryModel(nn.Module):
    def __init__(self, n_elements):
        super().__init__()
        self.n_elements = n_elements

        x = 1 / np.array([2**i for i in range(1, n_elements + 1)])
        self.register_buffer('factors', torch.from_numpy(x).float())
        self.p = nn.Parameter(torch.zeros(n_elements).uniform_(-6, 6))

    def forward(self):
        x = torch.sigmoid(self.p) @ self.factors
        return x


def hierarchical_dirac(elements: torch.Tensor):
    log2, _ = elements.shape
    total_size = 2 ** log2
    
    # chosen = torch.softmax(elements, dim=-1)
    chosen = sparse_softmax(elements, normalize=True, dim=-1)
    # chosen = F.gumbel_softmax(elements, tau=0.1, hard=True, dim=-1)
    
    signal = torch.zeros(1)
    
    for i in range(log2):
        
        if i == 0:
            signal = chosen[i]
        else:
            new_size = signal.shape[-1] * 2
            
            # first, upsample
            new_signal = torch.zeros(new_size)
            new_signal[::2] = signal
            
            diff = new_size - 2
            
            # pad the selection
            current = torch.cat([chosen[i], torch.zeros(diff)])
            
            signal = fft_convolve(new_signal, current)
    
    return signal

def hiearchical_fft_shift(elements: torch.Tensor):
    log2, = elements.shape

    signal = dirac_impulse(2, 0)

    for i in range(log2):
        shift_factor = 1 / (2 ** i)

        if i == 0:
            signal = fft_shift(signal, elements[i] * shift_factor)
        else:
            new_size = signal.shape[-1] * 2
            # first, upsample
            new_signal = torch.zeros(new_size)
            new_signal[::2] = signal
            signal = fft_shift(new_signal, elements[i] * shift_factor)

    return signal


def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:

    # this is here to make the shift value interpretable
    shift = (1 - shift)

    n_samples = a.shape[-1]

    shift_samples = (shift * n_samples * 0.5)

    # a = F.pad(a, (0, n_samples * 2))

    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs

    shift = torch.exp(shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    # samples = torch.relu(samples)
    return samples


def dirac_impulse(size, position, device=None):
    x = torch.zeros(size, device=device)
    x[position] = 1
    return x


def look_at_gradients():
    n_points = 1024

    target = dirac_impulse(n_points, position=768)

    positions = torch.linspace(0, 1, steps=n_points)[:, None]
    positions.requires_grad = True

    source = dirac_impulse(n_points, 0)[None, :]


    shifts = fft_shift(source, positions)

    loss = torch.abs(target[None, :] - shifts).sum()
    loss.backward()

    g = positions.grad

    plt.plot(positions.data.cpu().numpy().squeeze())
    plt.show()

    plt.plot(g.data.cpu().numpy().squeeze())
    plt.show()



class HierarchicalDiracModel(nn.Module):
    def __init__(self, signal_size: int):
        super().__init__()
        self.signal_size = signal_size
        n_elements = int(np.log2(signal_size))
        self.elements = nn.Parameter(torch.zeros(n_elements, 2).uniform_(-1, 1))

    def forward(self):
        x = hierarchical_dirac(self.elements)
        return x

class HiearchicalFFTShiftModel(nn.Module):
    def __init__(self, signal_size: int):
        super().__init__()
        self.signal_size = signal_size
        n_elements = int(np.log2(signal_size))
        self.elements = nn.Parameter(torch.zeros(n_elements).uniform_(-1, 1))

    def forward(self):
        x = hiearchical_fft_shift(self.elements)
        return x


class Model(nn.Module):
    def __init__(self, multiscale: bool = False):
        super().__init__()
        self.multiscale = multiscale

        if self.multiscale:
            self.position = BinaryModel(16)
        else:
            self.position = nn.Parameter(torch.zeros(1).fill_(0.1))

    @property
    def pos(self):
        if self.multiscale:
            return self.position.forward().item()
        else:
            return self.position.item()

    def forward(self, size):
        imp = dirac_impulse(size, 0)
        imp.requires_grad = True

        if self.multiscale:
            pos = self.position.forward()
            imp = fft_shift(imp, pos)
        else:
            imp = fft_shift(imp, self.position)


        return imp

def experiment_hieararchical_fft_shift():
    raster_size = 1024

    model = HiearchicalFFTShiftModel(raster_size)
    optim = Adam(model.parameters(), lr=1e-3)
    target = dirac_impulse(raster_size, 768)

    i = 0

    while True:
        optim.zero_grad()
        recon = model.forward()
        index = torch.argmax(recon, dim=-1)
        loss = torch.abs(recon - target).sum()
        loss.backward()
        optim.step()
        print(i, loss.item(), index.item())

        # if i % 1000 == 0:
        #     plt.plot(recon.data.cpu().numpy())
        #     plt.show()

        #     plt.matshow(model.elements.data.cpu().numpy())
        #     plt.show()

        i += 1




def experiment_hierarchical_dirac():
    
    raster_size = 1024
    
    model = HierarchicalDiracModel(raster_size)
    optim = Adam(model.parameters(), lr=1e-3)
    target = dirac_impulse(raster_size, 768)

    scale = torch.linspace(1, 0.001, steps=33)[None, None, :] ** 2
    pe = pos_encoded(1, raster_size, n_freqs=16)
    pe = pe * scale

    dist = torch.cdist(pe.view(1024, 33), pe.view(1024, 33))
    plt.matshow(dist.data.cpu().numpy())
    plt.show()

    t =  target @ pe

    i = 0
    
    while True:
        optim.zero_grad()
        recon = model.forward()

        r = recon @ pe

        index = torch.argmax(recon, dim=-1)

        # loss = torch.abs(recon - target).sum()

        loss = torch.abs(t - r).sum()

        loss.backward()
        optim.step()
        print(loss.item(), index.item())
        
        if i % 1000 == 0:
            plt.plot(recon.data.cpu().numpy())
            plt.show()
            
            plt.matshow(model.elements.data.cpu().numpy())
            plt.show()
        
        i += 1

def experiment():
    model = Model(
        multiscale=False, 
    )
    optim = Adam(model.parameters(), lr=1e-3)
    
    raster_size = 1024
    target = dirac_impulse(raster_size, 768)



    while True:
        optim.zero_grad()
        recon = model.forward(raster_size)
        index = torch.argmax(recon, dim=-1)
        loss = torch.abs(recon - target).sum()
        loss.backward()
        optim.step()
        print(loss.item(), model.pos, index.item())


# def pos_encodings(time_dim: int, n_sinusoids: int) -> torch.Tensor:
#     pass

if __name__ == '__main__':
    # experiment()
    # look_at_gradients()

    audio = get_one_audio_segment(2**17).to('cpu')
    audio = audio.view(1, 1, -1)
    spec = stft(audio, 2048, 256, pad=True)

    print(spec.min(), spec.max())
    plt.matshow(spec.view(-1, 1025))
    plt.show()

    th = torch.tanh(spec)
    print(th.min(), th.max())
    plt.matshow(th.view(-1, 1025))
    plt.show()

    sqrt = torch.sqrt(spec)
    print(sqrt.min(), sqrt.max())
    plt.matshow(sqrt.view(-1, 1025))
    plt.show()

    lspec = torch.log(spec + 1e-8)
    print(lspec.min(), lspec.max())
    plt.matshow(lspec.view(-1, 1025))
    plt.show()


    # experiment_hierarchical_dirac()
    # experiment_hieararchical_fft_shift()
    
    