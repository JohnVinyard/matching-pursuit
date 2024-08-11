import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import numpy as np
from modules.activation import unit_sine
from modules.decompose import fft_frequency_decompose
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
    samples = torch.relu(samples)
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

if __name__ == '__main__':
    experiment()
    # look_at_gradients()