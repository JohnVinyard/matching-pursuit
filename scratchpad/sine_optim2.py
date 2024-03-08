import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam

# from train.optim import optimizer

n_samples = 2**15

def stft(
        x: torch.Tensor,
        ws: int = 512,
        step: int = 256,
        pad: bool = False,
        log_epsilon: float = 1e-4):

    frames = x.shape[-1] // step

    if pad:
        x = F.pad(x, (0, ws))

    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')

    x = torch.abs(x)

    x = x[:, :, :frames, :]
    return x

class Model(nn.Module):
    def __init__(self, n_osc=3, mode='complex'):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(n_osc, 2).uniform_(0.0001, 0.99999))
        self.n_osc = n_osc
        self.mode = mode

    def forward(self, x):

        if self.mode == 'clamp':
            freq = torch.tanh(self.params[:, 0])
            amp = torch.sigmoid(self.params[:, 1])
        elif self.mode == 'sin':
            freq = torch.sin(self.params[:, 0])
            amp = torch.sigmoid(self.params[:, 1])
        elif self.mode == 'complex':
            freq = self.params[:, 0]
            amp = self.params[:, 1]

            amp = torch.norm(self.params, dim=1)
            # freq = torch.cosine_similarity(self.params, torch.ones_like(self.params))
            freq = (torch.angle(torch.complex(amp, freq)) / np.pi)
            # print('amp', amp.item(), 'freq', freq.item())
        else:
            raise ValueError('Unsupported mode')

        accum = torch.zeros(1, self.n_osc, n_samples)
        accum[:, :] = freq[None, :, None]

        signal = amp[None, :, None] * torch.sin(torch.cumsum(accum, dim=-1) * np.pi)
        signal = signal.sum(dim=1)
        return signal, self.params[:, 0]


model = Model(n_osc=3)
# optim = optimizer(model, lr=1e-3)
optim = Adam(model.parameters(), lr=1e-3)

if __name__ == '__main__':

    freq = np.array([220, 440, 880])
    freq = (torch.from_numpy(freq) / 11025)
    
    freq = freq.view(3, 1).repeat(1, n_samples)    
    
    samples = torch.sin(torch.cumsum(freq * np.pi, dim=-1))
    samples = torch.sum(samples, dim=0).view(1, 1, n_samples)
    
    i = 0
    
    while True:
        optim.zero_grad()
        inp, p = model.forward(None)
        fake_spec = stft(inp)
        real_spec = stft(samples)
        
        # recon_loss = F.mse_loss(inp, samples)
        
        recon_loss = F.mse_loss(fake_spec, real_spec)
        
        param_loss = torch.abs(p).sum() * 10
        l = recon_loss + param_loss
        l.backward()
        optim.step()
        print(l.item())
        
        if i > 0 and i % 1000 == 0:
            real_spec = stft(samples.view(1, 1, -1), 2048, 256, pad=True)
            fake_spec = stft(inp.view(1, 1, -1), 2048, 256, pad=True)
            
            plt.matshow(real_spec.data.cpu().numpy()[0].squeeze())
            plt.show()
            plt.matshow(fake_spec.data.cpu().numpy()[0].squeeze())
            plt.show()
        i += 1
