from typing import ForwardRef
import torch
from torch.optim.adam import Adam
import zounds
from scipy.fftpack import dct, idct
from torch import nn

from datastore import batch_stream
from modules import PositionalEncoding, pos_encode_feature
from modules3 import LinearOutputStack
from enum import Enum
from torch.nn import functional as F


sr = zounds.SR22050()
batch_size = 1
n_samples = 2**15

path = '/hdd/musicnet/train_data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True

def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-0.01, 0.01)
        except AttributeError:
            pass

        

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.factor = nn.Parameter(torch.FloatTensor(out_channels).fill_(10))
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.factor * x)
        return x

class Network(nn.Module):
    def __init__(self, layers, in_channels, hidden_channels):
        super().__init__()
        self.layers = layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Layer(in_channels, hidden_channels),
            *[Layer(hidden_channels, hidden_channels) for layer in range(layers)],
            Layer(hidden_channels, 1)
        )
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.net(x)
        return x

def to_pairs(signal):

    signal = torch.from_numpy(signal).to(device)
    pos = torch.linspace(-1, 1, n_samples).view(-1, 1).to(device)

    pos = pos_encode_feature(pos, 1, n_samples, 16)

    pos = pos[None, :]
    samples = signal[None, :]

    print(pos.shape, samples.shape)
    return pos, samples

def real():
    return zounds.AudioSamples(samples.data.cpu().numpy().reshape(-1), sr).pad_with_silence()

def fake():
    return zounds.AudioSamples(recon.data.cpu().numpy().reshape(-1), sr).pad_with_silence()


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)
    sig = next(batch_stream(path, '*.wav', batch_size, n_samples))
    sig /= (sig.max(axis=-1) + 1e-12)
    

    net = Network(4, 33, 128).to(device)
    optim = Adam(net.parameters(), lr=1e-4, betas=(0, 0.9))

    pos, samples = to_pairs(sig.reshape(-1))
    while True:
        optim.zero_grad()
        recon = net(pos)


        loss = F.mse_loss(recon.view(batch_size, n_samples), samples.view(batch_size, n_samples))
        loss.backward()
        optim.step()
        print(loss.item())
    

    
