from typing import ForwardRef
import torch
from torch.optim.adam import Adam
import zounds
from torch import nn

from datastore import batch_stream
from modules import pos_encode_feature
from modules3 import LinearOutputStack
from torch.nn import functional as F
from itertools import chain
import numpy as np

sr = zounds.SR22050()
batch_size = 2
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


def activation(x):
    return torch.sin(x)




class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.embed_pos = nn.Conv1d(33, 8, 1, 1, 0, bias=False)
        self.embed_sample = nn.Conv1d(1, 8, 7, 1, 3, bias=False)

        self.factor = nn.Parameter(torch.FloatTensor(1).fill_(10))

        self.net = nn.Sequential(
            nn.Conv1d(16, 16, 8, 8, 0, bias=False),
            nn.Conv1d(16, 32, 8, 8, 0, bias=False),
            nn.Conv1d(32, 64, 8, 8, 0, bias=False),
            nn.Conv1d(64, 128, 8, 8, 0, bias=False),
            nn.Conv1d(128, 128, 8, 8, 0, bias=False),
        )

        self.apply(init_weights)
    
    def forward(self, pos, sample):

        pos = pos.permute(0, 2, 1)
        sample = sample.permute(0, 2, 1)

        pe = self.embed_pos(pos)
        se = self.embed_sample(sample)

        x = torch.cat([pe, se], dim=1)
        x = activation(x * self.factor)

        for layer in self.net:
            x = layer(x)
            x = activation(x * self.factor)
        
        x = x.view(-1, 128)
        return x



class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, apply_activation=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.factor = nn.Parameter(torch.FloatTensor(out_channels).fill_(10))
        self.apply_activation = apply_activation
    
    def forward(self, x):
        x = self.linear(x)
        if self.apply_activation:
            x = activation(self.factor * x)
        return x

class Network(nn.Module):
    def __init__(self, layers, in_channels, hidden_channels):
        super().__init__()
        self.layers = layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.pos_bias = LinearOutputStack(hidden_channels, 3, out_channels=33, activation=activation)

        self.mod_bias = nn.Sequential(*[LinearOutputStack(hidden_channels, 3, activation=activation) for layer in range(layers + 1)])

        self.net = nn.Sequential(
            Layer(in_channels, hidden_channels),
            *[Layer(hidden_channels, hidden_channels) for layer in range(layers)],
            Layer(hidden_channels, 1, apply_activation=True)
        )
        self.apply(init_weights)
    
    def forward(self, x, latent):

        pb = self.pos_bias(latent)
        x = x + pb[:, None, :]

        for i, layer in enumerate(self.net):
            try:
                mod_bias = self.mod_bias[i]
                bias = mod_bias(latent)
                x = layer(x)
                x = x + bias[:, None, :]
            except IndexError:
                x = layer(x)
        return x

def to_pairs(signal):

    signal = torch.from_numpy(signal).to(device)

    pos = torch.linspace(-1, 1, n_samples).view(-1, 1).to(device)
    pos = pos_encode_feature(pos, 1, n_samples, 16).repeat(batch_size, 1, 1)

    samples = signal[..., None]

    return pos, samples

def real():
    index = np.random.randint(0, batch_size)
    return zounds.AudioSamples(samples[index].data.cpu().numpy().reshape(-1), sr).pad_with_silence()

def fake():
    index = np.random.randint(0, batch_size)
    return zounds.AudioSamples(recon[index].data.cpu().numpy().reshape(-1), sr).pad_with_silence()


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)
    
    

    encoder = Encoder(128).to(device)
    net = Network(4, 33, 128).to(device)
    optim = Adam(chain(net.parameters(), encoder.parameters()), lr=1e-4, betas=(0, 0.9))


    sig = next(batch_stream(path, '*.wav', batch_size, n_samples))
    sig /= (sig.max(axis=-1, keepdims=True) + 1e-12)

    while True:
        optim.zero_grad()

        
        pos, samples = to_pairs(sig)

        latent = encoder(pos, samples)
        z = latent.data.cpu().numpy().squeeze()
        recon = net(pos, latent)

        loss = F.mse_loss(recon.view(batch_size, n_samples), samples.view(batch_size, n_samples))
        loss.backward()
        optim.step()
        print(loss.item())
    

    
