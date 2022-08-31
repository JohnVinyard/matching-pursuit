from itertools import chain
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from config.dotenv import Config
from data.datastore import batch_stream
from modules.linear import LinearOutputStack

from modules.pos_encode import ExpandUsingPosEncodings
from modules.transformer import Transformer
from util import device

import zounds
from torch.optim import Adam

from util.weight_init import make_initializer


init_weights = make_initializer(0.1)


class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            mode='nearest'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.mode = mode

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.conv(x)
        return x


class Nearest(UpsampleBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 'nearest')


class Linear(UpsampleBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 'linear')


class LearnedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FFTUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, factor=2, infer=False):
        super().__init__()
        self.channels = in_channels
        self.size = size
        self.factor = factor
        self.new_time = self.size * self.factor 
        self.orig_coeffs = self.size // 2 + 1
        self.n_coeffs = self.new_time // 2 + 1

        inferred = self.n_coeffs - self.orig_coeffs

        self.infer = infer
        r = torch.zeros(self.orig_coeffs, inferred).uniform_(-0.01, 0.01)
        i = torch.zeros(self.orig_coeffs, inferred).uniform_(-0.01, 0.01)
        c = torch.complex(r, i)

        self.inferred = nn.Parameter(c)
        self.final = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
    
    def upsample(self, x):
        batch = x.shape[0]

        x = x.reshape(-1, self.channels, self.size)

        coeffs = torch.fft.rfft(x, axis=-1, norm='ortho')

        r = torch.zeros(batch, self.channels, self.n_coeffs).to(x.device)
        i = torch.zeros(batch, self.channels, self.n_coeffs).to(x.device)

        new_coeffs = torch.complex(r, i)

        new_coeffs[:, :, :self.orig_coeffs] = coeffs

        if self.infer:
            inferred = coeffs @ self.inferred
            new_coeffs[:, :, self.orig_coeffs:] = inferred

        x = torch.fft.irfft(new_coeffs, n=self.new_time, norm='ortho')
        # x = self.final(x)
        return x

    def forward(self, x):
        x = self.upsample(x)
        x = self.final(x)
        return x


def iter_layers(start_size, end_size):
    for i in range(int(np.log2(start_size)), int(np.log2(end_size))):
        yield i, 2**i


class PosEncodedUpsample(nn.Module):
    def __init__(
            self,
            latent_dim,
            channels,
            size,
            out_channels,
            layers,
            multiply=False,
            learnable_encodings=False,
            transformer=False,
            concat=False,
            filter_bank=False):

        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.size = size
        self.out_channels = out_channels
        self.multiply = multiply
        self.learnable_encodings = learnable_encodings
        self.layers = layers
        self.transformer = transformer
        self.filter_bank = filter_bank

        self.expand = ExpandUsingPosEncodings(
            channels, size, 16, latent_dim, multiply, learnable_encodings, concat=concat, filter_bank=filter_bank)

        if self.transformer:
            encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
            self.net = nn.Sequential(
                    nn.TransformerEncoder(encoder, layers, norm=None),
                    nn.Linear(channels, 1)
                )
            
            # self.net = nn.Sequential(
            #     Transformer(channels, layers),
            #     nn.Linear(channels, out_channels)
            # )
        else:
            self.net = LinearOutputStack(
                channels, layers, out_channels=out_channels)

            

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        x = self.expand(x)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x


class ConvUpsample(nn.Module):
    def __init__(
            self,
            latent_dim,
            channels,
            start_size,
            end_size,
            mode,
            out_channels,
            from_latent=True,
            batch_norm=False):

        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.mode = mode
        self.start_size = start_size
        self.end_size = end_size
        self.n_layers = int(np.log2(end_size) - np.log2(start_size))
        self.from_latent = from_latent

        self.begin = nn.Linear(
            self.latent_dim, self.channels * self.start_size)
        self.out_channels = out_channels

        if mode == 'learned':
            def layer(channels, size): return LearnedUpsampleBlock(
                channels, channels)
        elif mode == 'nearest':
            def layer(channels, size): return Nearest(channels, channels)
        elif mode == 'linear':
            def layer(channels, size): return Linear(channels, channels)
        elif mode == 'fft':
            def layer(channels, size): return FFTUpsampleBlock(
                channels, channels, size)
        elif mode == 'fft_learned':
            def layer(channels, size): return FFTUpsampleBlock(
                channels, channels, size, infer=True)

        self.net = nn.Sequential(*[nn.Sequential(
            layer(channels, size),
            nn.BatchNorm1d(channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
        ) for _, size in iter_layers(start_size, end_size)])

        self.final = nn.Conv1d(channels, self.out_channels, 3, 1, 1)

        self.apply(init_weights)

    def forward(self, x):
        if self.from_latent:
            x = x.reshape(-1, self.latent_dim)
            x = self.begin(x)
            x = x.view(-1, self.channels, self.start_size)
        
        x = self.net(x)
        x = self.final(x)
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, channels, input_size, latent_dim):
        super().__init__()
        self.channels = channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        layers = int(np.log2(input_size))

        self.start = nn.Conv1d(in_channels, self.channels, 3, 1, 1)

        self.net = nn.Sequential(*[nn.Sequential(
            nn.Conv1d(channels, channels, 3, 2, 1),
            nn.LeakyReLU(0.2)
        ) for _ in range(layers)])
        self.final = nn.Conv1d(channels, latent_dim, 1, 1, 0)

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.input_size)
        x = self.start(x)
        x = self.net(x)
        x = self.final(x)
        return x.view(-1, self.latent_dim)


def training_batch_stream(batch_size, n_samples, size):
    stream = batch_stream(Config.audio_path(), '*.wav', batch_size, n_samples)
    step = n_samples // size
    for b in stream:
        b = np.abs(b)
        b /= (b.max() + 1e-12)
        b = torch.from_numpy(b).to(device).view(batch_size, 1, n_samples)
        b = b.unfold(-1, step, step)
        b, _ = b.max(dim=-1)
        yield b


class ExperimentParams(object):
    def __init__(
            self,
            name,
            make_decoder,
            batch_size=16,
            n_samples=2**14,
            size=128,
            iterations=10000):

        super().__init__()
        self.name = name
        self.make_decoder = make_decoder
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.size = size
        self.iterations = iterations


class Experiment(object):
    def __init__(self, batch_size, n_samples, size, iterations, name, make_decoder):
        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.size = size
        self.iterations = iterations
        self.make_decoder = make_decoder
        self.name = name

    def run(self):
        encoder = SimpleEncoder(1, 32, self.size, 16).to(device)
        decoder = self.make_decoder().to(device)
        optim = Adam(
            chain(encoder.parameters(), decoder.parameters()),
            lr=1e-4,
            betas=(0, 0.9))

        losses = []
        print('===========================================================================')
        print('----------------------------------------------------------------')
        print('Beginning', self.name)
        print('----------------------------------------------------------------')

        for i, b in enumerate(training_batch_stream(self.batch_size, self.n_samples, self.size)):
            optim.zero_grad()
            encoded = encoder(b)
            decoded = decoder(encoded)

            loss = F.mse_loss(decoded, b)
            loss.backward()
            optim.step()

            if i % 1000 == 0:
                print(i, loss.item() if len(losses) == 0 else np.mean(losses[-10:]))

            losses.append(loss.item())

            if i == self.iterations:
                break

        print('----------------------------------------------------------')
        print(self.name, np.mean(losses[-10:]))
        print('----------------------------------------------------------')
        return self.name, b.data.cpu().numpy().squeeze(), decoded.data.cpu().numpy().squeeze()


if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    results = {}


    latent_dim = 16
    channels = 32
    size = 128

    params = [
        ExperimentParams('learned', lambda: ConvUpsample(
            latent_dim, channels, 4, size, 'learned', 1)), 
        ExperimentParams('pos_multiply_learned', lambda: PosEncodedUpsample(
            latent_dim, channels, size, 1, 5, transformer=False, multiply=True, learnable_encodings=True)),
        
        ExperimentParams('fft_learned', lambda: ConvUpsample(
            latent_dim, channels, 4, size, 'fft_learned', 1)),
        ExperimentParams('fft', lambda: ConvUpsample(
            latent_dim, channels, 4, size, 'fft', 1)),
        
        ExperimentParams('pos', lambda: PosEncodedUpsample(
            latent_dim, channels, size, 1, 5, transformer=False)),
        ExperimentParams('pos_learned', lambda: PosEncodedUpsample(
            latent_dim, channels, size, 1, 5, transformer=False, learnable_encodings=True)),
        ExperimentParams('pos_multiply', lambda: PosEncodedUpsample(
            latent_dim, channels, size, 1, 5, transformer=False, multiply=True)),
        ExperimentParams('pos_transformer', lambda: PosEncodedUpsample(
            latent_dim, channels, size, 1, 5, transformer=True)),
        

        ExperimentParams('linear', lambda: ConvUpsample(
            latent_dim, channels, 4, size, 'linear', 1)),
        ExperimentParams('nearest', lambda: ConvUpsample(
            latent_dim, channels, 4, size, 'nearest', 1))
        
    ]

    for param in params:
        exp = Experiment(**param.__dict__)
        name, real, fake = exp.run()
        results[name] = (real, fake)
    
    best = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print('==========================================')
    for b in best:
        print(b)
    
    input('waiting....')
