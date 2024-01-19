import torch
import zounds
from config.experiment import Experiment
from modules import filter_bank
from modules.ddsp import overlap_add
from modules.fft import fft_shift
from modules.linear import LinearOutputStack
from modules.pos_encode import hard_pos_encoding
from modules.stft import stft
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from util import device, playable
from util.readmedocs import readme
from torch import nn
import numpy as np
from torch.nn import functional as F
from modules.phase import morlet_filter_bank


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


operator = lambda a, b: a

def activation(x): 
    return torch.sin(x)


scale = zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist - 1000), 512)

fb = morlet_filter_bank(
    exp.samplerate, 
    512, 
    scale=scale, 
    scaling_factor=0.01, 
    normalize=False)

fb = torch.from_numpy(fb.real).to(device) #* torch.hamming_window(512, device=device)[None, :]

class PerceptualAudioModel(nn.Module):
    def __init__(self):
        super().__init__()

        scale = zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist), 128)

        orig_filters = filters = morlet_filter_bank(
            exp.samplerate, exp.kernel_size, scale, scaling_factor=0.1, normalize=True)
        filters = np.fft.rfft(filters, axis=-1, norm='ortho')

        self.register_buffer('orig', torch.from_numpy(orig_filters))

        padded = np.pad(orig_filters, [(0, 0), (0, exp.n_samples - exp.kernel_size)])
        full_size_filters = np.fft.rfft(
            padded, axis=-1, norm='ortho')
        

        self.register_buffer('filters', torch.from_numpy(filters))
        self.register_buffer('full_size_filters',
                             torch.from_numpy(full_size_filters))
        

    def loss(self, a, b):
        a1, a2 = self.forward(a)
        b1, b2 = self.forward(b)

        l1 = F.mse_loss(a1, b1)
        l2 = F.mse_loss(a2, b2)

        return l1 + l2

    def forward(self, x):

        x = x.view(-1, 1, exp.n_samples)

        spec = torch.fft.rfft(x, dim=-1, norm='ortho')

        conv = spec * self.full_size_filters[None, ...]

        spec = torch.fft.irfft(conv, dim=-1, norm='ortho').float()

        # half-wave rectification
        spec = torch.relu(spec)

        # compression
        spec = torch.sqrt(spec)

        # loss of phase locking above 5khz (TODO: make this independent of sample rate)
        spec = F.avg_pool1d(spec, kernel_size=3, stride=1, padding=1)

        # compute within-band periodicity
        spec = F.pad(spec, (0, 256)).unfold(-1, 512, 256)
        spec = spec * torch.hamming_window(512, device=spec.device)[None, None, None, :]

        real = spec @ self.orig.real.T
        imag = spec @ self.orig.imag.T

        spec = torch.complex(real, imag)
        spec = torch.abs(spec)

        # pooled = spec[..., 0]
        pooled = torch.mean(spec, dim=-1, keepdim=True)

        # only frequencies below the current band matter
        spec = torch.tril(spec)

        # we care about the *shape* and not the magnitude here
        norms = torch.norm(spec, dim=-1, keepdim=True)
        spec = spec / (norms + 1e-8)

        return pooled, spec


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('pos', hard_pos_encoding(exp.n_frames, device, operator=operator))
        self.net = LinearOutputStack(
            exp.model_dim, 
            4, 
            out_channels=1024, 
            in_channels=int(np.log2(exp.n_frames)) + 1)
        
        self.register_buffer('latent', torch.zeros(1, 128).normal_(0, 1))
        self.net = PosEncodedUpsample(
            exp.model_dim, exp.model_dim, exp.n_frames, out_channels=1024, layers=4, activation=activation, multiply=True)
        # self.net = ConvUpsample(
        #     exp.model_dim, 
        #     exp.model_dim, 
        #     8, 
        #     end_size=exp.n_frames, 
        #     mode='learned', 
        #     out_channels=1024)

        self.magnitudes = nn.Parameter(torch.zeros(exp.n_frames, 512).uniform_(0, 1))
        self.phases = nn.Parameter(torch.zeros(exp.n_frames, 512).normal_(0, 1))

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):

        # return self.net(self.latent)

        x = self.net(self.latent).permute(0, 2, 1)

        # x = self.net(self.pos.permute(0, 2, 1))

        p = x[..., :512].permute(1, 2, 0).reshape(exp.n_frames * 512, 1, 1)
        m = x[..., 512:].permute(1, 2, 0).reshape(exp.n_frames * 512, 1, 1)

        p = torch.tanh(p)
        m = torch.relu(m)

        # f = fb.view(512 * 128, 1, 512)
        # f = torch.cat([fb] * 128, axis=0)
        f = fb[None, :, :] * torch.ones(128, 1, 1, device=device)
        f = f.view(exp.n_frames * 512, 1, 512)

        # p = torch.tanh(self.phases.view(exp.n_frames * 512, 1, 1))
        # m = self.magnitudes.view(exp.n_frames * 512, 1, 1)

        shifted = fft_shift(f, p)

        mags = shifted * m # (frames * channels, 1, window)

        mags = mags.view(1, exp.n_frames, 512, 512).permute(0, 2, 1, 3)
        mags = overlap_add(mags, apply_window=True)[..., :exp.n_samples]
        mags = torch.mean(mags, dim=1, keepdim=True)
        return mags


loss_model = PerceptualAudioModel().to(device)

model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)

    loss = exp.perceptual_loss(recon, batch)
    # loss = loss_model.loss(recon, batch)

    loss.backward()
    optim.step()

    return loss, recon


@readme
class HardPosEncodingExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
        self.pos = hard_pos_encoding(exp.n_frames, device, operator=operator)
    
    def filters(self):
        return fb.data.cpu().numpy()

    def orig(self):
        return playable(self.real, exp.samplerate)

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def hard_pos(self):
        return self.pos.data.cpu().numpy().reshape((-1, exp.n_frames))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item

            l, r = train(item)
            self.fake = r

            print(l.item())
