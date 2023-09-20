
import numpy as np
from torch import nn
from conjure import numpy_conjure, SupportedContentType
import torch
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.overlap_add import overlap_add
from modules.phase import windowed_audio
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_bands = 512
kernel_size = 512
freq_band = zounds.FrequencyBand(50, exp.samplerate.nyquist - 1000)
scale = zounds.GeometricScale(freq_band.start_hz, freq_band.stop_hz, 0.01, n_bands)
filter_bank = morlet_filter_bank(
    exp.samplerate, kernel_size, scale, 0.1, normalize=False).astype(np.complex64)
basis = torch.from_numpy(filter_bank).to(device) * 0.025

center_frequencies = (np.array(list(scale.center_frequencies)) / int(exp.samplerate)).astype(np.float32)
center_frequencies = torch.from_numpy(center_frequencies).to(device)

def to_frequency_domain(audio):
    batch_size = audio.shape[0]

    windowed = windowed_audio(audio, kernel_size, kernel_size // 2)
    real = windowed @ basis.real.T
    imag = windowed @ basis.imag.T
    freq_domain = torch.complex(real, imag)

    freq_domain = freq_domain.view(batch_size, -1, freq_domain.shape[-1])

    mag = torch.abs(freq_domain)
    phase = torch.angle(freq_domain)
    phase = torch.diff(
        phase,
        dim=1,
        prepend=torch.zeros(batch_size, 1, freq_domain.shape[-1], device=freq_domain.device))
    phase = phase % (2 * np.pi)


    # subtract the expected value
    freqs = center_frequencies * 2 * np.pi
    phase = phase - freqs[None, None, :]

    return mag, phase

def to_time_domain(spec):
    mag, phase = spec


    # add expected value
    freqs = center_frequencies * 2 * np.pi
    phase = phase + freqs[None, None, :]


    imag = torch.cumsum(phase, dim=1)
    imag = (imag + np.pi) % (2 * np.pi) - np.pi
    spec = mag * torch.exp(1j * imag)
    windowed = torch.flip((spec @ basis).real, dims=(-1,))
    td = overlap_add(windowed[:, None, :, :], apply_window=False)
    return td



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.atoms = nn.Parameter(torch.zeros(1, 2, ))

        self.down = nn.Sequential(
            # (64, 256)
            nn.Sequential(
                nn.Conv2d(2, 16, (3, 3), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(16)
            ),

            # (32, 128)
            nn.Sequential(
                nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(32)
            ),

            # (16, 64)
            nn.Sequential(
                nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64)
            ),

            # (8, 32)
            nn.Sequential(
                nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(128)
            ),
        )


        self.up = nn.Sequential(

            # (16, 64)
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64)
            ),

            # (32, 128)
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(32)
            ),

            # (64, 256)
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(16)
            ),

            # (128, 512)
            nn.Sequential(
                nn.ConvTranspose2d(16, 2, (4, 4), (2, 2), (1, 1)),
                # nn.LeakyReLU(0.2),
                # nn.BatchNorm2d(16)
            ),
        )

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        batch, channels, time, freq = x.shape
        act = {}

        for layer in self.down:
            x = layer(x)
            act[x.shape[-1]] = x
        
        for layer in self.up:
            x = layer(x)
        
        mag = x[:, :1, :, :]
        phase = x[:, 1:, :, :]

        mag = torch.abs(mag)

        x = torch.cat([mag, phase], dim=1)
        
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()

    with torch.no_grad():
        spec = to_frequency_domain(batch)

        mag, phase = spec
        # print(mag.min().item(), mag.max().item(), phase.min().item(), phase.max().item())

        cat = torch.cat([mag[:, None, :, :], phase[:, None, :, :]], dim=1)

    
    recon = model.forward(cat)

    # loss = F.mse_loss(recon, cat)

    mag_loss = F.mse_loss(
        recon[:, 0, :, :], 
        cat[:, 0, :, :]
    ) * 100

    phase_loss = F.mse_loss(recon[:, 1, :, :], cat[:, 1, :, :])
    loss = mag_loss + phase_loss

    print(mag_loss.item(), phase_loss.item(), loss.item())
    loss.backward()
    optim.step()

    with torch.no_grad():
        recon = to_time_domain((recon[:, 0, :, :], recon[:, 1, :, :]))[..., :exp.n_samples]

    return loss, recon, spec[0], spec[1]


def make_conjure(experiment: BaseExperimentRunner):

    @numpy_conjure(experiment.collection, SupportedContentType.Spectrogram.value)
    def geom_spec(x: torch.Tensor):
        return x.data.cpu().numpy()[0]
    
    return (geom_spec,)

def make_phase_conj(experiment: BaseExperimentRunner):

    @numpy_conjure(experiment.collection, SupportedContentType.Spectrogram.value)
    def geom_phase(x: torch.Tensor):
        return x.data.cpu().numpy()[0]
    
    return (geom_phase,)

@readme
class MatchingPursuitV3(BaseExperimentRunner):

    geom_spec = MonitoredValueDescriptor(make_conjure)
    geom_phase = MonitoredValueDescriptor(make_phase_conj)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r, spec, phase = train(item, i)
            self.geom_spec = spec
            self.geom_phase = phase
            self.fake = r
            self.after_training_iteration(l)
    