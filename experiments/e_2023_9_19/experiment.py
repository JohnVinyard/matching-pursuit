
import numpy as np
from conjure import numpy_conjure, SupportedContentType
import torch
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.overlap_add import overlap_add
from modules.phase import windowed_audio
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
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
freq_band = zounds.FrequencyBand(40, exp.samplerate.nyquist)
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


def train(batch, i):
    with torch.no_grad():
        spec = to_frequency_domain(batch)
        recon = to_time_domain(spec)[..., :exp.n_samples]
        loss = F.mse_loss(recon, batch)
        print(loss.item())
        return loss, recon, spec[0]


def make_conjure(experiment: BaseExperimentRunner):

    @numpy_conjure(experiment.collection, SupportedContentType.Spectrogram.value)
    def geom_spec(x: torch.Tensor):
        return x.data.cpu().numpy()[0]
    
    return (geom_spec,)

@readme
class MatchingPursuitV3(BaseExperimentRunner):

    geom_spec = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r, spec = train(item, i)
            self.geom_spec = spec
            self.fake = r
            self.after_training_iteration(l)
    