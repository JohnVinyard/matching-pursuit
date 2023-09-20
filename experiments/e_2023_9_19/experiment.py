
import numpy as np
from conjure import numpy_conjure, SupportedContentType
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.overlap_add import overlap_add
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose, fft_resample
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from modules.normalization import unit_norm
from modules.phase import windowed_audio
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from util import device
from util.readmedocs import readme
from random import choice

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

def to_frequency_domain(audio):
    windowed = windowed_audio(audio, kernel_size, kernel_size // 2)
    real = windowed @ basis.real.T
    imag = windowed @ basis.imag.T
    freq_domain = torch.complex(real, imag)
    return freq_domain

def to_time_domain(spec):
    windowed = torch.flip((spec @ basis).real, dims=(-1,))
    td = overlap_add(windowed[:, None, :, :], apply_window=False)
    return td

# class MelScale(object):
#     def __init__(self):
#         super().__init__()
#         self.samplerate = zounds.SR22050()
#         self.fft_size = 512
#         self.freq_band = zounds.FrequencyBand(20, self.samplerate.nyquist)
#         self.scale = zounds.MelScale(self.freq_band, self.fft_size // 2)
#         self.basis = torch.from_numpy(morlet_filter_bank(
#             self.samplerate, self.fft_size, self.scale, 0.01)).to(device)
    
#     def transformation_basis(self, other_scale):
#         return other_scale._basis(self.scale, zounds.OggVorbisWindowingFunc())

#     def n_time_steps(self, n_samples):
#         return n_samples // (self.fft_size // 2)

#     def to_time_domain(self, spec):
#         # spec = spec.data.cpu().numpy()
#         # windowed = (spec @ self.basis.data.cpu().numpy()).real[..., ::-1]
#         # windowed = torch.from_numpy(windowed.copy())

#         windowed = torch.flip((spec @ self.basis).real, dims=(-1,))
#         td = overlap_add(windowed[:, None, :, :], apply_window=False)
#         return td

#     def to_frequency_domain(self, audio_batch):
#         windowed = windowed_audio(
#             audio_batch, self.fft_size, self.fft_size // 2)

#         # KLUDGE: torch doesn't seem to support real-to-complex
#         # multiplication at the moment;  it expects the two terms
#         # to be homogeneous
#         real = windowed @ self.basis.real.T
#         imag = windowed @ self.basis.imag.T

#         freq_domain = torch.complex(real, imag)
#         return freq_domain

    @property
    def center_frequencies(self):
        return (np.array(list(self.scale.center_frequencies)) / int(self.samplerate)).astype(np.float32)



# iterations = 64


# def is_fft_resample_part_of_the_problem(tick_size, final_size):
#     """
#     TODO: listen to each
#     """
#     signal = torch.zeros(1, 1, tick_size, device=device)
#     start = tick_size // 2
#     end = start + 10
#     signal[:, :, start: end] = torch.zeros(10, device=device).uniform_(-1, 1)
#     resampled = fft_resample(signal, final_size, is_lowest_band=True)
#     return resampled


def init_atoms(n_atoms, atom_size):
    raw = torch.zeros(n_atoms, atom_size, device=device).uniform_(-1, 1)
    return unit_norm(raw)


# iterations = {
#     512: 16,
#     1024: 16,
#     2048: 16,
#     4096: 32,
#     8192: 32,
#     16384: 32,
#     32768: 64,
# }

# atom_dict = {
#     512: init_atoms(1024, 64),
#     1024: init_atoms(512, 128),
#     2048: init_atoms(256, 256),
#     4096: init_atoms(128, 512),
#     8192: init_atoms(64, 1024),
#     16384: init_atoms(32, 2048),
#     32768: init_atoms(16, 4096),
# }

# total_iterations = sum(iterations.values())
# dict_size = sum([x.shape[0] for x in atom_dict.values()])

# print('TOTAL ITERATIONS', total_iterations)
# print('DICT SIZE', dict_size)

# def learn(batch):
#     bands = fft_frequency_decompose(batch, 512)
#     for size, band in bands.items():
#         print(f'learning band {size} with shape {band.shape}')
#         new_d = dictionary_learning_step(band, atom_dict[size], iterations[size], device=device)
#         atom_dict[size][:] = new_d

# def code(batch):
#     batch_size = batch.shape[0]

#     bands = fft_frequency_decompose(batch, 512)
#     coded = {size: sparse_code(band, atom_dict[size], iterations[size], device=device, flatten=True) for size, band in bands.items()}

#     recon_bands = {}
#     for size, encoded in coded.items():
#         print(f'coding band {size}')
#         events, scatter = encoded
#         recon = scatter((batch_size, 1, bands[size].shape[-1]), events)
#         recon_bands[size] = recon
    
#     recon = fft_frequency_recompose(recon_bands, exp.n_samples)    
#     return recon


def train(batch, i):
    with torch.no_grad():
        spec = to_frequency_domain(batch)
        spec = spec.view(batch.shape[0], -1, len(scale))
        recon = to_time_domain(spec)[..., :exp.n_samples]
        loss = F.mse_loss(recon, batch)
        print(loss.item())
        return loss, recon, torch.abs(spec)


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
    