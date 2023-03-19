from typing import List
import numpy as np
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.matchingpursuit import build_scatter_segments, dictionary_learning_step, sparse_code, compare_conv
from modules.normalization import unit_norm
from modules.pointcloud import decode_events, encode_events
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from torch import nn
from util.readmedocs import readme
import zounds
from util import device, playable
import torch
from torch.nn import functional as F


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class BandSpec(object):
    def __init__(self, size, n_atoms, atom_size, slce=None, device=None):
        super().__init__()
        self.size = size
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.slce = slce
        self.device = device

        d = torch.zeros(
            n_atoms, atom_size, requires_grad=False).uniform_(-1, 1).to(device)
        self.d = unit_norm(d)

    @property
    def filename(self):
        return f'band_{self.size}.dat'

    def load(self):
        try:
            d = torch.load(self.filename)
            self.d = d
            print(f'loaded {self.filename}')
        except IOError:
            print(f'failed to load ${self.filename}')

    def store(self):
        torch.save(self.d, self.filename)

    def learn(self, batch, steps=16):
        d = dictionary_learning_step(
            batch, self.d, steps, device=self.device, approx=self.slce)
        self.d = unit_norm(d)
        return d

    def encode(self, batch, steps=16):
        instances, scatter = sparse_code(
            batch, self.d, steps, device=self.device, approx=self.slce)
        all_instances = []
        for k, v in instances.items():
            all_instances.extend(v)
        return all_instances, scatter, batch.shape

    def decode(self, shape, all_instances, scatter):
        return scatter(shape, all_instances)

    def recon(self, batch, steps=16):
        # instances, scatter = sparse_code(
        #     batch, self.d, steps, device=self.device, approx=self.slce)
        # all_instances = []
        # for k, v in instances.items():
        #     all_instances.extend(v)

        all_instances, scatter, shape = self.encode(batch, steps)

        # recon = scatter(batch.shape, all_instances)
        recon = self.decode(shape, all_instances, scatter)
        return recon, all_instances, scatter


class MultibandDictionaryLearning(object):
    def __init__(self, specs: List[BandSpec]):
        super().__init__()
        self.bands = {spec.size: spec for spec in specs}
        self.min_size = min(map(lambda spec: spec.size, specs))
        self.n_samples = exp.n_samples

    @property
    def band_dicts(self):
        return {size: band.d for size, band in self.bands.items()}

    def partial_decoding_dict(self, batch_size):
        return {
            size: (build_scatter_segments(size, self.bands[size].atom_size), (batch_size, 1, size)) 
            for size in self.bands.keys()
        }

    def store(self):
        for band in self.bands.values():
            band.store()

    def load(self):
        for band in self.bands.values():
            band.load()

    def learn(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)
        for size, band in bands.items():
            self.bands[size].learn(band, steps)

    def encode(self, batch, steps):
        bands = fft_frequency_decompose(batch, self.min_size)
        return {size: band.encode(bands[size], steps) for size, band in self.bands.items()}

    def decode(self, d):
        output = {}
        for size, tup in d.items():
            all_instances, scatter, shape = tup
            recon = self.bands[size].decode(shape, all_instances, scatter)
            output[size] = recon
        recon = fft_frequency_recompose(output, self.n_samples)
        return recon

    def recon(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)

        recon_bands = {}
        events = {}
        scatter = {}
        for size, spec in self.bands.items():
            r, e, s = self.bands[size].recon(bands[size], steps)
            recon_bands[size] = r
            events[size] = e
            scatter[size] = s

        recon = fft_frequency_recompose(recon_bands, batch.shape[-1])
        return recon, events


class Predictor(nn.Module):
    """
    Analyze a sequence of atoms with absolute positions
    and magnitudes.

    Output a new atom, relative position and relative magnitude
    """

    def __init__(self, channels, n_atoms=512 * 7):
        super().__init__()

        self.embed = nn.Embedding(n_atoms, embedding_dim=channels)
        self.pos_amp = nn.Linear(2, channels)

        self.reduce = nn.Conv1d(channels * 2, channels, 1, 1, 0)

        self.net = DilatedStack(channels, [1, 3, 9, 27, 81, 1], dropout=0.1)

        self.to_atom = LinearOutputStack(channels, 3, out_channels=n_atoms)
        self.to_pos_amp_pred = LinearOutputStack(channels, 3, out_channels=2)

        self.apply(lambda x: exp.init_weights(x))

    def generate(self, x, steps):


        output = x
        batch_size = x.shape[0]

        with torch.no_grad():
            for i in range(steps):
                seed = output[:, :, i:]

                a, pa = self.forward(seed)
                a = a.view(batch_size, -1)
                p = pa.view(batch_size, 2, 1)
                a = torch.argmax(a, dim=-1, keepdim=True)

                last_pos_amp = seed[:, 1:3, -1:]
                new_pos_amp = last_pos_amp + p

                next_one = torch.zeros(batch_size, 4, device=x.device)
                next_one[:, 0] = a

                next_one[:, 1:3] = new_pos_amp.view(1, 2)
                next_one = next_one.view(batch_size, 4, 1)

                output = torch.cat([output, next_one], dim=-1)


        first = output[:, :, :steps // 2]
        second = output[:, :, steps // 2:]

        return first, second

    def forward(self, x):
        x = x.permute(0, 2, 1)
        atoms = x[:, :, 0].long()
        atoms = self.embed.forward(atoms)

        pos_amp = x[:, :, 1:3]
        pos_amp = self.pos_amp.forward(pos_amp)

        x = torch.cat([atoms, pos_amp], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.reduce.forward(x)
        x = self.net(x)
        x = x.permute(0, 2, 1)

        a = self.to_atom(x)[:, -1:, :]
        pa = self.to_pos_amp_pred(x)[:, -1:, :]
        return a, pa


def to_slice(n_samples, percentage):
    n_coeffs = n_samples // 2 + 1
    start = n_coeffs // 2
    total = n_coeffs - start
    size = int(percentage * total)
    end = start + size
    return slice(start, end)


n_atoms = 512
steps = 64
n_training_atoms = 8
dictionary_learning_iterations = 100

model = MultibandDictionaryLearning([
    BandSpec(512,   n_atoms, 128,  slce=None, device=device),
    BandSpec(1024,  n_atoms, 128,  slce=None, device=device),
    BandSpec(2048,  n_atoms, 128,  slce=None, device=device),
    BandSpec(4096,  n_atoms, 128, slce=None, device=device),
    BandSpec(8192,  n_atoms, 128, slce=None, device=device),
    BandSpec(16384, n_atoms, 128, slce=None, device=device),
    BandSpec(32768, n_atoms, 128, slce=None, device=device),
])
model.load()

total_atoms = n_atoms * len(model.bands)

predictor = Predictor(exp.model_dim, total_atoms).to(device)
optim = optimizer(predictor)


def train():
    pass


@readme
class BasicMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.encoded = None

    def recon(self, steps=steps):
        recon, events = model.recon(self.real[:1, ...], steps=steps)
        return playable(recon, exp.samplerate)

    def generate(self, steps=total_atoms):

        coding_steps = 64

        with torch.no_grad():
            encoded = self.encode_for_transformer(self.real[:1, ...], coding_steps)

        f, s = predictor.generate(encoded, steps=steps)

        fa = self.decode_to_audio(f, steps=coding_steps)
        fs = self.decode_to_audio(s, steps=coding_steps)

        n = zounds.AudioSamples.concat(fa, fs)
        return zounds.AudioSamples(n, exp.samplerate)

    def encode_for_transformer(self, batch, steps):
        # size -> (all_instances, scatter, shape)
        encoding = model.encode(batch, steps=steps)
        e = {k: v[0] for k, v in encoding.items()}  # size -> all_instances
        events = encode_events(e, steps)  # tensor (batch, 4, N)
        return events
    
    def decode_to_audio(self, events, steps):
        with torch.no_grad():
            encoding = model.partial_decoding_dict(self.real.shape[0])
            d = decode_events(events, model.band_dicts, steps)  # size -> all_instances
            # size -> (all_instances, scatter, shape)
            d = {k: (d[k], *encoding[k]) for k in d.keys()}
            recon = model.decode(d)
            return playable(recon, exp.samplerate)

    def round_trip(self, steps):
        # size -> (all_instances, scatter, shape)
        encoding = model.encode(self.real[:1, ...], steps=steps)
        e = {k: v[0] for k, v in encoding.items()}  # size -> all_instances
        events = encode_events(e, steps)  # tensor

        d = decode_events(events, model.band_dicts, steps)  # size -> all_instances
        # size -> (all_instances, scatter, shape)
        d = {k: (d[k], *encoding[k][1:]) for k in d.keys()}
        recon = model.decode(d)
        return playable(recon, exp.samplerate)

    def store(self):
        model.store()

    def spec(self, steps=steps):
        return np.abs(zounds.spectral.stft(self.recon(steps)))

    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            batch_size = item.shape[0]

            print('========================================')

            if i < dictionary_learning_iterations:
                with torch.no_grad():
                    model.learn(item, steps=steps)
                    print(i, 'sparse coding step')

            transformer_encoded = self.encode_for_transformer(
                item, steps=steps)
            transformer_encoded = transformer_encoded[:, :3, :]

            inputs = transformer_encoded[:, :, :-1]
            atom_targets = transformer_encoded[:, 0, -1:]
            rel_targets = torch.diff(transformer_encoded[:, 1:, -2:], dim=-1)

            pred_atoms, pred_pos_amp = predictor.forward(inputs)
            pred_atoms = pred_atoms.view(batch_size, -1)

            atom_loss = F.cross_entropy(
                pred_atoms,
                atom_targets.view(-1).long()
            )

            print(
                torch.argmax(pred_atoms, dim=-1).view(-1),
                atom_targets.view(-1).long())

            rel_loss = F.mse_loss(
                pred_pos_amp.view(batch_size, 2),
                rel_targets.view(batch_size, 2)
            )
            print(rel_loss.item())

            loss = atom_loss + rel_loss
            loss.backward()
            print(i, 'MODEL LOSS', loss.item())
            optim.step()

            self.generate()
