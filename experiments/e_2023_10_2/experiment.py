from typing import Callable, Dict
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.angle import windowed_audio
from modules.decompose import fft_frequency_decompose
from modules.mixer import MixerStack
from modules.overlap_add import overlap_add
from modules.ddsp import NoiseModel
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.pif import fft_based_pif
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from scipy.signal import square, sawtooth


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=512)


n_events = 64
context_dim = 16
impulse_size = 4096
resonance_size = 32768
base_resonance = 0.02


def make_waves(n_samples, f0s, samplerate):
    sawtooths = []
    squares = []
    triangles = []

    for f0 in f0s:
        f0 = f0 / (samplerate // 2)
        rps = f0 * np.pi
        radians = np.linspace(0, rps * n_samples, n_samples)
        sq = square(radians)[None, ...]
        squares.append(sq)
        st = sawtooth(radians)[None, ...]
        sawtooths.append(st)
        tri = sawtooth(radians, 0.5)[None, ...]
        triangles.append(tri)
    
    sawtooths = np.concatenate(sawtooths, axis=0)
    squares = np.concatenate(squares, axis=0)
    triangles = np.concatenate(triangles, axis=0)
    return sawtooths, squares, triangles


class RecurrentResonanceModelWithComplexWaveforms(nn.Module):
    def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.resonance_samples = resonance_samples
        self.filter_coeffs = window_size // 2 + 1

        n_atoms = 2048
        self.n_frames = resonance_samples // (window_size // 2)
        self.res_factor = (1 - base_resonance) * 0.95

        n_f0s = n_atoms // 3
        f0s = np.linspace(40, 4000, n_f0s)
        sq, st, tri = make_waves(self.resonance_samples, f0s, int(exp.samplerate))


        atoms = np.concatenate([sq, st, tri], axis=0)
        bank = torch.from_numpy(atoms).float()
        n_atoms = bank.shape[0]

        self.register_buffer('atoms', bank)

        # we don't want the filter to dominate the spectral shape, just roll off highs, mostly
        self.to_filter = ConvUpsample(
            latent_dim, channels, 4, end_size=self.n_frames, mode='learned', out_channels=32, from_latent=True, batch_norm=True)
        self.selection = nn.Linear(latent_dim, n_atoms)
        self.to_momentum = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):

        # compute resonance/sustain
        # computing each frame independently makes something like "damping" possible
        mom = base_resonance + \
            (torch.sigmoid(self.to_momentum(x)) * self.res_factor)
        mom = torch.log(1e-12 + mom)
        mom = torch.cumsum(mom, dim=-1)
        mom = torch.exp(mom)
        new_mom = mom

        sel = torch.relu(self.selection(x))
        atoms = self.atoms
        res = sel @ atoms

        # compute low-pass filter time-series
        filt = self.to_filter(x).view(-1, 32, self.n_frames).permute(0, 2, 1)
        filt = F.interpolate(filt, size=self.filter_coeffs, mode='linear').view(-1, n_events, self.n_frames, self.filter_coeffs)
        filt = torch.sigmoid(filt)

        windowed = windowed_audio(res, self.window_size, self.window_size // 2)
        windowed = unit_norm(windowed, dim=-1)
        windowed = windowed * new_mom[..., None]
        windowed = torch.fft.rfft(windowed, dim=-1)
        windowed = windowed * filt
        windowed = torch.fft.irfft(windowed, dim=-1)
        windowed = overlap_add(
            windowed, apply_window=False)[..., :self.resonance_samples]

        return windowed


class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels, mixer_channels = 2):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        mixer_channels = mixer_channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=mixer_channels, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

    def forward(self, x):
        x = self.to_mix(x)
        x = x.view(-1, self.encoding_channels, 1)
        x = torch.softmax(x, dim=-1)
        return x


class GenerateImpulse(nn.Module):
    """
    Given a latent vector, generate a noisy, energetic impulse
    """

    def __init__(self, latent_dim, channels, n_samples, n_filter_bands, encoding_channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.n_frames = n_samples // 256
        self.n_filter_bands = n_filter_bands
        self.channels = channels
        self.filter_kernel_size = 16
        self.encoding_channels = encoding_channels

        self.to_frames = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            mode='learned',
            end_size=self.n_frames,
            out_channels=channels,
            batch_norm=True
        )

        self.noise_model = NoiseModel(
            channels,
            self.n_frames,
            self.n_frames * 16,
            self.n_samples,
            self.channels,
            batch_norm=True,
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
        )

    def forward(self, x):
        x = self.to_frames(x)
        x = self.noise_model(x)
        return x.view(-1, n_events, self.n_samples)



class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.down = nn.Sequential(
            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 4
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.up = nn.Sequential(
            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 128
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.proj = nn.Conv1d(1024, 4096, 1, 1, 0)

    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x

        for layer in self.up:
            x = layer(x)
            size = x.shape[-1]
            if size in context:
                x = x + context[size]

        x = self.proj(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNet(1024)

        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)

        ResonanceModel = RecurrentResonanceModelWithComplexWaveforms

        self.res = ResonanceModel(
            n_events, 256, 64, 1024, resonance_samples=resonance_size)

        self.mix = GenerateMix(256, 128, n_events, mixer_channels=2)
        self.to_amp = nn.Linear(256, 1)

        self.verb = ReverbGenerator(
            context_dim, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((context_dim,)))

        self.to_context_mean = nn.Linear(4096, context_dim)
        self.to_context_std = nn.Linear(4096, context_dim)
        self.embed_memory_context = nn.Linear(4096, context_dim)

        self.from_context = nn.Linear(context_dim, 4096)

        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(
            self.refractory_period, power=10, device=device))

        self.apply(lambda x: exp.init_weights(x))

    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded = self.encoder.forward(x)
        encoded = F.dropout(encoded, 0.05)

        ref = F.pad(self.refractory,
                    (0, encoded.shape[-1] - self.refractory_period))
        encoded = fft_convolve(encoded, ref)[..., :encoded.shape[-1]]

        return encoded

    def generate(self, encoded, one_hot, packed):
        
        ctxt = torch.sum(encoded, dim=-1)
        dense = self.embed_memory_context(ctxt) # (batch, context_dim)

        # ctxt is a single vector
        ce = self.embed_context(ctxt)
        
        # one hot is n_events vectors
        oh = self.embed_one_hot(one_hot)

        embeddings = ce[:, None, :] + oh

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        res = self.res.forward(embeddings)

        # mixes
        mx = self.mix.forward(embeddings)

        conv = fft_convolve(padded, res)[..., :resonance_size]

        stacked = torch.cat([padded[..., None], conv[..., None]], dim=-1)
        mixed = stacked @ mx.view(-1, n_events, 2, 1)
        mixed = mixed.view(-1, n_events, resonance_size)

        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps

        # TODO: I could also try:
        # - positioning by best fit
        # - positioning using scalar positions
        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]

        final = self.verb.forward(dense, final)

        return final, imp
    
    # def sparse_encode(self, x):
    #     encoded = self.encode(x)
    #     encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
    #     return encoded

    def forward(self, x):
        encoded = self.encode(x)
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        encoded = torch.relu(encoded)
        final, imp = self.generate(encoded, one_hot, packed)
        return final, encoded, imp


model = Model().to(device)
optim = optimizer(model, lr=1e-3)



def dict_op(
        a: Dict[int, torch.Tensor],
        b: Dict[int, torch.Tensor],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[int, torch.Tensor]:

    return {k: op(v, b[k]) for k, v in a.items()}


def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d2 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    return dict(**d1, **d2)

    # pif = exp.perceptual_feature(x)
    # pif = fft_based_pif(x, 256, 64)
    # return { 'pif': pif }


# def ratio_loss(target: torch.Tensor, recon: torch.Tensor):
#     target = stft(target, 2048, 256, pad=True)
    
#     residual = target
    
#     loss = 0
    
#     indices = np.random.permutation(n_events)
#     for i in indices:
#         ch = recon[:, i: i + 1, :]
#         ch = stft(ch, 2048, 256, pad=True)
#         start_norm = torch.norm(residual, dim=(1, 2))
#         residual = residual - ch
#         end_norm = torch.norm(residual, dim=(1, 2))
#         loss = loss + (end_norm / (start_norm + 1e-12)).mean()    
    
#     return loss

def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

    target = multiband_transform(target)

    full = torch.sum(recon, dim=1, keepdim=True)
    full = multiband_transform(full)

    residual = dict_op(target, full, lambda a, b: a - b)
    
    loss = 0

    # for i in range(n_events):
    i = np.random.randint(0, n_events)
    ch = recon[:, i: i + 1, :]
    ch = multiband_transform(ch)

    t = dict_op(residual, ch, lambda a, b: a + b)

    diff = dict_op(ch, t, lambda a, b: a - b)
    loss = loss + sum([torch.abs(y).sum() for y in diff.values()])

    return loss




def train(batch, i):
    optim.zero_grad()

    recon, encoded, imp = model.forward(batch)
    # print('========================================')
    # sparsity_loss = torch.abs(encoded).sum() * 0.001
    # nz = (encoded > 0).view(batch.shape[0], -1).sum(dim=-1, dtype=torch.float32).mean()
    # print('AVERAGE SPARSITY', nz.item())
    
    energy_loss = torch.abs(imp).sum(dim=-1).mean() * 1e-5
    print('ENERGY LOSS', energy_loss.item())
    
    recon_summed = torch.sum(recon, dim=1, keepdim=True)

    loss = (single_channel_loss(batch, recon) * 1e-6) #+ energy_loss
    
    
    loss.backward()
    optim.step()

    print('GEN', loss.item())

    recon = max_norm(recon_summed)
    encoded = max_norm(encoded)
    return loss, recon, encoded


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x[:, None, :, :]
        x = F.max_pool2d(x, (16, 8), (16, 8))
        x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        return x
    return (encoded,)


@readme
class GraphRepresentation(BaseExperimentRunner):

    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None, save_weights=False, load_weights=True, model=model):
        
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            save_weights=save_weights, 
            load_weights=load_weights, 
            model=model)

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)
            

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l, i)
