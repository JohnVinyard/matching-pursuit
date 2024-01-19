from typing import Callable, Dict

import torch
import zounds
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F

from angle import windowed_audio
from config.experiment import Experiment
from modules.ddsp import NoiseModel
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.fft import fft_convolve, fft_shift
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_events = 16
context_dim = 16
impulse_size = 4096
resonance_size = 32768

use_dense_context = False
use_rnn = True


def fft_conv_with_padding(a: torch.Tensor, b: torch.Tensor):
    diff = b.shape[-1] - a.shape[-1]
    a = F.pad(a, (0, diff))
    result = fft_convolve(a, b)[..., :b.shape[-1]]
    return result


class GenerateImpulse(nn.Module):

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


class InstrumentModel(nn.Module):
    """
    From latent, generates:
        - control signal (noise, has time dim)
        - lookback window exponent
        - initial shape
        - shape delta (has time dim)

    f(cs, lookback, shape) => window

    """

    def __init__(self, latent_dim, channels):
        super().__init__()
        self.resonance_samples = 32768
        self.n_frames = 128

        self.latent_dim = latent_dim
        self.channels = channels

        self.embed_latent = nn.Linear(latent_dim, channels)
        self.embed_imp = nn.Linear(512, channels)

        # self.to_samples = ConvUpsample(
        #     channels,
        #     channels,
        #     start_size=self.n_frames,
        #     mode='nearest',
        #     end_size=self.resonance_samples,
        #     out_channels=1,
        #     batch_norm=True,
        #     from_latent=False
        # )
        
        self.to_samples = PosEncodedUpsample(
            channels,
            64,
            size=self.resonance_samples,
            out_channels=1,
            layers=5,
            concat=False,
            multiply=False,
            learnable_encodings=False,
            filter_bank=False)

    def forward(self, latent, impulse):

        imp = F.pad(impulse, (0, self.resonance_samples - impulse.shape[-1]))
        imp = windowed_audio(imp, 512, 256).permute(
            0, 1, 3, 2).view(-1, n_events, 512, self.n_frames)

        embedded = self.embed_imp(imp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        lat = self.embed_latent(latent).view(-1, n_events, self.channels, 1).repeat(1, 1, 1, self.n_frames)

        x = (embedded + lat).view(-1, self.channels, self.n_frames).permute(0, 2, 1)
        samples = self.to_samples(x).reshape(-1, n_events, self.resonance_samples)
        
        return samples


class MicroEvent(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()

        self.resonance_samples = 32768

        self.impulse_size = 8192

        self.n_res1 = 2048
        self.res1_size = self.resonance_samples

        self.n_res2 = 64
        self.res2_size = self.resonance_samples

        self.instr = InstrumentModel(latent_dim, channels)

        self.imp = GenerateImpulse(
            latent_dim, channels // 4, self.impulse_size, None, None)

        res1 = torch.zeros(
            1, self.n_res1, self.res1_size).uniform_(-1, 1) * (torch.linspace(1, 0, self.res1_size)[None, None, :] ** 14)

        res2 = torch.zeros(
            1, self.n_res2, self.res2_size).uniform_(-1, 1) * (torch.linspace(1, 0, self.res1_size)[None, None, :] ** 14)

        self.res1 = nn.ParameterDict(
            {str(k): v for k, v in fft_frequency_decompose(res1, 512).items()})
        self.res2 = nn.ParameterDict(
            {str(k): v for k, v in fft_frequency_decompose(res2, 512).items()})

        # self.impulse_selection = self.to_initial = LinearOutputStack(
        #     channels, layers=3, out_channels=self.n_impulses, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

        self.res1_selection = self.to_initial = LinearOutputStack(
            channels, layers=3, out_channels=self.n_res1, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

        self.res2_selection = self.to_initial = LinearOutputStack(
            channels, layers=3, out_channels=self.n_res2, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

        self.to_mix = self.to_initial = LinearOutputStack(
            channels, layers=3, out_channels=3, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

        self.to_shift = nn.Linear(latent_dim, 1)

    def forward(self, latent):

        res1_sel = torch.relu(self.res1_selection(latent))
        res2_sel = torch.relu(self.res2_selection(latent))

        imp = self.imp(latent)
        instr = self.instr.forward(latent, imp)

        imp = unit_norm(imp)

        # res1 = fft_frequency_recompose(
        #     {int(k): v for k, v in self.res1.items()}, self.resonance_samples)
        res2 = fft_frequency_recompose(
            {int(k): v for k, v in self.res2.items()}, self.resonance_samples)

        # res1 = res1.permute(0, 2, 1) @ res1_sel.permute(0, 2, 1)
        # res1 = res1.permute(0, 2, 1).view(-1, n_events, self.res1_size)
        # res1 = unit_norm(res1)

        res2 = res2.permute(0, 2, 1) @ res2_sel.permute(0, 2, 1)
        res2 = res2.permute(0, 2, 1).view(-1, n_events, self.res2_size)
        res2 = unit_norm(res2)

        # conv1 = fft_conv_with_padding(imp, res1)
        conv1 = instr
        conv2 = fft_conv_with_padding(conv1, res2)

        imp = F.pad(imp, (0, self.res2_size - self.impulse_size))
        stacked = torch.concatenate(
            [imp[..., None], conv1[..., None], conv2[..., None]], dim=-1)

        mx = torch.softmax(self.to_mix(latent), dim=-1)

        final = stacked @ mx[:, :, :, None]
        final = final.view(-1, n_events, self.res2_size)

        env = torch.concatenate([res2], dim=1)

        shift = torch.sigmoid(self.to_shift(latent)) * 0.01

        final = fft_shift(final, shift.view(-1, n_events, 1))
        return final, env


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

        # self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            context_dim, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((context_dim,)))

        self.to_amp = nn.Linear(256, 1)

        self.to_events = MicroEvent(256, 256)

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

    def generate(self, encoded, one_hot, packed, dense):
        if use_dense_context:
            ctxt = self.from_context(dense)
        else:
            # ctxt = make_memory_context(encoded, 4) # (batch, encoding_channels, time)
            # print(ctxt.shape)
            ctxt = torch.sum(encoded, dim=-1)
            dense = self.embed_memory_context(ctxt)  # (batch, context_dim)

        # ctxt is a single vector
        ce = self.embed_context(ctxt)

        # one hot is n_events vectors
        oh = self.embed_one_hot(one_hot)

        embeddings = ce[:, None, :] + oh

        # generate...

        # TODO: This is where we generate events
        mixed, env = self.to_events(embeddings)
        amps = self.to_amp(embeddings)
        mixed = mixed * amps

        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events,
                         exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]

        final = self.verb.forward(dense, final)

        return final, env

    def forward(self, x):
        encoded = self.encode(x)

        non_sparse = torch.mean(encoded, dim=-1)

        non_sparse_mean = self.to_context_mean(non_sparse)
        non_sparse_std = self.to_context_std(non_sparse)
        non_sparse = non_sparse_mean + \
            (torch.zeros_like(non_sparse_mean).normal_(0, 1) * non_sparse_std)

        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)

        final, env = self.generate(encoded, one_hot, packed, dense=non_sparse)
        return final, encoded, env, non_sparse


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

    # bands = stft(x, 2048, 256, pad=True)
    # return dict(bands=bands)


def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

    # target = stft(target, ws, ss, pad=True)
    target = multiband_transform(target)

    full = torch.sum(recon, dim=1, keepdim=True)
    # full = stft(full, ws, ss, pad=True)
    full = multiband_transform(full)

    # residual = target - full
    residual = dict_op(target, full, lambda a, b: a - b)

    loss = 0

    for i in range(n_events):
        ch = recon[:, i: i + 1, :]
        # ch = stft(ch, ws, ss, pad=True)
        ch = multiband_transform(ch)

        # t = residual + ch
        t = dict_op(residual, ch, lambda a, b: a + b)

        # loss = loss + F.mse_loss(ch, t.clone().detach())
        diff = dict_op(ch, t, lambda a, b: a - b)
        loss = loss + sum([torch.abs(y).sum() for y in diff.values()])

        # loss = loss + torch.abs(ch - t.clone().detach()).sum()

    return loss


def train(batch, i):
    optim.zero_grad()

    recon, encoded, env, context = model.forward(batch)

    env = F.avg_pool1d(torch.abs(env), 512, 256, padding=256)
    env = torch.diff(env, dim=-1)
    env_loss = env.mean()

    recon_summed = torch.sum(recon, dim=1, keepdim=True)

    # loss = (single_channel_loss(batch, recon) * 1e-6) + env_loss
    real_spec = stft(batch, 2048, 256, pad=True)
    fake_spec = stft(recon_summed, 2048, 256, pad=True)
    loss = F.mse_loss(fake_spec, real_spec)

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
class GraphRepresentation2(BaseExperimentRunner):

    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)

            if l is None:
                continue

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l)
