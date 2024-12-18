import torch
from torch import nn
from torch.nn import functional as F

from modules import ReverbGenerator
from modules.ddsp import NoiseModel
from modules.eventgenerators.generator import EventGenerator, ShapeSpec
from modules.eventgenerators.schedule import DiracScheduler
from modules.linear import LinearOutputStack
from modules.normalization import unit_norm
from modules.transfer import ResonanceChain, make_waves
from modules.upsample import ConvUpsample
from util import device
from util.music import musical_scale_hz

class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels, mixer_channels=2):
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
            weight_norm=True
        )

        self.noise_model = NoiseModel(
            channels,
            self.n_frames,
            self.n_frames * 4,
            self.n_samples,
            self.channels,
            weight_norm=True,
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
        )

        self.to_env = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):
        batch_size = x.shape[0]

        env = self.to_env(x) ** 2
        env = F.interpolate(env, mode='linear', size=self.n_samples)

        x = self.to_frames(x)
        x = self.noise_model(x)
        x = x.view(batch_size, -1, self.n_samples)

        x = x * env
        return x


class ConvImpulseEventGenerator(EventGenerator, nn.Module):



    def __init__(
            self,
            context_dim: int,
            impulse_size: int,
            resonance_size: int,
            samplerate: int,
            n_samples: int,
            n_events: int = 1):

        super().__init__()
        self.n_samples = n_samples
        self.impulse_size = impulse_size
        self.context_dim = context_dim
        self.n_events = n_events
        self.resonance_size = resonance_size
        self.samplerate = samplerate

        self.imp = GenerateImpulse(
            256, 128, impulse_size, 16, n_events)
        total_atoms = 4096
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_atoms // 4)
        waves = make_waves(resonance_size, f0s, int(samplerate))

        self.from_context = nn.Linear(context_dim, 256)

        self.scheduler = DiracScheduler(
            n_events=n_events, start_size=self.n_samples // 256, n_samples=self.n_samples)

        self.res = ResonanceChain(
            1,
            n_atoms=total_atoms,
            window_size=512,
            n_frames=256,
            total_samples=resonance_size,
            mix_channels=16,
            channels=64,
            latent_dim=256,
            initial=waves,
            learnable_resonances=False)


        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((self.context_dim,)))

    @property
    def shape_spec(self) -> ShapeSpec:
        return dict(vecs=(self.context_dim,))

    def forward(self, vecs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        batch_size = vecs.shape[0]
        embeddings = self.from_context(vecs)
        amps = torch.sum(times, dim=-1, keepdim=True)

        # impulses
        imp = self.imp.forward(embeddings)
        imp = unit_norm(imp)

        # # resonances
        mixed = self.res.forward(embeddings, imp)
        mixed = mixed.view(batch_size, -1, self.resonance_size)

        mixed = unit_norm(mixed)
        mixed = mixed * amps

        # coarse positioning
        # up = upsample_with_holes(times, self.n_samples)
        # final = F.pad(mixed, (0, self.n_samples - mixed.shape[-1]))
        #
        # final = fft_convolve(final, up)[..., :self.n_samples]

        final = self.scheduler.schedule(times, mixed)

        final = self.verb.forward(unit_norm(vecs, dim=-1), final)

        return final

    # def random_sequence(self) -> torch.Tensor:
    #     vecs = torch.zeros(1, 1, self.context_dim, device=device).uniform_(-1, 1)
    #     times = self.scheduler.random_params()
    #     final = self.forward(vecs, times)
    #     return final


