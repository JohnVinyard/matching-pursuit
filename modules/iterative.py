from typing import Callable
import torch

from modules.anticausal import AntiCausalStack
from modules.fft import fft_convolve
from modules.impulse import GenerateImpulse
from modules.normalization import unit_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify, sparsify_vectors
from modules.transfer import ResonanceChain, make_waves
from util.music import musical_scale_hz
from torch import nn
from torch.nn import functional as F

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def sort_channels_descending_norm(x: torch.Tensor) -> torch.Tensor:
    diff = torch.norm(x, dim=(-1), p=1)
    indices = torch.argsort(diff, dim=-1, descending=True)
    srt = torch.take_along_dim(x, indices[:, :, None], dim=1)
    return srt

def iterative_loss(
        target_audio: torch.Tensor,
        recon_channels: torch.Tensor,
        transform: TensorTransform,
        return_residual: bool = False,
        ratio_loss: bool = False):
    batch, _, time = target_audio.shape

    batch_size, n_events, time = recon_channels.shape

    # perform a transform on the target audio and flatten everything
    # but the batch dimension
    target = transform(target_audio.view(batch, 1, time)) \
        .reshape(target_audio.shape[0], -1)

    # perform the same transform on each reconstruction channel
    channels = transform(recon_channels.view(batch, n_events, time)) \
        .reshape(batch, n_events, -1)

    residual = target

    # sort channels from loudest to softest
    diff = torch.norm(channels, dim=(-1), p=1)
    indices = torch.argsort(diff, dim=-1, descending=True)

    srt = torch.take_along_dim(channels, indices[:, :, None], dim=1)

    loss = 0

    for i in range(n_events):
        current = srt[:, i, :]
        start_norm = torch.norm(residual, dim=-1, p=1)
        # TODO: should the residual be cloned and detached each time,
        # so channels are optimized independently?
        residual = residual - current
        end_norm = torch.norm(residual, dim=-1, p=1)
        if ratio_loss:
            loss = loss + (end_norm / (start_norm + 1e-12)).sum()
        else:
            diff = -(start_norm - end_norm)
            loss = loss + diff.sum()

    if return_residual:
        return residual, loss

    return loss


class IterativeDecomposer(nn.Module):

    def __init__(
            self,
            context_dim: int,
            n_events: int,
            impulse_size: int,
            resonance_size: int,
            samplerate: int,
            n_samples: int,
            transform: TensorTransform):

        super().__init__()

        self.context_dim = context_dim
        self.n_events = n_events
        self.impulse_size = impulse_size
        self.resonance_size = resonance_size
        self.samplerate = samplerate
        self.n_samples = n_samples
        self.transform = transform

        # TODO: Channels should be configurable
        self.encoder = AntiCausalStack(1024, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1])
        self.to_event_vectors = nn.Conv1d(1024, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(1024, 1, 1, 1, 0)
        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)
        self.embed_latent = nn.Linear(1024, context_dim)
        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)

        total_atoms = 4096
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_atoms // 4)
        waves = make_waves(resonance_size, f0s.tolist(), int(samplerate))

        # TODO: Channels should be configurable
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
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))

        self.from_context = nn.Linear(context_dim, 256)

        self.atom_bias = nn.Parameter(torch.zeros(4096).uniform_(-1, 1))

        # self.apply(lambda x: exp.init_weights(x))
        raise NotImplementedError('Initialize weights')

    def encode(self, x, n_events=1):
        batch_size = x.shape[0]

        if x.shape[1] == 1:
            x = self.transform(x)

        encoded = self.encoder.forward(x)

        event_vecs = self.to_event_vectors(encoded).permute(0, 2, 1)  # batch, time, channels

        event_switch = self.to_event_switch(encoded)
        attn = torch.relu(event_switch).permute(0, 2, 1).view(batch_size, 1, -1)

        attn, attn_indices, values = sparsify(attn, n_to_keep=n_events, return_indices=True)

        vecs, indices = sparsify_vectors(event_vecs.permute(0, 2, 1), attn, n_to_keep=n_events)

        scheduling = torch.zeros(batch_size, n_events, encoded.shape[-1], device=encoded.device)
        for b in range(batch_size):
            for j in range(n_events):
                index = indices[b, j]
                scheduling[b, j, index] = attn[b, 0][index]

        return vecs, scheduling

    def generate(self, vecs, scheduling):

        batch_size = vecs.shape[0]

        embeddings = self.from_context(vecs)

        amps = torch.sum(scheduling, dim=-1, keepdim=True)

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        imp = unit_norm(imp)

        # # resonances
        mixed = self.res.forward(embeddings, imp)
        mixed = mixed.view(batch_size, -1, self.resonance_size)

        mixed = unit_norm(mixed)

        mixed = mixed * amps

        # coarse positioning

        # TODO: use upsample with holes here

        final = F.pad(mixed, (0, self.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], final.shape[1], self.n_samples, device=final.device)
        up[:, :, ::256] = scheduling

        final = fft_convolve(final, up)[..., :self.n_samples]

        final = self.verb.forward(unit_norm(vecs, dim=-1), final)

        return final, imp, amps, mixed

    def iterative(self, x):
        channels = []
        schedules = []
        vecs = []

        spec = self.transform(x)

        for i in range(self.n_events):
            v, sched = self.encode(spec, n_events=1)
            vecs.append(v)
            schedules.append(sched)
            ch, _, _, _ = self.generate(v, sched)
            current = self.transform(ch)
            spec = (spec - current).clone().detach()
            channels.append(ch)

        channels = torch.cat(channels, dim=1)
        vecs = torch.cat(vecs, dim=1)
        schedules = torch.cat(schedules, dim=1)

        return channels, vecs, schedules

    def forward(self, x, return_context=True):

        channels, vecs, scheduling = self.iterative(x)

        print(f'In forward, calling generate with {vecs.shape} and {scheduling.shape}')
        final, imp, amps, mixed = self.generate(vecs, scheduling)

        if return_context:
            return final, vecs, imp, scheduling, amps, mixed
        else:
            # return channels, vecs, imp, scheduling, amps
            raise NotImplementedError('This code path is no longer supported')
