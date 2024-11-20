from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, loggers
from data import AudioIterator
from modules import stft, sparsify, sparsify_vectors, iterative_loss, max_norm, flattened_multiband_spectrogram, \
    DownsamplingDiscriminator, sparse_softmax
from modules.anticausal import AntiCausalAnalysis
from modules.eventgenerators.convimpulse import ConvImpulseEventGenerator
from modules.eventgenerators.generator import EventGenerator
from modules.eventgenerators.overfitresonance import OverfitResonanceModel
from modules.eventgenerators.splat import SplattingEventGenerator
from modules.eventgenerators.ssm import StateSpaceModelEventGenerator
from modules.multiheadtransform import MultiHeadTransform
from util import device, encode_audio, make_initializer

# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 15
samples_per_event = 2048
n_events = n_samples // samples_per_event
context_dim = 16

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256


n_frames = n_samples // transform_step_size

initializer = make_initializer(0.05)


def transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, n_frames, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


class Discriminator(nn.Module):
    def __init__(self, disc_type='dilated'):
        super().__init__()
        if disc_type == 'dilated':
            self.disc = DownsamplingDiscriminator(
                window_size=2048, step_size=256, n_samples=n_samples, channels=256)
        elif disc_type == 'unet':
            self.disc = DownsamplingDiscriminator(2048, 256, n_samples=n_samples, channels=256)

        self.apply(initializer)

    def forward(self, transformed: torch.Tensor):
        x = self.disc(transformed)
        return x


class Model(nn.Module):
    def __init__(
            self,
            resonance_model: EventGenerator,
            in_channels: int = 1024,
            hidden_channels: int = 256):

        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.encoder = AntiCausalAnalysis(
            in_channels=in_channels,
            channels=hidden_channels,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 64, 1],
            pos_encodings=False,
            do_norm=True)
        self.to_event_vectors = nn.Conv1d(hidden_channels, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(hidden_channels, 1, 1, 1, 0)
        self.resonance = resonance_model

        self.multihead = MultiHeadTransform(
            latent_dim=context_dim,
            hidden_channels=hidden_channels,
            n_layers=2,
            shapes=self.resonance.shape_spec
        )
        self.apply(initializer)

    def encode(self, transformed: torch.Tensor):
        n_events = 1

        batch_size = transformed.shape[0]

        if transformed.shape[1] == 1:
            transformed = transform(transformed)

        x = transformed

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

    def generate(self, vecs: torch.Tensor, scheduling: torch.Tensor):
        choices = self.multihead.forward(vecs)
        choices_with_scheduling = dict(**choices, times=scheduling)
        events = self.resonance.forward(**choices_with_scheduling)
        return events

    def random_sequence(self) -> torch.Tensor:
        vecs = torch.zeros(1, 1, context_dim, device=device).uniform_(-1, 1)
        times = sparse_softmax(
            torch.zeros(1, 1, n_frames, device=device).uniform_(-1, 1), normalize=True, dim=-1)
        final = self.generate(vecs, times)
        return final

    def iterative(self, audio: torch.Tensor):
        channels = []
        schedules = []
        vecs = []

        spec = transform(audio)

        for i in range(n_events):
            v, sched = self.encode(spec)
            vecs.append(v)
            schedules.append(sched)
            ch = self.generate(v, sched)
            current = transform(ch)
            spec = (spec - current).clone().detach()
            channels.append(ch)

        channels = torch.cat(channels, dim=1)
        vecs = torch.cat(vecs, dim=1)
        schedules = torch.cat(schedules, dim=1)

        return channels, vecs, schedules

    def forward(self, audio: torch.Tensor):
        raise NotImplementedError()


def train_and_monitor(
        batch_size: int = 8,
        overfit: bool = False,
        disc_type: str = 'dilated',
        model_type: str = 'conv',
        wipe_old_data: bool = True):
    stream = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True,
        overfit=overfit)

    collection = LmdbCollection(path='iterativedecomposition')

    if wipe_old_data:
        print('Wiping previous experiment data')
        collection.destroy()

    collection = LmdbCollection(path='iterativedecomposition')

    recon_audio, orig_audio, random_audio = loggers(
        ['recon', 'orig', 'random'],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure([
        orig_audio,
        recon_audio,
        random_audio
    ], port=9999, n_workers=1)

    print(f'training on {n_seconds} of audio and {n_events} with {model_type} event generator and {disc_type} disc')

    def train():
        hidden_channels = 512
        if model_type == 'lookup':
            resonance_model = OverfitResonanceModel(
                n_noise_filters=32,
                noise_expressivity=8,
                noise_filter_samples=128,
                noise_deformations=16,
                instr_expressivity=8,
                n_events=1,
                n_resonances=512,
                n_envelopes=128,
                n_decays=32,
                n_deformations=32,
                n_samples=n_samples,
                n_frames=n_frames,
                samplerate=samplerate,
                hidden_channels=hidden_channels
            )
        elif model_type == 'conv':
            resonance_model = ConvImpulseEventGenerator(
                context_dim=context_dim,
                impulse_size=8192,
                resonance_size=n_samples,
                samplerate=samplerate,
                n_samples=n_samples,
            )
        elif model_type == 'splat':
            resonance_model = SplattingEventGenerator(
                n_samples=n_samples,
                samplerate=samplerate,
                n_resonance_octaves=64,
                n_frames=n_frames
            )
        elif model_type == 'ssm':
            resonance_model = StateSpaceModelEventGenerator(
                context_dim=context_dim,
                control_plane_dim=16,
                input_dim=512,
                state_dim=32,
                hypernetwork_dim=16,
                hypernetwork_latent=16,
                samplerate=samplerate,
                n_samples=n_samples,
                n_frames=n_frames,
            )
        else:
            raise ValueError(f'Unknown model type {model_type}')

        print(resonance_model.shape_spec)

        model = Model(
            resonance_model=resonance_model,
            in_channels=1024,
            hidden_channels=hidden_channels).to(device)
        optim = Adam(model.parameters(), lr=1e-3)

        disc = Discriminator(disc_type=disc_type).to(device)
        disc_optim = Adam(disc.parameters(), lr=1e-3)

        for i, item in enumerate(iter(stream)):
            optim.zero_grad()
            disc_optim.zero_grad()

            target = item.view(batch_size, 1, n_samples).to(device)
            orig_audio(target)
            recon, encoded, scheduling = model.iterative(target)
            recon_summed = torch.sum(recon, dim=1, keepdim=True)
            recon_audio(max_norm(recon_summed))

            loss = iterative_loss(target, recon, loss_transform)

            loss = loss + (torch.abs(encoded).sum() * 1e-3)

            mask = torch.zeros(target.shape[0], n_events, 1, device=recon.device).bernoulli_(p=0.5)
            for_disc = torch.sum(recon * mask, dim=1, keepdim=True)
            j = disc.forward(for_disc)
            d_loss = torch.abs(1 - j).mean()
            print('G', d_loss.item())
            loss = loss + d_loss

            loss.backward()
            optim.step()
            print(i, loss.item())

            disc_optim.zero_grad()
            r_j = disc.forward(target)
            f_j = disc.forward(recon_summed.clone().detach())
            d_loss = torch.abs(0 - f_j).mean() + torch.abs(1 - r_j).mean()
            d_loss.backward()
            print('D', d_loss.item())
            disc_optim.step()


            with torch.no_grad():
                # TODO: this should be collecting statistics from reconstructions
                # so that random reconstructions are within the expected distribution
                rnd = model.random_sequence()
                rnd = torch.sum(rnd, dim=1, keepdim=True)
                rnd = max_norm(rnd)
                random_audio(rnd)

    train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--overfit',
        required=False,
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['lookup', 'conv', 'splat', 'ssm'])
    parser.add_argument(
        '--disc-type',
        type=str,
        default='dilated',
        choices=['dilated', 'unet']
    )
    parser.add_argument(
        '--save-data',
        required=False,
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        required=False
    )
    args = parser.parse_args()
    train_and_monitor(
        batch_size=1 if args.overfit else args.batch_size,
        overfit=args.overfit,
        model_type=args.model_type,
        disc_type=args.disc_type,
        wipe_old_data=not args.save_data)
