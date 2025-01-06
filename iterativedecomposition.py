from argparse import ArgumentParser
from typing import Union, Dict

import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, loggers, SupportedContentType, NumpySerializer, NumpyDeserializer
from data import AudioIterator, get_one_audio_segment
from modules import stft, sparsify, sparsify_vectors, iterative_loss, max_norm, flattened_multiband_spectrogram, \
    DownsamplingDiscriminator, sparse_softmax, fft_frequency_decompose, positional_encoding
from modules.anticausal import AntiCausalAnalysis
from modules.eventgenerators.convimpulse import ConvImpulseEventGenerator
from modules.eventgenerators.generator import EventGenerator
from modules.eventgenerators.overfitresonance import OverfitResonanceModel, FFTResonanceLookup, MultiSSM
from modules.eventgenerators.splat import SplattingEventGenerator
from modules.eventgenerators.ssm import StateSpaceModelEventGenerator
from modules.infoloss import CorrelationLoss
from modules.multiheadtransform import MultiHeadTransform
from util import device, encode_audio, make_initializer
from torch.nn import functional as F
import numpy as np

# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 17
samples_per_event = 2048

# this is cut in half since we'll mask out the second half of encoder activations
n_events = (n_samples // samples_per_event) // 2
context_dim = 32

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256

n_frames = n_samples // transform_step_size

initializer = make_initializer(0.1)


def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # this is here to make the shift value interpretable
    shift = (1 - shift)

    n_samples = a.shape[-1]

    shift_samples = (shift * n_samples * 0.5)

    # a = F.pad(a, (0, n_samples * 2))

    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs

    shift = torch.exp(shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    # samples = torch.relu(samples)
    return samples


def transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, -1, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
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


def all_at_once_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    t = transform(target)
    r = transform(recon)
    return torch.abs(t - r).sum()


# def loss_transform(x: torch.Tensor) -> torch.Tensor:
#     batch, channels, time = x.shape
#     spec = stft(x, 2048, 256, pad=True)
#     spec = spec.view(batch, channels, -1)
#     return spec

# def multiband_spectrogram(samples: torch.Tensor, min_size=512) -> Dict[int, torch.Tensor]:
#     batch = samples.shape[0]
#
#     bands = fft_frequency_decompose(samples, min_size=min_size)
#
#     specs = {}
#
#     for size, band in bands.items():
#         spec = stft(band, ws=64, step=16, pad=True)
#         specs[size] = spec
#
#     return specs
#
#
# def iterative_loss2(
#         target: Dict[int, torch.Tensor],
#         recon: Dict[int, torch.Tensor],
#         n_events: int) -> torch.Tensor:
#     residual = {k: t.clone() for k, t in target.items()}
#
#     loss = 0
#
#     for i in range(n_events):
#         for size, band in residual.items():
#             r = recon[size][:, i: i + 1, :]
#
#             t_norm = torch.norm(band, dim=-1, keepdim=True)
#
#             res = band - r
#
#             new_norm = torch.norm(res, dim=-1, keepdim=True)
#             residual[size] = res
#
#             loss = loss + (new_norm / t_norm).sum()
#
#     return loss


# def full_iterative_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
#     spec = multiband_spectrogram(target)
#     rspec = multiband_spectrogram(recon)
#     return iterative_loss2(spec, rspec, n_events)


class Discriminator(nn.Module):
    def __init__(self, disc_type='dilated'):
        super().__init__()
        if disc_type == 'dilated':
            self.disc = DownsamplingDiscriminator(
                window_size=2048, step_size=256, n_samples=n_samples // 2, channels=128)
        elif disc_type == 'unet':
            self.disc = DownsamplingDiscriminator(
                2048, 256, n_samples=n_samples // 2, channels=128)
        # elif disc_type == 'multiband':
        #     self.disc = MultibandDownsamplingDiscriminator(n_frames=n_frames, in_channels=1160, channels=256)
        else:
            raise ValueError(f'Unknown discriminator type: {disc_type}')

        self.apply(initializer)

    def forward(self, transformed: torch.Tensor):
        t = transformed.shape[-1]
        x = self.disc(transformed[..., :t // 2])
        return x


class Model(nn.Module):
    def __init__(
            self,
            resonance_model: Union[EventGenerator, nn.Module],
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
            do_norm=False)

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

    def embed_events(self, vectors: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        pe = positional_encoding(sequence_length=n_frames, n_freqs=context_dim, device=vectors.device)
        times = times @ pe.T
        embeddings = torch.cat([vectors, times], dim=-1)
        return embeddings

    def event_similarity(self, vectors: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed_events(vectors, times)
        self_sim = embeddings[:, :, None, :] - embeddings[:, :, None, :, None]
        return self_sim

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

        frame_count = attn.shape[-1]
        half_frames = frame_count // 2

        # we mask out the second half, so we're always choosing from the first
        # half for the next event
        mask = torch.ones_like(attn)
        mask[:, :, half_frames:] = 0
        attn = attn * mask

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

    def random_sequence(self, device=device) -> torch.Tensor:
        vecs = torch.zeros(1, 1, context_dim, device=device).uniform_(-1, 1)
        times = sparse_softmax(
            torch.zeros(1, 1, n_frames, device=device).uniform_(-1, 1), normalize=True, dim=-1)
        final = self.generate(vecs, times)
        return final

    def streaming(self, audio: torch.Tensor, return_event_vectors: bool = False):
        samps = audio.shape[-1]

        window_size = n_samples
        step_size = n_samples // 2

        print('========================')
        spec = transform(audio)
        print(spec.shape)
        batch, channels, time = spec.shape

        frame_window_size = n_frames
        frame_step_size = n_frames // 2

        segments = torch.zeros(1, n_events, samps, device=audio.device, requires_grad=False)

        all_event_vectors = []
        all_times = []
        all_events = []

        for i in range(0, time - frame_window_size, frame_step_size):
            print(f'streaming chunk {i}')

            channels, vecs, schedules, residual_spec = self.iterative(
                spec[:, :, i: i + frame_window_size], do_transform=False, return_residual=True)

            all_events.append(channels)
            all_event_vectors.append(vecs)
            all_times.append(schedules)

            spec[:, :, i: i + frame_window_size] = residual_spec

            # KLUDGE: this step should be derived
            start_sample = i * 256
            end_sample = start_sample + window_size
            segments[:, :, start_sample: end_sample] += channels

        final = torch.sum(segments, dim=1, keepdim=True)

        if not return_event_vectors:
            return final[..., :samps]
        else:
            x = final[..., :samps]
            vecs = torch.cat(all_event_vectors, dim=1)
            times = torch.cat(all_times, dim=1)
            events = torch.cat(all_events, dim=1)

            return x, vecs, times, events

    def iterative(
            self,
            audio: torch.Tensor,
            do_transform: bool = True,
            return_residual: bool = False,
            return_all_residuals: bool = False):

        channels = []
        schedules = []
        vecs = []
        residuals = []

        if do_transform:
            spec = transform(audio)
        else:
            spec = audio

        print(f'iterative {spec.shape}')

        for i in range(n_events):


            # normalize current spectrogram
            # mx = torch.amax(spec, dim=(1, 2), keepdim=True)
            # spec = spec / (mx + 1e-8)


            v, sched = self.encode(spec)
            vecs.append(v)
            schedules.append(sched)
            ch = self.generate(v, sched)
            current = transform(ch)

            # return spectrogram to original scale
            # spec = spec * mx

            # scale proposed "atom" by the same amount
            # current = current * mx

            spec = (spec - current).clone().detach()


            channels.append(ch)
            residuals.append(spec[:, None, :, :].clone().detach())


        channels = torch.cat(channels, dim=1)
        vecs = torch.cat(vecs, dim=1)
        schedules = torch.cat(schedules, dim=1)
        residuals = torch.cat(residuals, dim=1)

        if return_all_residuals:
            return channels, vecs, schedules, residuals
        elif return_residual:
            return channels, vecs, schedules, spec
        else:
            return channels, vecs, schedules

    def forward(self, audio: torch.Tensor):
        raise NotImplementedError()


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def train_and_monitor(
        batch_size: int = 8,
        overfit: bool = False,
        disc_type: str = 'dilated',
        model_type: str = 'conv',
        wipe_old_data: bool = True,
        fine_positioning: bool = False,
        save_and_load_weights: bool = False,
        adversarial_loss: bool = True):
    torch.backends.cudnn.benchmark = True

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

    recon_audio, orig_audio, random_audio, streaming = loggers(
        ['recon', 'orig', 'random', 'streaming'],
        'audio/wav',
        encode_audio,
        collection)

    envelopes, latents = loggers(
        ['envelopes', 'latents'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer())

    serve_conjure([
        orig_audio,
        recon_audio,
        random_audio,
        envelopes,
        latents,
        streaming
    ], port=9999, n_workers=1)

    print('==========================================')
    print(
        f'training on {n_seconds} of audio and {n_events} events with {model_type} event generator and {disc_type} disc')
    print('==========================================')

    model_filename = 'iterativedecomposition9.dat'
    disc_filename = 'iterativedecompositiondisc9.dat'

    def train():

        scaler = torch.cuda.amp.GradScaler()

        hidden_channels = 512
        if model_type == 'lookup':
            resonance_model = OverfitResonanceModel(
                n_noise_filters=64,
                noise_expressivity=4,
                noise_filter_samples=128,
                noise_deformations=32,
                instr_expressivity=4,
                n_events=1,
                n_resonances=4096,
                n_envelopes=64,
                n_decays=64,
                n_deformations=64,
                n_samples=n_samples,
                n_frames=n_frames,
                samplerate=samplerate,
                hidden_channels=hidden_channels,
                wavetable_device=device,
                fine_positioning=fine_positioning,
                fft_resonance=True
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
                n_frames=n_frames,
                wavetable_resonance=True,
                hierarchical_scheduler=False
            )
        elif model_type == 'ssm':
            # resonance_model = StateSpaceModelEventGenerator(
            #     context_dim=context_dim,
            #     control_plane_dim=context_dim,
            #     input_dim=1024,
            #     state_dim=128,
            #     hypernetwork_dim=32,
            #     hypernetwork_latent=context_dim,
            #     samplerate=samplerate,
            #     n_samples=n_samples,
            #     n_frames=n_frames,
            # )
            window_size = 1024
            step_size = window_size // 2
            resonance_model = MultiSSM(
                context_dim=context_dim,
                control_plane_dim=64,
                n_frames=n_samples // step_size,
                state_dim=128,
                window_size=window_size,
                n_models=8,
                n_control_planes=512,
                n_samples=n_samples
            )
        else:
            raise ValueError(f'Unknown model type {model_type}')

        print(resonance_model.shape_spec)

        model = Model(
            resonance_model=resonance_model,
            in_channels=1024,
            hidden_channels=hidden_channels).to(device)

        disc = Discriminator(disc_type=disc_type).to(device)

        if save_and_load_weights:
            # KLUDGE: Unless the same command line arguments are used, this will
            # require manual intervention to delete old weights, e.g., if a different
            # event generator is used
            try:
                model.load_state_dict(torch.load(model_filename))
                print('loaded model weights')
            except IOError:
                print('No model weights to load')

            try:
                disc.load_state_dict(torch.load(disc_filename))
                print('loaded discriminator weights')
            except IOError:
                print('no discriminator weights to load')

        optim = Adam(model.parameters(), lr=1e-4)
        disc_optim = Adam(disc.parameters(), lr=1e-3)

        loss_model = CorrelationLoss(n_elements=256).to(device)

        for i, item in enumerate(iter(stream)):
            optim.zero_grad()
            disc_optim.zero_grad()

            with torch.cuda.amp.autocast():
                target = item.view(batch_size, 1, n_samples).to(device)
                orig_audio(target)
                recon, encoded, scheduling = model.iterative(target)
                print(encoded.shape)

                recon_summed = torch.sum(recon, dim=1, keepdim=True)
                recon_audio(max_norm(recon_summed))

                envelopes(max_norm(scheduling[0]))
                latents(max_norm(encoded[0]).float())

                # print(target.shape, recon.shape)

                weighting = torch.ones_like(target)
                weighting[..., n_samples // 2:] = torch.linspace(1, 0, n_samples // 2, device=weighting.device) ** 8

                target = target * weighting
                recon_summed = recon_summed * weighting

                # loss = all_at_once_loss(target, recon_summed)
                loss = iterative_loss(target, recon, loss_transform)
                # loss = loss + loss_model.noise_loss(target, recon_summed)
                # loss = loss + loss_model.multiband_noise_loss(target, recon_summed, 128, 32)

                loss = loss + (torch.abs(encoded).sum() * 1e-4)

                if adversarial_loss:
                    # mask half of the events, at random. Each event should be realistic
                    # and stand on its own
                    mask = torch.zeros(target.shape[0], n_events, 1, device=recon.device).bernoulli_(p=0.5)
                    for_disc = torch.sum(recon * mask, dim=1, keepdim=True)
                    j = disc.forward(for_disc)
                    d_loss = torch.abs(1 - j).mean()
                    print('G', d_loss.item())
                    loss = loss + (d_loss * 1000)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            print(i, loss.item())

            if adversarial_loss:
                with torch.cuda.amp.autocast():
                    disc_optim.zero_grad()
                    r_j = disc.forward(target)
                    f_j = disc.forward(recon_summed.clone().detach())
                    d_loss = torch.abs(0 - f_j).mean() + torch.abs(1 - r_j).mean()

                scaler.scale(d_loss).backward()
                # d_loss.backward()
                print('D', d_loss.item())
                # disc_optim.step()
                scaler.step(disc_optim)
                scaler.update()

            with torch.no_grad():
                # TODO: this should be collecting statistics from reconstructions
                # so that random reconstructions are within the expected distribution
                rnd = model.random_sequence()
                rnd = torch.sum(rnd, dim=1, keepdim=True)
                rnd = max_norm(rnd)
                random_audio(rnd)

            if i % 50 == 0:
                with torch.no_grad():
                    s = get_one_audio_segment(n_samples * 4, device=device)
                    s = s.view(1, 1, -1)
                    s = model.streaming(s)
                    print(s.shape)
                    streaming(max_norm(s))

            if save_and_load_weights and i % 100 == 0:
                torch.save(model.state_dict(), model_filename)
                torch.save(disc.state_dict(), disc_filename)

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
        choices=['dilated', 'unet', 'multiband']
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
    parser.add_argument(
        '--fine-positioning',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--save-and-load-weights',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--no-adversarial-loss',
        action='store_true',
        default=False
    )

    args = parser.parse_args()
    train_and_monitor(
        batch_size=1 if args.overfit else args.batch_size,
        overfit=args.overfit,
        model_type=args.model_type,
        disc_type=args.disc_type,
        wipe_old_data=not args.save_data,
        adversarial_loss=not args.no_adversarial_loss,
        fine_positioning=bool(args.fine_positioning),
        save_and_load_weights=args.save_and_load_weights
    )
