from argparse import ArgumentParser
from typing import Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from config import Config
from conjure import LmdbCollection, serve_conjure, loggers, SupportedContentType, NumpySerializer, NumpyDeserializer
from data import AudioIterator, get_one_audio_segment
from data.audioiter import get_one_audio_batch
from modules import stft, sparsify, sparsify_vectors, iterative_loss, max_norm, flattened_multiband_spectrogram, \
    sparse_softmax, positional_encoding, NeuralReverb, fft_frequency_decompose, CanonicalOrdering
from modules.anticausal import AntiCausalAnalysis
from modules.eventgenerators.convimpulse import ConvImpulseEventGenerator
from modules.eventgenerators.generator import EventGenerator, ShapeSpec
from modules.eventgenerators.overfitresonance import OverfitResonanceModel, Lookup
from modules.eventgenerators.schedule import DiracScheduler
from modules.eventgenerators.splat import SplattingEventGenerator
from modules.infoloss import CorrelationLoss
from modules.iterative import sort_channels_descending_norm
from modules.mixer import MixerStack
from modules.multiheadtransform import MultiHeadTransform
from modules.transfer import fft_convolve
from util import device, encode_audio, make_initializer
from torch.nn import functional as F

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

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

# log_amplitude = True

n_frames = n_samples // transform_step_size

initializer = make_initializer(0.02)


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


def run_layer(
        control_plane: torch.Tensor,
        mapping: torch.Tensor,
        decays: torch.Tensor,
        out_mapping: torch.Tensor,
        audio_mapping: torch.Tensor,
        gains: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size, control_plane_dim, frames = control_plane.shape


    x = mapping @ control_plane
    orig = x
    decays = decays.view(batch_size, control_plane_dim, 1).repeat(1, 1, frames)

    decays = torch.log(1e-12 + decays)
    decays = torch.cumsum(decays, dim=-1)
    decays = torch.exp(decays)

    # decays = decays.cumprod(dim=-1)
    # decays = torch.flip(decays, dims=[-1])
    x = fft_convolve(x, decays)
    x = (out_mapping @ x) + orig

    cp = torch.tanh(x * gains.view(batch_size, control_plane_dim, 1))

    audio = audio_mapping @ cp

    # TODO: This should be mapped to audio outside of this layer, probably
    # each layer by a single mapping network
    audio = audio.permute(0, 2, 1)

    audio = audio.reshape(batch_size, 1, -1)

    return audio, cp


class Block(nn.Module):
    def __init__(
            self,
            block_size,
            base_resonance: float = 0.5,
            max_gain: float = 5,
            window_size: Union[int, None] = None):

        super().__init__()
        self.block_size = block_size
        self.base_resonance = base_resonance
        self.resonance_span = 1 - base_resonance
        self.max_gain = max_gain
        self.window_size = window_size or block_size

        self.w1 = nn.Parameter(torch.zeros(block_size, block_size).uniform_(-0.01, 0.01))
        self.w2 = nn.Parameter(torch.zeros(block_size, block_size).uniform_(-0.01, 0.01))
        self.audio = nn.Parameter(torch.zeros(window_size, block_size).uniform_(-0.01, 0.01))


        self.decays = nn.Parameter(torch.zeros(block_size).uniform_(0.001, 0.99))
        self.gains = nn.Parameter(torch.zeros(block_size).uniform_(-3, 3))

    def forward(self, cp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, cp = run_layer(
            torch.relu(cp),
            self.w1,
            self.base_resonance + torch.sigmoid(self.decays) * self.resonance_span,
            self.w2,
            self.audio,
            torch.sigmoid(self.gains) * self.max_gain)
        return output, cp


class Stack(nn.Module):
    def __init__(
            self,
            n_blocks,
            block_size,
            base_resonance: float = 0.5,
            max_gain: float = 5,
            window_size: Union[int, None] = None):
        super().__init__()
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.window_size = window_size

        self.mix = nn.Parameter(torch.zeros(n_blocks).uniform_(-1, 1))
        self.blocks = nn.ModuleList([
            Block(
                block_size,
                base_resonance,
                max_gain,
                window_size = window_size
            ) for _ in range(n_blocks)
        ])

    def forward(self, cp):
        batch_size, channels, frames = cp.shape

        working_control_plane = cp

        total_samples = frames * self.window_size

        channels = torch.zeros(
            batch_size, self.n_blocks, total_samples, device=cp.device)

        for i, block in enumerate(self.blocks):
            output, working_control_plane = block(working_control_plane)
            channels[:, i: i + 1, :] = output

        mix = torch.softmax(self.mix, dim=-1)
        mixed = channels.permute(0, 2, 1) @ mix
        mixed = mixed.view(batch_size, 1, total_samples)
        return max_norm(mixed)


# stats_batch = get_one_audio_batch(32, n_samples, samplerate, device)
# spec = stft(stats_batch, 2048, 256, pad=True)
# stds = torch.std(spec, dim=-2, keepdim=True)



def transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    # x = x / (stds + 1e-8)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, -1, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    batch, n_events, time = x.shape
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    # x = x / (stds + 1e-8)
    x = x.view(batch, n_events, -1)
    return x



def all_at_once_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    t = loss_transform(target)
    r = loss_transform(recon)
    loss = torch.abs(t - r).sum()
    return loss

class PhysicalModel(nn.Module):

    def __init__(
            self,
            control_plane_dim: int = 16,
            hidden_dim: int = 128,
            max_active: int = 64,
            window_size: int = 1024):
        super().__init__()
        self.control_plane_dim = control_plane_dim
        self.hidden_dim = hidden_dim
        self.max_active = max_active
        self.window_size = window_size

        self.net = Stack(
            3, self.control_plane_dim, base_resonance=0.5, max_gain=5, window_size=self.window_size)


    def forward(self, control: torch.Tensor) -> torch.Tensor:
        batch, cpd, frames = control.shape
        # control = control.permute(0, 2, 1)

        control = sparsify(control / (control.sum() + 1e-8), self.max_active)
        # control = torch.relu(control)

        # control = control[:, :, :frames // 4]

        # proj = self.in_projection(control)
        result = self.net.forward(control)
        # result = self.out_projection(result)

        samples = result.view(batch, -1, n_samples)
        # samples = torch.sin(samples)
        return samples


class MultiRNN(EventGenerator, nn.Module):

    def __init__(
            self,
            n_voices: int = 8,
            n_control_planes: int = 512,
            control_plane_dim: int = 16,
            hidden_dim: int = 128,
            n_frames: int = 128,
            control_plane_sparsity: int = 32,
            window_size: int = 1024,
            n_events: int = 1,
            n_samples: int = n_samples,
            fine_positioning: bool = False):
        super().__init__()
        self.n_voices = n_voices
        self.n_control_planes = n_control_planes
        self.control_plane_dim = control_plane_dim
        self.n_frames = n_frames
        self.fine_positioning = fine_positioning
        self.n_events = n_events
        self.n_samples = n_samples

        self.samples_per_frame = n_samples // n_frames
        self.frame_ratio = self.samples_per_frame / n_samples

        verbs = NeuralReverb.tensors_from_directory(Config.impulse_response_path(), n_samples)
        n_verbs = verbs.shape[0]
        self.n_verbs = n_verbs

        self.voices = nn.ModuleList([
            PhysicalModel(
                control_plane_dim=control_plane_dim,
                hidden_dim=hidden_dim,
                max_active=control_plane_sparsity,
                window_size=window_size, )
            for _ in range(self.n_voices)
        ])

        self.room_shape = (1, n_events, n_verbs)

        self.control_planes = Lookup(
            n_control_planes,
            n_frames * control_plane_dim,
            initialize=lambda x: torch.zeros_like(x).uniform_(0, 1))

        self.verb = Lookup(n_verbs, n_samples, initialize=lambda x: verbs, fixed=True)

        self.scheduler = DiracScheduler(
            self.n_events, start_size=n_frames, n_samples=self.n_samples, pre_sparse=True)

    @property
    def shape_spec(self) -> ShapeSpec:
        params = dict(
            voice=(self.n_voices,),
            control_plane_choice=(self.n_control_planes,),
            amplitudes=(1,),
            room_choice=(self.n_verbs,),
            room_mix=(2,),
        )

        if self.fine_positioning:
            params['fine'] = (1,)

        return params

    def forward(
            self,
            voice: torch.Tensor,
            control_plane_choice: torch.Tensor,
            amplitudes: torch.Tensor,
            room_choice: torch.Tensor,
            room_mix: torch.Tensor,
            times: torch.Tensor,
            fine: Union[torch.Tensor, None] = None) -> torch.Tensor:

        batch, n_events, _ = voice.shape

        final_events = torch.zeros((batch, n_events, self.n_samples), device=voice.device)

        hard_voice_choice = sparse_softmax(voice, normalize=True)
        voice_indices = torch.argmax(hard_voice_choice, dim=-1, keepdim=True)

        for b in range(batch):
            for e in range(n_events):
                active_voice = voice_indices[b, e]

                cp = self\
                    .control_planes.forward(control_plane_choice[b, e].view(1, 1, self.n_control_planes))\
                    .view(1, self.control_plane_dim, self.n_frames)

                samples = self.voices[active_voice.item()].forward(cp)

                # as in switch transformer, multiply samples by winning (sparse) softmax element
                # so gradients can flow through to the "router"
                final = samples * hard_voice_choice[b, e, active_voice]

                # apply reverb
                verb = self.verb.forward(room_choice[b: b + 1, e: e + 1, :])
                wet = fft_convolve(verb, final.view(*verb.shape))
                verb_mix = torch.softmax(room_mix[b: b + 1, e: e + 1, :], dim=-1)[:, :, None, :]
                stacked = torch.stack([wet, final.view(*verb.shape)], dim=-1)
                stacked = stacked * verb_mix
                final = stacked.sum(dim=-1)

                final = final * torch.abs(amplitudes[b, e])

                scheduled = self.scheduler.schedule(times[b: b + 1, e: e + 1], final)

                if fine is not None:
                    fine_shifts = torch.tanh(fine) * self.frame_ratio
                    scheduled = fft_shift(scheduled, fine_shifts[b: b + 1, e: e + 1, :])
                    scheduled = scheduled[..., :self.n_samples]


                final_events[b: b + 1, e: e + 1, :] = scheduled

        return final_events



class MixerEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.encoder = MixerStack(
            in_channels=in_channels,
            channels=hidden,
            sequence_length=n_frames,
            layers=4,
            attn_blocks=2,
            channels_last=False)
        self.out = nn.Conv1d(hidden, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.out(x)
        return x

class Model(nn.Module):

    def __init__(
            self,
            resonance_model: Union[EventGenerator, nn.Module],
            in_channels: int = 1024,
            hidden_channels: int = 256,
            with_activation_norm: bool = False):

        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.encoder = AntiCausalAnalysis(
            in_channels=in_channels,
            channels=hidden_channels,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 64, 1],
            pos_encodings=False,
            do_norm=False,
            with_activation_norm=with_activation_norm)

        # self.encoder = MixerEncoder(in_channels=in_channels, hidden=hidden_channels, out_channels=hidden_channels)

        self.to_event_vectors = nn.Conv1d(hidden_channels, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(hidden_channels, 1, 1, 1, 0)
        self.resonance = resonance_model

        self.context_dim = context_dim

        self.multihead = MultiHeadTransform(
            latent_dim=context_dim,
            hidden_channels=hidden_channels,
            n_layers=2,
            shapes=self.resonance.shape_spec,
        )

        self.reservoir_size = 256
        self.reservoir = np.zeros((self.reservoir_size, self.context_dim), dtype=np.float32)
        # self.scheduling_mean: float = 1e-3
        # self.scheduling_std: float = 1e-3

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

    def generate(self, vecs: torch.Tensor, scheduling: torch.Tensor, include_intermediates: bool = False):
        choices = self.multihead.forward(vecs)
        choices_with_scheduling = dict(**choices, times=scheduling)

        if include_intermediates:
            events, intermediates = self.resonance.forward_with_intermediates(**choices_with_scheduling)
            return events, intermediates
        else:
            events = self.resonance.forward(**choices_with_scheduling)
            return events

    def random_sequence(
            self,
            device=device,
            batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        total_needed = batch_size * n_events
        indices = np.random.permutation(self.reservoir_size)[:total_needed]
        vecs = self.reservoir[indices]
        vecs = torch.from_numpy(vecs).to(device).float()
        vecs = vecs.view(batch_size, n_events, self.context_dim)


        raw_times = torch.zeros(batch_size, n_events, n_frames, device=device).normal_(-1, 1)
        raw_times[:, :, n_frames // 2:] = 0
        times = sparse_softmax(
            raw_times, normalize=True, dim=-1)

        times = \
            times * \
            torch.zeros_like(times).uniform_(0, 1) \
            * torch.zeros_like(times).bernoulli_(p=0.5)

        final = torch.cat(
            [self.generate(vecs[:, i:i + 1, :], times[:, i:i + 1, :]) for i in range(n_events)], dim=1)

        return final, vecs, times

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

        batch_size = audio.shape[0]

        channels = []
        schedules = []
        vecs = []
        residuals = []

        if do_transform:
            spec = transform(audio)
        else:
            spec = audio

        for i in range(n_events):

            v, sched = self.encode(spec)
            vecs.append(v)
            schedules.append(sched)
            ch = self.generate(v, sched)

            # perform transform and scale proportionally
            current = transform(ch)

            spec = (spec - current)#.clone().detach()

            channels.append(ch)
            residuals.append(spec[:, None, :, :].clone().detach())

        channels = torch.cat(channels, dim=1)
        vecs = torch.cat(vecs, dim=1)
        schedules = torch.cat(schedules, dim=1)
        residuals = torch.cat(residuals, dim=1)

        # put vectors into the reservoir to support random generations
        v = vecs.view(-1, self.context_dim)
        indices = np.random.permutation(self.reservoir_size)[:v.shape[0]]
        v = v.data.cpu().numpy()
        self.reservoir[indices, :] = v

        # record mean and std of scheduling impulses
        # active = schedules[schedules > 0] + 1e-8
        # self.scheduling_mean = active.mean().item()
        # self.scheduling_std = active.std().item()

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
        model_type: str = 'conv',
        wipe_old_data: bool = True,
        fine_positioning: bool = False,
        save_and_load_weights: bool = False):
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

    recon_audio, orig_audio, random_audio = loggers(
        ['recon', 'orig', 'random',],
        'audio/wav',
        encode_audio,
        collection)

    envelopes, latents, reservoir = loggers(
        ['envelopes', 'latents', 'reservoir'],
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
        reservoir,
    ], port=9999, n_workers=1)

    print('==========================================')
    print(
        f'training on {n_seconds} of audio and {n_events} events with {model_type} event generator')
    print('==========================================')

    model_filename = 'iterativedecomposition16.dat'

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

            window_size = 512
            rnn_frames = n_samples // window_size

            resonance_model = MultiRNN(
                n_voices=16,
                n_control_planes=512,
                control_plane_dim=64,
                hidden_dim=64,
                control_plane_sparsity=128,
                window_size=window_size,
                n_frames=rnn_frames,
                fine_positioning=False)
        else:
            raise ValueError(f'Unknown model type {model_type}')

        print(resonance_model.shape_spec)

        model = Model(
            resonance_model=resonance_model,
            in_channels=1024,
            hidden_channels=hidden_channels,
            with_activation_norm=True).to(device)

        # disc = Discriminator(disc_type=disc_type).to(device)

        if save_and_load_weights:
            # KLUDGE: Unless the same command line arguments are used, this will
            # require manual intervention to delete old weights, e.g., if a different
            # event generator is used
            try:
                model.load_state_dict(torch.load(model_filename))
                print('loaded model weights')
            except IOError:
                print('No model weights to load')


        optim = Adam(model.parameters(), lr=1e-4)


        for i, item in enumerate(iter(stream)):
            optim.zero_grad()

            with torch.cuda.amp.autocast():
                target = item.view(batch_size, 1, n_samples).to(device)
                orig_audio(target)
                recon, encoded, scheduling = model.iterative(target)

                recon_summed = torch.sum(recon, dim=1, keepdim=True)
                recon_audio(max_norm(recon_summed))

                envelopes(max_norm(scheduling[0]))
                latents(max_norm(encoded[0]).float())


                weighting = torch.ones_like(target)
                weighting[..., n_samples // 2:] = torch.linspace(1, 0, n_samples // 2, device=weighting.device) ** 8

                target = target * weighting
                recon_summed = recon_summed * weighting

                loss = iterative_loss(target, recon, loss_transform, ratio_loss=True)
                # loss = loss_model.forward(target, recon_summed)


            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            print(i, loss.item())

            optim.zero_grad()

            # TODO: sample scale/amplitude of one-hot time vectors
            # for completely realistic ground-truth generations.
            # Then, sort both by descending norm and compare the spectrograms
            # of each
            with torch.no_grad():
                self_supervised_batch_size = 1
                # Generate a random sequence using random event vectors from the recent past
                rnd, vecs, times = model.random_sequence(batch_size=self_supervised_batch_size)

                # normalize such that the loudest channel peaks at 1
                random_seq = rnd
                random_seq = random_seq.view(self_supervised_batch_size, -1)
                random_seq = max_norm(random_seq)
                random_seq = random_seq.view(self_supervised_batch_size, n_events, n_samples)

                rnd = torch.sum(random_seq, dim=1, keepdim=True)


                random_audio(max_norm(rnd))
                reservoir(max_norm(torch.from_numpy(model.reservoir)))
                # random_vecs(max_norm(vecs[0]))

            # encode the random sequence
            # recon, encoded, scheduling = model.iterative(rnd)
            #
            # self_supervised(max_norm(torch.sum(recon, dim=1, keepdim=True)))
            #
            # # sort the random sequence and the reconstruction by descending norms
            # random_seq = sort_channels_descending_norm(random_seq)
            # random_seq = loss_transform(random_seq)
            #
            # recon = sort_channels_descending_norm(recon)
            # recon = loss_transform(recon)
            #
            # print('SELF-SUPERVISED', random_seq.shape, recon.shape)
            #
            # # compare the difference between the spectrogram at each individual channel
            # loss = torch.abs(random_seq - recon).sum() * 1e-2


            # loss.backward()
            # optim.step()
            # print('SELF-SUPERVISED', loss.item())


            # if i % 1000 == 0:
            #     with torch.no_grad():
            #         s = get_one_audio_segment(n_samples * 4, device=device)
            #         s = s.view(1, 1, -1)
            #         s = model.streaming(s)
            #         print(s.shape)
            #         streaming(max_norm(s))

            if save_and_load_weights and i % 100 == 0:
                torch.save(model.state_dict(), model_filename)
                # torch.save(disc.state_dict(), disc_filename)

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

    args = parser.parse_args()
    train_and_monitor(
        batch_size=1 if args.overfit else args.batch_size,
        overfit=args.overfit,
        model_type=args.model_type,
        wipe_old_data=not args.save_data,
        fine_positioning=bool(args.fine_positioning),
        save_and_load_weights=args.save_and_load_weights
    )
