from argparse import ArgumentParser
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.optim import Adam

from config import Config
from conjure import LmdbCollection, serve_conjure, loggers, SupportedContentType, NumpySerializer, NumpyDeserializer
from data import AudioIterator, get_one_audio_segment
from modules import stft, sparsify, sparsify_vectors, iterative_loss, max_norm, flattened_multiband_spectrogram, \
    sparse_softmax, positional_encoding, NeuralReverb
from modules.anticausal import AntiCausalAnalysis
from modules.eventgenerators.convimpulse import ConvImpulseEventGenerator
from modules.eventgenerators.generator import EventGenerator, ShapeSpec
from modules.eventgenerators.overfitresonance import OverfitResonanceModel, Lookup
from modules.eventgenerators.schedule import DiracScheduler
from modules.eventgenerators.splat import SplattingEventGenerator
from modules.multiheadtransform import MultiHeadTransform
from modules.transfer import fft_convolve
from spiking import AutocorrelationLoss
from util import device, encode_audio, make_initializer

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


def transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)

    # if log_amplitude:
    #     x = torch.relu(torch.log(x + 1e-8) + 27)

    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, -1, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    batch, n_events, time = x.shape

    # return flattened_multiband_spectrogram(
    #     x,
    #     stft_spec={
    #         'long': (128, 64),
    #         'short': (64, 32),
    #         'xs': (16, 8),
    #     },
    #     smallest_band_size=512)

    x = stft(x, transform_window_size, transform_step_size, pad=True)
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

        self.net = nn.RNN(
            input_size=window_size,
            hidden_size=hidden_dim,
            num_layers=1,
            nonlinearity='tanh',
            bias=False,
            batch_first=True)

        self.in_projection = nn.Linear(control_plane_dim, window_size, bias=False)
        self.out_projection = nn.Linear(hidden_dim, window_size, bias=False)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        batch, cpd, frames = control.shape
        control = control.permute(0, 2, 1)

        control = sparsify(control, self.max_active)
        control = torch.relu(control)

        # control = control[:, :, :frames // 4]

        proj = self.in_projection(control)
        result, hidden = self.net.forward(proj)
        result = self.out_projection(result)

        samples = result.view(batch, -1, n_samples)
        samples = torch.sin(samples)
        return samples


class MultiRNN(EventGenerator, nn.Module):

    def __init__(
            self,
            n_voices: int = 8,
            n_control_planes: int = 512,
            control_plane_dim: int = 16,
            hidden_dim: int = 128,
            n_frames: int = 128,
            control_plane_sparsity: int = 128,
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
                window_size=window_size, ) for _ in range(self.n_voices)
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
            v, sched = self.encode(spec)
            vecs.append(v)
            schedules.append(sched)
            ch = self.generate(v, sched)
            current = transform(ch)

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
        f'training on {n_seconds} of audio and {n_events} events with {model_type} event generator')
    print('==========================================')

    model_filename = 'iterativedecomposition14.dat'

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

            window_size = 128
            rnn_frames = n_samples // window_size

            resonance_model = MultiRNN(
                n_voices=16,
                n_control_planes=512,
                control_plane_dim=32,
                hidden_dim=128,
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
            hidden_channels=hidden_channels).to(device)

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

        loss_model = AutocorrelationLoss(n_channels=64, filter_size=64).to(device)

        for i, item in enumerate(iter(stream)):
            optim.zero_grad()
            # disc_optim.zero_grad()

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


                # _ = loss_model.compute_multiband_loss(target, recon_summed)
                loss = iterative_loss(target, recon, loss_transform, ratio_loss=True)
                # loss = loss + (all_at_once_loss(target, recon_summed) * 1e-4)

                # loss = loss_model.compute_loss(target, recon_summed)
                # loss = loss_model.compute_multiband_loss(target, recon_summed)


            scaler.scale(loss).backward()

            if model_type == 'ssm':
                clip_grad_value_(model.parameters(), 0.5)

            scaler.step(optim)
            scaler.update()
            print(i, loss.item())


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
