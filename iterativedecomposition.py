from argparse import ArgumentParser
from typing import Union, Tuple

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, loggers, SupportedContentType, NumpySerializer, NumpyDeserializer
from data import AudioIterator
from modules import stft, sparsify, sparsify_vectors, iterative_loss, max_norm, \
    sparse_softmax, positional_encoding, LinearOutputStack, pos_encoded
from modules.anticausal import AntiCausalAnalysis
from modules.eventgenerators.generator import EventGenerator, ShapeSpec
from modules.eventgenerators.overfitresonance import OverfitResonanceModel, Lookup
from modules.eventgenerators.schedule import DiracScheduler
from modules.eventgenerators.splat import SplattingEventGenerator
from modules.iterative import sort_channels_descending_norm
from modules.mixer import MixerStack
from modules.multiheadtransform import MultiHeadTransform
from modules.overlap_add import overlap_add
from ssm import control_plane_dim
from util import device, encode_audio, make_initializer

matplotlib.use('Qt5Agg')

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

initializer = make_initializer(0.02)

class RNNEventGenerator(nn.Module):
    """
    A state-space model-like module, with one additional matrix, used to project the control
    signal into the shape of each audio frame.

    The final output is produced by overlap-adding the windows/frames of audio into a single
    1D signal.
    """

    def __init__(
            self,
            control_plane_dim: int,
            input_dim: int,
            state_matrix_dim: int,
            state_dim: int,
            window_size: int):

        super().__init__()
        self.state_matrix_dim = state_matrix_dim
        self.input_dim = input_dim
        self.control_plane_dim = control_plane_dim

        self.net = nn.RNN(
            input_size=input_dim,
            hidden_size=state_matrix_dim,
            num_layers=1,
            nonlinearity='tanh',
            bias=False,
            batch_first=True)

        # matrix mapping control signal to audio frame dimension
        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        )

        self.out_proj = nn.Linear(state_dim, window_size, bias=False)

        self.apply(initializer)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        """
        (batch, control_plane, time) -> (batch, window_size, time)
        """

        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)

        # try to ensure that the input signal only includes low-frequency info
        proj = control @ self.proj

        # proj = F.interpolate(proj, size=self.input_dim, mode='linear')
        # proj = proj * torch.zeros_like(proj).uniform_(-1, 1)
        # proj = proj * torch.hann_window(self.input_dim, device=proj.device)

        assert proj.shape == (batch, frames, self.input_dim)
        result, hidden = self.net.forward(proj)

        result = self.out_proj(result)

        result = result.view(batch, 1, -1)
        result = torch.sin(result)
        return result



def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """
    Time shift a by shift in the frequency domain
    """

    # this is here to make the shift value interpretable
    shift = (1 - shift)

    n_samples = a.shape[-1]

    shift_samples = (shift * n_samples * 0.5)

    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs

    shift = torch.exp(shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    return samples




def transform(x: torch.Tensor):
    batch_size, n_events = x.shape[:2]

    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, n_events, -1, n_coeffs)
    x = x.permute(0, 1, 3, 2).view(batch_size, n_coeffs, -1)
    return x


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    batch, n_events, time = x.shape
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    x = x.view(batch, n_events, -1)
    return x


def reconstruction_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = stft(a, 2048, 256, pad=True)
    b = stft(b, 2048, 256, pad=True)
    return torch.abs(a - b).sum()




class RNNEventGeneratorWrapper(EventGenerator, nn.Module):

    @property
    def shape_spec(self) -> ShapeSpec:
        return dict(
            params=(context_dim,)
        )

    def __init__(self):
        super().__init__()

        # TODO: This needs works and is very confused, state_dim and state_matrix_dim
        # are "tied" (must be the same) without being explicit about that, or just collapsing
        # them into a single parameter.
        n_items = 256
        state_matrix_dim = 128
        # the size of each, half-lapped audio "frame"
        window_size = 128
        # the dimensionality of the control plane or control signal
        self.control_plane_dim = 128
        # the dimensionality of the state vector, or hidden state for the RNN
        state_dim = state_matrix_dim
        input_dim = control_plane_dim
        self.n_frames = n_samples // input_dim
        self.proj = nn.Linear(context_dim, n_items)

        self.net = RNNEventGenerator(
            control_plane_dim=self.control_plane_dim,
            input_dim=input_dim,
            state_matrix_dim=state_matrix_dim,
            state_dim=state_dim,
            window_size=window_size)

        self.control_plane_lookup = Lookup(
            n_items=n_items, n_samples=control_plane_dim * self.n_frames, selection_type='relu')

        self.scheduler = DiracScheduler(
            n_events, start_size=n_frames, n_samples=n_samples, pre_sparse=True)

    def forward(self, params: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        batch_size = params.shape[0]
        x = params
        x = self.proj(x)
        x = self.control_plane_lookup.forward(x).view(batch_size, self.control_plane_dim, -1)
        x = self.net(x)
        x = self.scheduler.schedule(times, x)

        return x


class MixerEncoder(nn.Module):
    """
    MLP-Mixer-like architecture for the encoder/analysis portion of the network
    """
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

        # TODO: Try out different encoder architectures here, including:
        # - transformer
        # - MLP Mixer
        # - UNet
        # - RNN ?
        self.encoder = AntiCausalAnalysis(
            in_channels=in_channels,
            channels=hidden_channels,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 64, 1],
            pos_encodings=False,
            do_norm=False,
            with_activation_norm=with_activation_norm)


        self.to_event_vectors = nn.Conv1d(
            hidden_channels, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(
            hidden_channels, 1, 1, 1, 0)
        self.resonance = resonance_model

        self.context_dim = context_dim

        self.multihead = MultiHeadTransform(
            latent_dim=context_dim,
            hidden_channels=hidden_channels,
            n_layers=2,
            shapes=self.resonance.shape_spec,
        )

        # hang on to recent event vectors and use reservoir sampling
        # to grab them at random
        self.reservoir_size = 256
        self.reservoir = np.zeros((self.reservoir_size, self.context_dim), dtype=np.float32)
        self.apply(initializer)

    def embed_events(self, vectors: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Create an embedding of the concatenation of the event and one-hot time vectors
        """
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
            events, intermediates = self.resonance.forward_with_intermediate_steps(**choices_with_scheduling)
            return events, intermediates
        else:
            events = self.resonance.forward(**choices_with_scheduling)
            return events

    def random_sequence(
            self,
            device=device,
            batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # how many total events will we need for this batch size?
        total_needed = batch_size * n_events

        # select total_needed events from the reservoir
        indices = np.random.permutation(self.reservoir_size)[:total_needed]
        vecs = self.reservoir[indices]

        vecs = torch.from_numpy(vecs).to(device).float()
        vecs = vecs.view(batch_size, n_events, self.context_dim)

        # generate random times for the events
        raw_times = torch.zeros(batch_size, n_events, n_frames, device=device).normal_(-1, 1)
        raw_times[:, :, n_frames // 2:] = 0
        times = sparse_softmax(
            raw_times, normalize=True, dim=-1)

        # assign amplitudes to the dirac impulses and "sparsify", making
        # approximately 50% of the events silent
        times = \
            times * \
            torch.zeros_like(times).uniform_(0, 1) \
            * torch.zeros_like(times).bernoulli_(p=0.5)

        # generate the events
        final = torch.cat(
            [self.generate(vecs[:, i:i + 1, :], times[:, i:i + 1, :]) for i in range(n_events)], dim=1)

        return final, vecs, times

    def streaming(self, audio: torch.Tensor, return_event_vectors: bool = False):
        samps = audio.shape[-1]

        window_size = n_samples

        spec = transform(audio)
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

            # KLUDGE: this step size should be derived
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
            current = transform(ch)
            spec = (spec - current).clone().detach()

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

    envelopes, latents, reservoir, dist = loggers(
        ['envelopes', 'latents', 'reservoir', 'dist'],
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
        dist
    ], port=9999, n_workers=1)

    print('==========================================')
    print(
        f'training on {n_seconds} of audio and {n_events} events with {model_type} event generator')
    print('==========================================')

    model_filename = 'iterativedecomposition19.dat'

    def train():

        scaler = torch.cuda.amp.GradScaler()

        hidden_channels = 512

        # decide which event decoder variant we should build
        if model_type == 'lookup':
            resonance_model = OverfitResonanceModel(
                n_noise_filters=64,
                noise_expressivity=2,
                noise_filter_samples=128,
                noise_deformations=32,
                instr_expressivity=4,
                n_events=1,
                n_resonances=4096,
                n_envelopes=256,
                n_decays=64,
                n_deformations=256,
                n_samples=n_samples,
                n_frames=n_frames,
                samplerate=samplerate,
                hidden_channels=hidden_channels,
                wavetable_device=device,
                fine_positioning=fine_positioning,
                fft_resonance=True,
                context_dim=context_dim
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
        elif model_type == 'rnn':
            resonance_model = RNNEventGeneratorWrapper()

        else:
            raise ValueError(f'Unknown model type {model_type}')

        model = Model(
            resonance_model=resonance_model,
            in_channels=1025,
            hidden_channels=hidden_channels,
            with_activation_norm=True).to(device)


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

                # sparsity_loss = torch.abs(scheduling).sum() * 10

                norms = torch.norm(recon, dim=-1)
                norms, indices = torch.sort(norms, dim=-1)
                norms = max_norm(norms)
                dist(norms)

                recon_summed = torch.sum(recon, dim=1, keepdim=True)
                recon_audio(max_norm(recon_summed))

                envelopes(max_norm(scheduling[0]))
                latents(max_norm(encoded[0]).float())


                weighting = torch.ones_like(target)
                weighting[..., n_samples // 2:] = torch.linspace(1, 0, n_samples // 2, device=weighting.device) ** 8

                target = target * weighting
                recon_summed = recon_summed * weighting

                # loss from iterative_loss will be negative since we're maximizing
                # the amount of energy removed
                loss = iterative_loss(
                    target,
                    recon,
                    loss_transform,
                    ratio_loss=False,
                    sort_channels=True)
                # loss = loss + sparsity_loss
                # loss = loss + reconstruction_loss(target, recon_summed)


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

            if i > 100:
                # self-supervised loss kicks in
                ss_target = sort_channels_descending_norm(random_seq)
                ss_target = loss_transform(ss_target)

                ss, _, _ = model.iterative(rnd)
                ss_recon = sort_channels_descending_norm(ss)
                ss_recon = loss_transform(ss_recon)

                ss_loss = torch.abs(ss_target - ss_recon).sum() * 1e-3
                ss_loss.backward()
                optim.step()
                print('SS', i, ss_loss.item())


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
        choices=['lookup', 'splat', 'rnn'])
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
