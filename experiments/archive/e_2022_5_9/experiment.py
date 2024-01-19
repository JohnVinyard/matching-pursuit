
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules import stft
from modules.ddsp import NoiseModel, OscillatorBank
from util.readmedocs import readme

import numpy as np
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from sklearn.cluster import MiniBatchKMeans
from modules.phase import MelScale, AudioCodec
import zounds
import torch
from torch import nn
from torch.nn import functional as F

from util.weight_init import make_initializer

n_clusters = 512
n_samples = 2 ** 14
samplerate = zounds.SR22050()
small_batch = 4

latent_dim = 128
band = zounds.FrequencyBand(20, samplerate.nyquist)
mel_scale = zounds.MelScale(band, latent_dim)
fb = zounds.learn.FilterBank(
    samplerate, 512, mel_scale, 0.1, normalize_filters=True, a_weighting=False).to(device)

aim = AuditoryImage(512, 64, do_windowing=False, check_cola=True).to(device)

def perceptual_features(x):
    # x = fb.forward(x, normalize=False)
    # x = fb.temporal_pooling(x, 512, 256)
    # x = aim(x)
    # x = stft(x)
    x = codec.to_frequency_domain(x.squeeze())
    x = x[..., 0]
    return x

scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256

n_events = 8
sequence_length = n_frames
n_harmonics = 64
n_rooms = 8


init_weights = make_initializer(0.1)


def to_frames(batch):
    spec = codec.to_frequency_domain(batch)[..., 0]
    norms = torch.norm(spec, dim=-1, keepdim=True)
    spec = spec / (norms + 1e-8)
    return spec, norms


def update_kmeans(i, kmeans, frames):
    if i > 500:
        return
    frames = frames.view(-1, n_freq_bins).data.cpu().numpy()
    kmeans.partial_fit(frames)


def encode_batch(kmeans, frames):
    frames = frames.data.cpu().numpy().reshape(-1, n_freq_bins)
    indices = kmeans.predict(frames)
    return indices.reshape(-1, n_frames, 1)


def decode_batch(kmeans, indices, norms):
    b, length, _ = indices.shape
    indices = indices.reshape((-1))
    frames = kmeans.cluster_centers_[indices]
    frames = frames.reshape((b, length, n_freq_bins))
    frames = torch.from_numpy(frames).to(device).float()
    return frames * norms


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = F.leaky_relu(x + orig, 0.2)
        return x


# class Discriminator(nn.Module):
#     def __init__(self, channels=latent_dim):
#         super().__init__()
#         c = channels

#         self.embedding = nn.Embedding(n_clusters, channels)
#         self.channels = channels

#         self.embed_spec = LinearOutputStack(c, 2, out_channels=c, in_channels=256)

#         self.amp = nn.Sequential(
#             nn.Conv1d(1, 16, 7, 1, 3),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(16, 32, 7, 1, 3),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(32, self.channels, 7, 1, 3),
#         )

#         c = self.channels

#         self.net = nn.Sequential(
#             DilatedBlock(c, 1),
#             DilatedBlock(c, 3),
#             DilatedBlock(c, 9),
#             DilatedBlock(c, 1),
#             DilatedBlock(c, 3),
#             DilatedBlock(c, 9),
#             nn.Conv1d(c, 1, 3, dilation=1, padding=1),
#         )
#         self.apply(init_weights)
    
#     def forward(self, audio, indices, norms):

#         # initial shape assertions
#         indices = indices.view(-1, n_frames)
#         norms = norms.view(-1, 1, n_frames)

#         embedded = self\
#             .embedding(indices).view(-1, n_frames, self.channels)\
#             .permute(0, 2, 1)\
#             .view(-1, self.channels, n_frames)

#         amp = self.amp(norms)

#         x = embedded + amp

#         z = perceptual_features(audio)
#         z = self.embed_spec(z).permute(0, 2, 1)

#         x = x + z
#         features = [x]
#         for layer in self.net:
#             x = layer(x)
#             features.append(x)
#         return x, features


class SyntheticGradient(nn.Module):
    def __init__(self, channels=latent_dim):
        super().__init__()

        c = channels

        self.amp = nn.Sequential(
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 1),
        )

        self.freq = nn.Sequential(
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 1),
        )

        self.noise =nn.Sequential(
             nn.Conv1d(33, c, 3, 2, 1), # 256
             nn.LeakyReLU(0.2),
             nn.Conv1d(c, c, 3, 2, 1), # 128
             nn.LeakyReLU(0.2),
             nn.Conv1d(c, c, 3, 2, 1), # 64
        )
        # self.spec = nn.Sequential(
        #     nn.Conv1d(256, c, 1, 1, 0),
        #     DilatedBlock(c, 1),
        #     DilatedBlock(c, 3),
        #     DilatedBlock(c, 1),
        # )

        # self.verb = LinearOutputStack(c, 3, out_channels=c, in_channels=n_rooms + 1)

        self.net = nn.Sequential(
            nn.Conv1d(c * 3, c, 3, 1, 1),
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            nn.Conv1d(c, 256, 3, 1, dilation=1, padding=1)
        )

        self.apply(init_weights)
    
    def forward(self, amp_params, freq_params, noise_params, verb_params, audio):
        # spec = perceptual_features(audio).permute(0, 2, 1)
        # spec = self.spec(spec)

        # verb = self.verb(verb_params).view(-1, 128, 1)

        amp = self.amp(amp_params)
        freq = self.freq(freq_params)
        noise = self.noise(noise_params)
        # x = amp + freq + noise + spec

        x = torch.cat([amp, freq, noise], dim=1)

        x = self.net(x)

        # estimate of spectrogram
        return torch.relu(x.permute(0, 2, 1))


class Generator(nn.Module):
    def __init__(self, channels=latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_clusters, channels)
        self.channels = channels

        self.amp = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, self.channels, 7, 1, 3),
        )

        c = self.channels

        self.to_params = nn.Sequential(
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1),
        )

        self.to_noise = nn.ConvTranspose1d(c, c, 4, 2, 1)

        self.to_verb_mix = nn.Conv1d(c, 1, 3, 1, 1)

        self.to_verb_params = LinearOutputStack(c, 3, out_channels=n_rooms)
        self.to_verb_mix = LinearOutputStack(c, 3, out_channels=1)

        self.osc = OscillatorBank(
            input_channels=128,
            n_osc=128,
            n_audio_samples=n_samples,
            activation=torch.sigmoid,
            amp_activation=torch.abs,
            return_params=True,
            constrain=True,
            log_frequency=False,
            lowest_freq=40 / samplerate.nyquist,
            sharpen=False,
            compete=False)

        self.noise = NoiseModel(
            input_channels=128,
            input_size=128,
            n_noise_frames=512,
            n_audio_samples=n_samples,
            channels=128,
            activation=lambda x: x,
            squared=False,
            mask_after=1,
            return_params=True)

        self.verb = NeuralReverb(n_samples, n_rooms)

        self.apply(init_weights)

    def forward(self, indices, norms, add_noise=False):
        # initial shape assertions
        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        x = embedded + amp

        ap = self.to_params(x)
        nzp = self.to_noise(x)

        verb = self.to_verb_params.forward(x.mean(dim=-1))
        mix = self.to_verb_mix.forward(x.mean(dim=-1))
        mix = torch.sigmoid(mix).view(-1, 1, 1)

        osc, freq_params, amp_params = self.osc(ap, add_noise)
        noise, noise_params = self.noise(nzp, add_noise)
        
        signal = osc + noise


        verb_params = torch.cat([verb, mix.view(-1, 1)], dim=-1)
        # wet = self.verb.forward(signal, verb)

        
        # x = (signal * mix) + (wet * (1 - mix))

        x = signal

        return x, amp_params, freq_params, noise_params, verb_params


gen = Generator(channels=latent_dim).to(device)
gen_optim = optimizer(gen, lr=1e-3)


grad = SyntheticGradient(channels=latent_dim).to(device)
grad_optim = optimizer(grad, lr=1e-3)

# disc = Discriminator(channels=latent_dim).to(device)
# disc_optim = optimizer(disc, lr=1e-4)

def train_gen(batch, indices, norms):
    gen_optim.zero_grad()
    fake, amp_params, freq_params, noise_params, verb_params = gen.forward(indices, norms)
    real_spec = perceptual_features(batch)
    fake_spec = grad.forward(amp_params, freq_params, noise_params, verb_params, batch)
    loss = torch.abs(real_spec - fake_spec).sum()
    loss.backward()
    gen_optim.step()
    return fake, loss


def train_grad(batch, indices, norms):
    grad_optim.zero_grad()
    fake, amp_params, freq_params, noise_params, verb_params = gen.forward(indices, norms)
    spec_pred = grad.forward(amp_params, freq_params, noise_params, verb_params, batch)
    fake_spec = perceptual_features(fake)
    loss = torch.abs(spec_pred - fake_spec).sum()
    loss.backward()
    grad_optim.step()
    return loss


# def train_disc(batch, indices, norms):
#     disc_optim.zero_grad()
#     fake = gen.forward(indices, norms)
    
#     fj, _ = disc(fake, indices, norms)
#     rj, _ = disc(batch, indices, norms)
#     loss = least_squares_disc_loss(rj, fj)
#     loss.backward()
#     disc_optim.step()
#     return loss

# def train_disc_wrong_audio(correct, wrong, indices, norms):
#     disc_optim.zero_grad()
    
#     fj, _ = disc(wrong, indices, norms)
#     rj, _ = disc(correct, indices, norms)
#     loss = least_squares_disc_loss(rj, fj)
#     loss.backward()
#     disc_optim.step()
#     return loss

@readme
class DiscreteToSynthGenerator(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None
        self.fake = None

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.gen = gen

    def view_spec(self):
        return self.spec.data.cpu().numpy()[0]

    # def view_recon(self):
    #     return self.recon.data.cpu().numpy()[0]

    def view_indices(self):
        return self.indices[0].squeeze()

    def view_norms(self):
        return self.norms.data.cpu().numpy()[0].squeeze()

    def view_clusters(self):
        return self.kmeans.cluster_centers_

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.norms = norms
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            self.indices = indices
            # decoded = decode_batch(self.kmeans, indices, norms)
            # self.recon = decoded

            indices = torch.from_numpy(indices).long()[:small_batch].to(device)

            norms = norms[:small_batch].to(device)
            real = item[:small_batch].view(-1, 1, n_samples)

            # wrong_audio = item[small_batch:].view(-1, 1, n_samples)


            if i % 2 == 0:
                grad_loss = train_grad(real, indices, norms)
            else:
                self.fake, gen_loss = train_gen(real, indices, norms)

            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())
                print('GRAD', i, grad_loss.item())

