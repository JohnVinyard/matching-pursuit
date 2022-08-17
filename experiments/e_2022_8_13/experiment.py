import numpy as np
from config.dotenv import Config
from modules.atoms import unit_norm
from modules.linear import LinearOutputStack
from modules.pos_encode import pos_encoded
from modules.reverb import NeuralReverb
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from util import device, playable, readme
import zounds
import torch
from modules.ddsp import overlap_add
from torch import nn
from torch.nn import functional as F
from torch import jit
from modules.psychoacoustic import PsychoacousticFeature
from util import make_initializer
from modules.stft import stft
from torch.nn.utils.clip_grad import clip_grad_value_, clip_grad_norm_

n_samples = 2 ** 15
samplerate = zounds.SR22050()

window_size = 512
step_size = window_size // 2
n_coeffs = (window_size // 2) + 1

n_frames = n_samples // step_size

model_dim = 64
latent_dim = 64
n_atoms = 16

n_bands = 64
kernel_size = 128

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


pif = PsychoacousticFeature([128] * 6).to(device)

init_weights = make_initializer(0.1)

def perceptual_feature(x):
    bands = pif.compute_feature_dict(x)
    return torch.cat(bands, dim=-2)

    # return stft(x, 512, 256)

    # x = torch.abs(torch.fft.rfft(x, dim=-1, norm='ortho'))
    # return x

def perceptual_loss(a, b):
    return F.mse_loss(a, b)
    # return torch.abs(a - b).sum()


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        return x


# def polar2z(r,theta):
#     return r * torch.exp( 1j * theta )

def generate_event(envelope, transfer_functions, envelope_transfer):
    output = []


    # norm = torch.norm(transfer_functions, dim=2, keepdim=True)

    # mx, _ = torch.max(norm.view(envelope.shape[0], -1), dim=-1)
    # scaled_norm = norm / (mx[:, None, None, None] + 1e-8)

    # unit_norm = transfer_functions / (norm + 1e-8)

    # norm = 0.9 + (scaled_norm * 0.0999)
    # transfer_functions = norm * unit_norm

    # mag = 0.9 + torch.sigmoid(transfer_functions[..., 0, :]) * 0.09999
    # phase = transfer_functions[..., 1, :]
    # phase = torch.cumsum(phase, dim=-1)
    # transfer_functions = mag * torch.exp(1j * phase)

    transfer_functions = torch.fft.rfft(transfer_functions, dim=1, norm='ortho')
    transfer_functions = transfer_functions / (torch.abs(transfer_functions) + 1e-8)

    # transfer_functions = torch.complex(
    #     transfer_functions[..., 0, :], transfer_functions[..., 1, :])

    transfer_functions = transfer_functions.view(-1, n_coeffs, n_frames)

    # envelope = \
    #     torch.zeros(envelope.shape[0], window_size, n_frames, device=envelope.device).uniform_(-1, 1) \
    #     * envelope

    envelope = envelope * torch.zeros_like(envelope).uniform_(-1, 1)
    
    
    for i in range(n_frames):
        start = i * step_size
        end = start + window_size

        env = envelope[:, :, start: end]
        if env.shape[-1] < window_size:
            env = F.pad(env, (0, 256))
        env = env.reshape(-1, 1, window_size).reshape(-1, window_size, 1)

        if i > 0:
            current = output[-1]
        else:
            current = torch.zeros_like(env)

        
        nxt = env + current
        spec = torch.fft.rfft(nxt, dim=1, norm='ortho')
        spec = spec * transfer_functions[:, :, i: i + 1]
        nxt = torch.fft.irfft(spec, dim=1, norm='ortho')


        output.append(
            nxt * torch.hamming_window(window_size, device=envelope.device)[None, :, None])

    output = torch.cat(output, dim=-1)
    output = output.permute(0, 2, 1)[:, None, :, :]

    final = overlap_add(output)

    final = final[..., :n_samples]

    # mx, _ = final.max(dim=-1, keepdim=True)
    # final = final / (mx + 1e-8)

    return final


class EventGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.gen_env = ConvUpsample(
            latent_dim, model_dim, 8, n_frames * 8, mode='nearest', out_channels=1)
        
        # self.gen_env = PosEncodedUpsample(
        #     latent_dim, model_dim, n_frames, out_channels=1, layers=4, learnable_encodings=True, concat=True)

        # self.gen_env = LinearOutputStack(model_dim, 3, out_channels=window_size)
        
        self.gen_transfer = ConvUpsample(
            latent_dim, model_dim, 8, n_frames, mode='nearest', out_channels=window_size)
        # self.gen_transfer = PosEncodedUpsample(
        #     latent_dim, model_dim, n_frames, out_channels=n_coeffs * 2, layers=4, learnable_encodings=False, concat=True)

        # self.gen_transfer = LinearOutputStack(model_dim, 3, out_channels=n_coeffs * 2)

        self.gen_impulse_transfer = LinearOutputStack(
            model_dim, 3, out_channels=n_coeffs * 2)

    def forward(self, x):
        x = x.view(-1, latent_dim)

        env = torch.abs(self.gen_env(x))
        env = F.upsample(env, size=n_samples, mode='linear')

        mx, _ = env.max(dim=-1, keepdim=True)
        env = env / (mx + 1e-8)
        

        transfer = self.gen_transfer(x)#.view(-1, n_coeffs, 2, n_frames)

        impulse = self.gen_impulse_transfer(x).reshape(-1, n_coeffs, 2)


        return env, transfer, impulse


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        
        # self.context = nn.Sequential(
        #     DilatedBlock(model_dim, 1),
        #     nn.LayerNorm((model_dim, n_frames)),
        #     DilatedBlock(model_dim, 3),
        #     nn.LayerNorm((model_dim, n_frames)),
        #     DilatedBlock(model_dim, 9),
        #     nn.LayerNorm((model_dim, n_frames)),
        #     DilatedBlock(model_dim, 27),
        #     nn.LayerNorm((model_dim, n_frames)),
        #     DilatedBlock(model_dim, 1),
        # )

        encoder = nn.TransformerEncoderLayer(model_dim, 4, model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(encoder, 4, norm=None)

        self.to_env = nn.Conv1d(model_dim, 1, 1, 1, 0)

        self.gen_events = EventGenerator()

        self.embed = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate, n_samples)

        self.to_room = LinearOutputStack(
            model_dim, 3, out_channels=self.verb.n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)

        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]

        orig = x

        n = fb.forward(orig, normalize=False)
        spec = n = fb.temporal_pooling(n, 512, 256)[..., :n_frames]

        pos = pos_encoded(batch, n_frames, n_freqs=16, device=n.device).permute(0, 2, 1)
        n = torch.cat([pos, n], dim=1)
        n = self.embed(n)

        n = n.permute(0, 2, 1)
        x = self.context(n)
        x = x.permute(0, 2, 1)

        # x = F.dropout(x, 0.1)

        x = x + spec

        norms = self.to_env(x).view(batch, -1)
        # norms = torch.norm(x, dim=1)

        values, indices = torch.topk(norms, k=n_atoms, dim=-1)

        latents = []
        for b in range(batch):
            for i in range(n_atoms):
                latents.append(x[b, :, indices[b, i]][None, :])
            
        # latents = torch.gather(x, dim=-1, index=indices)
        latents = torch.cat(latents, dim=0).view(batch * n_atoms, latent_dim)

        env, transfer, impulse = self.gen_events(latents)

        atoms = generate_event(env, transfer, impulse).view(batch, n_atoms, n_samples)

        output = torch.zeros(orig.shape[0], 1, n_samples * 2, device=atoms.device)

        for b in range(x.shape[0]):
            for i in range(n_atoms):
                v = values[b, i]
                start = indices[b, i] * 256
                end = start + n_samples
                output[b, :, start: end] += atoms[b, i]  * v
                
        final = output[..., :n_samples]

        # reverb
        # agg, _ = torch.max(x, dim=-1)
        # r = torch.softmax(self.to_room(agg), dim=-1)
        # m = torch.sigmoid(self.to_mix(agg).view(-1, 1, 1))
        # wet = self.verb.forward(final, r)
        # final = (m * wet) + (final * (1 - m))

        mx, _ = torch.max(final, dim=-1, keepdim=True)
        final = final / (mx + 1e-8)

        return final


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)

    
    loss.backward()
    # clip_grad_norm_(model.parameters(), 1)

    # 0.55 is too high, 0.5 never learns
    # clip_grad_value_(model.parameters(), 0.525)

    optim.step()

    return recon, loss


@readme
class WaveguideExpiriment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None

    def listen(self):
        return playable(self.fake, samplerate)

    def orig(self):
        return playable(self.real, samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            self.fake, loss = train(item)

            print(i, loss.item())
