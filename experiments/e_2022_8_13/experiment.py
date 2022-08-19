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
from modules.ddsp import NoiseModel, OscillatorBank, overlap_add
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

    # return x

def perceptual_loss(a, b):
    return F.mse_loss(a, b)
    # return torch.abs(a - b).sum()



class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim, 
            n_bands, 
            n_samples, 
            constrain=True, 
            lowest_freq=40 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=False)
        
        self.noise = NoiseModel(
            model_dim,
            n_frames,
            n_frames * 8,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)
        
    
    def forward(self, x):
        x = x.view(-1, model_dim, n_frames)
        harm = self.osc.forward(x)
        noise = self.noise(x)
        signal = harm + noise
        return signal
        
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

    mx, _ = final.max(dim=-1, keepdim=True)
    final = final / (mx + 1e-8)

    return final


class EventGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # self.gen_env = ConvUpsample(
        #     latent_dim, model_dim, 8, n_frames, mode='learned', out_channels=1)

        self.env_up = nn.Linear(latent_dim, model_dim * 8)
        self.gen_env = nn.Sequential(
            nn.ConvTranspose1d(model_dim, 64, 8, 4, 2), # 32
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, 8, 4, 2), # 128
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, 8, 4, 2), # 512
        )

        # self.gen_env = nn.Sequential(
        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(model_dim, model_dim, 7, 1, 3),
        #     nn.LeakyReLU(0.2),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(model_dim, 64, 7, 1, 3),
        #     nn.LeakyReLU(0.2),
        #     # nn.BatchNorm1d(64),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(64, 32, 7, 1, 3),
        #     nn.LeakyReLU(0.2),
        #     # nn.BatchNorm1d(32),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(32, 1, 7, 1, 3),
            

        # )
        
        # self.gen_transfer = ConvUpsample(
        #     latent_dim, model_dim, 8, n_samples, mode='learned', out_channels=1)
        
        self.up = nn.Linear(latent_dim, model_dim * 8)

        self.gen_transfer = nn.Sequential(
            nn.ConvTranspose1d(model_dim, model_dim, 8, 4, 2), # 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(model_dim, 64, 8, 4, 2), # 128
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, 8, 4, 2), # 512
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, 8, 4, 2), # 2048
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 8, 8, 4, 2), # 8192
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 1, 8, 4, 2), # 32768
        )

        # self.gen_transfer = nn.Sequential(

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(model_dim, model_dim, 7, 1, 3),
        #     nn.LeakyReLU(0.2),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(model_dim, 64, 7, 1, 3),
        #     nn.LeakyReLU(0.2),
        #     # nn.BatchNorm1d(64),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(64, 32, 7, 1, 3),
        #     nn.LeakyReLU(0.2),
        #     # nn.BatchNorm1d(32),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(32, 16, 7, 1, 3),
        #     nn.LeakyReLU(0.2),
        #     # nn.BatchNorm1d(16),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(16, 8, 7, 1, 3),
        #     nn.LeakyReLU(0.2),
        #     # nn.BatchNorm1d(8),

        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv1d(8, 1, 7, 1, 3),

        # )

        
        
        # self.gen_impulse_transfer = LinearOutputStack(
        #     model_dim, 3, out_channels=n_coeffs * 2)


    def forward(self, x):
        x = x.view(-1, latent_dim)
        batch = x.shape[0]

        e = self.env_up(x).view(batch, model_dim, 8)
        env = self.gen_env(e) ** 2
        envelope = env = F.upsample(env, size=n_samples, mode='linear')

        noise = torch.zeros(batch, 1, n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise

        mx, _ = torch.max(env, dim=-1, keepdim=True)
        env = env / (mx + 1e-8)
        

        # transfer = self.gen_transfer(x)
        t = self.up(x).view(batch, model_dim, 8)
        transfer = self.gen_transfer(t)

        mx, _ = torch.max(transfer, dim=-1, keepdim=True)
        transfer = transfer / (mx + 1e-8)
        
        # transfer = unit_norm(transfer, axis=-1)


        t = torch.fft.rfft(transfer, dim=-1, norm='ortho')
        e = torch.fft.rfft(env, dim=-1, norm='ortho')
        spec = e * t
        sig = torch.fft.irfft(spec, dim=-1, norm='ortho')
        sig = sig.view(batch, n_samples)

        
        # impulse = self.gen_impulse_transfer(x).reshape(-1, n_coeffs, 2)
        impulse = None

        sig = transfer
        return env, transfer, impulse, sig


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
        # norms = F.dropout(norms, 0.05)
        norms = torch.softmax(norms, dim=-1)
        # norms = torch.norm(x, dim=1)

        values, indices = torch.topk(norms, k=n_atoms, dim=-1)

        latents = []
        for b in range(batch):
            for i in range(n_atoms):
                latents.append(x[b, :, indices[b, i]][None, :])
            
        # latents = torch.gather(x, dim=-1, index=indices)
        latents = torch.cat(latents, dim=0).view(batch * n_atoms, latent_dim)
        # latents = unit_norm(latents)

        env, transfer, impulse, atoms = self.gen_events(latents)

        # atoms = generate_event(env, transfer, impulse).view(batch, n_atoms, n_samples)
        atoms = atoms.view(batch, n_atoms, n_samples)

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

        # mx, _ = torch.max(final, dim=-1, keepdim=True)
        # final = final / (mx + 1e-8)

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
class WaveguideExperiment(object):
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
