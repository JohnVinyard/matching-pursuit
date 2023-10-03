import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from angle import windowed_audio
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.overlap_add import overlap_add
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from scratch3 import sparsify2
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_events = 64
impulse_size = 512
resonance_size = 16384

recurrent_resonance_model = False
base_resonance = 0.2
apply_group_delay_to_dither = True




class RecurrentResonanceModel(nn.Module):
    def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.resonance_samples = resonance_samples

        n_res = 512
        self.n_frames = resonance_samples // (window_size // 2)
        self.res_factor = (1 - base_resonance) * 0.9

        band = zounds.FrequencyBand(40, 2000)
        scale = zounds.MelScale(band, n_res)
        bank = morlet_filter_bank(exp.samplerate, resonance_samples, scale, 0.01, normalize=True).real.astype(np.float32)
        bank = torch.from_numpy(bank)
        self.res = nn.Parameter(bank)

        # self.res = nn.Parameter(torch.zeros(n_res, resonance_samples).uniform_(-1, 1))
        self.to_momentum = LinearOutputStack(channels, 3, out_channels=1, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_selection = LinearOutputStack(channels, 3, out_channels=n_res, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):

        mom = base_resonance + (torch.sigmoid(self.to_momentum(x)) * self.res_factor)
        mom = torch.log(1e-12 + mom)
        mom = mom.repeat(1, 1, self.n_frames)
        mom = torch.cumsum(mom, dim=-1)
        mom = torch.exp(mom)

        sel = self.to_selection(x)
        sel = torch.softmax(sel, dim=-1)
        res = sel @ self.res
        windowed = windowed_audio(res, self.window_size, self.window_size // 2)
        windowed = unit_norm(windowed, dim=-1)

        windowed = windowed * mom[..., None]

        windowed = overlap_add(windowed, apply_window=False)[..., :self.resonance_samples]
        windowed = max_norm(windowed)

        return windowed


# class RecurrentResonanceModel(nn.Module):
#     def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.channels = channels
#         self.window_size = window_size
#         self.step = window_size // 2
#         self.n_coeffs = window_size // 2 + 1
#         self.resonance_samples = resonance_samples
#         self.n_frames = resonance_samples // self.step
#         self.encoding_channels = encoding_channels

#         self.base_resonance = base_resonance
#         self.resonance_factor = (1 - self.base_resonance) * 0.9

#         self.register_buffer('group_delay', torch.linspace(0, np.pi, self.n_coeffs))

#         self.to_initial = LinearOutputStack(
#             channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
#         self.to_resonance = LinearOutputStack(
#             channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
#         self.to_phase_dither = LinearOutputStack(
#             channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
#     def forward(self, latents):
#         batch_size = latents.shape[0]

#         initial = self.to_initial(latents)
#         res = self.base_resonance + (torch.sigmoid(self.to_resonance(latents)) * self.resonance_factor)
#         dither = torch.sigmoid(self.to_phase_dither(latents)) 

#         if apply_group_delay_to_dither:
#             dither = dither * self.group_delay[None, None, :]


#         # print(res.shape, dither.shape, initial.shape)

#         first_frame = initial[:, None, :]

#         if not recurrent_resonance_model:
#             # Non-recurrent
#             full_res = res[..., None].repeat(1, 1, 1, self.n_frames)
#             full_res = torch.cumprod(full_res, dim=-1)
#             mags = full_res * initial[..., None]
#             mags = mags.permute(0, 1, 3, 2)

#             full_delay = self.group_delay[None, None, :, None].repeat(1, 1, 1, self.n_frames)
#             full_delay = torch.cumsum(full_delay, dim=-1)
#             full_delay = full_delay + (torch.zeros_like(full_delay).uniform_(-np.pi, np.pi) * dither[:, :, :, None])
#             phases = full_delay
#             phases = phases.permute(0, 1, 3, 2)
#             # end non-recurrent
#         else:

#             # recurrent

#             frames = [first_frame]
#             phases = [torch.zeros_like(first_frame).uniform_(-np.pi, np.pi)]

#             # TODO: This should also incorporate impulses, i.e., new excitations
#             # beyond the original
#             for i in range(self.n_frames - 1):

#                 mag = frames[i]
#                 phase = phases[i]

#                 # compute next polar coordinates
#                 nxt_mag = mag * res

#                 nxt_phase = \
#                     phase \
#                     + self.group_delay[None, None, None, :] \
#                     + (dither * torch.zeros_like(dither).uniform_(-np.pi, np.pi)[:, None, :, :]) 


#                 frames.append(nxt_mag)
#                 phases.append(nxt_phase)


#             mags = torch.cat(frames, dim=1)
#             phases = torch.cat(phases, dim=1)

#         # end recurrent

#         frames = torch.complex(
#             mags * torch.cos(phases),
#             mags * torch.sin(phases)
#         )

#         windowed = torch.fft.irfft(frames, dim=-1, norm='ortho')
        
#         if recurrent_resonance_model:
#             windowed = windowed.permute(0, 2, 1, 3)
        
#         samples = overlap_add(windowed, apply_window=True)[..., :self.resonance_samples]
#         samples = max_norm(samples, dim=-1)

#         assert samples.shape == (batch_size, self.encoding_channels, self.resonance_samples)
#         return samples
                        


class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=2, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        x = self.to_mix(x)
        x = x.view(-1, self.encoding_channels, 2)
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

        scale = zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist), n_filter_bands)
        filters = morlet_filter_bank(
            exp.samplerate, self.filter_kernel_size, scale, 0.75, normalize=True).real.astype(np.float32)
        self.register_buffer('filters', torch.from_numpy(filters).view(n_filter_bands, self.filter_kernel_size))

        self.to_filter_mix = LinearOutputStack(
            channels, 3, out_channels=self.n_filter_bands, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        
        self.to_envelope = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            mode='nearest',
            end_size=self.n_frames,
            out_channels=1,
            batch_norm=True,
        )

    
    def forward(self, x):

        batch_size = x.shape[0]

        # generate envelopes
        frames = self.to_envelope(x.view(-1, self.latent_dim))
        frames = torch.abs(frames)
        frames = frames * torch.hamming_window(frames.shape[-1], device=x.device)[None, None, :]
        frames = F.interpolate(frames, size=self.n_samples, mode='linear')
        frames = frames.view(-1, self.encoding_channels, self.n_samples)

        # generate filters
        mix = self.to_filter_mix(x).view(-1, self.n_filter_bands, 1)

        filt = self.filters.view(-1, self.n_filter_bands, self.filter_kernel_size)
        filters = (mix * filt).view(-1, self.n_filter_bands, self.filter_kernel_size).sum(dim=1)
        filters = F.pad(filters, (0, self.n_samples - self.filter_kernel_size))
        filters = filters.view(-1, self.encoding_channels, self.n_samples)

        # generate noise
        noise = torch.zeros(batch_size, self.encoding_channels, self.n_samples, device=x.device).uniform_(-1, 1)

        # filter the noise
        filtered = fft_convolve(noise, filters)[..., :self.n_samples]

        # apply envelope
        impulse = filtered * frames
        impulse = max_norm(impulse)
        return impulse


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_period = nn.Linear(257, 8)
        

        self.embed_cond = nn.Conv1d(4096, 256, 1, 1, 0)
        self.embed_spec = nn.Conv1d(1024, 256, 1, 1, 0)

        self.net = nn.Sequential(

            # 64
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 32
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 8
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
            ),


            nn.Conv1d(256, 1, 1, 1, 0)            
        )

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, cond, audio):
        batch_size = cond.shape[0]

        spec = exp.perceptual_feature(audio)

        x = self.embed_period(spec)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        spec = self.embed_spec(x)

        # TODO: Weirdly, this was doing some interesting things
        # as an unconditional discriminator (due to a mistake)
        cond = self.embed_cond(cond)
        x = cond + spec
        j = self.net(x)
        return j




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(257, 8)
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 2), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 6), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 18), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=9),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 2), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),


            nn.Conv1d(1024, 4096, 1, 1, 0)
        )

        self.embed_context = LinearOutputStack(256, 3, in_channels=4096, norm=nn.LayerNorm((256,)))
        self.embed_one_hot = LinearOutputStack(256, 3, in_channels=4096, norm=nn.LayerNorm((256,)))
        
        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)
        self.res = RecurrentResonanceModel(n_events, 256, 128, 1024, resonance_samples=resonance_size)
        self.mix = GenerateMix(256, 128, n_events)
        self.to_shift = LinearOutputStack(256, 3, out_channels=1)
        self.to_amp = LinearOutputStack(256, 3, out_channels=1)


        self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            32, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((32,)))
    
        self.apply(lambda x: exp.init_weights(x))

    def encode(self, x):
        batch_size = x.shape[0]

        if len(x.shape) != 4:
            x = exp.perceptual_feature(x)

        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        encoded = self.encoder.forward(x)
        encoded = F.dropout(encoded, 0.02)

        return encoded
    
    def generate(self, encoded, one_hot, packed):
        ctxt = torch.sum(encoded, dim=-1)
        ce = self.embed_context(ctxt)
        ctxt = self.verb_context.forward(ctxt)

        # first embed context and one hot and combine them
        oh = self.embed_one_hot(one_hot)
        embeddings = ce[:, None, :] + oh

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        res = self.res.forward(embeddings)

        # mixes
        mx = self.mix.forward(embeddings)

        conv = fft_convolve(padded, res)[..., :resonance_size]


        stacked  = torch.cat([padded[..., None], conv[..., None]], dim=-1)
        mixed = stacked @ mx.view(-1, n_events, 2, 1)
        mixed = mixed.view(-1, n_events, resonance_size)

        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps

        # here I need indices + additional shifts
        # encoded_frames = encoded.shape[-1]
        # total_samples = exp.n_samples
        # step = total_samples // encoded_frames

        # # mostly shifted within a frame
        # fine_shifts = torch.tanh(self.to_shift(embeddings)) * (step / total_samples)
        # fine_shifts = fine_shifts.view(-1, n_events)

        # indices = torch.argmax(packed, dim=-1)
        # shifts = (indices * step) / total_samples

        # final_shifts = shifts + fine_shifts

        # final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        # final = fft_shift(final, final_shifts[..., None])[..., :exp.n_samples]

        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]

        final = torch.sum(final, dim=1, keepdim=True)


        final = self.verb.forward(ctxt, final)
        return final


    def forward(self, x):
        encoded = self.encode(x)

        # a = full sparse representation
        # b = packed (just active channels)
        # c = one_hot
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)



        final = self.generate(encoded, one_hot, packed)
        return final, encoded


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)



def train(batch, i):
    optim.zero_grad()
    disc_optim.zero_grad()

    with torch.no_grad():
        feat = exp.perceptual_feature(batch)
    

    if i % 2 == 0:
        recon, encoded = model.forward(feat)
        r = exp.perceptual_feature(recon)

        spec_loss = F.mse_loss(r, feat)

        # a, b, c = exp.perceptual_triune(recon)
        # d, e, f = exp.perceptual_triune(batch)
        # spec_loss = F.mse_loss(a, d) + F.mse_loss(b, e) + F.mse_loss(c, f)

        j = disc.forward(encoded.clone().detach(), recon)
        
        loss = (torch.abs(1 - j).mean() * 1) + spec_loss

        loss.backward()
        optim.step()

        print('GEN', loss.item())
        return loss, recon, encoded
    else:
        with torch.no_grad():
            recon, encoded = model.forward(feat)
        
        rj = disc.forward(encoded, batch)
        fj = disc.forward(encoded, recon)
        loss = (torch.abs(1 - rj) + torch.abs(0 - fj)).mean()
        loss.backward()
        disc_optim.step()
        print('DISC', loss.item())
        return None, None, None


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x[:, None, :, :]
        x = F.max_pool2d(x, (16, 8), (16, 8))
        x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        return x
    return (encoded,)



@readme
class GraphRepresentation(BaseExperimentRunner):
    
    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)
            

            if l is None:
                continue

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l)