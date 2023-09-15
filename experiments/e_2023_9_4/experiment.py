
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.overlap_add import overlap_add
from modules.reverb import ReverbGenerator
from modules.sparse import encourage_sparsity_loss
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from modules.normalization import max_norm, unit_norm


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

# Get rid of this setting
conv_only_dict = True

static_dict = True
convolve_impulse_and_resonance = True
resonance_model = True
full_atom_size = 2048
recurrent_resonance_model = False
do_reverb = False
base_resonance = 0.1
apply_group_delay_to_dither = True
encoding_channels = 512

class RecurrentResonanceModel(nn.Module):
    def __init__(self, encoding_channels, impulse_samples, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.impulse_samples = impulse_samples
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.step = window_size // 2
        self.n_coeffs = window_size // 2 + 1
        self.resonance_samples = resonance_samples
        self.n_frames = resonance_samples // self.step
        self.encoding_channels = encoding_channels

        self.base_resonance = base_resonance
        self.resonance_factor = (1 - self.base_resonance) * 0.999

        self.register_buffer('group_delay', torch.linspace(0, np.pi, self.n_coeffs))

        self.to_initial = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_resonance = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_phase_dither = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, latents):
        batch_size = latents.shape[0]

        initial = self.to_initial(latents)
        res = self.base_resonance + (torch.sigmoid(self.to_resonance(latents)) * self.resonance_factor)
        dither = torch.sigmoid(self.to_phase_dither(latents)) 

        if apply_group_delay_to_dither:
            dither = dither * self.group_delay[None, None, :]


        # print(res.shape, dither.shape, initial.shape)

        first_frame = initial[:, None, :]

        if not recurrent_resonance_model:
            # Non-recurrent
            full_res = res[..., None].repeat(1, 1, 1, self.n_frames)
            full_res = torch.cumprod(full_res, dim=-1)
            mags = full_res * initial[..., None]
            mags = mags.permute(0, 1, 3, 2)

            full_delay = self.group_delay[None, None, :, None].repeat(1, 1, 1, self.n_frames)
            full_delay = torch.cumsum(full_delay, dim=-1)
            full_delay = full_delay + (torch.zeros_like(full_delay).uniform_(-np.pi, np.pi) * dither[:, :, :, None])
            phases = full_delay
            phases = phases.permute(0, 1, 3, 2)
            # end non-recurrent
        else:

            # recurrent

            frames = [first_frame]
            phases = [torch.zeros_like(first_frame).uniform_(-np.pi, np.pi)]

            # TODO: This should also incorporate impulses, i.e., new excitations
            # beyond the original
            for i in range(self.n_frames - 1):

                mag = frames[i]
                phase = phases[i]

                # compute next polar coordinates
                nxt_mag = mag * res

                nxt_phase = \
                    phase \
                    + self.group_delay[None, None, None, :] \
                    + (dither * torch.zeros_like(dither).uniform_(-np.pi, np.pi)[:, None, :, :]) 


                frames.append(nxt_mag)
                phases.append(nxt_phase)


            mags = torch.cat(frames, dim=1)
            phases = torch.cat(phases, dim=1)

        # end recurrent

        frames = torch.complex(
            mags * torch.cos(phases),
            mags * torch.sin(phases)
        )

        windowed = torch.fft.irfft(frames, dim=-1, norm='ortho')
        
        if recurrent_resonance_model:
            windowed = windowed.permute(0, 2, 1, 3)
        
        samples = overlap_add(windowed, apply_window=True)[..., :self.resonance_samples]

        assert samples.shape == (batch_size, self.encoding_channels, self.resonance_samples)
        return samples
                        


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        x = F.dropout(x, 0.1)

        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x

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
            exp.samplerate, self.filter_kernel_size, scale, 0.5, normalize=True).real.astype(np.float32)
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
        return impulse


class Model(nn.Module):
    def __init__(
            self, 
            channels, 
            encoding_channels, 
            impulse_samples, 
            resonance_samples,
            latent_dim,
            resonance_model=False):

        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.impulse_samples = impulse_samples
        self.resonance_samples = resonance_samples
        self.latent_dim = latent_dim

        self.stack = nn.Sequential(
            nn.Conv1d(exp.n_bands, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),
            DilatedBlock(channels, 1),    
            DilatedBlock(channels, 3),    
            DilatedBlock(channels, 9),    
            DilatedBlock(channels, 27),    
            DilatedBlock(channels, 81),    
            DilatedBlock(channels, 243),    
            DilatedBlock(channels, 1),    
        )

        self.up = nn.Conv1d(channels, encoding_channels, 1, 1, 0)


        self.static_dict = nn.Parameter(torch.zeros(1, encoding_channels, full_atom_size).uniform_(-1, 1))

        self.impulse_latent = nn.Parameter(
            torch.zeros(1, encoding_channels, latent_dim).uniform_(-1, 1))
        self.resonance_latent = nn.Parameter(
            torch.zeros(1, encoding_channels, latent_dim).uniform_(-1, 1))
        self.mix_latent = nn.Parameter(
            torch.zeros(1, encoding_channels, latent_dim).uniform_(-1, 1))

        self.embed_impulse_latent = LinearOutputStack(
             channels, 
             layers=3, 
             in_channels=encoding_channels, 
             out_channels=latent_dim, 
             norm=nn.LayerNorm((channels,))
        )

        self.embed_resonance_latent = LinearOutputStack(
             channels, 
             layers=3, 
             in_channels=encoding_channels, 
             out_channels=latent_dim, 
             norm=nn.LayerNorm((channels,))
        )

        self.embed_mix_latent = LinearOutputStack(
             channels, 
             layers=3, 
             in_channels=encoding_channels, 
             out_channels=latent_dim, 
             norm=nn.LayerNorm((channels,))
        )


        if resonance_model:
            self.to_resonance = RecurrentResonanceModel(
                self.encoding_channels,
                self.impulse_samples,
                self.latent_dim,
                self.channels,
                512,
                self.resonance_samples)
        else:
            self.to_resonance = ConvUpsample(
                latent_dim, 
                channels, 
                8, 
                end_size=resonance_samples, 
                out_channels=1, 
                mode='nearest', 
                batch_norm=True)
        
        
        self.to_impulse = GenerateImpulse(
            latent_dim, channels, self.impulse_samples, 8, encoding_channels)
        
        self.to_mix = GenerateMix(
            latent_dim, channels, encoding_channels)
        
        self.embed_verb_latent = nn.Linear(encoding_channels, latent_dim)
        self.verb = ReverbGenerator(latent_dim, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((latent_dim,)))

        self.apply(lambda x: exp.init_weights(x))
    
        
    def forward(self, x):
        batch_size = x.shape[0]

        x = exp.fb.forward(x, normalize=False)
        x = self.stack(x)

        x = self.up(x)
        x = F.dropout(x, 0.01)

        # TODO: A softmax operation prior to the RELU might provide better signal 
        # around sparsity

        encoding = x = torch.relu(x)

        ctxt = torch.sum(encoding, dim=-1)

        if static_dict:
            d = self.static_dict.repeat(batch_size, 1, 1)
        else:
            impulse_latent = self.embed_impulse_latent(ctxt)[:, None, :] + self.impulse_latent
            resonance_latent = self.embed_resonance_latent(ctxt)[:, None, :] + self.resonance_latent
            mix_latent = self.embed_mix_latent(ctxt)[:, None, :] + self.mix_latent

            assert impulse_latent.shape == (batch_size, self.encoding_channels, self.latent_dim)
            assert resonance_latent.shape == (batch_size, self.encoding_channels, self.latent_dim)
            assert mix_latent.shape == (batch_size, self.encoding_channels, self.latent_dim)

            impulses = self.to_impulse.forward(impulse_latent)
            assert impulses.shape == (batch_size, self.encoding_channels, self.impulse_samples)
            impulses = F.pad(impulses, (0, self.resonance_samples - self.impulse_samples))

            resonances = self.to_resonance.forward(resonance_latent).view(-1, self.encoding_channels, self.resonance_samples)
            assert resonances.shape == (batch_size, self.encoding_channels, self.resonance_samples)
            
            mix = self.to_mix(mix_latent)
            assert mix.shape == (batch_size, self.encoding_channels, 2)

            imp_amp = mix[..., :1]
            res_amp = mix[..., 1:]

            if conv_only_dict:
                d = resonances
                d = d * torch.hamming_window(d.shape[-1], device=x.device)[None, None, :]
            else:
                # TODO: what happens if I skip this convolution?
                # is it better to just add the two together
                if convolve_impulse_and_resonance:
                    d = fft_convolve(impulses, resonances)
                else:
                    d = resonances
                d = (impulses * imp_amp) + (d * res_amp)
                assert d.shape == (batch_size, self.encoding_channels, self.resonance_samples)
            
        d = F.pad(d, (0, exp.n_samples - self.resonance_samples))
        d = unit_norm(d, dim=-1)

        # TODO: What if I convert the feature map to sparse before doing this?
        x = fft_convolve(d, encoding)[..., :exp.n_samples]
            # x = torch.sum(x, dim=1, keepdim=True)

        if do_reverb:
            # TODO: apply reverb to each individual channel.  This could
            # help to remove more norm by extending notes
            verb_ctxt = self.embed_verb_latent(ctxt)
            x = self.verb.forward(verb_ctxt, x)
        return x, encoding

model = Model(
    channels=128, 
    encoding_channels=encoding_channels, 
    impulse_samples=256 * 8, 
    resonance_samples=full_atom_size,
    latent_dim=16,
    resonance_model=resonance_model
).to(device)

optim = optimizer(model, lr=1e-3)

def train(batch, i):
    batch_size = batch.shape[0]

    optim.zero_grad()
    recon, encoding = model.forward(batch)

    residual = stft(batch, 512, 256, pad=True)
    loss = 0

    # TODO: try JIT here
    print('==========================')
    # TODO: try sorting from loudest to quietest channel
    for i in range(encoding_channels):
        start_norm = torch.norm(residual.view(batch_size, -1), dim=-1)
        channel_spec = stft(recon[:, i: i + 1, :], 512, 256, pad=True)
        residual = residual - channel_spec
        end_norm = torch.norm(residual.view(batch_size, -1), dim=-1)
        diff = start_norm - end_norm
        loss = loss + -diff.mean()
    
    

    # with torch.no_grad():
    #     for i in range(batch_size):
    #         for j in range(encoding.shape[1]):
    #             channel = recon[i, j, :]
    #             start_norm = torch.norm(residual[i, j])
    #             energy = torch.sum(channel)
    #             if energy.item() == 0:
    #                 continue
    #             nz = (channel > 0).sum()
    #             print(f'batch {i}, channel {j} has {nz.item()} non-zero elements, {(nz / channel.nelement())} sparse')
    

    recon = torch.sum(recon, dim=1, keepdim=True)

    # encoding = encoding.view(batch_size, -1)
    # srt, indices = torch.sort(encoding, dim=-1, descending=True)

    # the first 128 atoms may be as large/loud as they need to be
    # TODO: This number could slowly drop over training time
    # penalized = srt[:, 128:]

    # non_zero = (encoding > 0).sum()
    # sparsity = non_zero / encoding.nelement()
    # print('sparsity', sparsity.item(), 'n_elements', (non_zero / batch_size).item())

    # sparsity_loss = torch.abs(penalized).sum() * 0.00001
    sparsity_loss = encourage_sparsity_loss(
        encoding, 
        n_unpenalized=128, 
        sparsity_loss_weight=0.0001)

    loss = loss + sparsity_loss
    # loss = exp.perceptual_loss(recon, batch) + sparsity_loss
    # loss = F.mse_loss(exp.pooled_filter_bank(recon), exp.pooled_filter_bank(batch)) + sparsity_loss

    loss.backward()
    optim.step()
    return loss, recon


@readme
class SparseResonance(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    