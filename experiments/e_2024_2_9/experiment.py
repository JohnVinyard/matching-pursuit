
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType

from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.decompose import fft_frequency_decompose

from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.ddsp import NoiseModel
from modules.fft import fft_convolve, fft_shift
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, sparse_softmax
from modules.sparse import sparsify2
from modules.stft import stft
from modules.transfer import ImpulseGenerator, ResonanceChain
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from torch.nn.utils.weight_norm import weight_norm
from torch.distributions import Normal


exp = Experiment(
    samplerate=22050,
    n_samples=2 ** 15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=512)

n_events = 16
context_dim = 16
impulse_size = 16384
resonance_size = 32768
samplerate = 22050
n_samples = 32768

alternate_model = False


def gumbel_straight_through_estimator(x: torch.Tensor):
    # return sparse_softmax(x, normalize=True)
    return F.gumbel_softmax(x, tau=1, hard=True)


class ResonanceModel2(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        channels, 
        resonance_size, 
        n_atoms, 
        n_piecewise, 
        init_atoms=None, 
        learnable_atoms=False,
        mixture_over_time=False,
        n_frames = 128):
        
        super().__init__()
        
        self.n_frames = n_frames
        self.latent_dim = latent_dim
        self.channels = channels
        self.resonance_size = resonance_size
        self.n_atoms = n_atoms
        self.n_piecewise = n_piecewise
        self.init_atoms = init_atoms
        self.learnable_atoms = learnable_atoms
        self.mixture_over_time = mixture_over_time
        
        self.n_coeffs = (self.resonance_size // 2) + 1
        self.coarse_coeffs = 257
        
        self.base_resonance = 0.02
        self.res_factor = (1 - self.base_resonance) * 0.99
        
        low_hz = 40
        high_hz = 4000
        
        low_samples = int(samplerate) // low_hz
        high_samples = int(samplerate) // high_hz
        spacings = torch.linspace(low_samples, high_samples, self.n_atoms)
        print('SMALLEST SPACING', low_samples, 'HIGHEST SPACING', high_samples)
        oversample_rate = 8
        
        if init_atoms is None:
            atoms = torch.zeros(self.n_atoms, self.resonance_size * oversample_rate)
            for i, spacing in enumerate(spacings):
                sp = int(spacing * oversample_rate)
                atoms[i, ::(sp + 1)] = 1
            
            atoms = F.avg_pool1d(atoms.view(1, 1, -1), kernel_size=oversample_rate, stride=oversample_rate).view(self.n_atoms, self.resonance_size)
            if learnable_atoms:
                self.atoms = nn.Parameter(atoms)
            else:
                self.register_buffer('atoms', atoms)
        else:
            if learnable_atoms:
                self.atoms = nn.Parameter(init_atoms)
            else:
                self.register_buffer('atoms', init_atoms)
        
        self.selections = nn.ModuleList([
            nn.Linear(latent_dim, self.n_atoms) 
            for _ in range(self.n_piecewise)
        ])
        
        self.decay = nn.Linear(latent_dim, self.n_frames)
        
        
        self.to_filter = ConvUpsample(
            latent_dim,
            channels,
            start_size=8,
            end_size=self.n_frames,
            mode='nearest',
            out_channels=self.coarse_coeffs,
            from_latent=True,
            batch_norm=False,
            weight_norm=True
        )
        
        self.to_mixture = ConvUpsample(
            latent_dim, 
            channels, 
            start_size=8, 
            end_size=self.n_frames, 
            mode='nearest', 
            out_channels=n_piecewise, 
            from_latent=True, 
            batch_norm=False,
            weight_norm=True
        )
        
        
        self.final_mix = nn.Linear(latent_dim, 2)
        
    
    def forward(self, latent, impulse):
        """
        Generate:
            - n selections
            - n decay exponents
            - n filters
            - time-based mixture
        """
        
        batch_size = latent.shape[0]
        
        
        # TODO: There should be another collection for just resonances
        convs = []
        
        imp = F.pad(impulse, (0, self.resonance_size - impulse.shape[-1]))
        
        
        decay = torch.sigmoid(self.decay(latent))
        decay = self.base_resonance + (decay * self.res_factor)
        decay = torch.log(1e-12 + decay)
        decay = torch.cumsum(decay, dim=-1)
        decay = torch.exp(decay)
        decay = F.interpolate(decay, size=self.resonance_size, mode='linear')
        

        # produce time-varying, frequency-domain filter coefficients        
        filt = self.to_filter(latent).view(-1, self.coarse_coeffs, self.n_frames).permute(0, 2, 1)
        filt = torch.sigmoid(filt)
        filt = F.interpolate(filt, size=257, mode='linear')
        filt = filt.view(batch_size, n_events, self.n_frames, 257)
        
        
        for i in range(self.n_piecewise):
            # choose a linear combination of resonances
            # and convolve the impulse with each
            
            sel = self.selections[i].forward(latent)
            sel = torch.softmax(sel, dim=-1)
            res = sel @ self.atoms
            res = res * decay
            conv = fft_convolve(res, imp)
            convs.append(conv[:, None, :, :])
            
        
        # TODO: Concatenate both the pure resonances and the convolved audio
        convs = torch.cat(convs, dim=1)
        
        # produce a linear mixture-over time
        mx = self.to_mixture(latent)
        mx = F.interpolate(mx, size=self.resonance_size, mode='linear')
        # mx = F.avg_pool1d(mx, 9, 1, 4)
        mx = torch.softmax(mx, dim=1)
        mx = mx.view(-1, n_events, self.n_piecewise, self.resonance_size).permute(0, 2, 1, 3)
        
        final_convs = (mx * convs).sum(dim=1)
        
        # apply time-varying filter
        # TODO: To avoid windowing artifacts, this is really just the 
        # same process again:  Convole the entire signal with N different
        # filters and the produce a smooth mixture over time
        windowed = windowed_audio(final_convs, 512, 256)
        windowed = unit_norm(windowed, dim=-1)
        windowed = torch.fft.rfft(windowed, dim=-1)
        windowed = windowed * filt
        windowed = torch.fft.irfft(windowed)
        final_convs = overlap_add(windowed, apply_window=False)[..., :self.resonance_size]\
            .view(-1, n_events, self.resonance_size)
        
        final_mx = self.final_mix(latent)
        final_mx = torch.softmax(final_mx, dim=-1)
        
        # final_convs = unit_norm(final_convs)
        # imp = unit_norm(imp)
        
        stacked = torch.cat([final_convs[..., None], imp[..., None]], dim=-1)
        
        final = stacked @ final_mx[..., None]
        final = final.view(-1, n_events, self.resonance_size)
    
        return final



def make_waves(n_samples, f0s, samplerate):
    sawtooths = []
    squares = []
    triangles = []
    sines = []
    
    total_atoms = len(f0s) * 4

    for f0 in f0s:
        f0 = f0 / (samplerate // 2)
        rps = f0 * np.pi
        radians = np.linspace(0, rps * n_samples, n_samples)
        sq = square(radians)[None, ...]
        squares.append(sq)
        st = sawtooth(radians)[None, ...]
        sawtooths.append(st)
        tri = sawtooth(radians, 0.5)[None, ...]
        triangles.append(tri)
        sin = np.sin(radians)
        sines.append(sin[None, ...])
    
    waves = np.concatenate([sawtooths, squares, triangles, sines], axis=0)
    waves = torch.from_numpy(waves).view(total_atoms, n_samples).float()
    return waves


class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels, mixer_channels=2):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        mixer_channels = mixer_channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=mixer_channels, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

    def forward(self, x):
        x = self.to_mix(x)
        x = x.view(-1, self.encoding_channels, 1)
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

        self.to_frames = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            mode='learned',
            end_size=self.n_frames,
            out_channels=channels,
            batch_norm=False,
            weight_norm=True
        )

        self.noise_model = NoiseModel(
            channels,
            self.n_frames,
            self.n_frames * 4,
            self.n_samples,
            self.channels,
            batch_norm=False,
            weight_norm=True,
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
        )
        
        self.to_env = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):
        
        env = self.to_env(x) ** 2
        env = F.interpolate(env, mode='linear', size=self.n_samples)
        
        x = self.to_frames(x)
        x = self.noise_model(x)
        x = x.view(-1, n_events, self.n_samples)
        
        x = x * env
        return x




class UNet(nn.Module):
    def __init__(self, channels, return_latent=False):
        super().__init__()
        self.channels = channels
        
        self.return_latent = return_latent
        
        if self.return_latent:
            self.to_latent = nn.Linear(channels * 4, channels)
        
        
        self.embed_spec = nn.Conv1d(1024, 1024, 1, 1, 0)
        self.pos = nn.Parameter(torch.zeros(1, 1024, 128).uniform_(-0.01, 0.01))
        

        self.down = nn.Sequential(
            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(channels)
            ),

            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(channels)
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(channels)
            ),

            # 4
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(channels)
            ),
        )

        self.up = nn.Sequential(
            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.ConvTranspose1d(channels, channels, 4, 2, 1)),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(channels)
            ),
            # 16
            nn.Sequential(
                weight_norm(nn.ConvTranspose1d(channels, channels, 4, 2, 1)),
            ),
        )
        
                
        self.proj = nn.Conv1d(channels, 256, 1, 1, 0)

        

    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}
        
        batch_size = x.shape[0]
        
        if x.shape[1] == 1:
            x = stft(x, 2048, 256, pad=True).view(
                batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
        
        x = self.embed_spec(x)
        x = x + self.pos
        
        batch_size = x.shape[0]

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x
        
        if self.return_latent:
            z = self.to_latent(x.view(-1, self.channels * 4))
        

        for layer in self.up:
            x = layer(x)
            size = x.shape[-1]
            if size in context:
                x = x + context[size]

        x = self.proj(x).permute(0, 2, 1)
        
        if self.return_latent:
            return x, z
        
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNet(1024, return_latent=True)
    

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)

        total_atoms = 2048
        f0s = np.linspace(40, 4000, total_atoms // 4)
        waves = make_waves(resonance_size, f0s, int(samplerate))
        
        self.res = ResonanceModel2(
            256, 
            128, 
            resonance_size, 
            n_atoms=total_atoms, 
            n_piecewise=4, 
            init_atoms=waves, 
            learnable_atoms=False, 
            mixture_over_time=True,
            n_frames=128)
        
        
        self.to_pos = nn.Linear(256, 1024)
        
        self.imp_gen = ImpulseGenerator(exp.n_samples, lambda x: gumbel_straight_through_estimator(x))
        self.to_amp = nn.Linear(256, 1)
        
        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))

        self.to_context_mean = nn.Linear(256, context_dim)
        self.to_context_std = nn.Linear(256, context_dim)
        

        self.from_context = nn.Linear(context_dim, 256)
        
        self.from_latent = nn.Linear(1024, context_dim)
        
        self.embbedding_bottleneck_down = nn.Linear(256, 16)
        
        
        self.embbedding_bottleneck_up = nn.Linear(16, 256)
        
        self.apply(lambda x: exp.init_weights(x))
        
        
        
    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded, z = self.encoder.forward(x)
        return encoded, z

    
    
    def generate(self, embeddings, dense):
        # impulses
        imp = self.imp.forward(embeddings)
        # resonances
        mixed = self.res.forward(embeddings, imp)
        mixed = mixed.view(-1, n_events, resonance_size)
        mixed = unit_norm(mixed)
        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps
        final = self.verb.forward(dense, mixed)
        return final

    def forward(self, x):
        embeddings, z = self.encode(x)
        dense = self.from_latent(z)
        b = self.embbedding_bottleneck_down(embeddings)
        embeddings = self.embbedding_bottleneck_up(b)
        final = self.generate(embeddings, dense)
        return final, embeddings

    
    
    

model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def test_conv_spec():
    
    frame = 32
    
    event = torch.hamming_window(512) * torch.zeros(512).uniform_(-1, 1)
    
    
    signal = torch.zeros(1, 1, n_samples)
    padding = torch.zeros_like(signal)
    
    signal[:, :, frame * 256: (frame * 256) + 512] = event
    
    padded_signal = torch.cat([padding, signal, padding])
    spec = transform(padded_signal)
    
    
    ch = torch.zeros_like(signal)
    ch[:, :, :512] = event
    padded_ch = torch.cat([ch, padding, padding])
    ch_spec = transform(padded_ch)
    
    start_norm = torch.norm(spec).item()
    
    amps, indices, residual = convolve_spectrograms(spec, ch_spec)
    
    residual_norm = torch.norm(residual).item()
    
    print(start_norm, residual_norm, indices)
    
    
    raise Exception('Done')
    


def convolve_spectrograms(batch: torch.Tensor, channels: torch.Tensor):
    """
    
    """
    
    batch = batch.reshape(-1, 128 * 3, 1025)
    channels = channels.reshape(*batch.shape)
    
    b, time, ch = batch.shape
    
    
    # give channels unit norm
    norm = torch.norm(channels, dim=(1, 2), keepdim=True)
    channels = channels / (norm + 1e-4)
    
    # convolve channels with batch
    batch_spec = torch.fft.rfft(batch, dim=1, norm='ortho')
    channels_spec = torch.fft.rfft(channels, dim=1, norm='ortho')
    
    conv = batch_spec * channels_spec
    fm = torch.fft.irfft(conv, dim=1, norm='ortho')
    
    
    # bias_start = torch.linspace(2, 1, steps=time // 3, device=batch.device)
    # bias_end = torch.linspace(1, 2, steps=time // 3, device=batch.device)
    # bias_mid = torch.ones_like(bias_start)
    
    # bias = torch.cat([bias_start, bias_mid, bias_end])
    
    # aggregate over all channels
    fm = fm.norm(dim=-1)
    
    # from matplotlib import pyplot as plt
    # plt.plot(fm.data.cpu().numpy().squeeze())
    # plt.savefig('view.png')
    # plt.clf()
    
    # biased = fm * bias[None, :]
    
    values, indices = torch.max(fm, dim=-1, keepdim=True)
    
    amps = torch.gather(fm, dim=-1, index=indices)
    
    residual = batch.clone()
    
    workspace = torch.zeros_like(residual)
    
    
    for i in range(b):
        index = indices[i]
        amp = fm[i, index: index + 1]
        size = workspace[i, index:, :].shape
        workspace[i, index:, :] = channels[i, :size[0], :] * amp
    
    residual = residual - workspace
    
    
    return amps, indices, residual


def transform(x: torch.Tensor):
    return stft(x, 2048, 256, pad=True).reshape(x.shape[0], -1, 1025)

def train(batch, i):
    optim.zero_grad()
    
    print('=========================================')

    batch_size = batch.shape[0]
    
    recon, embeddings = model.forward(batch)
    
    padding = torch.zeros_like(batch)
    padded = residual = torch.cat([padding, batch, padding], dim=-1)
    residual = transform(residual)
    
    start_norm = torch.norm(residual[:, 128:-128, :], dim=(1, 2))
    
    if (start_norm == 0).sum().item() > 0:
        print(f'WARNING, skpping batch {i} due to 0 norm')
        return None, None
    
    # choose a random order for the channels
    indices = np.random.permutation(n_events)
    
    output = torch.zeros_like(padded)
    
    loss = 0
    
    for i in indices:
        ch = recon[:, i: i + 1, :]
        ch_samples = unit_norm(ch)
        
        ch = torch.cat([ch, padding, padding], dim=-1)
        ch = transform(ch)
        
        start = residual
        amps, indices, residual = convolve_spectrograms(residual, ch)
        
        local_loss = -(start[:, 128:-128, :] - residual[:, 128:-128, :]).sum()
        loss = loss + local_loss
        
        sample_index = indices * 256
        
        
        for b in range(batch_size):
            si = sample_index[b]
            a = amps[b]
            
            
            # print(f'channel {i}, batch {b} index {si.item()} amp {a.item()}')
            
            size = output[b, :, si: si + n_samples].shape
            output[b, :, si: si + n_samples] = output[b, :, si: si + n_samples] + (ch_samples[b, :, :size[1]] * a)
    
    
    
    # print(f'start norm {start_norm.item()}, end_norm {loss.item()}')
        
    loss.backward()
    optim.step()

    
    recon = max_norm(output[:, :, n_samples:-n_samples])
    
    
    return loss, recon





@readme
class NeuralMatchingPursuit(BaseExperimentRunner):

    def __init__(self, stream, port=None, load_weights=True, save_weights=False, model=model):
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            load_weights=load_weights, 
            save_weights=save_weights, 
            model=model)

    def run(self):
        
        # test_conv_spec()
        
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, n_samples)
            l, r = train(item, i)
            
            if l is None:
                continue

            self.real = item
            self.fake = r
            
            print(i, l.item())
            self.after_training_iteration(l, i)
