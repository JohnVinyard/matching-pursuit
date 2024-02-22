
from typing import Tuple
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType

from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.decompose import fft_frequency_decompose
from modules.infoloss import MultiBandSpectralInfoLoss, MultiWindowSpectralInfoLoss, PatchBandSpec, SpectralInfoLoss

from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme
from torch.nn.utils.weight_norm import weight_norm

exp = Experiment(
    samplerate=22050,
    n_samples=2 ** 15,
    weight_init=0.02,
    model_dim=256,
    kernel_size=512)

n_events = 16
context_dim = 16
impulse_size = 16384
resonance_size = 32768
samplerate = 22050
n_samples = 32768

def anticausal_inhibition(x: torch.Tensor, inhibitory_area: Tuple[int, int]):
    batch, channels, time = x.shape
    width, height = inhibitory_area
    x = x[:, None, :, :]
    
    # give 0 mean
    mean = F.avg_pool2d(x, inhibitory_area, stride=(1, 1), padding=(width // 2, height // 2))
    x = x - mean
    
    new_mean = F.avg_pool2d(x, inhibitory_area, stride=(1, 1), padding=(width // 2, height // 2))
    
    # give unit std deviation
    deviations = (x - new_mean) ** 2
    std_deviations = F.avg_pool2d(deviations, inhibitory_area, stride=(1, 1), padding=(width // 2, height // 2))
    std_deviations = torch.sqrt(std_deviations)
    
    x = x / (std_deviations + 1e-4)
    
    return x.view(batch, channels, time)


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
            layer_norm=False,
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
            layer_norm=False,
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
            sel = torch.relu(sel)
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
        final_convs = unit_norm(final_convs)
        
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

class SimpleGenerateImpulse(nn.Module):
    
    def __init__(self, latent_dim, channels, n_samples, n_filter_bands, encoding_channels):
        super().__init__()
        
        self.n_samples = n_samples
        
        self.filter_size = 64
        
        self.to_envelope = LinearOutputStack(channels, layers=3, out_channels=self.n_samples // 128, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        
        self.to_filt = LinearOutputStack(channels, layers=3, out_channels=self.filter_size, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        env = self.to_envelope(x)
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        
        # TODO: consider making this a hard choice via gumbel softmax as well
        env = torch.abs(env).view(-1, n_events, self.n_samples)
        
        filt = self.to_filt(x).view(-1, n_events, self.filter_size)
        
        noise = torch.zeros(x.shape[0], n_events, self.n_samples, device=x.device).uniform_(-1, 1)
        
        noise = noise * env
        
        filt = F.pad(filt, (0, self.n_samples - self.filter_size))
        
        final = fft_convolve(noise, filt)
        return final


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
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
            ),

            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
            ),

            # 4
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
                nn.LeakyReLU(0.2),
            ),
        )

        self.up = nn.Sequential(
            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.ConvTranspose1d(channels, channels, 4, 2, 1)),
                nn.LeakyReLU(0.2),
            ),
            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.ConvTranspose1d(channels, channels, 4, 2, 1)),
                nn.LeakyReLU(0.2),
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.ConvTranspose1d(channels, channels, 4, 2, 1)),
                nn.LeakyReLU(0.2),
            ),

            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.ConvTranspose1d(channels, channels, 4, 2, 1)),
                nn.LeakyReLU(0.2),
            ),

            # 128
            nn.Sequential(
                nn.Dropout(0.1),
                weight_norm(nn.ConvTranspose1d(channels, channels, 4, 2, 1)),
                nn.LeakyReLU(0.2),
            ),
        )
        
        self.bias = nn.Conv1d(1024, 4096, 1, 1, 0)
        self.proj = nn.Conv1d(1024, 4096, 1, 1, 0)
        

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

        b = self.bias(x)
        x = self.proj(x)
        x = x - b
                
        if self.return_latent:
            return x, z
        
        return x




class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNet(1024, return_latent=True)
    
        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)
        self.down = nn.Linear(512, 256)
        self.embed_latent = nn.Linear(1024, context_dim)

        self.imp = SimpleGenerateImpulse(256, 128, impulse_size, 16, n_events)

        
        total_atoms = 2048
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_atoms // 4)
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
        
        
        
        # total_atoms = 1024
        # f0s = musical_scale_hz(40, 4000, total_atoms // 4)
        # waves = make_waves(resonance_size, f0s, int(samplerate))
        
        # self.res = ResonanceChain(
        #     2, 
        #     n_atoms=1024, 
        #     window_size=512, 
        #     n_frames=128, 
        #     total_samples=resonance_size, 
        #     mix_channels=4, 
        #     channels=128, 
        #     latent_dim=256,
        #     initial=waves,
        #     learnable_resonances=False)
        

        # self.mix = GenerateMix(256, 128, n_events, mixer_channels=3)
        self.to_amp = nn.Linear(256, 1)

        # self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))

        self.to_context_mean = nn.Linear(4096, context_dim)
        self.to_context_std = nn.Linear(4096, context_dim)
        
        self.embed_memory_context = nn.Linear(4096, context_dim)

        self.from_context = nn.Linear(context_dim, 256)
        
        # self.fine_shift = nn.Linear(256, 1)
        # self.shift_factor = (256 / resonance_size) * 0.5
        
        

        self.atom_bias = nn.Parameter(torch.zeros(4096).uniform_(-1, 1))

        self.apply(lambda x: exp.init_weights(x))
        

    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        
        encoded, z = self.encoder.forward(x)
        encoded = F.dropout(encoded, p=0.05, training=self.training)
        encoded = anticausal_inhibition(encoded, (9, 9))
        encoded = torch.relu(encoded)
        
        
        return encoded, z

    def from_sparse(self, sparse, ctxt):
        encoded, packed, one_hot = sparsify2(sparse, n_to_keep=n_events)
        x, imp = self.generate(encoded, one_hot, packed, ctxt)
        return x, imp, encoded
    
    
    def generate(self, encoded, one_hot, packed, dense):
        
        # one hot is n_events vectors
        proj = self.from_context(dense).view(-1, 1, 256).repeat(1, n_events, 1)
        
        # amps, amp_indices = torch.max(one_hot, dim=-1, keepdim=True)
        
        oh = self.embed_one_hot(one_hot)
        
        # allow amps to be exactly 0
        amps = torch.relu(self.to_amp(oh))
        
        # oh = unit_norm(oh, dim=-1)
        # proj = unit_norm(proj, dim=-1)
        # embeddings = unit_norm(proj + oh, dim=-1)
        embeddings = oh
        
        embeddings = torch.cat([proj, oh], dim=-1)
        embeddings = self.down(embeddings)

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        imp = unit_norm(imp)
        # padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        mixed = self.res.forward(embeddings, imp)
        mixed = mixed.view(-1, n_events, resonance_size)
        mixed = unit_norm(mixed)
        

        mixed = mixed * amps
        # mixed = unit_norm(mixed)
        
        # shift = torch.tanh(self.fine_shift(embeddings)) * self.shift_factor

        # coarse positioning
        final = F.pad(mixed, (0, n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, n_samples, device=final.device)
        up[:, :, ::256] = packed
        
        # fine positioning
        # up = fft_shift(up, shift)

        final = fft_convolve(final, up)[..., :n_samples]

        final = self.verb.forward(unit_norm(dense, dim=-1), final)

        return final, imp

    def forward(self, x):
        
        encoded, z = self.encode(x)
        
        # dense = torch.mean(encoded, dim=-1)
        # mean = self.to_context_mean(dense)
        # std = self.to_context_std(dense)
        # dense = mean + (torch.zeros_like(mean).normal_(0, 1) * std)
        
        dense = self.embed_latent(z)
        
        # encoded = encoded + self.atom_bias[None, :, None]
        # encoded = torch.relu(encoded)

        # note that some of the "top" channels will be zero, the eventual
        # scheduling convolution will make the event silent        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        
        final, imp = self.generate(encoded, one_hot, packed, dense)
        
        # print('ENCODED', encoded.max().item())
        # print('DENSE', dense.mean().item(), dense.std().item())
        
        
        return final, encoded, imp
    
    def random_generation(self, exponent=2):
        with torch.no_grad():
            # generate context latent
            z = torch.zeros(1, context_dim).normal_(0, 0.2)
            
            # generate events
            events = torch.zeros(1, 1, 4096, 128).uniform_(0, 3)
            events = F.avg_pool2d(events, (7, 7), (1, 1), (3, 3))
            events = events.view(1, 4096, 128)
            
            ch, _, encoded = self.from_sparse(events, z)
            ch = torch.sum(ch, dim=1, keepdim=True)
            ch = max_norm(ch)
        
        return ch, encoded
    
    def derive_events_and_context(self, x: torch.Tensor):
        
        print('STARTING WITH', x.shape)
        
        encoded, z = self.encode(x)

        
        dense = self.embed_latent(z)
        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        
        
        return encoded, dense
    

model = Model().to(device)
optim = optimizer(model, lr=1e-4)


# loss_model = SpectralInfoLoss(
#     stft_window_size=2048, 
#     stft_step_size=256, 
#     patch_size=(16, 16), 
#     patch_step=(8, 8), 
#     embedding_channels=32, 
#     n_centroids=1024).to(device)

# loss_model = MultiBandSpectralInfoLoss([
#     PatchBandSpec(512, 64, 16, (4, 4), (2, 2), 256),
#     PatchBandSpec(1024, 128, 32, (4, 4), (2, 2), 256),
#     PatchBandSpec(2048, 256, 64, (8, 8), (4, 4), 256),
#     PatchBandSpec(2048, 256, 64, (8, 8), (4, 4), 256),
#     PatchBandSpec(4096, 512, 128, (8, 8), (4, 4), 256),
#     PatchBandSpec(8192, 1024, 256, (8, 8), (4, 4), 256),
#     PatchBandSpec(16384, 1024, 256, (16, 16), (8, 8), 256),
#     PatchBandSpec(32768, 2048, 256, (16, 16), (8, 8), 256),
# ]).to(device)

loss_model = MultiWindowSpectralInfoLoss([
    [(16, 16), (8, 8)],
    [(3, 128), (1, 32)],
    [(64, 3), (16, 1)],
]).to(device)
loss_optim = optimizer(loss_model, lr=1e-4)



def train(batch, i):
    optim.zero_grad()
    loss_optim.zero_grad()
    
    # train loss
    spec_recon, orig_normed = loss_model.forward(batch)
    loss_loss = F.mse_loss(spec_recon, orig_normed.detach())
    loss_loss.backward()
    loss_optim.step()
    print('LOSS MODEL', loss_loss.item())
    
    

    b = batch.shape[0]
    
    recon, encoded, imp = model.forward(batch)  
    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    
    loss = loss_model.loss(batch, recon_summed)
    loss.backward()
    optim.step()
    
    
    # with torch.no_grad():
        
    #     # switch to cpu evaluation
    #     model.to('cpu')
    #     model.eval()
        
    #     # generate context latent
    #     z = torch.zeros(1, context_dim).normal_(0, 12)
        
    #     # generate events
    #     events = torch.zeros(1, 4096, 128).uniform_(0, 8.7)
    #     # events = F.avg_pool2d(events, (7, 7), (1, 1), (3, 3))
    #     # events = events.view(1, 4096, 128)
        
    #     ch, _, encoded = model.from_sparse(events, z)
    #     ch = torch.sum(ch, dim=1, keepdim=True)
        
    #     # switch back to GPU training
    #     model.to('cuda')
    #     model.train()
    
    # recon = max_norm(ch)
    
    recon = max_norm(recon_summed)
    encoded = max_norm(encoded)
    
    return loss, recon, encoded


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
class SpecInfoLoss(BaseExperimentRunner):
    encoded = MonitoredValueDescriptor(make_conjure)

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
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, n_samples)
            l, r, e = train(item, i)

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l, i)


    