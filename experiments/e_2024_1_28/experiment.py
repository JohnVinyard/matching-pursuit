
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType

from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.decompose import fft_frequency_decompose
from modules.mixer import MixerStack

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
    # return F.gumbel_softmax(x, tau=1, hard=False)
    return sparse_softmax(x, normalize=True)


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
        
        
        # self.register_buffer('env', torch.linspace(1, 0, self.resonance_size))
        # self.max_exp = 20
        
    
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
    
        # self.embed_context = nn.Linear(4096, 256)
        # self.embed_one_hot = nn.Linear(4096, 256)

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
        
        
        # total_atoms = 1024
        # f0s = np.linspace(40, 4000, total_atoms // 4)
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
        
        


    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded, z = self.encoder.forward(x)
        return encoded, z

    def from_sparse(self, sparse, ctxt):
        encoded, packed, one_hot = sparsify2(sparse, n_to_keep=n_events)
        x, imp = self.generate(encoded, one_hot, packed, ctxt)
        return x, imp, encoded
    
    
    def generate(self, embeddings, dense, hard=False, random_timings=False):
        
        # generate...
        
        # impulses
        imp = self.imp.forward(embeddings)
        # padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        mixed = self.res.forward(embeddings, imp)
        mixed = mixed.view(-1, n_events, resonance_size)
        mixed = unit_norm(mixed)

        amps = torch.abs(self.to_amp(embeddings))
        
        mixed = mixed * amps

        pos = self.to_pos.forward(embeddings)
        
        if random_timings:
            pos = torch.zeros_like(pos).uniform_(0, 1)
        
        loc, logits = self.imp_gen.forward(
            pos.view(-1, 1024), 
            return_logits=True)
        
        loc = loc.view(-1, n_events, exp.n_samples)
        final = fft_convolve(mixed, loc)

        final = self.verb.forward(dense, final)

        return final, embeddings, logits, amps.squeeze()

    def forward(self, x, hard=False, random_events=False, random_context=False, random_timings=False):
        
        
        embeddings, z = self.encode(x)
        dense = self.from_latent(z)
        if random_context:
            means = torch.mean(dense, dim=-1)
            stds = torch.std(dense, dim=-1)
            norm = Normal(means, stds)
            dense = norm.sample((x.shape[0],))
        
        b = self.embbedding_bottleneck_down(embeddings)
        
        if random_events:
            means = torch.mean(b, dim=(0, 1))
            stds = torch.std(b, dim=(0, 1))
            norm = Normal(means, stds)
            print(means.shape, stds.shape)
            b = norm.sample((x.shape[0], n_events))
            print(b.shape)
            
        
        embeddings = self.embbedding_bottleneck_up(b)
        
        # dense = torch.mean(embeddings, dim=1)
        # mean = self.to_context_mean(dense)
        # std = self.to_context_std(dense)
        # dense = mean + (torch.zeros_like(mean).normal_(0, 1) * std)
        
        final, imp, logits, amps = self.generate(embeddings, dense, hard=hard, random_timings=random_timings)
        
        
        return final, b, logits, amps
    
    # def random_generation(self, exponent=2):
    #     with torch.no_grad():
    #         # generate context latent
    #         z = torch.zeros(1, context_dim).normal_(0, 0.2)
            
    #         # generate events
    #         events = torch.zeros(1, 1, 4096, 128).uniform_(0, 3)
    #         events = F.avg_pool2d(events, (7, 7), (1, 1), (3, 3))
    #         events = events.view(1, 4096, 128)
            
    #         ch, _, encoded = self.from_sparse(events, z)
    #         ch = torch.sum(ch, dim=1, keepdim=True)
    #         ch = max_norm(ch)
        
    #     return ch, encoded
    
    # def derive_events_and_context(self, x: torch.Tensor):
        
    #     print('STARTING WITH', x.shape)
        
    #     encoded = self.encode(x)

    #     dense = torch.mean(encoded, dim=-1)
    #     mean = self.to_context_mean(dense)
    #     std = self.to_context_std(dense)
    #     dense = mean + (torch.zeros_like(mean).normal_(0, 1) * std)
        
        
    #     encoded = torch.relu(encoded)
        
    #     encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        
        
    #     return encoded, dense
    

model = Model().to(device)
optim = optimizer(model, lr=1e-3)




def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d2 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    return dict(**d1, **d2)
    
    
def transform(x: torch.Tensor):
    batch, channels, time = x.shape
    d = multiband_transform(x)
    feat = torch.cat([z.view(batch, channels, -1) for z in d.values()], dim=-1)
    return feat


def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor):
    
    target = transform(target).view(target.shape[0], -1)
    
    # full = torch.sum(recon, dim=1, keepdim=True)
    # full = transform(full).view(*target.shape)
    
    channels = transform(recon)
    
    residual = target
    
    # Try L1 norm instead of L@
    # Try choosing based on loudest patch/segment
    
    # sort channels from loudest to softest
    diff = torch.norm(channels, dim=(-1), p = 1)
    indices = torch.argsort(diff, dim=-1, descending=True)
    
    srt = torch.take_along_dim(channels, indices[:, :, None], dim=1)
    
    loss = 0
    for i in range(n_events):
        current = srt[:, i, :]
        start_norm = torch.norm(residual, dim=-1, p=1)
        # TODO: should the residual be cloned and detached each time,
        # so channels are optimized independently?
        residual = residual - current
        end_norm = torch.norm(residual, dim=-1, p=1)
        diff = -(start_norm - end_norm)
        loss = loss + diff.sum()
        
    
    return loss


# def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor):
#     target = transform(target)
#     full = torch.sum(recon, dim=1, keepdim=True)
#     full = transform(full)
    
#     channels = transform(recon)
    
#     residual = target - full
    
    
#     # find the channel closest to the residual
#     diff = torch.norm(channels - residual, dim=(-1))
#     indices = torch.argsort(diff, dim=-1)
    
#     srt = torch.take_along_dim(channels, indices[:, :, None], dim=1)
    
#     channel = 0
#     chosen = srt[:, channel, :]
    
#     # add the chosen channel back to the residual
#     with_residual = chosen + residual.view(*chosen.shape)
    
#     # make the chosen channel explain more of the residual    
#     loss = torch.abs(chosen - with_residual).sum()
    
#     return loss

    

def train(batch, i):
    optim.zero_grad()

    # mask = torch.zeros(batch.shape[0], n_events, 1, device=batch.device).bernoulli_(p=0.5)
    
    recon, encoded, logits, amps = model.forward(
        batch, 
        hard=True, 
        random_events=False, 
        random_context=False,
        random_timings=False)
    
    logits = logits.view(-1, n_events, 1024)
    
    values, indices = torch.max(logits, dim=-1)
    # recon = recon * mask
    
        
    print(indices.squeeze())
    
    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    
    loss = single_channel_loss_3(batch, recon) * 1e-5
    
        
    loss.backward()
    optim.step()
    
    
    recon = max_norm(recon_summed)
    # encoded = max_norm(encoded)
    return loss, recon, encoded, logits, amps


def make_conjure(experiment: BaseExperimentRunner):
    
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x - x.min()
        x = x / (x.max() + 1e-5)
        x = x.data.cpu().numpy()
        return x[0]

    return (encoded,)

def make_conjure_amps(experiment: BaseExperimentRunner):
    
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def amps(x: torch.Tensor):
        x = x - x.min()
        x = x / (x.max() + 1e-5)
        x = x.data.cpu().numpy()
        x = np.sort(x, axis=-1)
        return x

    return (amps,)


def make_conjure_logits(experiment: BaseExperimentRunner):
    
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def logits(x: torch.Tensor):
        x = x.view(-1, n_events, 1024)[0]
        x = x.data.cpu().numpy()
        indices = np.argmax(x, axis=-1)
        maxes = np.max(x, axis=-1)
        return x

    return (logits,)


@readme
class NoSparsity(BaseExperimentRunner):
    encoded = MonitoredValueDescriptor(make_conjure)
    logits = MonitoredValueDescriptor(make_conjure_logits)
    amps = MonitoredValueDescriptor(make_conjure_amps)

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
            l, r, e, logits, amps = train(item, i)

            self.real = item
            self.fake = r
            self.encoded = e
            self.logits = logits
            self.amps = amps
            
            print(i, l.item())
            self.after_training_iteration(l, i)
