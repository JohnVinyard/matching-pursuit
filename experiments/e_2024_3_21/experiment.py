
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType

from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.anticausal import AntiCausalStack
from modules.ddsp import AudioModel, NoiseModel
from modules.decompose import fft_frequency_decompose

from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify, sparsify_vectors
from modules.stft import stft
from modules.transfer import ResonanceChain
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme
from torch.distributions import Normal

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


def experiment_spectrogram(x: torch.Tensor):
    batch_size = x.shape[0]
    
    x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
    return x

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
        filt = filt.view(batch_size, -1, self.n_frames, 257)
        
        
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
        mx = mx.view(batch_size, -1, self.n_piecewise, self.resonance_size).permute(0, 2, 1, 3)
        
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
            .view(batch_size, -1, self.resonance_size)
        final_convs = unit_norm(final_convs)
        
        final_mx = self.final_mix(latent)
        final_mx = torch.softmax(final_mx, dim=-1)
        
        # final_convs = unit_norm(final_convs)
        # imp = unit_norm(imp)
        
        stacked = torch.cat([final_convs[..., None], imp[..., None]], dim=-1)
        
        final = stacked @ final_mx[..., None]
        final = final.view(batch_size, -1, self.resonance_size)
        
    
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
            # batch_norm=True
            weight_norm=True
        )

        self.noise_model = NoiseModel(
            channels,
            self.n_frames,
            self.n_frames * 4,
            self.n_samples,
            self.channels,
            # batch_norm=True,
            weight_norm=True,
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
        )
        
        self.to_env = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):
        batch_size = x.shape[0]
        
        env = self.to_env(x) ** 2
        env = F.interpolate(env, mode='linear', size=self.n_samples)
        
        x = self.to_frames(x)
        x = self.noise_model(x)
        x = x.view(batch_size, -1, self.n_samples)
        
        x = x * env
        return x



class SimpleGenerateImpulse(nn.Module):
    
    def __init__(self, latent_dim, channels, n_samples, n_filter_bands, encoding_channels):
        super().__init__()
        
        self.n_samples = n_samples
        
        self.filter_size = 64
        
        self.to_envelope = LinearOutputStack(
            channels, layers=3, out_channels=self.n_samples // 128, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        
        self.to_filt = LinearOutputStack(
            channels, layers=3, out_channels=self.filter_size, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        env = self.to_envelope(x)
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        
        # TODO: consider making this a hard choice via gumbel softmax as well
        env = torch.abs(env).view(x.shape[0], -1, self.n_samples)
        
        filt = self.to_filt(x).view(x.shape[0], -1, self.filter_size)
        
        noise = torch.zeros(x.shape[0], 1, self.n_samples, device=x.device).uniform_(-1, 1)
        
        noise = noise * env
        
        filt = F.pad(filt, (0, self.n_samples - self.filter_size))
        
        final = fft_convolve(noise, filt)
        return final



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AntiCausalStack(1024, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1])
        
        self.to_event_vectors = nn.Conv1d(1024, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(1024, 1, 1, 1, 0)
        
    
        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)
        self.embed_latent = nn.Linear(1024, context_dim)
        

        # self.imp = SimpleGenerateImpulse(256, 128, impulse_size, 16, n_events)
        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)
        
        # self.events = ConvUpsample(256, 128, start_size=8, end_size=exp.n_samples, mode='learned', out_channels=1, from_latent=True, batch_norm=True)
        
        # total_atoms = 4096
        # f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_atoms // 4)
        # waves = make_waves(resonance_size, f0s, int(samplerate))
        
        # self.res = ResonanceModel2(
        #     256, 
        #     128, 
        #     resonance_size, 
        #     n_atoms=total_atoms, 
        #     n_piecewise=8, 
        #     init_atoms=waves, 
        #     learnable_atoms=False, 
        #     mixture_over_time=True,
        #     n_frames=128)
        
        total_atoms = 4096
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_atoms // 4)
        waves = make_waves(resonance_size, f0s, int(samplerate))
        
        self.res = ResonanceChain(
            2, 
            n_atoms=total_atoms, 
            window_size=512, 
            n_frames=128, 
            total_samples=resonance_size, 
            mix_channels=8, 
            channels=64, 
            latent_dim=256,
            initial=waves,
            learnable_resonances=False)
        
        
        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))


        self.from_context = nn.Linear(context_dim, 256)
        
        self.atom_bias = nn.Parameter(torch.zeros(4096).uniform_(-1, 1))

        self.apply(lambda x: exp.init_weights(x))
        

    def encode(self, x, n_events=n_events):
        batch_size = x.shape[0]

        if x.shape[1] == 1:
            x = experiment_spectrogram(x)

        
        encoded = self.encoder.forward(x)
        
        
        event_vecs = self.to_event_vectors(encoded).permute(0, 2, 1) # batch, time, channels
        
        event_switch = self.to_event_switch(encoded)
        attn = torch.relu(event_switch).permute(0, 2, 1).view(batch_size, 1, -1)
        
        attn, attn_indices, values = sparsify(attn, n_to_keep=n_events, return_indices=True)
        
        
        vecs, indices = sparsify_vectors(event_vecs.permute(0, 2, 1), attn, n_to_keep=n_events)
        
        
        scheduling = torch.zeros(batch_size, n_events, encoded.shape[-1], device=encoded.device)
        for b in range(batch_size):
            for j in range(n_events):
                index = indices[b, j]
                scheduling[b, j, index] = attn[b, 0][index]
                
        
        return vecs, scheduling

    
    def generate(self, vecs, scheduling):
        
        batch_size = vecs.shape[0]
        
        
        embeddings = self.from_context(vecs)
        
        amps = torch.sum(scheduling, dim=-1, keepdim=True)
        

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        imp = unit_norm(imp)

        # # resonances
        mixed = self.res.forward(embeddings, imp)
        mixed = mixed.view(batch_size, -1, resonance_size)
        
        # mixed = self.events(embeddings)
        # imp = None
        
        mixed = unit_norm(mixed)
        

        mixed = mixed * amps
        
        # coarse positioning
        final = F.pad(mixed, (0, n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], final.shape[1], n_samples, device=final.device)
        up[:, :, ::256] = scheduling
        
        final = fft_convolve(final, up)[..., :n_samples]

        final = self.verb.forward(unit_norm(vecs, dim=-1), final)
        

        return final, imp, amps, mixed
    
    def iterative(self, x):
        channels = []
        schedules = []
        vecs = []
        
        spec = experiment_spectrogram(x)
        
        for i in range(n_events):
            v, sched = self.encode(spec, n_events=1)
            vecs.append(v)
            schedules.append(sched)
            ch, _, _, _ = self.generate(v, sched)
            current = experiment_spectrogram(ch)
            spec = (spec - current).clone().detach()
            channels.append(ch)
    
        channels = torch.cat(channels, dim=1)        
        vecs = torch.cat(vecs, dim=1)        
        schedules = torch.cat(schedules, dim=1)        
        
        return channels, vecs, schedules
            
    

    def forward(self, x, random_timings=False, random_events=False, return_context=True):
        
        batch_size = x.shape[0]
        
        channels, vecs, scheduling = self.iterative(x)
        
        if random_events:
            means = torch.mean(vecs, dim=(0, 1))
            stds = torch.std(vecs, dim=(0, 1)) + 1e-4
            dist = Normal(means, stds)
            vecs = dist.sample((batch_size, n_events))
        
        if random_timings:
            orig_shape = scheduling.shape
            scheduling = torch.zeros_like(scheduling).view(batch_size * n_events, 1, -1).uniform_(scheduling.min(), scheduling.max())
            scheduling, indices, values = sparsify(scheduling, n_to_keep=1, return_indices=True)
            scheduling = scheduling.view(*orig_shape)
        
        
        final, imp, amps, mixed = self.generate(vecs, scheduling)
        
        
        if return_context:
            return final, vecs, imp, scheduling, amps, mixed
        else:
            return final, vecs, imp, scheduling, amps
    
    

model = Model().to(device)
optim = optimizer(model, lr=1e-4)



def transform(x: torch.Tensor):
    batch_size, channels, _ = x.shape
    bands = multiband_transform(x)
    return torch.cat([b.view(batch_size, channels, -1) for b in bands.values()], dim=-1)

        
def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    # TODO: each band should have 256 frequency bins and also 256 time bins
    # this requires a window size of (n_samples // 256) * 2
    # and a window size of 512, 256
    
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
    return dict(**d1, **d3, **d4, normal=stft(x, 2048, 256, pad=True))


# def patches(spec: torch.Tensor, size: int = 9, step: int = 3):
#     batch, channels, time = spec.shape
    
#     p = spec.unfold(1, size, step).unfold(2, size, step)
#     last_dim = np.prod(p.shape[-2:])
#     p = p.reshape(batch, -1, last_dim)
#     norms = torch.norm(p, dim=-1, keepdim=True)
#     normed = p / (norms + 1e-12)
    
#     return p, norms, normed

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



def train(batch, i):
    optim.zero_grad()
    
    b = batch.shape[0]
    
    recon, encoded, scheduling = model.iterative(batch)
    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    sparsity_loss = torch.abs(encoded).sum() * 1e-3
    
    
    scl = single_channel_loss_3(batch, recon) * 1e-4
    
    # _, _, fake_patches = patches(stft(recon_summed, 2048, 256, pad=True).view(-1, 128, 1025), 9, 3)
    # _, _, real_patches = patches(stft(batch, 2048, 256, pad=True).view(-1, 128, 1025), 9, 3)
    
    # patch_loss = F.mse_loss(fake_patches, real_patches) * 100
    
    
    loss = scl + sparsity_loss
        
    loss.backward()
    optim.step()
    
    
    
    recon = max_norm(recon_summed)
    encoded = max_norm(encoded)
    
    return loss, recon, encoded, scheduling


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)

def make_sched_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def events(x: torch.Tensor):
        x = torch.sum(x, dim=1)
        x = x.data.cpu().numpy().squeeze()
        return x

    return (events,)


@readme
class IterativeDecomposition8EventGenerator(BaseExperimentRunner):
    encoded = MonitoredValueDescriptor(make_conjure)
    sched = MonitoredValueDescriptor(make_sched_conjure)

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
            l, r, e, s = train(item, i)

            self.real = item
            self.fake = r
            self.encoded = e
            self.sched = s
            
            print(i, l.item())
            self.after_training_iteration(l, i)


    