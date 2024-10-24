
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
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.phase import morlet_filter_bank
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify, sparsify_vectors
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
    weight_init=0.05,
    model_dim=256,
    kernel_size=512)


import zounds
band = zounds.FrequencyBand(20, 10000)
scale = zounds.MelScale(band, 128)

fb = morlet_filter_bank(exp.samplerate, 256, scale, 0.1, normalize=True).real
fb = torch.from_numpy(fb).to(device)
fb = fb.view(1, 128, 256)
fb = F.pad(fb, (0, exp.n_samples - 256))



def pif(signal: torch.Tensor):
    signal = signal.view(-1, 1, exp.n_samples)
    spec = fft_convolve(signal, fb)
    spec = spec.view(signal.shape[0], -1, exp.n_samples)
    spec = F.pad(spec, (0, 256))
    windowed = spec.unfold(-1, 512, 256)
    ws = torch.abs(torch.fft.rfft(windowed))
    return ws
    

def transform(signal: torch.Tensor):
    return pif(signal)                


def experiment_spectrogram(signal: torch.Tensor):
    return pif(signal)

n_events = 16
context_dim = 16
impulse_size = 16384
resonance_size = 32768
samplerate = 22050
n_samples = 32768



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

        self.embed_periodicity = nn.Linear(257, 8)
        self.encoder = AntiCausalStack(1024, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1], do_norm=True)
        
        
        self.to_event_vectors = nn.Conv1d(1024, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(1024, 1, 1, 1, 0)
        
    
        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)
        self.embed_latent = nn.Linear(1024, context_dim)
        

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)
        
        
        total_atoms = 4096
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_atoms // 4)
        waves = make_waves(resonance_size, f0s, int(samplerate))
        
        self.res = ResonanceChain(
            1, 
            n_atoms=total_atoms, 
            window_size=512, 
            n_frames=256, 
            total_samples=resonance_size, 
            mix_channels=4, 
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
            print(x.shape)
        
        x = self.embed_periodicity(x)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, 1024, -1)

        
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
        # (batch, 1, 16)
        # (batch, n_events, 16)
        
        batch_size = vecs.shape[0]
        
        
        embeddings = self.from_context(vecs)
        
        
        amps = torch.sum(scheduling, dim=-1, keepdim=True)
        

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        imp = unit_norm(imp)

        # # resonances
        mixed = self.res.forward(embeddings, imp)
        # print(f'\t {imp.shape} {embeddings.shape} {mixed.shape}')
        mixed = mixed.view(batch_size, -1, resonance_size)
        
        # mixed = self.events(embeddings)
        # imp = None
        
        mixed = unit_norm(mixed)
        

        mixed = mixed * amps
        
        # coarse positioning
        final = F.pad(mixed, (0, n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], final.shape[1], n_samples, device=final.device)
        up[:, :, ::256] = scheduling
        
        # print(f'\t FINAL {final.shape} {up.shape}')
        
        final = fft_convolve(final, up)[..., :n_samples]
        
        # print(f'\t FINAL 2 {final.shape}')

        final = self.verb.forward(unit_norm(vecs, dim=-1), final)
        # print(f'\t WITH_VERB {final.shape}')
        

        return final, imp, amps, mixed
    
    def iterative(self, x):
        channels = []
        schedules = []
        vecs = []
        
        spec = experiment_spectrogram(x)
        
        for i in range(n_events):
            print(i, spec.shape)
            v, sched = self.encode(spec, n_events=1)
            vecs.append(v)
            schedules.append(sched)
            # print(f'In iterate {i}, calling generate() with {v.shape} and {sched.shape}')
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
        
        
        print(f'In forward, calling generate with {vecs.shape} and {scheduling.shape}')
        final, imp, amps, mixed = self.generate(vecs, scheduling)
        
        if not random_events and not random_timings and return_context:
            # Note that here we're returning the audio channels from 
            # the iterative process and not those from the all-at-once process
            return channels, vecs, imp, scheduling, amps, mixed
        
        if return_context:
            return final, vecs, imp, scheduling, amps, mixed
        else:
            # return channels, vecs, imp, scheduling, amps
            raise NotImplementedError('This code path is no longer supported')
    


model = Model().to(device)
optim = optimizer(model, lr=1e-3)




def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor):
    
    target = transform(target).reshape(target.shape[0], -1)
    
    channels = transform(recon.view(-1, 1, exp.n_samples)).reshape(target.shape[0], n_events, -1)
    
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
    
    
    loss = scl + sparsity_loss #+ d_loss
        
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
class IterativeDecomposition9EventGenerator(BaseExperimentRunner):
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


    