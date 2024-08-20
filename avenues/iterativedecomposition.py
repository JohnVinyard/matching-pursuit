
import torch
from conjure import numpy_conjure, SupportedContentType

from torch import nn
from torch.nn import functional as F
from modules.anticausal import AntiCausalStack
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.impulse import GenerateImpulse
from modules.iterative import IterativeDecomposer
from modules.normalization import max_norm, unit_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify, sparsify_vectors
from modules.stft import stft
from modules.transfer import ResonanceChain, make_waves
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme
from torch.distributions import Normal
from torch.nn.utils.weight_norm import weight_norm



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




class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: replace this with anticausal analysis
        self.encoder = AntiCausalStack(
            1024, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1])
        
        # self.encoder = UNet(1024, return_latent=False, is_disc=False)
        
        self.to_event_vectors = nn.Conv1d(1024, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(1024, 1, 1, 1, 0)
        
    
        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)
        self.embed_latent = nn.Linear(1024, context_dim)
        

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)
        
        
        total_atoms = 4096
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_atoms // 4)
        waves = make_waves(resonance_size, f0s.tolist(), int(samplerate))
        
        self.res = ResonanceChain(
            1, 
            n_atoms=total_atoms, 
            window_size=512, 
            n_frames=256, 
            total_samples=resonance_size, 
            mix_channels=16, 
            channels=64, 
            latent_dim=256,
            initial=waves,
            learnable_resonances=False)
        
        
        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))


        self.from_context = nn.Linear(context_dim, 256)
        
        self.atom_bias = nn.Parameter(torch.zeros(4096).uniform_(-1, 1))

        # self.apply(lambda x: exp.init_weights(x))
        raise NotImplementedError('Initialize weights')
        

    def encode(self, x, n_events=1):
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
    



model = IterativeDecomposer(
    context_dim=context_dim,
    n_events=n_events,
    impulse_size=impulse_size,
    resonance_size=resonance_size,
    samplerate=samplerate,
    n_samples=n_samples
)
optim = optimizer(model, lr=1e-4)


                

# def transform(x: torch.Tensor):
#     batch_size, channels, _ = x.shape
#     bands = multiband_transform(x)
#     return torch.cat([b.reshape(batch_size, channels, -1) for b in bands.values()], dim=-1)

        
# def multiband_transform(x: torch.Tensor):
#     bands = fft_frequency_decompose(x, 512)
#     # TODO: each band should have 256 frequency bins and also 256 time bins
#     # this requires a window size of (n_samples // 256) * 2
#     # and a window size of 512, 256
    
#     # window_size = 512
    
    
#     d1 = {f'{k}_xl': stft(v, 512, 64, pad=True) for k, v in bands.items()}
#     d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
#     d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
#     d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
    
#     normal = stft(x, 2048, 256, pad=True).reshape(-1, 128, 1025).permute(0, 2, 1)
#     # pooled = F.avg_pool1d(normal, kernel_size=128, stride=1, padding=64)[..., :128]
#     # residual = torch.relu(normal - pooled)
    
#     return dict(**d1, **d3, **d4, normal=normal)
    
#     # mbt = multiscale_pif(x)
#     # return dict(mbt=mbt)


# def patches(spec: torch.Tensor, size: int = 27, step: int = 9):
#     batch, channels, time = spec.shape
    
#     p = spec.unfold(1, size, step).unfold(2, size, step)
#     last_dim = np.prod(p.shape[-2:])
#     p = p.reshape(batch, -1, last_dim)
#     norms = torch.norm(p, dim=-1, keepdim=True)
#     normed = p / (norms + 1e-12)
    
#     return p, norms, normed

def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor):
    
    target = transform(target).reshape(target.shape[0], -1)
    
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
    
    
    mask = torch.zeros(b, n_events, 1, device=batch.device).bernoulli_(p=0.5)
    for_disc = torch.sum(recon * mask, dim=1, keepdim=True).clone().detach()    
    j = disc.forward(for_disc)
    d_loss = torch.abs(1 - j).mean()
    scl = single_channel_loss_3(batch, recon) * 1e-4
    
    
    loss = scl + sparsity_loss + d_loss
        
    loss.backward()
    optim.step()
    
    
    disc_optim.zero_grad()
    
    rj = disc.forward(batch)
    fj = disc.forward(for_disc)
    disc_loss = (torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()) * 0.5
    disc_loss.backward()
    disc_optim.step()
    print('DISC', disc_loss.item())
    
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


    