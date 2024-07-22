
from typing import Tuple
import torch

from torch import nn
from config.experiment import Experiment
from modules.anticausal import AntiCausalStack
from modules.decompose import fft_frequency_decompose

from modules.linear import LinearOutputStack
from modules.normalization import max_norm
from modules.reds import RedsLikeModel
from modules.sparse import sparsify, sparsify_vectors
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=22050,
    n_samples=2 ** 15,
    weight_init=0.02,
    model_dim=256,
    kernel_size=512)

n_events = 16
context_dim = 64
n_octaves = 64

impulse_size = 16384
# resonance_size = 32768
samplerate = 22050
n_samples = 32768


def experiment_spectrogram(x: torch.Tensor):
    batch_size = x.shape[0]
    
    x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
    return x

class Model(nn.Module):
    def __init__(self, use_wavetables: bool = False):
        super().__init__()
        self.encoder = AntiCausalStack(1024, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1])
        self.to_event_vectors = nn.Conv1d(1024, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(1024, 1, 1, 1, 0)
        self.use_wavetables = use_wavetables
        
        self.reds = RedsLikeModel(
            n_resonance_octaves=n_octaves, 
            n_samples=exp.n_samples, 
            samplerate=exp.samplerate,
            use_wavetables=True)
        
        
        self.to_mix = nn.Linear(context_dim, 2)
        
        self.resonance_choice = nn.Linear(context_dim, 4096)
        self.to_f0 = nn.Linear(context_dim, 1)
        self.decay_choice = nn.Linear(context_dim, 1)
        self.freq_spacing = nn.Linear(context_dim, 1)
        
        self.noise_filter = nn.Linear(context_dim, 2)
        self.filter_decays = nn.Linear(context_dim, 1)
        self.res_filter = nn.Linear(context_dim, 2)
        self.res_filter2 = nn.Linear(context_dim, 2)
        self.decays = nn.Linear(context_dim, 1)
        self.env = nn.Linear(context_dim, 2)
        self.verb_params = nn.Linear(context_dim, 4)
        self.amps = nn.Linear(context_dim, 1)
        
        self.apply(lambda x: exp.init_weights(x))
    
    def generate(self, vecs, scheduling):
        
        batch, n_events, _ = vecs.shape
        
        shifts = torch.zeros(batch, n_events, exp.n_samples, device=vecs.device)
        shifts[:, :, ::256] = scheduling
        
        
        final, amps = self.reds.forward(
            mix=self.to_mix(vecs),
            f0_choice=self.resonance_choice(vecs) if self.use_wavetables else torch.sigmoid(self.to_f0(vecs)) ** 2,
            decay_choice=torch.sigmoid(self.decay_choice(vecs)),
            freq_spacing=torch.abs(self.freq_spacing(vecs)),
            noise_filter=torch.sigmoid(self.noise_filter(vecs)),
            filter_decays=torch.sigmoid(self.filter_decays(vecs)),
            resonance_filter=torch.sigmoid(self.res_filter(vecs)),
            resonance_filter2=torch.sigmoid(self.res_filter2(vecs)),
            decays=torch.sigmoid(self.decays(vecs)),
            shifts=shifts,
            env=torch.sigmoid(self.env(vecs)),
            verb_params=self.verb_params(vecs),
            # TODO: remove this, it's unused
            amplitudes=self.amps(vecs)
        )
        
        # final, amps = self.reds.forward(
        #     mix=self.to_mix(vecs),
        #     f0_choice=self.resonance_choice(vecs) if self.use_wavetables else self.to_f0(vecs),
        #     decay_choice=self.decay_choice(vecs),
        #     freq_spacing=self.freq_spacing(vecs),
        #     noise_filter=self.noise_filter(vecs),
        #     filter_decays=self.filter_decays(vecs),
        #     resonance_filter=self.res_filter(vecs),
        #     resonance_filter2=self.res_filter2(vecs),
        #     decays=self.decays(vecs),
        #     shifts=shifts,
        #     env=self.env(vecs),
        #     verb_params=self.verb_params(vecs),
        #     amplitudes=self.amps(vecs)
        # )
        
        return final        
        

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


    
    def iterative(self, x) -> torch.Tensor:
        channels = []
        
        spec = experiment_spectrogram(x)
        
        for i in range(n_events):
            v, sched = self.encode(spec, n_events=1)
            ch = self.generate(v, sched)
            # print(torch.argmax(sched, dim=-1))
            current = experiment_spectrogram(ch)
            spec = (spec - current).clone().detach()
            channels.append(ch)
    
        channels = torch.cat(channels, dim=1)        
        
        return channels
            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = experiment_spectrogram(x)
        v, sched = self.encode(spec, n_events=n_events)
        channels = self.generate(v, sched)
        return channels


model = Model(use_wavetables=True).to(device)
optim = optimizer(model, lr=1e-3)
                

def transform(x: torch.Tensor):
    batch_size, channels, _ = x.shape
    bands = multiband_transform(x)
    return torch.cat([b.reshape(batch_size, channels, -1) for b in bands.values()], dim=-1)
        
def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_xl': stft(v, 512, 64, pad=True) for k, v in bands.items()}
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
    normal = stft(x, 2048, 256, pad=True).reshape(-1, 128, 1025).permute(0, 2, 1)
    # return dict(normal=normal)
    return dict(**d1, **d3, **d4, normal=normal)


def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    
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



def train(batch, i) -> Tuple[torch.Tensor, torch.Tensor]:
    optim.zero_grad()
    recon = model.iterative(batch)
    # recon = model.forward(batch)
    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    scl = single_channel_loss_3(batch, recon)
    loss = scl
    loss.backward()
    optim.step()
    recon = max_norm(recon_summed)
    return loss, recon





@readme
class ZeroParameterDecoder(BaseExperimentRunner):

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
            l, r = train(item, i)

            self.real = item
            self.fake = r
            
            print(i, l.item())
            self.after_training_iteration(l, i)


    