
from typing import Tuple
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType

from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.anticausal import AntiCausalStack
from modules.decompose import fft_frequency_decompose

from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax, step_func
from modules.sparse import sparsify, sparsify2, sparsify_vectors
from modules.stft import stft
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme
from torch.nn.utils.weight_norm import weight_norm
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



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_atoms = 2048
        self.atom_size = 32768

        self.atoms = nn.Parameter(torch.zeros(1, self.n_atoms, self.atom_size).uniform_(-0.01, 0.01) * (torch.linspace(1, 0, self.atom_size) ** 2)[None, None, :])
        
        
        self.encoder = AntiCausalStack(1024, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1])
        
        self.up = nn.Conv1d(1024, 256, 1, 1, 0)
        
        
        # self.to_samples = ConvUpsample(
        #     latent_dim=256,
        #     channels=256,
        #     start_size=128,
        #     end_size=n_samples, 
        #     mode='nearest',
        #     out_channels=self.n_atoms,
        #     from_latent=False,
        #     weight_norm=True
        # )
    
        # self.embed_context = nn.Linear(4096, 256)
        # self.embed_one_hot = nn.Linear(4096, 256)
        self.embed_latent = nn.Linear(1024, context_dim)


        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))

        # self.from_context = nn.Linear(context_dim, 256)
        

        # self.atom_bias = nn.Parameter(torch.zeros(4096).uniform_(-1, 1))

        self.apply(lambda x: exp.init_weights(x))
        

    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
        
        
        encoded = self.encoder.forward(x)
        z = torch.mean(encoded, dim=-1)
        
        
        return encoded, z

    
    def generate(self, encoded, dense):
        encoded = self.up(encoded)
        encoded = F.dropout(encoded, p=0.02)
        
        full = torch.zeros(encoded.shape[0], encoded.shape[1], exp.n_samples, device=encoded.device)
        full[:, :, ::256] = encoded
        encoded = full
        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=16)
        indices = torch.argmax(one_hot, dim=-1, keepdim=True)
        
        # TODO: use one_hot to select subset of dictionary
        # one_hot (batch, n_to_keep, channels)
        print(self.atoms.shape, indices.shape)
        subset = torch.take_along_dim(self.atoms, indices, dim=1)
        
        print(one_hot.shape, subset.shape, packed.shape)
        
        final = fft_convolve(packed, subset)[..., :n_samples]
        
        
        final = self.verb.forward(unit_norm(dense, dim=-1), final)
        return final
    

    def forward(self, x):
        encoded, z = self.encode(x)
        dense = self.embed_latent(z)
        final = self.generate(encoded, dense)
        return final
    
    

class UNet(nn.Module):
    def __init__(self, channels, return_latent=False, is_disc=False):
        super().__init__()
        self.channels = channels
        self.is_disc = is_disc
        
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
        
        if self.is_disc:
            self.judge = nn.Linear(self.channels * 4, 1)

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
        
        if self.is_disc:
            self.apply(lambda x: exp.init_weights(x))
        

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
        
        if self.is_disc:
            j = self.judge(x.view(-1, self.channels * 4))
            return j

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


model = Model().to(device)
optim = optimizer(model, lr=1e-4)

disc = UNet(1024, return_latent=False, is_disc=True).to(device)
disc_optim = optimizer(disc)

try:
    disc.load_state_dict(torch.load('disc.dat'))
    print('Loaded disc weights')
except IOError:
    print('No saved disc weights')



def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()
    forward = mask
    backward = x
    y = backward + (forward - backward).detach()
    return y.sum()

def l1_norm(x: torch.Tensor):
    return torch.abs(x).sum()

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
    return dict(**d1, **d3, **d4)



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
    
    recon = model.forward(batch)
    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    
    # fake_specs = stft(recon, 2048, 256, pad=True).view(b, n_events, -1)
    # sim = fake_specs @ fake_specs.permute(0, 2, 1)
    # sim = torch.triu(sim, diagonal=1)
    # sim_loss = sim.mean() 
    
    
    # # summary of audio channels
    # acs = windowed_audio(recon[:1, None, :, :], 512, 256)
    # acs = torch.norm(torch.abs(acs).view(n_events, 128, 512), dim=-1)
    
    # low_prob_mask = mask = torch.zeros(b, n_events, 1, device=batch.device).bernoulli_(p=0.9)
    # for_recon = torch.sum(recon * low_prob_mask, dim=1, keepdim=True)
    
    # # randomly drop events.  Events should stand on their own
    # mask = torch.zeros(b, n_events, 1, device=batch.device).bernoulli_(p=0.5)
    # for_disc = torch.sum(recon * mask, dim=1, keepdim=True).clone().detach()    
    
    # j = disc.forward(for_disc)
    # d_loss = torch.abs(1 - j).mean()
    recon_loss = single_channel_loss_3(batch, recon) * 1e-2
    # recon_loss = exp.perceptual_loss(recon_summed, batch, norm='l1') * 1e-7
    
    # # print(sim_loss.item(), recon_loss.item(), d_loss.item())
    # loss = d_loss + recon_loss + sim_loss
    
    loss = recon_loss
        
    loss.backward()
    optim.step()
    
    # if i % 100 == 0:
    #     torch.save(disc.state_dict(), 'disc.dat')
    #     print('saving dem disc weights')
    

    # disc_optim.zero_grad()
    
    # rj = disc.forward(batch)
    # fj = disc.forward(for_disc)
    # disc_loss = (torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()) * 0.5
    # disc_loss.backward()
    # disc_optim.step()
    # print('DISC', disc_loss.item())
    
    
    recon = max_norm(recon_summed)
    
    return loss, recon


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


def make_event_vec_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def event_vecs(x: torch.Tensor):
        x = x[0].data.cpu().numpy()
        return x

    return (event_vecs,)


def make_acs_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def acs(x: torch.Tensor):
        x = x.data.cpu().numpy()
        return x

    return (acs,)

@readme
class UnsupervisedEventSeparation(BaseExperimentRunner):
    # encoded = MonitoredValueDescriptor(make_conjure)
    # sched = MonitoredValueDescriptor(make_sched_conjure)
    # event_vecs = MonitoredValueDescriptor(make_event_vec_conjure)
    # acs = MonitoredValueDescriptor(make_acs_conjure)

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
            # self.acs = acs
            
            print(i, l.item())
            self.after_training_iteration(l, i)


    