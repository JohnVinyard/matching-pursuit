
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType

from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.anticausal import AntiCausalStack

from modules.normalization import max_norm
from modules.stft import stft
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from torch.nn.utils.weight_norm import weight_norm
from scipy.ndimage import rotate

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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AntiCausalStack(1024, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1])
        self.decoder = nn.Conv1d(1024, 1024 * n_events, 1, 1, 0, groups=n_events)
        
        self.to_latent = nn.Conv1d(1024, 64, 1, 1, 0)
        self.to_samples = ConvUpsample(
            64, 
            64, 
            start_size=128, 
            end_size=exp.n_samples, 
            mode='nearest', 
            out_channels=1, 
            from_latent=False, 
            batch_norm=True)
        
        self.apply(lambda x: exp.init_weights(x))
        

    def encode(self, x):
        batch_size = x.shape[0]
        # x = stft(x, 2048, 256, pad=True).view(
        #     batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
        
        if x.shape[1] == 1:
            x = experiment_spectrogram(x)
        
        encoded = self.encoder.forward(x)
        return encoded, x

    
    def forward(self, x):
        encoded, orig_spec = self.encode(x)
        decoded = self.decoder.forward(encoded)
        decoded = torch.abs(decoded)
        
        decoded = decoded.view(-1, 1024, 128)
        
        samples = self.to_latent(decoded)
        samples = self.to_samples(samples).view(-1, n_events, 1, exp.n_samples)
        
        decoded = decoded.view(x.shape[0], n_events, 1024, 128)
        
        return decoded, orig_spec, samples

    

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
            # x = stft(x, 2048, 256, pad=True).view(
            #     batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
            x = experiment_spectrogram(x)
        
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


def train(batch, i):
    optim.zero_grad()
    
    b = batch.shape[0]
    
    recon, orig_spec, samples = model.forward(batch)
    
    norms = torch.norm(recon, dim=(2, 3))
    print(norms.shape)
    diffs = torch.abs(norms[:, None, :] - norms[:, :, None])
    print(diffs.shape)
    diffs = torch.triu(diffs, diagonal=1)
    diff_loss = diffs.mean()
    
    recon_summed = torch.sum(recon, dim=1)
    
    summed_samples = torch.sum(samples, dim=1)
    sample_spec = experiment_spectrogram(summed_samples)
    
    
    
    # ensure that generated samples equal acutal samples
    # in the frequency domain
    sample_loss = F.mse_loss(sample_spec, orig_spec) * 100
    
    assert orig_spec.shape == recon_summed.shape
    
    # ensure that the sum of spectrograms equals the original
    # spectrogram
    recon_loss = F.mse_loss(recon_summed, orig_spec) * 100
    
    # randomly drop events.  Events should stand on their own
    mask = torch.zeros(b, n_events, 1, 1, device=batch.device).bernoulli_(p=0.5)
    masked = recon * mask
    
    with torch.no_grad():
        # generate audio from masked channels
        masked_spec = masked.view(-1, 1024, 128)
        _, _, masked_samples = model.forward(masked_spec[:1, ...])
        masked_samples = torch.sum(masked_samples, dim=1)
    
    # ensure the sum of masked channels passes the disc test    
    for_disc = torch.sum(masked, dim=1).clone().detach()    
    j = disc.forward(for_disc)
    d_loss = torch.abs(1 - j).mean()
    
    loss = recon_loss + d_loss + sample_loss + diff_loss
        
    loss.backward()
    optim.step()
    
    
    disc_optim.zero_grad()
    rj = disc.forward(batch)
    fj = disc.forward(for_disc)
    disc_loss = (torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()) * 0.5
    disc_loss.backward()
    disc_optim.step()
    print('DISC', disc_loss.item())
    
    recon_spec = max_norm(for_disc.view(b, -1)).view(b, 1024, 128)
    
    recon = max_norm(masked_samples)
    
    return loss, recon, recon_spec


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x.data.cpu().numpy()[0]
        # Holy god, why can't I orient this correclty?
        # x = rotate(x, -270, axes=(0, 1))
        return x

    return (encoded,)



@readme
class UnsupervisedSourceSeparation2(BaseExperimentRunner):
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
            l, samples, r = train(item, i)

            self.real = item
            self.fake = samples
            self.fake = torch.zeros_like(item)
            
            self.encoded = r
            
            print(i, l.item())
            self.after_training_iteration(l, i)


    