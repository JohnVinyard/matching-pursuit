
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.sparse import encourage_sparsity_loss
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from torch import jit


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


resonance_samples = 2048
encoding_channels = 512


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


class Model(nn.Module):
    def __init__(
            self, 
            channels, 
            encoding_channels, 
            resonance_samples,
            latent_dim):

        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.resonance_samples = resonance_samples
        self.latent_dim = latent_dim

        self.embed_periodicity = nn.Linear(257, 8)
        self.to_channel_dim = nn.Conv1d(exp.n_bands * 8, channels, 1, 1, 0)

        self.stack = nn.Sequential(
            DilatedBlock(channels, 1),    
            DilatedBlock(channels, 3),    
            DilatedBlock(channels, 9),    
            DilatedBlock(channels, 1),    
        )

        self.to_salience = nn.Conv1d(channels, channels, 1, 1, 0)
        self.activation = nn.Conv1d(channels, channels, 1, 1, 0)

        self.up = nn.Conv1d(channels, encoding_channels, 1, 1, 0)

        # self.up = ConvUpsample(
        #     channels, 
        #     channels, 
        #     start_size=128, 
        #     end_size=exp.n_samples, 
        #     mode='learned', 
        #     out_channels=encoding_channels, 
        #     from_latent=False, 
        #     batch_norm=True)
        

        atoms = torch.zeros(1, encoding_channels, resonance_samples).uniform_(-1, 1)
        self.atoms = nn.Parameter(atoms)

        
    
        
    def forward(self, x):
        batch_size = x.shape[0]

        if len(x.shape) == 4:
            pif = x
        else:
            pif = exp.perceptual_feature(x)
        pif = self.embed_periodicity(pif).permute(0, 3, 1, 2).reshape(batch_size, 8 * exp.n_bands, -1)
        x = self.to_channel_dim(pif)

        # gather context
        x = self.stack(x)

        # use softmax to distribute gradient throughout
        # sal = torch.softmax(self.to_salience(x).view(batch_size, -1), dim=-1).view(batch_size, self.channels, -1)
        # act = self.activation(x)
        # x = sal * act

        # upsample to full samplerate
        x = self.up(x)
        # rectification (only positive activations)
        encoding = x = torch.relu(x)


        new_encoding = torch.zeros(batch_size, self.encoding_channels, exp.n_samples, device=x.device)
        ratio = exp.n_samples // encoding.shape[-1]
        new_encoding[:, :, ::ratio] = encoding
        

        d = self.atoms
        d = F.pad(d, (0, exp.n_samples - self.resonance_samples))
        d = unit_norm(d, dim=-1)
        x = fft_convolve(d, new_encoding)[..., :exp.n_samples]

        return x, encoding

model = Model(
    channels=128, 
    encoding_channels=encoding_channels, 
    resonance_samples=resonance_samples,
    latent_dim=16,
).to(device)

optim = optimizer(model, lr=1e-3)


class MatchingPursuitLoss(jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.encoding_channels = 512


    def forward(self, batch: torch.Tensor, recon: torch.Tensor):
        batch_size = batch.shape[0]

        residual = stft(batch, 512, 256, pad=True)


        loss = torch.zeros((1,), device=batch.device)

        for b in range(batch_size):
            x = recon[b]
            norms = torch.sum(torch.abs(x), dim=-1)
            indices = torch.argsort(norms, descending=True)

            r = residual[b: b + 1, :, : ]

            skipped = 0

            for i in indices:
                
                if norms[i].item() == 0:
                    skipped += 1
                    # continue

                start_norm = torch.abs(r.view(-1)).sum()

                channel_spec = stft(recon[b: b + 1, i: i + 1, :], 512, 256, pad=True)
                r = r - channel_spec
                end_norm = torch.abs(r.view(-1)).sum()
                diff = start_norm - end_norm
                loss = loss + -diff.mean()
            
            # how many channels have an l1 norm of 0?
            print(f'skipped {skipped} channels')
        
        return loss

mp_loss = MatchingPursuitLoss().to(device)





def train(batch, i):
    optim.zero_grad()


    with torch.no_grad():
        feat = exp.perceptual_feature(batch)

    recon, encoding = model.forward(feat)

    sparsity_loss = encourage_sparsity_loss(
        encoding=encoding, 
        n_unpenalized=128,
        sparsity_loss_weight=0.1
    )

    loss = mp_loss.forward(batch, recon) + sparsity_loss

    recon = torch.sum(recon, dim=1, keepdim=True)

    loss.backward()
    optim.step()
    return loss, recon



@readme
class SparseV3(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    