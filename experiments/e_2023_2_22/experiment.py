from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util.readmedocs import readme
import zounds
from torch import nn
from util import device
from torch.nn import functional as F
import torch

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

non_zero_elements = 64
zero_elements = (exp.model_dim * exp.n_samples) - non_zero_elements

class Discriminator(nn.Module):
    def __init__(self, channels, n_samples):
        super().__init__()
        self.channels = channels
        self.net = DilatedStack(
            channels, 
            [1, 3, 9, 27, 81, 1], 
            dropout=0.1, 
            soft_sparsity=False, 
            internally_sparse=False, 
            sparsity_amt=1)
        self.judge = nn.Conv1d(self.channels, 1, 1, 1, 0)
        self.apply(lambda p: exp.init_weights(p))
        
    def forward(self, x):
        spec = exp.fb.forward(x, normalize=False)
        encoded, features = self.net.forward(spec, return_features=True)
        final = self.judge(encoded)
        final = torch.sigmoid(final)
        return final, features


class Model(nn.Module):
    def __init__(self, channels, n_samples):
        super().__init__()
        self.channels = channels
        self.encoder = DilatedStack(
            channels, 
            [1, 3, 9, 27, 81, 1], 
            dropout=0.1, 
            padding='only-future', 
            soft_sparsity=False, 
            internally_sparse=False, 
            sparsity_amt=1)
    
        self.up = nn.Conv1d(self.channels, 2048, 1, 1, 0)
        self.down = nn.Conv1d(2048, self.channels, 1, 1, 0)

        self.verb = ReverbGenerator(channels, 3, exp.samplerate, n_samples)
        
        self.decoder = DilatedStack(
            channels,
            [1, 3, 9, 27, 81, 1], 
            dropout=0.1,
            padding='only-past',
            soft_sparsity=False,
            internally_sparse=False,
            sparsity_amt=1
        )

        inhibition = torch.ones(1, 1, 512)
        inhibition[:, :, 1:] = torch.linspace(-1, 0, 511)[None, None, :]
        self.register_buffer('inhibition', inhibition)


        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, x, debug=True):
        batch = x.shape[0]
        spec = exp.fb.forward(x, normalize=False)
        encoded = self.encoder(spec)

        context = torch.mean(encoded, dim=-1)

        # project to high dimension and sparsify
        encoded = self.up(encoded)

        # temporal inhibition
        # encoded = encoded.view(-1, 1, exp.n_samples)
        # encoded = F.pad(encoded, (0, self.inhibition.shape[-1]))
        # encoded = F.conv1d(encoded, self.inhibition, stride=1)[..., :exp.n_samples]
        # encoded = encoded.view(batch, -1, exp.n_samples)

        encoded = F.dropout(encoded, p=0.05)
        encoded = sparsify(encoded, non_zero_elements, return_indices=False, soft=True)
        encoded = self.down(encoded)

        # if debug:
        #     zeros = encoded == 0
        #     total = zeros.sum().item() 
        #     assert total == (zero_elements * batch)

        decoded = self.decoder(encoded)
        decoded = F.pad(decoded, (0, 1))
        final = exp.fb.transposed_convolve(decoded)
        # means = torch.mean(final, dim=-1, keepdim=True)
        # final = final - means
        # final = self.verb.forward(context, final)
        return final

model = Model(exp.model_dim, exp.n_samples).to(device)
optim = optimizer(model, lr=1e-3)


disc = Discriminator(exp.model_dim, exp.n_samples).to(device)
disc_optim = optimizer(disc, lr=1e-3)


def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

    if i % 2 == 0:
        optim.zero_grad()
        recon = model.forward(batch)
        j, features = disc.forward(recon)
        _, rf = disc.forward(batch)
        loss = F.mse_loss(features, rf)
        loss.backward()
        optim.step()
        print('G------------------------------------')
        return loss, recon
    else:
        disc_optim.zero_grad()
        recon = model.forward(batch)
        fj, ff = disc.forward(recon)
        rj, rf = disc.forward(batch)
        loss = torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()
        loss.backward()
        disc_optim.step()
        print('D------------------------------------')
        return loss, recon


@readme
class AdversarialSparseAutoencoder(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.real = None
        self.fake = None
    