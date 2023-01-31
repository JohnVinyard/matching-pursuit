
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.dilated import DilatedStack
from modules.perceptual import PerceptualAudioModel
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from torch import nn
import torch
from torch.nn import functional as F

from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_samples = exp.n_samples

        scale = zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist), 128)
        bank = morlet_filter_bank(exp.samplerate, 512, scale, 0.1, normalize=True)
        self.register_buffer('bank', torch.from_numpy(bank.real).float().view(128, 1, 512))

        self.k_sparse = 32
        self.n_atoms = 2048
        self.atom_size = 2048

        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)

        # TODO: In order to make the locations where atoms are placed valid,
        # this should probably be the opposite of causal convolutions
        self.net = nn.Sequential(
            DilatedStack(exp.model_dim, [1, 3, 9, 27, 1]),
        )

        self.to_atoms = nn.Conv1d(exp.model_dim, self.n_atoms, 1, 1, 0)
        self.to_transfer = nn.Conv1d(exp.model_dim, self.n_atoms, 1, 1, 0)


        self.apply(lambda x: exp.init_weights(x))
    

    def forward(self, orig, pooled, spec):
        batch = orig.shape[0]

        x = F.conv1d(orig, self.bank, padding=256)
        x = x[..., :self.n_samples]

        x = self.net(x)
        context, indices = torch.max(x, dim=-1)

        x = self.to_atoms(x)
        x = F.dropout(x, p=0.05)
        output, indices, values = sparsify(x, self.k_sparse, return_indices=True)

        atom_indices = indices // self.n_samples
        time_indices = indices % self.n_samples



        for i in range(batch):
            for e in range(self.k_sparse):
                atom = self.atoms[atom_indices[i, e]] * values[i, e]
                time_index = time_indices[i, e]
                output[:, :, time_index: time_index + self.atom_size] += atom
        
        output = output[..., :self.n_samples]

        verb = self.verb.forward(context, output)
        return verb


loss_model = PerceptualAudioModel(exp, norm_second_order=False)
loss_model = PsychoacousticFeature()

model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()

    recon = model.forward(batch, None, None)

    real, _ = loss_model.forward(batch)
    fake, _ = loss_model.forward(recon)

    loss = F.mse_loss(fake, real)
    loss.backward()
    optim.step()

    return loss, recon

@readme
class KSparse(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
