
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.atoms import unit_norm
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.matchingpursuit import sparse_code_to_differentiable_key_points, sparse_feature_map
from modules.normalization import max_norm
from modules.pos_encode import ExpandUsingPosEncodings
from modules.sparse import to_key_points_one_d
from scalar_scheduling import pos_encode_feature, pos_encoded
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

d_size = 256
kernel_size = 256
sparse_coding_iterations = 16

band = zounds.FrequencyBand(20, 2000)
scale = zounds.MelScale(band, d_size)
d = morlet_filter_bank(exp.samplerate, kernel_size, scale, np.linspace(0.25, 0.01, d_size)).real
d = torch.from_numpy(d).float().to(device)


def generate(batch_size):
    total_events = batch_size * sparse_coding_iterations

    amps = torch.zeros(total_events, device=device).uniform_(0, 1)
    positions = torch.zeros(total_events, device=device).uniform_(0, 1)
    atom_indices = torch.randperm(d_size)[:total_events]
    # print('GENERATING PROCESS', atom_indices)

    output = _inner_generate(
        batch_size, total_events, amps, positions, atom_indices)
    return output

    
def _inner_generate(batch_size, total_events, amps, positions, atom_indices):
    output = torch.zeros(total_events, exp.n_samples, device=device)

    for i in range(total_events):
        index = atom_indices[i]
        pos = positions[i]
        amp = amps[i]

        signal = torch.zeros(exp.n_samples, device=device)
        signal[:kernel_size] = unit_norm(d[index]) * amp
        signal = fft_shift(signal, pos)[..., :exp.n_samples]
        output[i] = signal
    
    output = output.view(batch_size, sparse_coding_iterations, exp.n_samples)
    output = torch.sum(output, dim=1, keepdim=True)
    return output


def extract_key_points(x, d):
    batch = x.shape[0]

    # fm = sparse_feature_map(
    #     x, 
    #     d, 
    #     n_steps=sparse_coding_iterations, 
    #     device=device,
    #     use_softmax=True
    # )

    # x = to_key_points_one_d(
    #     fm, 
    #     n_to_keep=sparse_coding_iterations
    # ).permute(0, 2, 1)

    x = sparse_code_to_differentiable_key_points(
        x, d, n_steps=sparse_coding_iterations)
    

    x = x.view(sparse_coding_iterations, batch, d_size + 2).permute(1, 2, 0)
    return x



class Encoder(nn.Module):
    def __init__(self, channels, d: torch.Tensor = None):
        super().__init__()

        self.channels = channels

        if d is not None:
            self.d = d
        else:
            self.d = nn.Parameter(torch.zeros(d_size, kernel_size).uniform_(-1, 1))
        
        self.reduce = nn.Conv1d(33 + channels, channels, 1, 1, 0)

        self.pos_amp = nn.Conv1d(2, channels, 1, 1, 0)
        self.atoms = nn.Conv1d(d_size, channels, 1, 1, 0)

        self.reduce_again = nn.Conv1d(channels * 2, channels, 1, 1, 0)

        self.stack = DilatedStack(channels, [1, 3, 9, 1])
    
    def forward(self, x):
        x = x.view(-1, 1, exp.n_samples)

        x = extract_key_points(x, self.d)

        pos_amp = x[:, :2, :]
        atoms = x[:, 2:, :]

        pa = self.pos_amp(pos_amp)
        at = self.atoms(atoms)

        x = torch.cat([pa, at], dim=1)
        x = self.reduce_again(x)

        # pe = pos_encoded(x.shape[0], x.shape[-1], n_freqs=16, device=device).permute(0, 2, 1)

        # x = torch.cat([x, pe], dim=1)
        # x = self.reduce(x)
        # x = self.stack(x)
        # x = torch.mean(x, dim=-1)

        return x


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = ExpandUsingPosEncodings(
            channels, time_dim=sparse_coding_iterations, n_freqs=16, latent_dim=channels)
        
        self.to_pos_amp = LinearOutputStack(channels, 3, out_channels=2)
        self.to_atom = LinearOutputStack(channels, 3, out_channels=d_size)

    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.net(x.permute(0, 2, 1))

        pos_amp = torch.sigmoid(self.to_pos_amp(x))

        atoms = self.to_atom(x)

        x = torch.cat([pos_amp, atoms], dim=-1).permute(0, 2, 1)
        # x = self.expand(x).permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = Encoder(channels, d=d)
        self.decoder = Decoder(channels)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return d

model = Model(channels=d_size).to(device)
optim = optimizer(model, lr=1e-3)


def loss_func(a, b):
    # a = extract_key_points(a, d)
    b = extract_key_points(b, d)
    # print(a.shape, b.shape)

    pos_amp = F.mse_loss(a[:, :2, :], b[:, :2, :])

    real_atoms = b[:, 2:, :]
    fake_atoms = a[:, 2:, :].permute(0, 2, 1)
    # fake_indices = torch.argmax(fake_atoms, dim=-1).view(-1)

    expected_indices = torch.argmax(real_atoms.permute(0, 2, 1), dim=-1).view(-1)
    actual = fake_atoms.reshape(-1, d_size)
    # print(expected_indices)
    # print(fake_indices)

    atom_loss = F.cross_entropy(actual, expected_indices)

    # print('POS_AMP', pos_amp.item(), 'ATOM', atom_loss.item())
    total_loss = (pos_amp * 1) + (atom_loss * 1)
    return total_loss

def train(batch, i):
    optim.zero_grad()

    r = model.forward(batch)
    l = loss_func(r, batch)
    l.backward()
    optim.step()

    with torch.no_grad():
        r = r.permute(0, 2, 1)
        amps = r[0, :, :1].view(-1)
        positions = r[0, :, 1:2].view(-1)
        indices = torch.argmax(r[0, :, 2:], dim=-1).view(-1)
        r = _inner_generate(
            1, sparse_coding_iterations, amps, positions, indices)
    
    return l, r
    

@readme
class KeyPointLossToyExperiment(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            # print('=============================')

            g = generate(item.shape[0])
            self.real = g
            l, r = train(g, i)
            self.fake = r
            print('ITER', i, l.item())
            self.after_training_iteration(l)

            
    