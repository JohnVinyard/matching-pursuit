import numpy as np
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.matchingpursuit import dictionary_learning_step, flatten_atom_dict, sparse_code
from modules.normalization import unit_norm
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util.readmedocs import readme
import zounds
from util import device, playable
import torch
from torch import nn
from torch.nn import functional as F



exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_atoms = 512
atom_size = 512
steps = 128
approx = None

d = torch.zeros(n_atoms, atom_size, device=device).uniform_(-1, 1)
d = unit_norm(d)


def events_to_tensor(events):
    """
    Convert from a list of 
        [ 
            (atom_index, batch, position, atom),
            ...
        ]
    
    To a tensor with shape (batch, 3, n_events), sorted
    by ascending time along the final axis
    """
    ai, b, p, a = zip(*events)
    batch_size = max(b) + 1
    n_events = len(events)

    t = torch.zeros(batch_size, 3, n_events, device=device)
    for i in range(n_events):
        b_index = b[i]
        at_index = ai[i]
        pos = p[i]
        amp = torch.norm(a[i])

        t[b_index, 0, i] = at_index
        t[b_index, 1, i] = pos / exp.n_samples
        t[b_index, 2, i] = amp
    

    p = t[:, 1:2, :]
    indices = torch.argsort(p, dim=-1).repeat(1, 3, 1)
    x = torch.gather(t, dim=-1, index=indices)
    return x

def tensor_to_events(events, d):
    """
    Convert a tensor with shape (batch, 3, n_events)

    to a list of
        [
            (atom_index, batch, position, atom),
            ...
        ]
    """

    output = []

    for b, seq in enumerate(events):
        for i in range(seq.shape[-1]):
            ai, p, a = events[b, :, i]
            output.append((ai, b, int(p * exp.n_samples), d[int(ai)] * a))
    
    return output


class Predictor(nn.Module):
    """
    Analyze a sequence of atoms with absolute positions
    and magnitudes.

    Output a new atom, relative position and relative magnitude
    """

    def __init__(self, channels, n_atoms=n_atoms):
        super().__init__()

        self.embed = nn.Embedding(n_atoms, embedding_dim=channels)
        self.pos_amp = nn.Linear(2, channels)

        self.reduce = nn.Conv1d(channels * 2, channels, 1, 1, 0)

        self.net = DilatedStack(channels, [1, 3, 9, 27, 81, 1], dropout=0.1)

        self.to_atom = LinearOutputStack(channels, 3, out_channels=n_atoms)
        self.to_pos_amp_pred = LinearOutputStack(channels, 3, out_channels=2)

        self.apply(lambda x: exp.init_weights(x))

    def generate(self, x, steps):

        output = x
        batch_size = x.shape[0]

        with torch.no_grad():
            for i in range(steps):
                seed = output[:, :, i:]

                a, pa = self.forward(seed)
                a = a.view(batch_size, -1)
                p = pa.view(batch_size, 2, 1)
                a = torch.argmax(a, dim=-1, keepdim=True)

                last_pos_amp = seed[:, 1:3, -1:]
                new_pos_amp = last_pos_amp + p

                next_one = torch.zeros(batch_size, 3, device=x.device)
                next_one[:, 0] = a

                next_one[:, 1:3] = new_pos_amp.view(1, 2)
                next_one = next_one.view(batch_size, 3, 1)

                output = torch.cat([output, next_one], dim=-1)


        first = output[:, :, :steps // 2]
        second = output[:, :, steps // 2:]

        return first, second

    def forward(self, x):
        x = x.permute(0, 2, 1)
        atoms = x[:, :, 0].long()
        atoms = self.embed.forward(atoms)

        pos_amp = x[:, :, 1:3]
        pos_amp = self.pos_amp.forward(pos_amp)

        x = torch.cat([atoms, pos_amp], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.reduce.forward(x)
        x = self.net(x)
        x = x.permute(0, 2, 1)

        a = self.to_atom(x)[:, -1:, :]
        pa = self.to_pos_amp_pred(x)[:, -1:, :]
        return a, pa



predictor = Predictor(exp.model_dim, n_atoms).to(device)
optim = optimizer(predictor, lr=1e-3)

def train():
    pass

@readme
class SingleBandMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.encoded = None
    
    def roundtrip(self, steps=steps):
        with torch.no_grad():
            target = self.real[:1, ...]
            instances, scatter_segments = sparse_code(
                target, d, steps, device=device, approx=approx)
            instances = flatten_atom_dict(instances)
            t = events_to_tensor(instances)
            events = tensor_to_events(t, d)
            recon = scatter_segments(target.shape, events)
            return playable(recon, exp.samplerate)
    
    
    def generate(self, steps=steps):
        target = self.real[:1, ...]

        instances, scatter_segments = sparse_code(
            target, d, n_steps=steps, device=device, approx=approx, flatten=True)
        inp = events_to_tensor(instances)
        f, s = predictor.generate(inp, steps=steps)

        fa = tensor_to_events(f, d)
        fa = scatter_segments(target.shape, fa)

        fs = tensor_to_events(s, d)
        fs = scatter_segments(target.shape, fs)

        x = torch.cat([fa.view(-1), fs.view(-1)])
        return playable(x, exp.samplerate)

    def recon(self, steps=steps):
        with torch.no_grad():
            target = self.real[:1, ...]
            instances, scatter_segments = sparse_code(
                target, d, steps, device=device, approx=approx)
            recon = scatter_segments(target.shape, flatten_atom_dict(instances))
            return playable(recon, exp.samplerate)

    def view_dict_spec(self):
        return np.abs(np.fft.rfft(d.data.cpu().numpy(), axis=-1)).T
    
    def view_dict(self):
        return d.data.cpu().numpy()

    def spec(self, steps=steps):
        return np.abs(zounds.spectral.stft(self.recon(steps)))

    def run(self):
        for i, item in enumerate(self.iter_items()):
            optim.zero_grad()

            batch_size = item.shape[0]

            # print('=============================')

            with torch.no_grad():
                self.real = item
                new_d = dictionary_learning_step(
                    item,
                    d,
                    steps,
                    device=device,
                    approx=approx)

                d[:] = unit_norm(new_d)
            

            instances, scatter = sparse_code(
                item, d, n_steps=steps, device=device, approx=approx, flatten=True)
            transformer_encoded = events_to_tensor(instances)

            inputs = transformer_encoded[:, :, :-1]

            atom_targets = transformer_encoded[:, 0, -1:]
            rel_targets = torch.diff(transformer_encoded[:, 1:, -2:], dim=-1)

            pred_atoms, pred_pos_amp = predictor.forward(inputs)

            pred_atoms = pred_atoms.view(batch_size, -1)

            atom_loss = F.cross_entropy(
                pred_atoms,
                atom_targets.view(-1).long()
            )

            print(
                torch.argmax(pred_atoms, dim=-1).view(-1),
                atom_targets.view(-1).long())

            rel_loss = F.mse_loss(
                pred_pos_amp.view(batch_size, 2),
                rel_targets.view(batch_size, 2)
            )

            print(rel_loss.item())

            loss = atom_loss + rel_loss
            loss.backward()
            print(i, 'MODEL LOSS', loss.item())
            optim.step()
