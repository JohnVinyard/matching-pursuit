import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.matchingpursuit import dictionary_learning_step, flatten_atom_dict, sparse_code
from modules.normalization import ExampleNorm, unit_norm
from modules.pointcloud import greedy_set_alignment
from modules.pos_encode import ExpandUsingPosEncodings, pos_encoded
from modules.softmax import hard_softmax
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device, playable
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)


def train(batch, i):
    pass


n_atoms = 512
atom_size = 512
iterations = 128
dict_learning_steps = 256
latent_dim = 128


class SetProcessor(nn.Module):
    def __init__(self, embedding_size, dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.dim = dim

        # encoder = nn.TransformerEncoderLayer(embedding_size, 4, dim, 0.1, batch_first=True)
        # self.net = nn.TransformerEncoder(encoder, 6, norm=ExampleNorm())
        self.net = DilatedStack(dim, [1, 3, 9, 27, 81, 1], 0.1)
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, internal_dim, n_events):
        super().__init__()
        self.latent_dim = latent_dim
        self.internal_dim = internal_dim
        self.n_events = n_events

        self.embed = nn.Linear(self.latent_dim + 33, internal_dim)
        self.process = SetProcessor(internal_dim, internal_dim)

        self.to_atom = nn.Linear(self.internal_dim, n_atoms)
        self.to_pos = nn.Linear(self.internal_dim, 1)
        self.to_amp = nn.Linear(self.internal_dim, 1)

        # self.net = ExpandUsingPosEncodings(
        #     internal_dim,
        #     n_events,
        #     n_freqs=16,
        #     latent_dim=latent_dim,
        #     multiply=False,
        #     learnable_encodings=False,
        #     concat=False)
        # self.process = LinearOutputStack(
        #     internal_dim, layers=4, out_channels=internal_dim)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        x = x.view(-1, 1, self.latent_dim).repeat(1, self.n_events, 1)
        pos = pos_encoded(x.shape[0], self.n_events, 16, device=x.device)

        x = torch.cat([x, pos], dim=-1)
        x = self.embed(x)
        x = self.process(x)
        # x = self.net(x)
        # x = self.process(x)

        atom = self.to_atom.forward(x)
        pos = torch.sigmoid(self.to_pos.forward(x))
        amp = torch.abs(self.to_amp.forward(x))

        final = torch.cat([pos, amp, atom], dim=-1)
        return final


class Discriminator(nn.Module):
    def __init__(self, n_atoms, internal_dim, out_dim=1):
        super().__init__()
        self.n_atoms = n_atoms
        self.internal_dim = internal_dim
        # self.embed = nn.Linear(n_atoms + 2, internal_dim)

        self.embed_pos_amp = nn.Linear(2, internal_dim // 2)
        self.embed_atom = nn.Linear(n_atoms, internal_dim // 2)

        self.processor = SetProcessor(internal_dim, internal_dim)
        self.judge = nn.Linear(internal_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):

        pos_amp = x[:, :, :2]
        atoms = x[:, :, 2:]

        pos_amp = self.embed_pos_amp(pos_amp)
        atoms = self.embed_atom(atoms)

        x = torch.cat([pos_amp, atoms], dim=-1)

        # x = self.embed(x)
        x = self.processor(x)
        x = self.judge(x)
        # x = x[:, -1, :]
        x = torch.mean(x, dim=1)
        return x


class Autoencoder(nn.Module):
    def __init__(self, n_atoms, internal_dim, latent_dim, n_events):
        super().__init__()
        self.n_atoms = n_atoms
        self.internal_dim = internal_dim
        self.latent_dim = latent_dim
        self.n_events = n_events

        self.encoder = Discriminator(n_atoms, internal_dim, latent_dim)
        self.decoder = Generator(latent_dim, internal_dim, n_events)
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        return decoded


def canonical_ordering(a: torch.Tensor):
    indices = torch.argsort(a[..., 1], dim=-1)[..., None].repeat(1, 1, a.shape[-1])
    values = torch.gather(a, dim=1, index=indices)
    return values

def set_alignment_loss(a: torch.Tensor, b: torch.Tensor):
    a = canonical_ordering(a)
    b = canonical_ordering(b)

    # indices = greedy_set_alignment(
        # a[..., :2], b[..., :2], return_indices=True, full_feature_count=a.shape[-1])
    # srt = torch.gather(b, dim=1, index=indices)

    srt = b

    apa = a[..., :2]
    bpa = srt[..., :2]

    ap_atoms = a[..., 2:]
    bp_atoms = srt[..., 2:]

    pos_amp_loss = F.mse_loss(apa, bpa)
    # atom_loss = F.binary_cross_entropy(ap_atoms, bp_atoms)
    targets = torch.argmax(bp_atoms, dim=-1).view(-1)
    atom_loss = F.cross_entropy(ap_atoms.view(
        targets.shape[0], n_atoms), targets)

    pos_amp_loss = pos_amp_loss * 1
    atom_loss = atom_loss * 1

    # print(pos_amp_loss.item(), atom_loss.item())

    loss = pos_amp_loss + atom_loss
    return loss


ae = Autoencoder(n_atoms, 512, latent_dim=128, n_events=iterations).to(device)
optim = optimizer(ae, lr=1e-3)


def train_ae(batch):
    optim.zero_grad()
    recon = ae.forward(batch)
    loss = set_alignment_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

# gen = Generator(latent_dim, 512, iterations).to(device)
# gen_optim = optimizer(gen)

# disc = Discriminator(n_atoms, 512).to(device)
# disc_optim = optimizer(disc)

# def make_latent(batch_size, latent_dim):
#     return torch.zeros(batch_size, latent_dim, device=device).uniform_(-1, 1)

# def train_gen(batch):
#     gen_optim.zero_grad()
#     latent = make_latent(batch.shape[0], latent_dim)
#     fake = gen.forward(latent)
#     fj = disc.forward(fake)
#     loss = torch.abs(1 - fj).mean()
#     loss.backward()
#     gen_optim.step()
#     return loss, fake

# def train_disc(batch):
#     disc_optim.zero_grad()
#     latent = make_latent(batch.shape[0], latent_dim)
#     fake = gen.forward(latent)
#     fj = disc.forward(fake)
#     rj = disc.forward(batch)

#     loss = (torch.abs(0 - fj).mean() + torch.abs(1 - rj).mean()) * 0.5
#     loss.backward()
#     disc_optim.step()
#     return loss


d = torch.zeros(n_atoms, atom_size, device=device).uniform_(-1, 1)
d = unit_norm(d)


def to_atom_vectors(instances):
    batch_size = max(map(lambda x: x[1], instances))
    x = torch.zeros(batch_size + 1, iterations, n_atoms + 2, device=device)

    for i, instance in enumerate(instances):
        atom_index, batch, pos, atom = instance
        amp = torch.norm(atom)

        oh = F.one_hot(
            torch.zeros(1, dtype=torch.long).fill_(atom_index),
            num_classes=n_atoms)

        x[batch, i % iterations, 2:] = oh
        x[batch, i % iterations, 0] = pos / exp.n_samples
        x[batch, i % iterations, 1] = amp
    return x


def to_instances(vectors):
    instances = []
    for b, batch in enumerate(vectors):
        for i, vec in enumerate(batch):
            atom_index = vec[..., 2:]
            index = torch.argmax(atom_index, dim=-1).view(-1).item()
            pos = int(vec[..., 0] * exp.n_samples)
            amp = vec[..., 1]
            atom = d[index] * amp
            event = (index, b, pos, atom)
            instances.append(event)
    return instances


@readme
class MatchingPursuitGAN(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.scatter = None
        self.fake = None

    def pos_amp(self):
        return self.fake.data.cpu().numpy()[0, :, :2]

    def atoms(self):
        return self.fake.data.cpu().numpy()[0, :, 2:]

    def listen(self):
        with torch.no_grad():
            inst = to_instances(self.fake)
            recon = self.scatter(self.real.shape, inst)
            return playable(recon, exp.samplerate)

    def round_trip(self):
        with torch.no_grad():
            target = self.real[:1, ...]
            instances, scatter_segments = sparse_code(
                target, d, iterations, device=device, flatten=True)
            vecs = to_atom_vectors(instances)
            instances = to_instances(vecs)
            recon = scatter_segments(target.shape, instances)
            return playable(recon, exp.samplerate)

    def recon(self, steps=iterations):
        with torch.no_grad():
            target = self.real[:1, ...]
            instances, scatter_segments = sparse_code(
                target, d, steps, device=device)
            recon = scatter_segments(
                target.shape, flatten_atom_dict(instances))
            return playable(recon, exp.samplerate)

    def view_dict_spec(self):
        return np.abs(np.fft.rfft(d.data.cpu().numpy(), axis=-1)).T

    def view_dict(self):
        return d.data.cpu().numpy()

    def spec(self, steps=iterations):
        return np.abs(zounds.spectral.stft(self.recon(steps)))

    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            if i < dict_learning_steps:
                with torch.no_grad():
                    d[:] = unit_norm(dictionary_learning_step(
                        item, d, n_steps=iterations, device=device))

            with torch.no_grad():
                instances, scatter = sparse_code(
                    item, d, iterations, device=device, flatten=True)
                vec = to_atom_vectors(instances)
                self.scatter = scatter

            l, fake = train_ae(vec)
            self.fake = fake
            print(l.item())

            # if i % 2 == 0:
            #     dl = train_disc(vec)
            #     print('D', dl.item())
            # else:
            #     gl, fake = train_gen(vec)
            #     print('G', gl.item())
            #     self.fake = fake
