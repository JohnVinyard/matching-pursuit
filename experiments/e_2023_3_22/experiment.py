from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.matchingpursuit import dictionary_learning_step, flatten_atom_dict, sparse_code
from modules.normalization import ExampleNorm, unit_norm
from modules.pointcloud import encode_events, greedy_set_alignment
from modules.pos_encode import ExpandUsingPosEncodings, pos_encoded
from modules.softmax import hard_softmax
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device, playable
from util.readmedocs import readme
import numpy as np
from experiments.e_2023_3_8.experiment import model

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)


def train(batch, i):
    pass


latent_dim = 128


class SetProcessor(nn.Module):
    def __init__(self, embedding_size, dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.dim = dim

        encoder = nn.TransformerEncoderLayer(embedding_size, 4, dim, 0.1, batch_first=True)
        self.net = nn.TransformerEncoder(encoder, 6, norm=ExampleNorm())
        # self.net = DilatedStack(dim, [1, 3, 9, 27, 81, 1], 0.1)
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.net(x)
        # x = x.permute(0, 2, 1)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, internal_dim, n_events):
        super().__init__()
        self.latent_dim = latent_dim
        self.internal_dim = internal_dim
        self.n_events = n_events

        # self.embed = nn.Linear(self.latent_dim + 33, internal_dim)
        # self.process = SetProcessor(internal_dim, internal_dim)

        self.to_atom = nn.Linear(self.internal_dim, model.total_atoms)
        self.to_pos = nn.Linear(self.internal_dim, 1)
        self.to_amp = nn.Linear(self.internal_dim, 1)

        self.net = ExpandUsingPosEncodings(
            internal_dim,
            n_events,
            n_freqs=16,
            latent_dim=latent_dim,
            multiply=False,
            learnable_encodings=False,
            concat=True)
        
        self.process = LinearOutputStack(
            internal_dim, layers=6, out_channels=internal_dim, in_channels=internal_dim * 2)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        # x = x.view(-1, 1, self.latent_dim).repeat(1, self.n_events, 1)
        # pos = pos_encoded(x.shape[0], self.n_events, 16, device=x.device)
        # x = torch.cat([x, pos], dim=-1)
        # x = self.embed(x)
        # x = self.process(x)

        x = self.net(x)
        x = self.process(x)

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
    # pos
    indices = torch.argsort(a[..., 0], dim=-1)[..., None].repeat(1, 1, a.shape[-1])

    # amp
    # indices = torch.argsort(a[..., 1], dim=-1)[..., None].repeat(1, 1, a.shape[-1])
    values = torch.gather(a, dim=1, index=indices)
    return values

def set_alignment_loss(a: torch.Tensor, b: torch.Tensor):
    a = canonical_ordering(a)
    b = canonical_ordering(b)


    srt = b

    apa = a[..., :2]
    bpa = srt[..., :2]

    ap_atoms = a[..., 2:]
    bp_atoms = srt[..., 2:]

    pos_amp_loss = F.mse_loss(apa, bpa)
    # atom_loss = F.binary_cross_entropy(ap_atoms, bp_atoms)
    targets = torch.argmax(bp_atoms, dim=-1).view(-1)
    atom_loss = F.cross_entropy(ap_atoms.view(
        targets.shape[0], model.total_atoms), targets)

    pos_amp_loss = pos_amp_loss * 1
    atom_loss = atom_loss * 1


    loss = pos_amp_loss + atom_loss
    return loss

steps = 32
n_events = steps * 7

ae = Autoencoder(model.total_atoms, 512, latent_dim=128, n_events=n_events).to(device)
optim = optimizer(ae, lr=1e-3)


def train_ae(batch):
    optim.zero_grad()
    recon = ae.forward(batch)
    loss = set_alignment_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon



def encode_for_transformer(encoding, steps=32, n_atoms=512):
    e = {k: v[0] for k, v in encoding.items()}  # size -> all_instances
    events = encode_events(e, steps, n_atoms)  # tensor (batch, 4, N)
    return events[:, :3, :]


def to_atom_vectors(instances):
    vec = encode_for_transformer(instances).permute(0, 2, 1)
    final = torch.zeros(vec.shape[0], vec.shape[1], 2 + model.total_atoms, device=device)

    for b, item in enumerate(vec):
        final[b, :, 0] = item[:, 1]
        final[b, :, 1] = item[:, 2]
        oh = F.one_hot(item[:, 0][None, ...].long(), num_classes=model.total_atoms)
        final[b, :, 2:] = oh
    
    return final


def to_instances(vectors):
    instances = defaultdict(list)

    for b, batch in enumerate(vectors):
        for i, vec in enumerate(batch):
            # atom index
            atom_index = vec[..., 2:]
            index = torch.argmax(atom_index, dim=-1).view(-1).item()
            size_key = index // 512
            size_key = model.size_at_index(size_key)

            index = index % 512
            pos = int(vec[..., 0] * size_key)
            amp = vec[..., 1]
            atom = model.get_atom(size_key, index, amp)
            event = (index, b, pos, atom)
            instances[size_key].append(event)
    
    return instances


@readme
class MatchingPursuitGAN(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.fake = None
        self.vec = None

    def pos_amp(self):
        return self.fake.data.cpu().numpy()[0, :, :2]

    def atoms(self):
        return self.fake.data.cpu().numpy()[0, :, 2:]

    def listen(self):
        with torch.no_grad():
            f = self.fake[:1, ...]
            inst = to_instances(f)
            recon = model.decode(inst, shapes=model.shape_dict(f.shape[0]))
            return playable(recon, exp.samplerate)

    def round_trip(self):
        with torch.no_grad():
            target = self.real[:1, ...]

            # encode
            instances = model.encode(target, steps=32)
            vecs = to_atom_vectors(instances)

            # decode
            instances = to_instances(vecs)
            recon = model.decode(instances, shapes=model.shape_dict(target.shape[0]))

            return playable(recon, exp.samplerate)


    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            with torch.no_grad():
                instances = model.encode(item, steps=steps)
                vec = to_atom_vectors(instances)
                self.vec = vec

            l, fake = train_ae(vec)
            self.fake = fake
            print(l.item())


