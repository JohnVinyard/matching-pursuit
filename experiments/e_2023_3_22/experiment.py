from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, unit_norm
from modules.pointcloud import encode_events
from modules.pos_encode import ExpandUsingPosEncodings
from scalar_scheduling import pos_encoded
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device, playable
from util.readmedocs import readme
from experiments.e_2023_3_8.experiment import model

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def train(batch, i):
    pass


latent_dim = 128


class TransformerSetProcessor(nn.Module):
    def __init__(self, embedding_size, dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.dim = dim

        encoder = nn.TransformerEncoderLayer(
            embedding_size, 4, dim, 0.1, batch_first=True)
        self.net = nn.TransformerEncoder(encoder, 6, norm=ExampleNorm())
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        x = self.net(x)
        return x


class DilatedStackSetProcessor(nn.Module):
    def __init__(self, embedding_size, dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.dim = dim
        self.net = DilatedStack(dim, [1, 3, 9, 27, 81, 1], 0.1)
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            internal_dim,
            n_events,
            nerf_like=True,
            set_processor=TransformerSetProcessor,
            softmax=lambda x: x):

        super().__init__()
        self.latent_dim = latent_dim
        self.internal_dim = internal_dim
        self.n_events = n_events

        self.nerf_like = nerf_like

        self.to_atom = nn.Linear(self.internal_dim, model.embedding_dim)
        self.to_pos = nn.Linear(self.internal_dim, 1)
        self.to_amp = nn.Linear(self.internal_dim, 1)
        self.softmax = softmax

        if nerf_like:
            self.net = ExpandUsingPosEncodings(
                internal_dim,
                n_events,
                n_freqs=16,
                latent_dim=latent_dim,
                multiply=False,
                learnable_encodings=False,
                concat=False)

            self.process = LinearOutputStack(
                internal_dim, layers=5, out_channels=internal_dim)
        else:
            self.embed = nn.Linear(self.latent_dim + 33, internal_dim)
            self.process = set_processor(internal_dim, internal_dim)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):

        if self.nerf_like:
            x = self.net(x[:, None, :])
            x = self.process(x)
        else:
            x = x.view(-1, 1, self.latent_dim).repeat(1, self.n_events, 1)
            pos = pos_encoded(x.shape[0], self.n_events, 16, device=x.device)
            x = torch.cat([x, pos], dim=-1)
            x = self.embed(x)
            x = self.process(x)

        atom = self.to_atom.forward(x)
        atom = self.softmax(atom)
        pos = torch.sigmoid(self.to_pos.forward(x))
        amp = torch.abs(self.to_amp.forward(x))

        final = torch.cat([pos, amp, atom], dim=-1)
        return final


class Discriminator(nn.Module):
    def __init__(
            self,
            n_atoms,
            internal_dim,
            out_dim=1,
            set_processor=TransformerSetProcessor,
            reduction='last',
            judgement_activation=lambda x: x,
            process_edges=False):

        super().__init__()
        self.n_atoms = n_atoms
        self.internal_dim = internal_dim

        self.embed_pos_amp = nn.Linear(2, internal_dim // 2)
        self.embed_atom = nn.Linear(n_atoms, internal_dim // 2)

        self.embed_edges = nn.Linear(2 + n_atoms, internal_dim)

        self.processor = set_processor(internal_dim, internal_dim)
        self.judge = nn.Linear(internal_dim, out_dim)
        self.out_dim = out_dim
        self.reduction = reduction
        self.judgement_activation = judgement_activation

        self.process_edges = process_edges
        self.edges = ProduceEdges(n_edges=256)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):

        if self.process_edges:
            x = self.edges.forward(x)
            x = self.embed_edges(x)
        else:
            pos_amp = x[:, :, :2]
            atoms = x[:, :, 2:]
            pos_amp = self.embed_pos_amp(pos_amp)
            atoms = self.embed_atom(atoms)
            x = torch.cat([pos_amp, atoms], dim=-1)

        x = self.processor(x)

        if self.reduction == 'last':
            x = x[:, -1, :]
        elif self.reduction == 'mean':
            x = torch.mean(x, dim=1)
        else:
            raise ValueError(f'Unknown reduction type {self.reduction}')
        
        x = self.judge(x)
        x = self.judgement_activation(x)
        
        return x


class CanonicalOrdering(nn.Module):
    """
    Project embeddings into a single dimension and order them
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.register_buffer(
            'projection', 
            torch.zeros(embedding_dim, 1).uniform_(-1, 1))
    
    def forward(self, x):
        x = x @ self.projection
        indices = torch.argsort(x, dim=-1)
        values = torch.gather(x, dim=-1, index=indices)
        return values


class ProduceEdges(nn.Module):
    def __init__(self, n_edges: float = None):
        super().__init__()
        self.n_edges = n_edges

    
    def forward(self, embeddings: torch.Tensor):

        # (Batch, n embeddings)
        batch, size, dim = embeddings.shape

        dist = torch.cdist(embeddings, embeddings)

        dist_flat = dist.view(batch, -1)
        srt = torch.argsort(dist_flat, dim=-1)

        rows = srt // size
        columns = srt % size


        output = torch.zeros(
            batch, self.n_edges, dim, device=embeddings.device)

        # KLUDGE: this will contain duplicates of each element
        for b in range(dist.shape[0]):
            for i in range(self.n_edges):
                x = embeddings[b, rows[b, i]]
                y = embeddings[b, columns[b, i]]
                edge = (x - y)
                output[b, i, :] = edge

        
        return output


class SetRelationshipLoss(nn.Module):
    def __init__(self, embedding_dim, threshold):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.edges = ProduceEdges(threshold)
        self.ordering = CanonicalOrdering(embedding_dim)
    
    def forward(self, x):
        pass
        


# TODO: Move into point cloud
def canonical_ordering(a: torch.Tensor):
    # pos
    indices = torch.argsort(a[..., 0], dim=-1)[..., None].repeat(1, 1, a.shape[-1])

    # amp
    # indices = torch.argsort(a[..., 1], dim=-1)[..., None].repeat(1, 1, a.shape[-1])
    values = torch.gather(a, dim=1, index=indices)
    return values

# TODO: move into point cloud
def set_alignment_loss(a: torch.Tensor, b: torch.Tensor):
    a = canonical_ordering(a)
    b = canonical_ordering(b)

    return F.mse_loss(a, b)

    srt = b

    apa = a[..., :2]
    bpa = srt[..., :2]

    ap_atoms = a[..., 2:]
    bp_atoms = srt[..., 2:]

    pos_amp_loss = F.mse_loss(apa, bpa)
    # atom_loss = F.binary_cross_entropy(ap_atoms, bp_atoms)
    targets = torch.argmax(bp_atoms, dim=-1).view(-1)
    atom_loss = F.cross_entropy(ap_atoms.view(
        targets.shape[0], model.total_atoms), targets, reduction='none')

    target_amp = b[..., 1].view(-1)
    atom_loss = (atom_loss * target_amp).mean()

    pos_amp_loss = pos_amp_loss * 1
    atom_loss = atom_loss * 1

    loss = pos_amp_loss + atom_loss
    return loss


steps = 32
n_events = steps * 7

gen = Generator(
    latent_dim=latent_dim, 
    internal_dim=512, 
    n_events=n_events, 
    nerf_like=False, 
    set_processor=TransformerSetProcessor,
    softmax=lambda x: unit_norm(torch.abs(x), dim=-1)).to(device)
gen_optim = optimizer(gen, lr=1e-4)


disc = Discriminator(
    n_atoms=model.embedding_dim,
    internal_dim=512,
    out_dim=1,
    set_processor=DilatedStackSetProcessor,
    reduction='mean',
    judgement_activation=lambda x: x,
    process_edges=True).to(device)
disc_optim = optimizer(disc, lr=1e-4)


def generate_latent(batch_size):
    return torch.zeros(batch_size, latent_dim, device=device).uniform_(-1, 1)

def train_gen(batch):
    gen_optim.zero_grad()
    x = generate_latent(batch.shape[0])
    fake = gen.forward(x)
    j = disc.forward(fake)
    loss = torch.abs(1 - j).mean()
    loss.backward()
    gen_optim.step()
    return loss, fake


def train_disc(batch):
    disc_optim.zero_grad()
    x = generate_latent(batch.shape[0])
    fake = gen.forward(x)
    fj = disc.forward(fake)
    rj = disc.forward(batch)
    loss = (torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()) * 0.5
    loss.backward()
    disc_optim.step()
    return loss



# ae = Autoencoder(model.total_atoms, 512, latent_dim=128,
#                  n_events=n_events).to(device)
# optim = optimizer(ae, lr=1e-3)


# def train_ae(batch):
#     optim.zero_grad()
#     recon, encoded = ae.forward(batch)
#     loss = set_alignment_loss(recon, batch)
#     loss.backward()
#     optim.step()
#     return loss, recon, encoded





def encode_for_transformer(encoding, steps=32, n_atoms=512):
    e = {k: v[0] for k, v in encoding.items()}  # size -> all_instances
    events = encode_events(e, steps, n_atoms)  # tensor (batch, 4, N)
    return events[:, :3, :]


def to_atom_vectors(instances):
    vec = encode_for_transformer(instances).permute(0, 2, 1)
    final = torch.zeros(vec.shape[0], vec.shape[1],
                        2 + model.embedding_dim, device=device)

    for b, item in enumerate(vec):
        final[b, :, 0] = item[:, 1]
        final[b, :, 1] = item[:, 2]
        # oh = F.one_hot(item[:, 0][None, ...].long(),
        #                num_classes=model.total_atoms)
        oh = model.to_embeddings(item[:, 0].long())
        final[b, :, 2:] = oh

    return final


def to_instances(vectors):
    instances = defaultdict(list)

    for b, batch in enumerate(vectors):
        for i, vec in enumerate(batch):

            # atom index (NOW EMBEDDINGS)
            atom_index = vec[..., 2:]

            # index = torch.argmax(atom_index, dim=-1).view(-1).item()
            index = model.to_indices(atom_index.view(-1, model.embedding_dim))

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
        self.encoded = None
    
    def view_embeddings(self, size: int = None):
        if size is None:
            return model.embeddings.data.cpu().numpy()
        
        band = model.bands[size]
        return band.embeddings.data.cpu().numpy()

    def z(self):
        return self.encoded.squeeze().data.cpu().numpy()

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
            recon = model.decode(
                instances, shapes=model.shape_dict(target.shape[0]))

            return playable(recon, exp.samplerate)

    def run(self):
        # initialize embeddings
        print('initializing embeddings')
        for size in model.band_sizes:
            self.view_embeddings(size)
            print(f'{size} initialized...')
        
        for i, item in enumerate(self.iter_items()):
            self.real = item

            with torch.no_grad():
                instances = model.encode(item, steps=steps)
                vec = to_atom_vectors(instances)
                self.vec = vec

            if i % 2 == 0:
                l, r = train_gen(vec)
                self.fake = r
                print('G', l.item())
            else:
                l = train_disc(vec)
                print('D', l.item())
            

            self.listen()