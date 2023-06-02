from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm, unit_norm
from modules.pointcloud import CanonicalOrdering, encode_events, extract_graph_edges
from modules.pos_encode import ExpandUsingPosEncodings, pos_encode_feature
from scalar_scheduling import pos_encoded
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device, playable
from util.readmedocs import readme
from experiments.e_2023_3_8.experiment import model
from conjure import numpy_conjure, SupportedContentType

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)



latent_dim = 128
steps = 32
n_events = steps * 7


class TransformerSetProcessor(nn.Module):
    def __init__(self, embedding_size, dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.dim = dim

        encoder = nn.TransformerEncoderLayer(
            embedding_size, 4, dim, 0.1, batch_first=True)
        self.net = nn.TransformerEncoder(encoder, 6, norm=None)
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

snap_embeddings = True # biggest positive contribution yet
# encode_edges = False
canonical_ordering_dim = 0 # canonical ordering by time seems to work best
nerf_generator = False
max_amp = 15
encoder_class = TransformerSetProcessor
decoder_class = TransformerSetProcessor
aggregation_method = 'mean'


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            internal_dim,
            n_events,
            nerf_like=True,
            set_processor=TransformerSetProcessor,
            softmax=lambda x: x,
            snap_embeddings=False):

        super().__init__()
        self.latent_dim = latent_dim
        self.internal_dim = internal_dim
        self.n_events = n_events

        self.snap_embeddings = snap_embeddings

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
                learnable_encodings=True,
                concat=True)

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

        if self.snap_embeddings:
            orig_shape = atom.shape
            atom = atom.view(-1, model.embedding_dim)
            indices = model.to_indices(atom)
            orig = model.to_embeddings(indices)

            forward = orig
            backward = atom
            y = backward + (forward - backward).detach()
            atom = y.view(*orig_shape)
        

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
        self.edges = ProduceEdges(threshold=0.2)

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



class ProduceEdges(nn.Module):
    def __init__(self, threshold: float = None):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, embeddings: torch.Tensor):
        edges = extract_graph_edges(embeddings, threshold=self.threshold)
        return edges



class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Discriminator(
            model.embedding_dim, 
            internal_dim=512, 
            out_dim=latent_dim, 
            set_processor=encoder_class, 
            reduction=aggregation_method, 
            judgement_activation=lambda x: x, 
            # process_edges=encode_edges
        )
        
        self.decoder = Generator(
            latent_dim=latent_dim,
            internal_dim=512,
            n_events=n_events,
            nerf_like=nerf_generator,
            set_processor=decoder_class,
            softmax=lambda x: torch.abs(x),
            snap_embeddings=snap_embeddings
        )

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        return decoded, encoded


class SetRelationshipLoss(nn.Module):
    def __init__(self, embedding_dim, n_edges):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_edges = n_edges

        self.edges = ProduceEdges(self.n_edges)
        self.ordering = CanonicalOrdering(embedding_dim)
    
    def _extract_and_order(self, x):
        edges = self.edges.forward(x)
        ordered = self.ordering.forward(edges)
        return ordered
    

    def forward(self, recon, target):
        r = self._extract_and_order(recon)
        t = self._extract_and_order(target)
        return F.mse_loss(r, t)

        
set_loss = SetRelationshipLoss(
    embedding_dim=model.embedding_dim + 2, 
    n_edges=256).to(device)


pos_amp_dim = 1
canonical = CanonicalOrdering(
    model.embedding_dim + (pos_amp_dim * 2), 
    dim=canonical_ordering_dim).to(device)

def pos_encode_feature(x, n_freqs):
    output = [x]
    for i in range(n_freqs):
        output.extend([
            torch.sin((2 ** i) * x),
            torch.cos((2 ** i) * x)
        ])
    x = torch.cat(output, dim=-1)
    return x

def compute_full_embedding(a: torch.Tensor):
    pos, amp, atom = a[..., :1], a[..., 1:2], a[..., 2:]

    pos = pos_encode_feature(pos * np.pi, n_freqs=16)
    amp = pos_encode_feature((amp / 20) * np.pi, n_freqs=16)

    final = torch.cat([pos, amp, atom], dim=-1)
    return final


gen = Generator(
    latent_dim=latent_dim, 
    internal_dim=512, 
    n_events=n_events, 
    nerf_like=False, 
    set_processor=DilatedStackSetProcessor,
    softmax=lambda x: torch.abs(x),
    snap_embeddings=True).to(device)
gen_optim = optimizer(gen, lr=1e-4)


disc = Discriminator(
    n_atoms=model.embedding_dim,
    internal_dim=512,
    out_dim=1,
    set_processor=DilatedStackSetProcessor,
    reduction=aggregation_method,
    judgement_activation=lambda x: x,
    process_edges=False).to(device)
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



ae = AutoEncoder().to(device)
optim = optimizer(ae, lr=1e-3)


def train_ae(batch):
    optim.zero_grad()
    recon, encoded = ae.forward(batch)
    # No set alignment, we expect atoms to produced in the same
    # order they were seen
    loss = F.mse_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon, encoded



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
        oh = model.to_embeddings(item[:, 0].long())
        final[b, :, 2:] = oh

    return final.data.cpu().numpy()


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

def train(batch):
    pass


def cached_encode(target, steps=32):
    """
    Encode individual audio segments
    """
    instances = model.encode(torch.from_numpy(target).to(device), steps=steps)
    vecs = to_atom_vectors(instances)
    return vecs


@readme
class MatchingPursuitGAN(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
        self.fake = None
        self.vec = None
        self.encoded = None
        self.model = ae

        def read_from_cache_hook(val):
            print('READING FROM CACHE')

        wrapper = numpy_conjure(self.collection, read_hook=read_from_cache_hook)
        wrapped = wrapper(cached_encode)
        self.cached_encode = wrapped
    

    def batch_encode(self, target):
        target = target.data.cpu().numpy()
        results = []

        for i in range(target.shape[0]):
            results.append(self.cached_encode(target[i: i + 1]))
        
        results = np.concatenate(results, axis=0)
        return results
    
    def view_embeddings(self, size: int = None):
        if size is None:
            return model.embeddings.data.cpu().numpy()
        
        band = model.bands[size]
        return band.embeddings.data.cpu().numpy()

    def z(self):
        return self.encoded.squeeze().data.cpu().numpy()


    def listen(self):
        with torch.no_grad():
            f = self.fake[:1, ...]
            inst = to_instances(f)
            recon = model.decode(inst, shapes=model.shape_dict(f.shape[0]))
            return playable(recon, exp.samplerate)

    # TODO: What happens to something like this?  It's typically something
    # I would have tried once or twice at the beginning of a session to ensure
    # that the implementation is correct
    def round_trip(self):
        with torch.no_grad():
            target = self.real[:1, ...]

            # encode
            vecs = self.batch_encode(target)
            vecs = torch.from_numpy(vecs).to(device)

            # decode
            instances = to_instances(vecs)
            recon = model.decode(
                instances, shapes=model.shape_dict(target.shape[0]))

            return playable(recon, exp.samplerate)
    
    @property
    def conjure_funcs(self):
        funcs = super().conjure_funcs

        @numpy_conjure(self.collection, content_type=SupportedContentType.Spectrogram.value)
        def latent(x: np.ndarray):
            return x / (np.abs(x.max()) + 1e-12)
        
        self._latent = latent
        return [
            self._latent, 
            *funcs
        ]

    def after_training_iteration(self, l):
        val = super().after_training_iteration(l)
        l = self.z()
        self._latent(l)
        return val

    def run(self):
        print('initializing embeddings')
        for size in model.band_sizes:
            self.view_embeddings(size)
            print(f'{size} initialized...')
        
        for i, item in enumerate(self.iter_items()):
            self.real = item

            with torch.no_grad():
                # encode as a dictionary where keys correspond to bands
                # and values are lists of atoms, times and amplitudes
                vec = self.batch_encode(item)
                vec = torch.from_numpy(vec).to(device)
                vec = canonical.forward(vec)
                self.vec = vec

            l, f, e = train_ae(vec)
            self.encoded = e
            self.fake = f
            print(i, l.item())

            self.after_training_iteration(l)