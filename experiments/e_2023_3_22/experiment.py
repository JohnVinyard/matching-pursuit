from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.activation import unit_sine
from modules.pointcloud import CanonicalOrdering, encode_events
from modules.softmax import hard_softmax
from scalar_scheduling import pos_encoded
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from upsample import ConvUpsample, PosEncodedUpsample
from util import device, playable
from util.readmedocs import readme
from experiments.e_2023_3_8.experiment import model
from conjure import numpy_conjure, SupportedContentType

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.02,
    model_dim=128,
    kernel_size=512)


latent_dim = 128
steps = 32
n_events = steps * 7 # encoding steps * number of bands


class TransformerSetProcessor(nn.Module):
    def __init__(self, embedding_size, dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.dim = dim

        encoder = nn.TransformerEncoderLayer(
            embedding_size, 4, dim, 0.1, batch_first=True)
        self.net = nn.TransformerEncoder(encoder, 6, norm=nn.LayerNorm((n_events, dim)))
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        x = self.net(x)
        return x

class Reduce(nn.Module):
    def __init__(self, channels, out_dim):
        super().__init__()
        self.channels = channels
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, 4, 3), # 64
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),

            nn.Conv1d(channels, channels, 7, 4, 3), # 16
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),

            nn.Conv1d(channels, channels, 7, 4, 3), # 4
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),

            nn.Conv1d(channels, out_dim, 4, 4, 0)
        )
    
    def forward(self, x):
        x = x.view(-1, n_events, self.channels)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (0, 256 - n_events))
        x = self.net(x)
        x = x.view(-1, self.out_dim)
        return x




do_canonical_ordering = True
canonical_ordering_dim = 0 # canonical ordering by time seems to work best
encoder_class = TransformerSetProcessor
decoder_class = TransformerSetProcessor
# aggregation_method = 'sum'


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            internal_dim,
            n_events,
            set_processor=TransformerSetProcessor,
            softmax=lambda x: x,
        ):

        super().__init__()
        self.latent_dim = latent_dim
        self.internal_dim = internal_dim
        self.n_events = n_events


        self.to_atom = nn.Linear(self.internal_dim, model.embedding_dim)
        self.to_pos = nn.Linear(self.internal_dim, 1)
        self.to_amp = nn.Linear(self.internal_dim, 1)
        self.softmax = softmax

        self.embed = nn.Linear(self.internal_dim + 33, internal_dim)
        self.process = set_processor(internal_dim, internal_dim)

        self.up = ConvUpsample(
            latent_dim, internal_dim, 4, 256, mode='learned', out_channels=internal_dim, batch_norm=False)
        
        self.up = PosEncodedUpsample(latent_dim, internal_dim, size=n_events, out_channels=internal_dim, layers=6, concat=True, learnable_encodings=True, multiply=True)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):

        # batch, time, channels = x.shape
        # factor = self.n_events // time

        # x = x.view(batch, time, self.latent_dim).repeat(1, factor, 1)
        x = x.view(-1, self.latent_dim)
        x = self.up(x).permute(0, 2, 1)[:, :n_events, :]
        # skip = x

        # pos = pos_encoded(x.shape[0], self.n_events, 16, device=x.device)
        # x = torch.cat([x, pos], dim=-1)
        # x = self.embed(x)
        # x = self.process(x)

        # x = skip + x

        atom = self.to_atom.forward(x)
        atom = self.softmax(atom)

        pos = torch.sigmoid(self.to_pos.forward(x))
        # pos = unit_sine(self.to_pos.forward(x))

        amp = torch.sigmoid(self.to_amp.forward(x)) * 15
        # amp = torch.relu(self.to_amp.forward(x))
        # amp = self.to_amp.forward(x) ** 2

        final = torch.cat([pos, amp, atom], dim=-1)
        return final


class Discriminator(nn.Module):
    def __init__(
            self,
            n_atoms,
            internal_dim,
            out_dim=1,
            set_processor=TransformerSetProcessor,
            judgement_activation=lambda x: x,
        ):

        super().__init__()
        self.n_atoms = n_atoms
        self.internal_dim = internal_dim

        self.embed_pos_amp = nn.Linear(2, internal_dim // 2)
        self.embed_atom = nn.Linear(n_atoms, internal_dim // 2)

        self.embed_edges = nn.Linear(2 + n_atoms, internal_dim)

        self.processor = set_processor(internal_dim, internal_dim)
        self.judge = nn.Linear(internal_dim, out_dim)
        self.out_dim = out_dim
        self.judgement_activation = judgement_activation

        self.collapse = Reduce(internal_dim, out_dim)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        pos_amp = x[:, :, :2]
        atoms = x[:, :, 2:]
        pos_amp = self.embed_pos_amp(pos_amp)
        atoms = self.embed_atom(atoms)
        x = torch.cat([pos_amp, atoms], dim=-1)
        # skip = x
        # x = self.processor(x)
        # x = x + skip
        x = self.collapse(x)

        # if self.reduction == 'last':
        #     x = x[:, -1:, :]
        # elif self.reduction == 'mean':
        #     x = torch.mean(x, dim=1, keepdim=True)
        # elif self.reduction == 'sum':
        #     x = torch.sum(x, dim=1, keepdim=True)
        # elif self.reduction == 'none':
        #     pass
        # else:
        #     raise ValueError(f'Unknown reduction type {self.reduction}')
        
        # x = self.judge(x)
        # x = self.judgement_activation(x)
        
        return x




class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Discriminator(
            model.embedding_dim, 
            internal_dim=512, 
            out_dim=latent_dim, 
            set_processor=encoder_class, 
            judgement_activation=lambda x: x, 
        )
        
        self.decoder = Generator(
            latent_dim=latent_dim,
            internal_dim=512,
            n_events=n_events,
            set_processor=decoder_class,
            softmax=lambda x: x,
        )

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        return decoded, encoded




pos_amp_dim = 1
canonical = CanonicalOrdering(
    model.embedding_dim + (pos_amp_dim * 2), 
    dim=canonical_ordering_dim, 
    no_op=not do_canonical_ordering).to(device)


ae = AutoEncoder().to(device)
optim = optimizer(ae, lr=1e-3)





def train_ae(batch):
    optim.zero_grad()
    recon, encoded = ae.forward(batch)
    

    # loss = F.mse_loss(batch, recon)
    targets = torch.argmax(batch[..., 2:], dim=-1).view(-1)

    # weight = targets // 512

    l1 = F.mse_loss(recon[..., :2], batch[..., :2])
    # weight = batch[..., 1:2]
    # diff = (recon[..., :2] - batch[..., :2])
    # l1 = (torch.norm(diff, dim=-1, keepdim=True) * weight).mean()
    l2 = F.cross_entropy(recon[..., 2:].view(-1, model.total_atoms), targets)

    loss = (l1 * 1) + (l2 * 1)
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

            # atom one-hot encoding
            atom_index = vec[..., 2:]

            index = torch.argmax(atom_index, dim=-1).view(-1).item()
            # index = model.to_indices(atom_index.view(-1, model.embedding_dim))

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
            # print('READING FROM CACHE')
            pass

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