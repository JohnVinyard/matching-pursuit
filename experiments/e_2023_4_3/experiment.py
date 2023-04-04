
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.matchingpursuit import sparse_code
from modules.normalization import max_norm
from modules.overfitraw import OverfitRawAudio
from modules.pointcloud import CanonicalOrdering
from modules.sparse import sparsify
from scalar_scheduling import pos_encode_feature
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from experiments.e_2023_3_8.experiment import model

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

trace = {}

def pos_encode_feature(x, n_freqs):
    output = [x]
    for i in range(n_freqs):
        output.extend([
            torch.sin((2 ** i) * x),
            torch.cos((2 ** i) * x)
        ])

    x = torch.cat(output, dim=-1)
    return x


def soft_dirac(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = torch.softmax(x, dim=dim)

    values, indices = torch.max(x, dim=dim, keepdim=True)

    output = torch.zeros_like(x, requires_grad=True)
    ones = torch.ones_like(values, requires_grad=True)

    output = torch.scatter(output, dim=dim, index=indices, src=ones)

    forward = output
    backward = x

    y = backward + (forward - backward).detach()
    return y


def extract_atom_embedding(fm: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Given a dense feature map of shape (batch, n_atoms, time),
    return an atom embedding concatenating an atom embedding, 
    a time embedding and an amplitude encoding
    """

    batch, n_features, time = fm.shape
    index = model.index_of_size(time)

    fm, indices, values = sparsify(fm, n_to_keep=1, return_indices=True)

    # sum the feature map along the feature dimension
    position = torch.sum(fm, dim=1, keepdim=True)
    position = soft_dirac(position, dim=-1)
    rng = torch.linspace(0, 1, position.shape[-1], device=position.device, requires_grad=True)
    position = position * rng[None, None, :]

    # scalar time value
    scalar_pos = position = torch.sum(position, dim=(1, 2)).view(batch, 1)
    # encoded time value
    position = pos_encode_feature(position, n_freqs=16)

    # scalar amp value
    scalar_amp = v = values.view(batch, 1)
    # encoded amp value
    v = pos_encode_feature(v, n_freqs=16)

    # atom embedding
    n_atoms, d_size = d.shape

    # index = model.index_of_dict_size(d_size)
    start = index * n_atoms
    stop = start + n_atoms

    full = torch.zeros(batch, model.total_atoms, device=fm.device)
    a = torch.sum(fm, dim=-1, keepdim=True) # (batch, n_atoms, 1)
    oh = a = soft_dirac(a, dim=1).view(-1, n_atoms)

    full[:, start: stop] = a
    a = full @ model.embeddings

    embedding = torch.cat([
        position, 
        v, 
        a
    ], dim=-1)[:, None, :]
    return embedding


class SparseCodingLoss(nn.Module):
    def __init__(self, embedding_dim, sparse_coding_steps=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sparse_coding_steps = sparse_coding_steps
        self.ordering = CanonicalOrdering(embedding_dim)
    
    def _extract_embeddings(self, x):
        embeddings = model.encode(
            x, 
            steps=self.sparse_coding_steps, 
            extract_embeddings=extract_atom_embedding)
        
        final_embeddings = []
        for e in embeddings.values():
            final_embeddings.extend(e)
        
        embeddings = torch.cat(final_embeddings, dim=1)
        embeddings = self.ordering.forward(embeddings)
        return embeddings
    
    def forward(self, a, b):

        ae = self._extract_embeddings(a)
        trace['recon_embeddings'] = ae

        be = self._extract_embeddings(b)
        trace['orig_embeddings'] = be

        return F.mse_loss(ae, be)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ConvUpsample(
            128, 128, 8, end_size=exp.n_samples, mode='learned', out_channels=1)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = self.net(x)
        # x = max_norm(x, dim=-1)
        return x


gen = OverfitRawAudio((1, 1, exp.n_samples), std=1e-5).to(device)
# gen = Generator().to(device)
optim = optimizer(gen, lr=1e-2)

loss_func = SparseCodingLoss(
    33 + 33 + model.embedding_dim, 
    sparse_coding_steps=32).to(device)

latent = torch.zeros(1, 128).uniform_(-1, 1).to(device)

def train(batch, i):
    optim.zero_grad()
    recon = gen.forward(latent.clone())
    loss = loss_func.forward(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class PointCloudAudioLoss(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)

        example = torch.zeros(4, 13, device=device).uniform_(-1, 1)
        example = soft_dirac(example, dim=-1)

        self.example = example.data.cpu().numpy()
    
    def recon_embeddings(self):
        return trace['recon_embeddings'].data.cpu().numpy()[0]
    
    def orig_embeddings(self):
        return trace['orig_embeddings'].data.cpu().numpy()[0]
    
    