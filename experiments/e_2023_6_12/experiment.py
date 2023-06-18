
from collections import defaultdict
from typing import List
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.atoms import unit_norm
from modules.matchingpursuit import dictionary_learning_step, flatten_atom_dict, sparse_code
from modules.pointcloud import encode_events
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

sparse_coding_iterations = 128
atom_size = 8
features_per_band = 8
feature_size = exp.n_bands * features_per_band
sparse_coding_phase = False
pointcloud_encoding_phase = False
n_atoms = 4096

def encode_events(
        inst: List[tuple], 
        batch_size: int, 
        n_frames: int, 
        n_points: int, 
        device: torch.device =None):
    """
    Convert from a flat list of tuples of (atom_index, batch, position, scaled_atom)

    to two tensors, one containing atom indices and the other containing 
    (pos, amp) pairs
    """

    srt = sorted(inst, key=lambda x: (x[2], x[1]))
    packed_indices = torch.zeros(batch_size * n_points, dtype=torch.long, device=device)
    packed_points = torch.zeros(batch_size * n_points, 2, dtype=torch.float, device=device)

    for i, tup in enumerate(srt):
        ai, j, p, a = tup
        packed_indices[i] = ai
        amp = torch.norm(a)
        pos = p / n_frames
        packed_points[i, :] = torch.cat([pos.view(1), amp.view(1)])
    
    indices = packed_indices.view(batch_size, n_points)
    points = packed_points.view(batch_size, n_points, 2)

    return indices, points, srt


def decode_events(indices: torch.Tensor, points: torch.Tensor, n_frames: int, d: torch.Tensor):
    """
    Convert back to a shape usable by the scatter function returned
    by sparse_code
    """
    inst = []
    batch, n_points = indices.shape
    for i in range(batch):
        for p in range(n_points):
            index = indices[i, p]
            pnt = points[i, p]
            inst.append((
                # atom index
                index.item(), 
                # batch
                i, 
                # frame number
                int(pnt[0] * n_frames),
                # atom, scaled by norm 
                unit_norm(d[index]) * pnt[1]
            ))        
    return inst


class SparseCode(nn.Module):
    def __init__(self, n_atoms, atom_size, channels, slow_movement=False):
        super().__init__()

        self.slow_movement = slow_movement
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.channels = channels

        self.d = nn.Parameter(torch.zeros((n_atoms, channels, atom_size)).uniform_(-1, 1))
    
    def learning_step(self, x, n_iterations):
        d = dictionary_learning_step(
            x, 
            self.d, 
            n_steps=n_iterations, 
            device=x.device, 
            d_normalization_dims=(-1, -2), 
            slow_movement=self.slow_movement
        )
        self.d.data[:] = d
    
    def feature_to_point_cloud(self, x, n_iterations):
        batch, channels, frames = x.shape
        inst, scatter = sparse_code(
            x, 
            self.d, 
            n_steps=n_iterations, 
            device=x.device, 
            d_normalization_dims=(-1, -2), 
            flatten=True)
        indices, points, srt = encode_events(
            inst, batch, frames, n_iterations, device=x.device)
        return indices, points, scatter

    def point_cloud_to_feature(self, indices, points, n_frames):
        return decode_events(indices, points, n_frames, self.d)
    
    def do_sparse_code(self, x, n_iterations):

        inst, scatter = sparse_code(
            x, 
            self.d, 
            n_steps=n_iterations, 
            device=x.device, 
            d_normalization_dims=(-1, -2), 
            flatten=True)

        recon = scatter(x.shape, inst)
        return recon
    
    def forward(self, x, n_iterations=32):
        return self.sparse_code(x, n_iterations=n_iterations)


class PointCloudAutencoder(nn.Module):
    def __init__(self, input_features, channels):
        super().__init__()
        self.input_features = input_features
        self.channels = channels

        self.atom_embedding = nn.Embedding(n_atoms, channels // 2)
        self.point_embedding = nn.Linear(2, channels // 2)

        self.embed = nn.Linear(channels + 33, channels)
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)

        self.encoder = nn.TransformerEncoder(
            encoder, 4, norm=nn.LayerNorm((sparse_coding_iterations, channels)))
        self.decoder = nn.TransformerEncoder(
            encoder, 4, norm=nn.LayerNorm((sparse_coding_iterations, channels)))
        
        self.to_indices = nn.Linear(channels, n_atoms)
        self.to_amps = nn.Linear(channels, 1)
        self.to_pos = nn.Linear(channels, 1)

        self.apply(lambda x: exp.init_weights(x))
        
    
    def forward(self, indices, points):
        batch, n_points = indices.shape

        index_embedding = self.atom_embedding(indices)
        point_embedding = self.point_embedding(points)

        x = torch.cat([index_embedding, point_embedding], dim=-1)
        pos = pos_encoded(batch, n_points, 16, device=indices.device)
        x = torch.cat([x, pos], dim=-1)

        x = self.embed(x)
        x = self.encoder(x)
        x = self.decoder(x)

        indices = self.to_indices(x)

        amps = torch.relu(self.to_amps(x))
        pos = torch.sigmoid(self.to_pos(x))
        points = torch.cat([pos, amps], dim=-1)
        return indices, points

class UpsampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch, channels, time = x.shape
        return F.interpolate(x, scale_factor=4, mode='nearest')
    


class NeuralSparseCode(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, padding=1, dilation=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=3, dilation=3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=9, dilation=9),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=27, dilation=27),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

        )
            
        self.up = nn.Conv1d(channels, n_atoms, 1, 1, 0)

        self.map = nn.Conv1d(n_atoms, channels, 1, 1, 0)

        self.decoder = nn.Sequential(
            

            nn.Conv1d(channels, channels, 3, 1, padding=1, dilation=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=3, dilation=3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=9, dilation=9),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=27, dilation=27),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 1, 1, 0)
        )


    def forward(self, x):
        batch, channels, time = x.shape

        skip = x
        x = self.encoder(x)
        x = skip + x


        x = self.up(x)
        # sm = torch.softmax(x.view(batch, n_atoms * time), dim=-1).view(batch, n_atoms, time)
        # x = sm * x
        x = sparsify(x, sparse_coding_iterations)
        x = self.map(x)


        skip = x
        x = self.decoder(x)
        x = skip + x

        return x

class Model(nn.Module):
    def __init__(self, sparse_code=False):
        super().__init__()
        self.sparse_code = sparse_code

        self.embed = nn.Linear(257, features_per_band)

        self.sparse = NeuralSparseCode(feature_size)

        self.up = nn.Sequential(

            nn.Conv1d(1024, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(512, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(256, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(128, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(64, 1, 7, 1, 3),
        )        
        self.apply(lambda x: exp.init_weights(x))
    
    def embed_features(self, x):
        # torch.Size([16, 128, 128, 257])
        batch, channels, time, period = x.shape
        x = self.embed(x).permute(0, 3, 1, 2).reshape(batch, 8 * channels, time)

        if self.sparse_code:
            x = self.sparse(x)
        
        return x

    def generate(self, x):
        x = self.up(x)
        return x
    
    def forward(self, x):
        # torch.Size([16, 128, 128, 257])
        x = self.embed_features(x)
        x = self.generate(x)
        return x

try:
    model = Model(sparse_code=True).to(device)
    model.load_state_dict(torch.load('model.dat'))
    print('Loaded model')
except IOError:
    pass
optim = optimizer(model, lr=1e-3)


try:
    sparse_model = SparseCode(n_atoms, atom_size, 1024, slow_movement=True).to(device)
    sparse_model.load_state_dict(torch.load('sparse_model.dat'))
    print('Loaded sparse model')
except IOError:
    pass


point_cloud = PointCloudAutencoder(n_atoms, 1024).to(device)
pc_optim = optimizer(point_cloud, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    with torch.no_grad():
        spec = exp.perceptual_feature(batch)

    recon = model.forward(spec)
    recon_spec = exp.perceptual_feature(recon)
    audio_loss = F.mse_loss(recon_spec, spec)
    loss = audio_loss
    loss.backward()
    optim.step()
    return loss, recon


def train_sparse_coding(batch, i):
    with torch.no_grad():
        spec = exp.perceptual_feature(batch)
        embedded = model.embed_features(spec)

        # do the recon _before_ a learning iteration on this batch, otherwise
        # batches always sound better than they should
        recon = sparse_model.do_sparse_code(embedded, n_iterations=sparse_coding_iterations)

        sparse_model.learning_step(embedded, n_iterations=sparse_coding_iterations)
        loss = F.mse_loss(recon, embedded)
    
    with torch.no_grad():
        recon = model.generate(recon)

    return loss, recon
    


def train_pointcloud_encoder(batch, i):
    """
    Encoding
    ------------------
    audio -> pif -> low-dim -> pointcloud

    Decoding
    -------------------
    pointcloud -> low-dim -> audio
    """
    pc_optim.zero_grad()

    with torch.no_grad():
        # compute the PIF feature
        spec = exp.perceptual_feature(batch)
        # compute the lower-dimensional embedding
        embedded = model.embed_features(spec)
        # sparse code the lower-dimensional feature
        target_indices, target_points, scatter = sparse_model.feature_to_point_cloud(
            embedded, n_iterations=sparse_coding_iterations)
    
    # encoded and decode the point cloud
    indices, points = point_cloud.forward(target_indices, target_points)

    # compute losses for the reconstruction
    index_loss = F.cross_entropy(indices.view(-1, n_atoms), target_indices.view(-1))
    point_loss = F.mse_loss(points, target_points)
    loss = index_loss + point_loss
    loss.backward()
    pc_optim.step()

    with torch.no_grad():
        indices = torch.argmax(indices, dim=-1)
        # first, back to instances
        inst = sparse_model.point_cloud_to_feature(indices, points, points.shape[1])
        # then reconstruct the low-dim feature
        feature = scatter(embedded.shape, inst)
        # finally, generate audio from the low-dim feature
        recon = model.generate(feature)
    
    return loss, recon


def return_train_func():
    if sparse_coding_phase:
        return train_sparse_coding
    
    if pointcloud_encoding_phase:
        return train_pointcloud_encoder

    return train



@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, return_train_func(), exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r = self.train(item, i)
            self.fake = r
            print(i, l.item())
            self.after_training_iteration(l)

            if i >= 100000 and not sparse_coding_phase:
                torch.save(model.state_dict(), 'model.dat')
                break

            if i > 1000 and sparse_coding_phase:
                torch.save(sparse_model.state_dict(), 'sparse_model.dat')
                break

                
    
    