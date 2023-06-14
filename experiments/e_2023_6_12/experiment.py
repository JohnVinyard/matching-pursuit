
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.atoms import unit_norm
from modules.matchingpursuit import dictionary_learning_step, flatten_atom_dict, sparse_code
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
do_sparse_coding = False

sparse_coding_phase = True

# TODO: Allow a much larger "vocabulary", along with 
# an initial mechanism to produce a mask (like switch transformers)
# to choose an appropriate subset
class SparseCode(nn.Module):
    def __init__(self, n_atoms, atom_size, channels):
        super().__init__()

        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.channels = channels

        self.d = nn.Parameter(torch.zeros((n_atoms, channels, atom_size)).uniform_(-1, 1))
    
    def learning_step(self, x, n_iterations):
        d = dictionary_learning_step(x, self.d, n_steps=n_iterations, device=x.device, d_normalization_dims=(-1, -2))
        self.d.data[:] = d
    
    def do_sparse_code(self, x, n_iterations):
        print(x.shape)
        inst, scatter = sparse_code(
            x, self.d, n_steps=n_iterations, device=x.device, d_normalization_dims=(-1, -2))
        inst = flatten_atom_dict(inst)
        recon = scatter(x.shape, inst)
        print(recon.shape)
        return recon
    
    def forward(self, x, n_iterations=32):
        return self.sparse_code(x, n_iterations=n_iterations)
    
    # def sparse_code(self, x, n_iterations=32):
    #     """
    #     To update:

    #     1. For each atom, add all instances back to the residual
    #     2. take the average of the residual at all positions
    #     3. subtract the newly updated atom from the residual
    #     """

    #     d = unit_norm(self.d, axis=(1, 2))
    #     batch, channels, time = x.shape

    #     recon = torch.zeros_like(x)

    #     x = x.clone()

    #     for i in range(n_iterations):
    #         # TODO: Allow atoms to start before the beginning of the sample
    #         fm = F.conv1d(F.pad(x, (0, self.atom_size)), d)[..., :time]
    #         values, indices = torch.max(fm.reshape(batch, -1), dim=-1)

    #         atom_indices = indices // time
    #         time_indices = indices % time

    #         # scaled atoms
    #         atoms = d[atom_indices]
    #         atoms = atoms * values[:, None, None]

    #         sparse = torch.zeros(batch, channels, time, device=x.device)

    #         for i, ti in enumerate(time_indices):
    #             start = ti
    #             end = torch.clamp(start + self.atom_size, 0, time - 1)
    #             size = end - start
    #             sparse[i, :, start:end] = atoms[i, :, :size]
            
    #         # remove the sparse atoms
    #         x = x - sparse

    #         # add the sparse atoms to the reconstruction
    #         recon = recon + sparse
        
    #     residual = x

    #     return residual, recon


class SparseCode2(nn.Module):
    def __init__(self, channels, k_sparse):
        super().__init__()
        self.channels = channels
        self.k_sparse = k_sparse

        self.embed = nn.Linear(33 + channels, channels)
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, channels)))

        self.embed_again = nn.Linear(33 + channels, channels)
        self.decoder = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, channels)))
        self.apply(lambda x: exp.init_weights(x))
    
    def sparse_code(self, x, n_iterations):
        return self.forward(x)

    def forward(self, x, n_iterations=None):
        batch, channels, time = x.shape
        x = x.permute(0, 2, 1)

        pos = pos_encoded(batch, time, 16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.embed(x)
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x = sparsify(x, self.k_sparse)
        x = x.permute(0, 2, 1)

        x = torch.cat([x, pos], dim=-1)
        x = self.embed_again(x)
        x = self.decoder(x)

        x = x.permute(0, 2, 1)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch, channels, time = x.shape

        return F.interpolate(x, scale_factor=4, mode='nearest')
        
        new_time = time * 4
        new_coeffs = new_time // 2 + 1
        spec = torch.fft.rfft(x, dim=-1, norm='ortho')
        new_spec = torch.zeros(batch, channels, new_coeffs, dtype=spec.dtype, device=spec.device)
        new_spec[:, :, :spec.shape[-1]] = spec
        result = torch.fft.irfft(new_spec, dim=-1, norm='ortho')
        return result

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(257, 8)

        self.sparse = SparseCode(1024, 32, 1024)

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
        return x

    def generate(self, x):
        x = self.up(x)
        return x
    
    def forward(self, x):
        # torch.Size([16, 128, 128, 257])
        # batch, channels, time, period = x.shape
        # x = self.embed(x).permute(0, 3, 1, 2).reshape(batch, 8 * channels, time)
        x = self.embed_features(x)

        orig = x

        if do_sparse_coding:
            res, rec = self.sparse.sparse_code(x, n_iterations=32)
        else:
            res, rec = None, orig

        # x = self.up(rec)
        x = self.generate(x)

        return x, res, rec, orig

try:
    model = Model().to(device)
    model.load_state_dict(torch.load('model.dat'))
    print('Loaded model')
except IOError:
    pass
optim = optimizer(model, lr=1e-3)


sparse_model = SparseCode(1024, 32, 1024).to(device)
# sparse_model = SparseCode2(1024, sparse_coding_iterations).to(device)


def train(batch, i):
    optim.zero_grad()
    with torch.no_grad():
        spec = exp.perceptual_feature(batch)

    recon, residual, latent_recon, orig_recon = model.forward(spec)
    recon_spec = exp.perceptual_feature(recon)

    audio_loss = F.mse_loss(recon_spec, spec)

    if do_sparse_coding:
        latent_loss = F.mse_loss(latent_recon, orig_recon.detach())
        loss = audio_loss + latent_loss
    else:
        loss = audio_loss
    
    loss.backward()
    optim.step()
    return loss, recon


def train_sparse_coding(batch, i):
    with torch.no_grad():
        spec = exp.perceptual_feature(batch)
        embedded = model.embed_features(spec)
        sparse_model.learning_step(embedded, n_iterations=sparse_coding_iterations)
        recon = sparse_model.do_sparse_code(embedded, n_iterations=sparse_coding_iterations)
        # loss = torch.abs(recon - embedded).sum()
        loss = F.mse_loss(recon, embedded)
    
    # res, rec = sparse_model.forward(embedded)

    # # loss = F.mse_loss(rec, embedded)
    # # TODO: Should I do norm of residual or mean-squared error?
    # # Is there any difference?
    # loss = torch.norm(res, dim=(1, 2)).mean()
    # loss.backward()
    # sparse_optim.step()

    with torch.no_grad():
        recon = model.generate(recon)

    return loss, recon
    

@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train_sparse_coding if sparse_coding_phase else train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r = self.train(item, i)
            self.fake = r
            print(l.item())
            self.after_training_iteration(l)

            if i >= 6500 and not sparse_coding_phase:
                torch.save(model.state_dict(), 'model.dat')
                break
    
    