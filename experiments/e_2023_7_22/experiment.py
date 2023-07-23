
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.ddsp import overlap_add
from modules.floodfill import flood_fill_loss
from modules.normalization import unit_norm
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512,
    a_weighting=True)

# pos = morlet_filter_bank(
#     exp.samplerate,
#     exp.n_samples,
#     zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist), 128),
#     0.01,
#     normalize=True
# ).real.astype(np.float32)
# pos = torch.from_numpy(pos).to(device)

mel = MelScale()
codec = AudioCodec(mel)

class AudioAnalysis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Conv1d(exp.n_bands, channels, 7, 1, 3)
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)
        x = self.net(x)
        return x

class AudioSynthesis(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.energy_model = False
        self.energy = nn.Conv1d(channels, exp.n_bands, 1, 1, 0)
        self.momentum = nn.Conv1d(channels, exp.n_bands, 1, 1, 0)

        self.up = ConvUpsample(
            channels, 
            channels, 
            128, 
            exp.n_samples, 
            mode='nearest', 
            out_channels=exp.n_bands, 
            batch_norm=True,
            from_latent=False)
        self.net = nn.Conv1d(128, 128, 7, 1, 3)

    
    def forward(self, x):
        if self.energy_model:
            batch, channels, time = x.shape

            e = self.energy(x)

            # limit momentum to be in the range 0-1
            m = self.momentum(x)
            m = torch.sigmoid(m)

            items = [e[:, :, :1]]

            for i in range(1, time):
                nxt = e[:, :, i] + (e[:, :, i - 1] * m[:, :, i])
                items.append(nxt[..., None])
            
            final = torch.cat(items, dim=-1)
            final = torch.relu(final)

            
            x = pos
            x = F.pad(x, (0, 256))
            x = x.unfold(-1, 512, 256) * torch.hamming_window(512, device=x.device)[None, None, None, :]
            x = unit_norm(x, dim=-1)
            x = x * final[..., None]
            x = overlap_add(x, apply_window=False)
            x = torch.sum(x, dim=1, keepdim=True)
            x = x[..., :exp.n_samples]

            return x, e
        else:
            x = self.up(x)
            x = self.net(x)
            x = F.pad(x, (0, 1))
            x = exp.fb.transposed_convolve(x * 0.001)
            return x, torch.zeros(1, device=x.device)


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.conv1 = nn.Conv1d(
            channels, channels, 3, 1, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x

class Stack(nn.Module):
    def __init__(self, channels, dilations):
        super().__init__()
        self.channels = channels
        self.dilation = dilations
        self.transformer_mode = False

        if self.transformer_mode:
            self.proj = nn.Linear(33, channels)
            encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
            self.net = nn.TransformerEncoder(encoder, 5, norm=nn.LayerNorm((128, channels)))
        else:
            self.net = nn.Sequential(*[Block(channels, d) for d in dilations])
        
    
    def forward(self, x):
        if self.transformer_mode:
            pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device)
            pos = self.proj(pos)

            x = x.permute(0, 2, 1)
            x = pos + x
            x = self.net(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.net(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, encoding_channels, k_sparse):
        super().__init__()
        self.in_channels = in_channels
        self.encoding_channels = encoding_channels
        self.k_sparse = k_sparse
        self.up = nn.Conv1d(in_channels, encoding_channels, 1, 1, 0)
        self.mask = nn.Conv1d(in_channels, encoding_channels, 1, 1, 0)

        self.summary = nn.Linear(encoding_channels, in_channels)
        self.down = nn.Conv1d(encoding_channels, in_channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(in_channels)
    
    def forward(self, x):
        batch = x.shape[0]
        orig = x

        x = self.up(x)

        # generate mask independent of vector scale
        mask = self.mask(unit_norm(orig, dim=1))
        sm = torch.softmax(mask.view(batch, -1), dim=-1)
        sm = sm.view(x.shape)
        sm = sparsify(sm, n_to_keep=self.k_sparse)

        # multiply mask by computed values
        sm = sm * x
        x = sm

        # summarize over all time and project
        summary = torch.sum(x, dim=-1)
        proj = self.summary(summary)

        # project each individual vector
        x = self.down(x)

        # add together global and local projections
        x = x + proj[..., None]

        x = self.norm(x)
        return x

class Model(nn.Module):
    def __init__(self, channels, encoding_channels, k_sparse):
        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.k_sparse = k_sparse
        self.analyze = AudioAnalysis(channels)
        self.encoder = Stack(channels, [1, 3, 9, 27, 1])
        self.bottleneck = Bottleneck(channels, encoding_channels, k_sparse)
        self.decoder = Stack(channels, [1, 3, 9, 27, 1])
        self.synthesize = AudioSynthesis(channels)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = self.analyze(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x, energy = self.synthesize(x)
        return x, energy

model = Model(512, 2048, k_sparse=512).to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i):
    optim.zero_grad()
    recon, energy = model.forward(batch)

    # energy = torch.abs(energy)
    # energy_loss = energy.sum()

    # recon_feat = exp.perceptual_feature(recon)[:, None, ...]
    # recon_local = F.avg_pool3d(recon_feat, (15, 15, 15), (1, 1, 1), (7, 7, 7))
    # recon_feat = recon_feat - recon_local

    # real_feat = exp.perceptual_feature(batch)[:, None, ...]
    # real_local = F.avg_pool3d(real_feat, (15, 15, 15), (1, 1, 1), (7, 7, 7))
    # real_feat = real_feat - real_local

    # # loss = exp.perceptual_loss(recon, batch) + energy_loss
    # loss = F.mse_loss(recon_feat, real_feat)

    recon = codec.to_frequency_domain(recon.view(-1, exp.n_samples))[:, None, :, :]
    batch = codec.to_frequency_domain(batch.view(-1, exp.n_samples))[:, None, :, :]


    loss = F.mse_loss(recon, batch)

    # loss = flood_fill_loss(recon[..., 0], real[..., 0], 2)
    # loss = F.mse_loss(recon[..., 0], real[..., 0])

    loss.backward()
    optim.step()
    return loss, recon

@readme
class SparseAutoencoder(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    