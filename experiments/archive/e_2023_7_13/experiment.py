
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules import stft
from modules.decompose import fft_frequency_decompose
from modules.floodfill import flood_fill_loss
from modules.matchingpursuit import dictionary_learning_step, sparse_code, sparse_code_to_differentiable_key_points, sparse_feature_map
from modules.normalization import max_norm, unit_norm
from modules.phase import AudioCodec, MelScale
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=1024,
    scaling_factor=0.25)


d_size = 512
kernel_size = 512
sparse_coding_steps = 128

d = torch.zeros(d_size, kernel_size, device=device).uniform_(-1, 1)

mel = MelScale()
codec = AudioCodec(mel)


# band = zounds.FrequencyBand(1, exp.samplerate.nyquist)
# scale = zounds.LinearScale(band, 64)
# fb = morlet_filter_bank(exp.samplerate, 64, scale, 0.25, normalize=True).real.astype(np.float32)
# fb = torch.from_numpy(fb).to(device)

# def pif(x):
#     batch = x.shape[0]

#     bands = fft_frequency_decompose(x, 512)
#     normalized = []
#     for band in bands.values():
#         spec = F.conv1d(band, fb.view(64, 1, 64), padding=32)
#         spec = torch.relu(spec)
#         spec = spec.unfold(-1, 32, 16)
#         spec = torch.abs(torch.fft.rfft(spec, dim=-1, norm='ortho'))
#         normalized.append(spec.view(batch, -1))
    
#     return torch.cat(normalized, dim=-1)

# def basic_matching_pursuit_loss(recon, target):

#     # fake_events, scatter = sparse_code(
#     #     recon, d, n_steps=sparse_coding_steps, device=device, flatten=True)
#     # recon = scatter(recon.shape, fake_events)
    
#     real_events, scatter = sparse_code(
#         target, d, n_steps=sparse_coding_steps, device=device, flatten=True)
    
#     real = scatter(recon.shape, real_events)

#     counter = defaultdict(int)
#     for _, j, _, _ in real_events:
#         counter[j] = 0
    
#     loss = 0

#     for i in range(len(real_events)):
#         ai, j, p, a = real_events[i]

#         start = p
#         stop = min(exp.n_samples, start + a.shape[-1])
#         # size = stop - start
#         channel = counter[j]

#         recon_spec = recon[j, channel, start: stop]
#         real_spec = real[j, channel, start: stop]
#         loss = loss + F.mse_loss(recon_spec, real_spec)
#         counter[j] += 1


#     return loss
#     # print(recon.shape, real.shape)

#     # return loss + F.mse_loss(recon, real)

#     # return exp.perceptual_loss(recon, real)
    



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(exp.n_bands, 128, 7, 4, 3), # 32
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 128, 7, 4, 3), # 8
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 128, 8, 8, 0)
        )

        
        self.net = ConvUpsample(
            128, 
            128, 
            start_size=4, 
            end_size=exp.n_samples, 
            mode='nearest', 
            out_channels=1, 
            batch_norm=True
        )
        
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        spec = exp.pooled_filter_bank(x)
        latent = self.encoder(spec)
        latent.view(-1, 128)
        result = self.net(latent)
        # LEARNING: normaliztion seems to cause noise
        # result = max_norm(result, dim=-1)
        # result = torch.clamp(result, -1, 1)
        return result

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def extract_windows(x, kernel, step):
    kw, kh = kernel
    sw, sh = step

    x = x.unfold(-1, kw, sw)
    x = x.unfold(1, kh, sh)

    # win = torch.hamming_window(kw, device=x.device)[:, None] * torch.hamming_window(kh, device=x.device)[None, :]
    # x = x * win[None, None, None, :, :]

    x = x.reshape(*x.shape[:-2], np.product(x.shape[-2:]))
    x = unit_norm(x, dim=-1)


    return x

def extract_feature(x):

    # spec = codec.to_frequency_domain(x.view(-1, exp.n_samples))[..., 0]
    # return spec


    p = exp.perceptual_feature(x)
    p = unit_norm(p, dim=-1)    

    

    spec = exp.pooled_filter_bank(x)
    t = extract_windows(spec, (15, 15), (7, 7))
    # p = extract_windows(spec, (3, 15), (1, 7))

    # batch, time, channels = spec.shape

    # w = spec.unfold(-1, 15, 7)
    # w = w.unfold(1, 15, 7)
    
    # w = w.reshape(*w.shape[:-2], np.product(w.shape[-2:]))
    # w = unit_norm(w, dim=-1)

    # # # https://en.wikipedia.org/wiki/Approximate_entropy


    return torch.cat([
        t.view(-1),
        p.view(-1)
    ])


def exp_loss(a, b):
    

    a = extract_feature(a)
    b = extract_feature(b)

    return F.mse_loss(a, b)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    orig_recon = recon

    loss = exp_loss(recon, batch)

    
    loss.backward()
    optim.step()
    return loss, max_norm(orig_recon)

@readme
class MatchingPursuitLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item


            # if i < 100:
            #     new_d = dictionary_learning_step(item, d, device=device, n_steps=sparse_coding_steps)
            #     d.data[:] = new_d

            #     encoded, scatter = sparse_code(item[:1, ...], d, n_steps=sparse_coding_steps, device=device, flatten=True)
            #     self.real = scatter(item[:1, ...].shape, encoded)
            

            l, r = self.train(item, i)
            print(i, l.item())
            self.fake = r
            self.after_training_iteration(l)
