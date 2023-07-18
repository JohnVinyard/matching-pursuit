
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules import stft
from modules.matchingpursuit import dictionary_learning_step, sparse_code, sparse_feature_map
from modules.normalization import max_norm
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
    kernel_size=512)


d_size = 512
kernel_size = 512
sparse_coding_steps = 128

d = torch.zeros(d_size, kernel_size, device=device).uniform_(-1, 1)


def basic_matching_pursuit_loss(recon, target):
    events, scatter = sparse_code(target, d, n_steps=sparse_coding_steps, flatten=True)
    # t = scatter(target.shape, events)

    # return F.mse_loss(
    #     stft(recon, 512, 256, pad=True),
    #     stft(t, 512, 256, pad=True)
    # )

    loss = 0

    for ai, j, p, a in events:
        start = p
        stop = min(exp.n_samples, p + kernel_size)
        size = stop - start

        r = torch.abs(torch.fft.rfft(recon[j, :, start: start + size], dim=-1))
        at = torch.abs(torch.fft.rfft(a[:, :size], dim=-1))

        loss = loss + torch.abs(r - at).sum()
    
    return loss



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(128, 128, 7, 4, 3), # 32
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
        return result

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = basic_matching_pursuit_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class MatchingPursuitLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item


            if i < 100:
                new_d = dictionary_learning_step(item, d, device=device, n_steps=sparse_coding_steps)
                d.data[:] = new_d

                encoded, scatter = sparse_code(item[:1, ...], d, n_steps=sparse_coding_steps, device=device, flatten=True)
                self.real = scatter(item[:1, ...].shape, encoded)
            

            l, r = self.train(item, i)
            print(i, l.item())
            self.fake = r
            self.after_training_iteration(l)
    