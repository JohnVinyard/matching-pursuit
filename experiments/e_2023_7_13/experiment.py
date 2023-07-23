
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules import stft
from modules.floodfill import flood_fill_loss
from modules.matchingpursuit import dictionary_learning_step, sparse_code, sparse_code_to_differentiable_key_points, sparse_feature_map
from modules.normalization import max_norm
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
    model_dim=128,
    kernel_size=512)


d_size = 512
kernel_size = 512
sparse_coding_steps = 128

d = torch.zeros(d_size, kernel_size, device=device).uniform_(-1, 1)

# mel = MelScale()
# codec = AudioCodec(mel)


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
    # loss = basic_matching_pursuit_loss(recon, batch)

    # recon = codec.to_frequency_domain(recon.view(-1, exp.n_samples))[..., 0]
    # batch = codec.to_frequency_domain(batch.view(-1, exp.n_samples))[..., 0]
    # loss = F.mse_loss(recon, batch)


    orig_recon = recon

    recon = exp.pooled_filter_bank(recon)
    batch = exp.pooled_filter_bank(batch)
    print(batch.min().item(), batch.max().item(), batch.std().item())
    loss = flood_fill_loss(recon[:, None, :, :], batch[:, None, :, :], threshold=1.25)

    # loss = F.mse_loss(recon, batch)
    
    loss.backward()
    optim.step()
    return loss, orig_recon

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
