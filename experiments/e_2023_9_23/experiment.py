
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.ddsp import AudioModel
from modules.normalization import max_norm
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512,
    windowed_pif=True)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_cond = nn.Conv1d(4096, 1024, 1, 1, 0)
        self.embed_spec = nn.Conv1d(exp.n_bands, 1024, 1, 1, 0)
        self.down = nn.Conv1d(1024 * 2, 1024, 1, 1, 0)

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=3, dilation=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=9, dilation=9),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            # 2
            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            # 1
            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 2, 2, 0),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(1024)
            ),

            nn.Conv1d(1024, 1, 1, 1, 0)
            
        )

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, cond: torch.Tensor, audio: torch.Tensor):
        spec = exp.pooled_filter_bank(audio)
        spec = self.embed_spec(spec)
        cond = self.embed_cond(cond)
        x = torch.cat([spec, cond], dim=1)
        # x = spec + cond
        x = self.down(x)
        x = self.encoder(x)
        
        return x

    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(257, 8)
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=3, dilation=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=9, dilation=9),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Conv1d(1024, 4096, 1, 1, 0)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(4096, 1024, 1, 1, 0),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 256, 1, 1, 0),
            nn.BatchNorm1d(256)
        )

        self.up = nn.Sequential(
            # 256
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(256, 128, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(128)
            ),

            # 512
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(128, 64, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(64)
            ),

            # 1024
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(64, 32, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32)
            ),

            # 2048
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(32, 16, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(16)
            ),

            # 4096
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(16, 8, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(8)
            ),

            # 8192
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(8, 4, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(4)
            ),

            # 16384
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(4, 2, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(2)
            ),

            # 32768
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(2, 1, 7, 1, 3),
            ),
        )
        



        self.apply(lambda x: exp.init_weights(x))
    
    def encode(self, x):
        batch_size = x.shape[0]

        if len(x.shape) != 4:
            x = exp.perceptual_feature(x)

        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        encoded = self.encoder.forward(x)
        return encoded

    def forward(self, x):

        # batch_size = x.shape[0]

        # if len(x.shape) != 4:
        #     x = exp.perceptual_feature(x)

        # x = self.embed(x)
        # x = x.permute(0, 3, 1, 2)
        # x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        # encoded = self.encoder.forward(x)
        encoded = self.encode(x)

        encoded = sparsify(encoded, n_to_keep=512)

        decoded = self.decoder.forward(encoded)
        final = self.up.forward(decoded)
        return final, encoded


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)


def train(batch, i):
    optim.zero_grad()
    disc_optim.zero_grad()

    with torch.no_grad():
        feat = exp.perceptual_feature(batch)

    if i % 2 == 0:

        # get conditioning signal
        with torch.no_grad():
            recon, encoded = model.forward(feat)
        
        rj = disc.forward(encoded, batch)
        fj = disc.forward(encoded, recon)
        loss = (torch.abs(1 - rj) + torch.abs(0 - fj)).mean()
        loss.backward()
        disc_optim.step()
        print(i, 'DISC', loss.item())
        return None, None, None
    else:
        recon, encoded = model.forward(feat)
        r = exp.perceptual_feature(recon)

        cond = encoded.clone().detach()
        fj = disc.forward(cond, recon)

        # MSE loss and adversarial loss
        loss = torch.abs(1 - fj).mean() + F.mse_loss(r, feat)
        loss.backward()
        optim.step()
        print(i, 'GEN', loss.item())
        return loss, recon, encoded


    # recon, encoded = model.forward(feat)

    # r = exp.perceptual_feature(recon)
    # loss = F.mse_loss(r, feat)

    # loss.backward()
    # optim.step()

    # recon = max_norm(recon)

    # return loss, recon, encoded


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x[:, None, :, :]
        x = F.max_pool2d(x, (4, 4), (4, 4))
        x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)


@readme
class SparseV5(BaseExperimentRunner):

    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)

            if l is None:
                continue

            self.real = item
            self.fake = r
            self.encoded = e
            # print(i, l.item())
            self.after_training_iteration(l)
