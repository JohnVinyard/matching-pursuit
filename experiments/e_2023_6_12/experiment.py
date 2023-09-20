
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import encourage_sparsity_loss
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512,
    a_weighting=False,
    windowed_pif=False,
    norm_periodicities=False)


# PIF will be (batch, bands, time, periodicity)


features_per_band = 8
transformer_context = False


class ChannelsLastBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x


class Contextual(nn.Module):
    def __init__(self, channels, n_layers=6, n_heads=4):
        super().__init__()
        self.channels = channels
        self.down = nn.Conv1d(channels + 33, channels, 1, 1, 0)
        encoder = nn.TransformerEncoderLayer(channels, n_heads, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, n_layers, norm=nn.LayerNorm((128, channels)))

    def forward(self, x):
        pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.down(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode='nearest')


class Model(nn.Module):
    def __init__(self):
        super().__init__()


        self.embed = nn.Linear(257, features_per_band)

        self.salience = nn.Conv1d(1024, 1024, 1, 1, 0)

        if transformer_context:
            self.context = Contextual(1024)
        else:
            self.context = nn.Sequential(
                nn.Sequential(
                    nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(1024)
                ),

                nn.Sequential(
                    nn.Conv1d(1024, 1024, 3, 1, padding=3, dilation=3),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(1024)
                ),

                nn.Sequential(
                    nn.Conv1d(1024, 1024, 3, 1, padding=9, dilation=9),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(1024)
                ),

                nn.Sequential(
                    nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(1024)
                ),
            )

        

        self.verb = ReverbGenerator(1024, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((1024,)))

        

        self.up = nn.Sequential(

            nn.Conv1d(1024, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            UpsampleBlock(512),

            nn.Conv1d(512, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            UpsampleBlock(256),

            nn.Conv1d(256, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            UpsampleBlock(128),

            nn.Conv1d(128, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            UpsampleBlock(64),

            nn.Conv1d(64, 1, 7, 1, 3),
        )

        


        self.norm = nn.LayerNorm((1024, 128))


        self.apply(lambda x: exp.init_weights(x))
    
    def embed_features(self, x, iteration):
        encoding = None

        # torch.Size([16, 128, 128, 257])
        batch, channels, time, period = x.shape
        x = self.embed(x) # (batch, channels, time, 8)

        x = x.permute(0, 3, 1, 2).reshape(batch, 8 * channels, time)

        salience = self.salience(x)
        salience = F.dropout(salience, 0.05)
        salience = torch.relu(salience)
        encoding = x = salience

        return x, encoding

    def generate(self, x):
        x = self.up(x)
        return x
    
    def forward(self, x, iteration):

        # torch.Size([16, 128, 128, 257])
        encoded, encoding = self.embed_features(x, iteration)
        
        
        ctx = torch.sum(encoded, dim=-1)

        x = self.generate(encoded)

        x = self.verb.forward(ctx, x)
        
        return x, encoding

model = Model().to(device)
optim = optimizer(model, lr=1e-3)



def train(batch, i):
    batch_size = batch.shape[0]

    optim.zero_grad()

    with torch.no_grad():
        feat = exp.perceptual_feature(batch)

    recon, encoding = model.forward(feat, i)
    r = exp.perceptual_feature(recon)


    sparsity_loss = encourage_sparsity_loss(encoding, 0, sparsity_loss_weight=0.001)

    loss = F.mse_loss(r, feat) + sparsity_loss

    loss.backward()
    optim.step()

    encoding = (encoding != 0).float()


    return loss, recon, encoding
    
def make_conjure(experiment: BaseExperimentRunner):

    @numpy_conjure(experiment.collection, SupportedContentType.Spectrogram.value)
    def encoding(x: torch.Tensor):
        return x[0].data.cpu().numpy()
    
    return (encoding,)

@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):

    encoding_view = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r, e = train(item, i)
            self.fake = r
            self.encoding_view = e
            print(l.item())
            self.after_training_iteration(l)
    
    
            

                
    
    