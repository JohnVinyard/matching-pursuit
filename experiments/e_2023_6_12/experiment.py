
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.ddsp import AudioModel
from modules.decompose import fft_frequency_decompose
from modules.linear import LinearOutputStack
from modules.normalization import max_norm
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
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


sparse_encoding = False
audio_model_bands = 128
features_per_band = 8
transformer_context = False
use_audio_model = False
k_sparse = 64


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
    def __init__(self, is_disc=False):
        super().__init__()
        self.is_disc = is_disc


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


        self.judge = nn.Sequential(
            # 64
            nn.Sequential(
                nn.Conv1d(1024, 512, 3, 2, 1),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(512)
            ),
            # 32
            nn.Sequential(
                nn.Conv1d(512, 512, 3, 2, 1),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(512)
            ),
            # 16
            nn.Sequential(
                nn.Conv1d(512, 512, 3, 2, 1),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(512)
            ),
            # 8
            nn.Sequential(
                nn.Conv1d(512, 512, 3, 2, 1),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(512)
            ),
             # 4
            nn.Sequential(
                nn.Conv1d(512, 512, 3, 2, 1),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(512)
            ),
            # 1
            nn.Sequential(
                nn.Conv1d(512, 512, 4, 4, 0),
                nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(512)
            ),

            nn.Conv1d(512, 512, 1, 1, 0),
            nn.Conv1d(512, 1, 1, 1, 0),
        )


        # self.to_verb_context = LinearOutputStack(1024, 3, out_channels=1024, norm=nn.LayerNorm((1024,)))
        self.verb = ReverbGenerator(1024, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((1024,)))

        if use_audio_model:
            self.up = nn.Sequential(
                nn.Conv1d(1024, audio_model_bands, 1, 1, 0),
                AudioModel(
                    exp.n_samples, 
                    audio_model_bands, 
                    exp.samplerate, 
                    128, 
                    512, 
                    batch_norm=True, 
                    use_wavetable=False,
                    complex_valued_osc=False
                )
            )
            
        else:        

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
        # x = self.norm(x)

        if not self.is_disc:

            if sparse_encoding:
                salience = self.salience(x)
                salience = F.dropout(salience, 0.05)
                salience = salience.reshape(batch, -1)
                salience = torch.softmax(salience, dim=-1)
                salience = salience.reshape(batch, 1024, -1)
                encoding = salience = sparsify(salience, n_to_keep=k_sparse)
                x = x * salience
                x = self.context(x)
                return x, encoding
            else:
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

        if self.is_disc:
            features = []
            x = encoded
            for layer in self.judge:
                x = layer(x)
                features.append(x)
            return x, features
        
        ctx = torch.sum(encoded, dim=-1)
        # ctx = self.to_verb_context(ctx)

        x = self.generate(encoded)

        x = self.verb.forward(ctx, x)
        
        # x = max_norm(x)

        return x, encoding

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

disc = Model(is_disc=True).to(device)
disc_optim = optimizer(disc, lr=1e-3)



def train(batch, i):
    batch_size = batch.shape[0]

    optim.zero_grad()
    disc_optim.zero_grad()

    with torch.no_grad():
        feat = exp.perceptual_feature(batch)

    recon, encoding = model.forward(feat, i)
    r = exp.perceptual_feature(recon)

    

    # TODO: move this into a stand-alone function
    encoding = encoding.view(batch_size, -1)
    srt, indices = torch.sort(encoding, dim=-1, descending=True)

    # the first 128 atoms may be as large/loud as they need to be
    # TODO: This number could slowly drop over training time
    penalized = srt[:, 128:]
    non_zero = (encoding > 0).sum()
    sparsity = non_zero / encoding.nelement()
    print('sparsity', sparsity.item(), 'n_elements', (non_zero / batch_size).item())


    sparsity_loss = torch.abs(penalized).sum() * 0.01

    loss = F.mse_loss(r, feat) + sparsity_loss

    loss.backward()
    optim.step()

    # with torch.no_grad():
    #     batch = exp.perceptual_feature(batch)

    # if i % 2 == 0:
    #     print('-----------------------------')
    #     print('G')
    #     recon = model.forward(batch, i)
    #     rec_feat = exp.perceptual_feature(recon)
    #     j, features = disc.forward(rec_feat, i)
    #     rj, rf = disc.forward(batch, i)
    #     g_loss = torch.abs(1 - j).mean()
    #     feat_loss = 0
    #     for a, b, in zip(features, rf):
    #         feat_loss = feat_loss + F.mse_loss(a, b)
        
    #     loss = feat_loss + g_loss
    #     loss.backward()
    #     optim.step()

    # else:
    #     with torch.no_grad():
    #         recon = model.forward(batch, i)
    #         recon_spec = exp.perceptual_feature(recon)

    #     fj, features = disc.forward(recon_spec, i)
    #     rj, rf = disc.forward(batch, i)
    #     loss = torch.abs(0 - fj).mean() + torch.abs(1 - rj).mean()
    #     loss.backward()
    #     disc_optim.step()
    #     print('D', fj.mean().item(), rj.mean().item())


    return loss, recon
    


@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    
            

                
    
    