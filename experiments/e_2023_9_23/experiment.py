
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose, fft_resample
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify, sparsify2
from modules.stft import stft
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
    windowed_pif=False)

scale = zounds.LinearScale.from_sample_rate(exp.samplerate, 512)
bank = morlet_filter_bank(exp.samplerate, 64, scale, 0.1, normalize=True).real.astype(np.float32)
bank = torch.from_numpy(bank).to(device).view(512, 1, 64)


# def log_magnitude(x):
#     x = F.relu(x)
#     x = 20 * torch.log10(1 + x)
#     return x

# def feature(x: torch.Tensor):
#     batch_size = x.shape[0]

#     bands = fft_frequency_decompose(x, 512)

#     specs = []
#     for size, band in bands.items():
#         spec = F.conv1d(band, bank, padding=bank.shape[-1] // 2)
#         spec = log_magnitude(spec)

#         window_size = 64
#         step = window_size // 2
#         windowed = spec.unfold(-1, window_size, step)
#         windowed = windowed * torch.hamming_window(window_size, device=device)[None, None, None, :]
#         spec = torch.abs(torch.fft.rfft(windowed, dim=-1))
#         specs.append(spec.view(batch_size, -1))
    
#     specs = torch.cat(specs, dim=-1)

#     return specs



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_period = nn.Linear(257, 8)
        

        self.embed_cond = nn.Conv1d(4096, 256, 1, 1, 0)
        self.embed_spec = nn.Conv1d(1024, 256, 1, 1, 0)

        self.net = nn.Sequential(

            # 64
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 32
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 8
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
            ),


            nn.Conv1d(256, 1, 1, 1, 0)            
        )

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, cond, audio):
        batch_size = cond.shape[0]

        spec = exp.perceptual_feature(audio)

        x = self.embed_period(spec)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        spec = self.embed_spec(x)

        # TODO: Weirdly, this was doing some interesting things
        # as an unconditional discriminator (due to a mistake)
        cond = self.embed_cond(cond)
        x = cond + spec
        j = self.net(x)
        return j


class ContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, 1, 1)
        self.embed_conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.embed_pooled = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)

    
    def forward(self, x):
        x = F.dropout(x, 0.05)
        x = self.conv(x)
        pooled = F.avg_pool1d(x, 7, 1, padding=3)
        pooled = self.embed_pooled(pooled)
        x = self.embed_conv(x)
        x = x + pooled
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
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

        self.to_samples = nn.ModuleDict({
            '512': nn.Conv1d(256, 256, 7, 1, 3),
            '1024': nn.Conv1d(256, 256, 7, 1, 3),
            '2048': nn.Conv1d(256, 256, 7, 1, 3),
            '4096': nn.Conv1d(256, 256, 7, 1, 3),
            '8192': nn.Conv1d(256, 256, 7, 1, 3),
            '16384': nn.Conv1d(256, 256, 7, 1, 3),
            '32768': nn.Conv1d(256, 256, 7, 1, 3),
        })

        self.atoms = nn.ParameterDict({
            '512': torch.zeros(1, 256, 64).uniform_(-0.1, 0.1),
            '1024': torch.zeros(1, 256, 64).uniform_(-0.1, 0.1),
            '2048': torch.zeros(1, 256, 64).uniform_(-0.1, 0.1),
            '4096': torch.zeros(1, 256, 64).uniform_(-0.1, 0.1),
            '8192': torch.zeros(1, 256, 64).uniform_(-0.1, 0.1),
            '16384': torch.zeros(1, 256, 64).uniform_(-0.1, 0.1),
            '32768': torch.zeros(1, 256, 64).uniform_(-0.1, 0.1),
        })


        self.up = ConvUpsample(
            256, 256, 128, exp.n_samples, mode='learned', out_channels=256, from_latent=False, batch_norm=True)

        self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            32, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((32,)))
    
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
    
    def generate(self, encoded):
        ctxt = torch.sum(encoded, dim=-1)
        ctxt = self.verb_context.forward(ctxt)

        decoded = self.decoder.forward(encoded)

        # final = self.up.forward(decoded)

        samples = {}
        for layer in self.up:
            decoded = layer(decoded)
            key = str(decoded.shape[-1])
            if key in self.to_samples:
                band = self.to_samples[key].forward(decoded)

                band = F.pad(band, (0, 64))
                band = F.conv1d(band, self.atoms[key])

                samples[int(key)] = band

        final = fft_frequency_recompose(samples, exp.n_samples)
        # bands = {size: AF.resample()}

        final = self.verb.forward(ctxt, final)
        return final


    def forward(self, x):
        encoded = self.encode(x)

        # a = full sparse representation
        # b = packed (just active channels)
        # c = one_hot
        # a, b, c = sparsify2(encoded, n_to_keep=64)
        # print(a.shape, b.shape, c.shape)

        encoded = sparsify(encoded, n_to_keep=512)


        final = self.generate(encoded)
        return final, encoded


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)


try:
    model.load_state_dict(torch.load('sparse_conditioned_gen.dat'))
    disc.load_state_dict(torch.load('sparse_conditioned_disc.dat'))
    print('loaded model weights')
except IOError as err:
    print(err)



def perlin():
    scale = 1.0

    start = torch.zeros(1, 4096, 4, device=device).uniform_(0, scale)

    while start.shape[-1] < 128:
        # start = F.interpolate(start, scale_factor=2, mode='linear')
        start = fft_resample(start, desired_size=start.shape[-1] * 2, is_lowest_band=True)
        scale = scale / 2
        start = start + torch.zeros_like(start).uniform_(0, scale)
    
    start = sparsify(start, n_to_keep=512)
    return start

def train(batch, i):
    optim.zero_grad()
    disc_optim.zero_grad()



    with torch.no_grad():
        feat = exp.perceptual_feature(batch)
    

    if i % 2 == 0:
        recon, encoded = model.forward(feat)
        r = exp.perceptual_feature(recon)

        spec_loss = F.mse_loss(r, feat)
        j = disc.forward(encoded.clone().detach(), recon)
        

        loss = torch.abs(1 - j).mean() + spec_loss

        # with torch.no_grad():
        #     enc = perlin()
        #     gen = model.generate(enc)

        loss.backward()
        optim.step()

        print('GEN', loss.item())
        return loss, recon, encoded
    else:
        with torch.no_grad():
            recon, encoded = model.forward(feat)
        
        rj = disc.forward(encoded, batch)
        fj = disc.forward(encoded, recon)
        loss = (torch.abs(1 - rj) + torch.abs(0 - fj)).mean()
        loss.backward()
        disc_optim.step()
        print('DISC', loss.item())
        return None, None, None


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
            
            if i % 1000 == 0 and i > 0:
                print('SAVING')
                torch.save(model.state_dict(), 'sparse_conditioned_gen.dat')
                torch.save(disc.state_dict(), 'sparse_conditioned_disc.dat')


            if l is None:
                continue

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l)
