import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from angle import windowed_audio
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.ddsp import NoiseModel
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.fft import fft_convolve
from modules.latent_loss import latent_loss
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.overlap_add import overlap_add
from modules.pos_encode import pos_encoded
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.stft import stft
from modules.sparse import sparsify2
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from torch.distributions import Uniform, Normal


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_events = 32
impulse_size = 4096
resonance_size = 16384

base_resonance = 0.2
apply_group_delay_to_dither = True



class RecurrentResonanceModel(nn.Module):
    def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.resonance_samples = resonance_samples

        n_res = 512
        self.n_frames = resonance_samples // (window_size // 2)
        self.res_factor = (1 - base_resonance) * 0.99

        band = zounds.FrequencyBand(40, exp.samplerate.nyquist)
        scale = zounds.MelScale(band, n_res)
        bank = morlet_filter_bank(exp.samplerate, resonance_samples, scale, 0.01, normalize=True).real.astype(np.float32)
        bank = torch.from_numpy(bank)
        res = bank.view(1, n_res, resonance_samples)
        bands = fft_frequency_decompose(res, 512)
        bands = {str(k): v for k, v in bands.items()}
        self.res = nn.ParameterDict(bands)
        
        self.to_momentum = LinearOutputStack(channels, 3, out_channels=1, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_selection = LinearOutputStack(channels, 3, out_channels=n_res, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

    
    def forward(self, x):

        # TODO: can I bring back this more physics-y thing?  Was this ever the problem?

        mom = base_resonance + (torch.sigmoid(self.to_momentum(x)) * self.res_factor)
        mom = torch.log(1e-12 + mom)
        mom = mom.repeat(1, 1, self.n_frames)
        mom = torch.cumsum(mom, dim=-1)
        mom = torch.exp(mom)
        new_mom = mom


        sel = self.to_selection(x)
        res = fft_frequency_recompose({int(k): v for k, v in self.res.items()}, self.resonance_samples)
        res = sel @ res


        windowed = windowed_audio(res, self.window_size, self.window_size // 2)
        windowed = unit_norm(windowed, dim=-1)
        windowed = windowed * new_mom[..., None]
        windowed = overlap_add(windowed, apply_window=False)[..., :self.resonance_samples]


        windowed = max_norm(windowed)

        return windowed, new_mom


                        


class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=2, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        x = self.to_mix(x)
        x = x.view(-1, self.encoding_channels, 1)
        x = torch.softmax(x, dim=-1)
        return x

class GenerateImpulse(nn.Module):

    def __init__(self, latent_dim, channels, n_samples, n_filter_bands, encoding_channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.n_frames = n_samples // 256
        self.n_filter_bands = n_filter_bands
        self.channels = channels
        self.filter_kernel_size = 16
        self.encoding_channels = encoding_channels


        self.to_frames = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            mode='learned',
            end_size=self.n_frames,
            out_channels=channels,
            batch_norm=True
        )

        self.noise_model = NoiseModel(
            channels,
            self.n_frames,
            self.n_frames * 32,
            self.n_samples,
            self.channels,
            batch_norm=True,
            squared=True
        )

    
    def forward(self, x):
        x = self.to_frames(x)
        x = self.noise_model(x)
        return x.view(-1, n_events, self.n_samples)


        

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

        cond = self.embed_cond(cond)
        x = cond + spec
        j = self.net(x)
        return j




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(257, 8)
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 2), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 6), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 18), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=9),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 2), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),


            nn.Conv1d(1024, 4096, 1, 1, 0)
        )


        self.embed_context = LinearOutputStack(256, 3, in_channels=4096, norm=nn.LayerNorm((256,)))
        self.embed_one_hot = LinearOutputStack(256, 3, in_channels=4096, norm=nn.LayerNorm((256,)))
        
        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)
        self.res = RecurrentResonanceModel(n_events, 256, 128, 1024, resonance_samples=resonance_size)
        self.mix = GenerateMix(256, 128, n_events)
        self.to_amp = LinearOutputStack(256, 3, out_channels=1, norm=nn.LayerNorm((256,)))


        self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            32, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((32,)))
        
        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(self.refractory_period, power=10, device=device))
    
        self.apply(lambda x: exp.init_weights(x))

    def encode(self, x):
        batch_size = x.shape[0]

        if len(x.shape) != 4:
            x = exp.perceptual_feature(x)

        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        encoded = self.encoder.forward(x)
        # encoded = F.dropout(encoded, 0.05)

        ref = F.pad(self.refractory, (0, encoded.shape[-1] - self.refractory_period))
        encoded = fft_convolve(encoded, ref)[..., :encoded.shape[-1]]
        
        return encoded
    
    def generate(self, encoded, one_hot, packed):
        batch_size = encoded.shape[0]

        ctxt = torch.mean(encoded, dim=-1)
        ce = self.embed_context(ctxt)
        
        ctxt = self.verb_context.forward(ctxt)

        # TODO: consider adding context back in and/or dense context
        # first embed context and one hot and combine them
        oh = self.embed_one_hot(one_hot)
        embeddings = ce[:, None, :] + oh

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        res, env = self.res.forward(embeddings)

        # mixes
        mx = self.mix.forward(embeddings)

        conv = fft_convolve(padded, res)[..., :resonance_size]


        stacked  = torch.cat([padded[..., None], conv[..., None]], dim=-1)
        mixed = stacked @ mx.view(-1, n_events, 2, 1)
        mixed = mixed.view(-1, n_events, resonance_size)


        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps


        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]


        final = torch.sum(final, dim=1, keepdim=True)


        final = self.verb.forward(ctxt, final)
        return final, env


    def forward(self, x):
        encoded = self.encode(x)


        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)


        final, env = self.generate(encoded, one_hot, packed)
        return final, encoded, env


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
        recon, encoded, env = model.forward(feat)

        diversity = latent_loss(encoded.sum(dim=-1), mean_weight=0, std_weight=0)


        r = exp.perceptual_feature(recon)
        spec_loss = F.mse_loss(r, feat)

        j = disc.forward(encoded.clone().detach(), recon)
        
        adv_loss = (torch.abs(1 - j).mean() * 1)

        loss = spec_loss + adv_loss + diversity


        loss.backward()
        optim.step()

        print('GEN', loss.item())

        recon = max_norm(torch.sum(recon, dim=1, keepdim=True))
        encoded = max_norm(encoded)
        return loss, recon, encoded
    else:
        with torch.no_grad():
            recon, encoded, env = model.forward(feat)
        
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
        x = F.max_pool2d(x, (16, 8), (16, 8))
        x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        return x
    return (encoded,)



@readme
class GraphRepresentation(BaseExperimentRunner):
    
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
            print(i, l.item())
            self.after_training_iteration(l)