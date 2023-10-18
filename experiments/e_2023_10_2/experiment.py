from types import FunctionType
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from angle import windowed_audio
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.overlap_add import overlap_add
from modules.ddsp import NoiseModel
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
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


n_events = 32
context_dim = 16
impulse_size = 4096
resonance_size = 32768

base_resonance = 0.02
apply_group_delay_to_dither = True


def control_signal_loss(
        target: torch.Tensor, 
        stems: torch.Tensor, 
        control: torch.Tensor, 
        rendered: torch.Tensor):
    
    batch, _, n_samples = target.shape
    _, n_events, stem_samples = stems.shape
    _, n_atoms, _ = control.shape

    
    # find the best position for each stem
    stems = F.pad(stems, (0, n_samples - stem_samples))
    fm = fft_convolve(target, stems)

    values, indices = torch.max(fm, dim=-1)
    sparse = torch.zeros_like(fm)
    sparse = torch.scatter(sparse, dim=-1, index=indices, src=values)
    re_positioned = fft_convolve(sparse, stems)

    # TODO: would gradient flow better if I just do this manually, 
    # by index?
    complete = torch.sum(re_positioned, dim=1, keepdim=True)
    residual = target - complete

    # how close are the samples when aligned optimally?
    stem_loss = 0
    for i in range(n_events):
        res = residual + re_positioned[:, i: i + 1, ...]
        stem_loss = stem_loss + F.mse_loss(res, re_positioned[:, i: i + 1, ...])
    

    # how close is the sparse control signal to the optimal alignment?
    # The problem here is that no gradient flow occurs
    control_loss = F.mse_loss(control, sparse)


    

    


class RecurrentResonanceModel(nn.Module):
    def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.resonance_samples = resonance_samples

        n_atoms = 1024
        self.n_frames = resonance_samples // (window_size // 2)
        self.res_factor = (1 - base_resonance) * 0.9

        # TODO: should this be multi-band, or parameterized differently?
        # What about a convolutional model to produce impulse responses?

        band = zounds.FrequencyBand(40, exp.samplerate.nyquist)
        scale = zounds.LinearScale(band, n_atoms)
        bank = morlet_filter_bank(exp.samplerate, resonance_samples, scale, 0.01, normalize=True).real.astype(np.float32)
        bank = torch.from_numpy(bank).view(n_atoms, resonance_samples)

        # bank = torch.zeros_like(bank).uniform_(-1, 1)
        # self.atoms = nn.ParameterDict({str(k): v for k, v in fft_frequency_decompose(bank, 512).items()})

        self.atoms = nn.Parameter(bank)
        # self.register_buffer('atoms', bank)

        self.to_res = ConvUpsample(
            latent_dim, channels, 4, end_size=resonance_samples, mode='nearest', out_channels=1, from_latent=True, batch_norm=True)

        self.selection = nn.Linear(latent_dim, n_atoms)
        self.to_momentum = nn.Linear(latent_dim, 1)

    
    def forward(self, x):

        # compute resonance/sustain
        mom = base_resonance + (torch.sigmoid(self.to_momentum(x)) * self.res_factor)
        mom = torch.log(1e-12 + mom)
        mom = mom.repeat(1, 1, self.n_frames)
        mom = torch.cumsum(mom, dim=-1)
        mom = torch.exp(mom)
        new_mom = mom

        # compute resonance shape/pattern
        sel = torch.softmax(self.selection(x), dim=-1)
        # atoms = fft_frequency_recompose({int(k): v for k, v in self.atoms.items()}, self.resonance_samples)
        atoms = self.atoms
        res = sel @ atoms
        # res = self.to_res(x).view(-1, n_events, self.resonance_samples)

        windowed = windowed_audio(res, self.window_size, self.window_size // 2)
        windowed = unit_norm(windowed, dim=-1)
        windowed = windowed * new_mom[..., None]
        windowed = overlap_add(windowed, apply_window=False)[..., :self.resonance_samples]


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
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
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
        self.embed_context = nn.Linear(context_dim, 256)

        self.net = nn.Sequential(

            # 128
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 128
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 128
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 1, 3),
                nn.LeakyReLU(0.2),
            ),


            nn.Conv1d(256, 1, 1, 1, 0)            
        )



        self.apply(lambda x: exp.init_weights(x))

    def forward(self, cond, audio, context):
        batch_size = cond.shape[0]

        context = self.embed_context(context)

        spec = stft(audio, 2048, 256, pad=True).view(batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        spec = self.embed_spec(spec)
        cond = self.embed_cond(cond)
        x = cond + spec + context[:, :, None]
        j = self.net(x)
        return j



class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.down = nn.Sequential(
            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 4
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.up = nn.Sequential(
            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 128
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.proj = nn.Conv1d(1024, 4096, 1, 1, 0)
    
    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x
        
        for layer in self.up:
            x = layer(x)
            size = x.shape[-1]
            if size in context:
                x = x + context[size]
            
        x = self.proj(x)
        return x



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.embed = nn.Linear(257, 8)

        # TODO: What is the best analysis architecture?
        # Transformer? Dilated conv?, U-Net?
        # self.encoder = nn.Sequential(
        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 2), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=1),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),

        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 6), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=3),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),

        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 18), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=9),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),

        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 2), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=1),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),


        #     nn.Conv1d(1024, 4096, 1, 1, 0)
        # )

        self.encoder = UNet(1024)

        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)
        self.res = RecurrentResonanceModel(n_events, 256, 64, 1024, resonance_samples=resonance_size)
        self.mix = GenerateMix(256, 128, n_events)
        self.to_amp = nn.Linear(256, 1)


        # self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            context_dim, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((context_dim,)))


        self.to_context_mean = nn.Linear(4096, context_dim)
        self.to_context_std = nn.Linear(4096, context_dim)

        self.from_context = nn.Linear(context_dim, 4096)


        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(self.refractory_period, power=10, device=device))
    
        self.apply(lambda x: exp.init_weights(x))

    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded = self.encoder.forward(x)
        encoded = F.dropout(encoded, 0.05)

        ref = F.pad(self.refractory, (0, encoded.shape[-1] - self.refractory_period))
        encoded = fft_convolve(encoded, ref)[..., :encoded.shape[-1]]
        
        return encoded
    
    def generate(self, encoded, one_hot, packed, dense):

        ctxt = self.from_context(dense)
        
        ce = self.embed_context(ctxt)


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
        # mixed = unit_norm(mixed, dim=-1)

        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps

        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]

        # final = torch.sum(final, dim=1, keepdim=True)
        
        final = self.verb.forward(dense, final)

        # TODO: lose the verb, for now
        '''
        Get the stems, give them unit norm, and find the optimal 
        position for each, as well as the optimal amplitude/norm.

        Stems are aligned and loss is computed

        Loss for sparse control signal is compute from best matching positions
        '''

        return final, env


    def forward(self, x):
        encoded = self.encode(x)

        # TODO: This should be a mean/std instead
        # mean + (std * N(0, 1))
        non_sparse = torch.mean(encoded, dim=-1)

        non_sparse_mean = self.to_context_mean(non_sparse)
        non_sparse_std = self.to_context_std(non_sparse)
        non_sparse = non_sparse_mean + (torch.zeros_like(non_sparse_mean).normal_(0, 1) * non_sparse_std)


        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)

        final, env = self.generate(encoded, one_hot, packed, dense=non_sparse)
        return final, encoded, env, non_sparse


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)


def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):
    ws = 2048
    ss = 256

    target = stft(target, ws, ss, pad=True)

    full = torch.sum(recon, dim=1, keepdim=True)
    full = stft(full, ws, ss, pad=True)

    residual = target - full

    loss = 0
    
    for i in range(n_events):
        ch = recon[:, i: i + 1, :]
        ch = stft(ch, ws, ss, pad=True)
        t = residual + ch
        loss = loss + F.mse_loss(ch, t.clone().detach())
    
    return loss


def train(batch, i):
    optim.zero_grad()
    disc_optim.zero_grad()

    # with torch.no_grad():
    #     feat = exp.perceptual_feature(batch)
    
    if True or i % 2 == 0:
        recon, encoded, env, context = model.forward(batch)

        recon_summed = torch.sum(recon, dim=1, keepdim=True)

        # compute spec loss
        # fake_spec = stft(recon_summed, 2048, 256, pad=True)
        # real_spec = stft(batch, 2048, 256, pad=True)
        # spec_loss = F.mse_loss(fake_spec, real_spec)
        

        loss = single_channel_loss(batch, recon) * 10


        # make sure random encodings also sound reasonable
        # with torch.no_grad():
        #     random_encoding = torch.zeros_like(encoded).uniform_(-1, 1)
        #     e, p, oh = sparsify2(random_encoding, n_to_keep=n_events)
        #     dense = torch.zeros(e.shape[0], context_dim, device=e.device).normal_(0, 1)
        
        # fake, _ = model.generate(e, oh, p, dense=dense)
        # fake = fake.sum(dim=1, keepdim=True)
        # j = disc.forward(encoded.clone().detach(), recon_summed, context)[..., None]
        # fj = disc.forward(e.clone().detach(), fake, dense)[..., None]
        # j = torch.cat([j, fj], dim=-1)
        
        # adv_loss = (torch.abs(1 - j).mean() * 1)

        # loss = loss + adv_loss

        loss.backward()
        optim.step()

        print('GEN', loss.item())

        recon = max_norm(recon_summed)
        encoded = max_norm(encoded)
        return loss, recon, encoded
    else:
        with torch.no_grad():
            recon, encoded, env, context = model.forward(batch)
            recon_summed = torch.sum(recon, dim=1, keepdim=True)
        
        rj = disc.forward(encoded, batch, context)
        fj = disc.forward(encoded, recon_summed, context)
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