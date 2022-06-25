from mimetypes import init
import zounds
from torch import nn
import torch
from modules.ddsp import NoiseModel, OscillatorBank, overlap_add
from modules.linear import LinearOutputStack
from train.gan import least_squares_disc_loss, least_squares_generator_loss
from modules.phase import windowed_audio
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from modules.pos_encode import pos_encoded
from train.optim import optimizer

from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer


n_samples = 2**15
samplerate = zounds.SR22050()

model_dim = 128
freq_bands = 128
kernel_size = 512
step_size = kernel_size // 2
n_rooms = 8
n_frames = n_samples // 256
n_noise_frames = n_samples // 32

frames_to_predict = 16
samples_to_predict = frames_to_predict * 256

band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, freq_bands)
fb = zounds.learn.FilterBank(
    samplerate, 
    kernel_size, scale, 
    0.01, 
    normalize_filters=True, 
    a_weighting=False).to(device)
aim = AuditoryImage(512, n_samples // step_size, do_windowing=False, check_cola=False).to(device)

init_weights = make_initializer(0.1)


class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim, 
            model_dim, 
            n_samples, 
            constrain=True, 
            lowest_freq=40 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=False)
        
        self.noise = NoiseModel(
            model_dim,
            n_frames,
            n_noise_frames,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)
        
        self.verb = NeuralReverb(n_samples, n_rooms)

        self.to_rooms = LinearOutputStack(model_dim, 3, out_channels=n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)

    
    def forward(self, x):
        x = x.view(-1, model_dim, n_frames)

        agg = x.mean(dim=-1)
        room = self.to_rooms(agg)
        mix = torch.sigmoid(self.to_mix(agg)).view(-1, 1, 1)

        harm = self.osc.forward(x)
        noise = self.noise(x)

        dry = harm + noise
        wet = self.verb(dry, room)
        signal = (dry * mix) + (wet * (1 - mix))
        return signal


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_scattering = LinearOutputStack(
            model_dim, 2, in_channels=kernel_size // 2 + 1, out_channels=8)
        self.reduce = LinearOutputStack(model_dim, 2, in_channels=model_dim * 8)

        self.reduce_again = LinearOutputStack(model_dim, 2, in_channels=model_dim + 33)

        layer = nn.TransformerEncoderLayer(model_dim, 4, model_dim)
        layer.norm1 = nn.Identity()
        layer.norm2 = nn.Identity()
        self.net = nn.TransformerEncoder(layer, 6)
        self.final = LinearOutputStack(model_dim, 2, out_channels=1)
        self.apply(init_weights)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = fb.forward(x, normalize=False)
        x = aim(x)
        x = self.reduce_scattering(x)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, model_dim * 8)
        x = self.reduce(x)

        pos = pos_encoded(batch_size, x.shape[1], 16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.reduce_again(x)

        x = self.net.forward(x)
        x = self.final(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_scattering = LinearOutputStack(
            model_dim, 2, in_channels=kernel_size // 2 + 1, out_channels=8)
        self.reduce = LinearOutputStack(model_dim, 2, in_channels=model_dim * 8)
        self.reduce_again = LinearOutputStack(model_dim, 2, in_channels=model_dim + 33)

        layer = nn.TransformerEncoderLayer(model_dim, 4, model_dim)
        layer.norm1 = nn.Identity()
        layer.norm2 = nn.Identity()
        self.net = nn.TransformerEncoder(layer, 6)
        
        self.audio = AudioModel(n_samples)
        self.apply(init_weights)
    
    def forward(self, x):
        orig = x

        batch_size = x.shape[0]
        x = fb.forward(x, normalize=False)
        x = aim(x)
        x = self.reduce_scattering(x)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, model_dim * 8)
        x = self.reduce(x)

        pos = pos_encoded(batch_size, x.shape[1], 16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.reduce_again(x)

        x = self.net.forward(x).permute(0, 2, 1)

        signal = self.audio(x)

        ow = windowed_audio(orig, kernel_size, step_size)
        sw = windowed_audio(signal, kernel_size, step_size)

        final = torch.zeros_like(ow)
        final[:, :, :-frames_to_predict, :] = ow[:, :, :-frames_to_predict, :]
        final[:, :, -frames_to_predict:, :] = sw[:, :, -frames_to_predict:, :]

        signal = overlap_add(final, apply_window=False)
        
        return signal
        

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-4)

gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-4)

def train_disc(batch):
    disc_optim.zero_grad()

    r = batch.clone()
    rj = disc(r)[:, -frames_to_predict:, :]

    f = batch.clone()
    f[:, :, -samples_to_predict:] = 0
    f = gen(f)
    fj = disc(f)[:, -frames_to_predict:, :]

    # loss = -(torch.mean(rj) - torch.mean(fj))
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()

    # for p in disc.parameters():
    #     p.data.clamp_(-0.01, 0.01)
    
    return loss

def train_gen(batch):
    gen_optim.zero_grad()

    f = batch.clone()
    f[:, :, -samples_to_predict:] = 0
    
    f = gen(f)
    fj = disc(f)[:, -frames_to_predict:, :]
    # loss = -torch.mean(fj)
    loss = least_squares_generator_loss(fj)
    loss.backward()
    gen_optim.step()
    return loss, f

@readme
class AdversarialAutoregressiveExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.fake = None
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            
            if i % 2 == 0:
                gen_loss, self.fake = train_gen(item)
            else:
                disc_loss = train_disc(item)
            

            if i > 0 and i % 10 == 0:
                print('G', gen_loss.item())
                print('D', disc_loss.item())
                print('===================================')
