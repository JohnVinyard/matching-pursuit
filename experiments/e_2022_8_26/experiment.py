import zounds
import torch
from torch import nn
from torch.nn import functional as F
from config.dotenv import Config
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale, morlet_filter_bank
import numpy as np
from modules.pif import AuditoryImage
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from modules.sparse import sparsify
from util import device
from util.readmedocs import readme
from train.optim import optimizer
from util.weight_init import make_initializer
from util import playable
from modules.stft import stft

n_bands = 128

model_dim = 128
window_size = 512
step_size = 256


n_samples = 2 ** 15
samplerate = zounds.SR22050()
n_frames = n_samples // step_size


band = zounds.FrequencyBand(40, samplerate.nyquist - 1000)
scale = zounds.MelScale(band, n_bands)


fb = zounds.learn.FilterBank(
    samplerate,
    512,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


init_weights = make_initializer(0.1)

pif = PsychoacousticFeature().to(device)
# aim = AuditoryImage(window_size, time_steps=n_frames, do_windowing=False, check_cola=False).to(device)

mel_scale = MelScale()
codec = AudioCodec(mel_scale)

def perceptual_feature(x):
    x = x.view(-1, 1, n_samples)
    bands = pif.compute_feature_dict(x)
    return torch.cat(list(bands.values()), dim=-1)
    # x = stft(x, window_size, step_size, pad=True, log_amplitude=True)
    # x = codec.to_frequency_domain(x.view(-1, n_samples))[..., 0]
    # return x


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)

# filters = morlet_filter_bank(
    # samplerate, n_samples, scale, 0.1, normalize=True).real


# noise = np.random.uniform(-1, 1, (1, n_samples))

# noise_spec = np.fft.rfft(noise, axis=-1, norm='ortho')
# filter_spec = np.fft.rfft(filters, axis=-1, norm='ortho')
# filtered_noise = noise_spec * filter_spec
# filtered_noise = np.fft.irfft(filtered_noise, norm='ortho')

# filters = torch.from_numpy(filters).view(n_bands, n_samples)
# noise = torch.from_numpy(filtered_noise).view(n_bands, n_samples)

# all_bands = torch.cat([filters, noise], dim=0).view(1, n_bands * 2, n_samples).float()

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
            n_frames * 8,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)
        
        # n_rooms = 8
        # self.verb = NeuralReverb(n_samples, n_rooms)
        self.verb = NeuralReverb.from_directory(Config.impulse_response_path(), samplerate, n_samples)

        self.to_rooms = LinearOutputStack(model_dim, 3, out_channels=self.verb.n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)
    
    def forward(self, x):
        x = x.view(-1, model_dim, n_frames)

        agg = x.mean(dim=-1)
        room = torch.softmax(self.to_rooms(agg), dim=-1)
        mix = torch.sigmoid(self.to_mix(agg)).view(-1, 1, 1)

        # harm = self.osc.forward(x)
        # noise = self.noise(x)

        # dry = harm + noise
        dry = x
        wet = self.verb(dry, room)
        signal = (dry * mix) + (wet * (1 - mix))
        return signal

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv1d(n_bands, model_dim, 7, 2, 3),  # 16384
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 8192
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 4096
            nn.LeakyReLU(0.2),
            # nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 2048
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 1024
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 512
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 256
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 128
            nn.Conv1d(model_dim, model_dim, 1, 1, 0)
            
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 1, 3)
        )

        self.verb = NeuralReverb.from_directory(Config.impulse_response_path(), samplerate, n_samples)

        self.to_rooms = LinearOutputStack(model_dim, 3, out_channels=self.verb.n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)
        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 1, n_samples)
        x = fb.forward(x, normalize=False)
        x = self.downsample(x)

        
        x = sparsify(x, n_to_keep=128)

        agg, _ = x.max(dim=-1)
        room = torch.softmax(self.to_rooms(agg), dim=-1)
        mix = torch.sigmoid(self.to_mix(agg)).view(-1, 1, 1)

        x = self.upsample(x)

        x = F.pad(x, (0, 1))
        dry = fb.transposed_convolve(x)

        wet = self.verb(dry, room)
        signal = (dry * mix) + (wet * (1 - mix))
        return signal


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return recon, loss


@readme
class SimpleWaveTableExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
        self.filters = fb.filter_bank.data.cpu().numpy().squeeze()
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def orig(self):
        return playable(self.real, samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            self.fake, loss = train(item)
            print(loss.item())
