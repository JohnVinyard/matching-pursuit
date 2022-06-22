from vector_quantize_pytorch import VectorQuantize
import zounds
import torch
from torch import nn
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable
from torch.nn import functional as F

from util.readmedocs import readme
from util.weight_init import make_initializer

n_rooms = 8
n_samples = 2 ** 14
n_frames = 64
n_noise_frames = 512
samplerate = zounds.SR22050()
batch_size = 8
model_dim = 128
n_clusters = 512

pif = PsychoacousticFeature(kernel_sizes=[128] * 6).to(device)

band = zounds.FrequencyBand(20, samplerate.nyquist)
mel_scale = zounds.MelScale(band, 256)
n_samples = 2**14

fb = zounds.learn.FilterBank(
    samplerate, 
    512, 
    mel_scale, 
    0.01, 
    normalize_filters=True, 
    a_weighting=True).to(device)

init_weights = make_initializer(0.05)

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

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)

    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = F.leaky_relu(x + orig, 0.2)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()


        self.encode = nn.Sequential(
            nn.Conv1d(256, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
        ) 
        self.vq = VectorQuantize(
            model_dim, 
            n_clusters, 
            decay=0.9, 
            commitment_weight=1, 
            channel_last=False)
        
        self.decode = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 1),
        )

        self.audio = AudioModel(n_samples)
        self.apply(init_weights)

    
    def forward(self, x):
        x = fb.forward(x, normalize=False)
        x = e = self.encode(x)
        q, i, loss = self.vq.forward(x)
        x = self.decode.forward(q)
        signal = self.audio(x)
        return e, i, loss, signal

ae = AutoEncoder().to(device)
optim = optimizer(ae, lr=1e-4)

def train_ae(batch):
    optim.zero_grad()
    encoded, indices, commit_loss, signal = ae.forward(batch)

    fake = pif.scattering_transform(signal)
    fake = torch.cat(list(fake.values()))

    real = pif.scattering_transform(batch)
    real = torch.cat(list(real.values()))


    loss = F.mse_loss(fake, real) + (commit_loss * 10)
    loss.backward()
    optim.step()
    return signal, indices, loss


@readme
class VectorQuantizeExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.indices = None
        self.signal = None
    
    def listen(self):
        return playable(self.signal, samplerate)
    
    def view_indices(self):
        return self.indices.data.cpu().numpy()
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.signal, self.indices, loss = train_ae(item)

            if i % 10 == 0:
                print(loss.item())