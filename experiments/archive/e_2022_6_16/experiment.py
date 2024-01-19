from imp import init_builtin
import torch
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable, readme
import zounds
from torch.nn import functional as F
from torch import nn

from util.weight_init import make_initializer

samplerate = zounds.SR22050()
band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)
n_samples = 2**14

fb = zounds.learn.FilterBank(samplerate, 512, scale, 0.1, normalize_filters=True, a_weighting=False).to(device)
aim = AuditoryImage(512, 64, do_windowing=True, check_cola=True).to(device)

n_clusters = 512

model_dim = 128
n_frames = 64
n_noise_frames = 512
n_rooms = 8


init_weights = make_initializer(0.1)

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


class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim, 
            len(scale), 
            n_samples, 
            constrain=False, 
            lowest_freq=40 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=True)
        
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


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.encode = nn.Sequential(
        #     nn.Conv2d(64, 64, 7, 4, 3),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 64, 7, 4, 3),
        #     nn.LeakyReLU(0.2),
        # )

        self.reduce = LinearOutputStack(model_dim, 3, out_channels=8, in_channels=257)
        self.encode = nn.Sequential(
            nn.Conv1d(8 * model_dim, model_dim, 1, 1, 0),
            DilatedBlock(128, 1),
            DilatedBlock(128, 3),
            DilatedBlock(128, 9),
            DilatedBlock(128, 1),
        )

        self.quantize = nn.Conv1d(model_dim, n_clusters, 1, 1, 0)

        self.decode = nn.Sequential(
            nn.Conv1d(n_clusters, 128, 1, 1, 0),
            DilatedBlock(128, 1),
            DilatedBlock(128, 3),
            DilatedBlock(128, 9),
            DilatedBlock(128, 1),
        )

        self.to_envelope = nn.Conv1d(model_dim, 1, 7, 1, 3)

        self.audio = AudioModel(n_samples)

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 1, n_samples)
        x = fb.forward(x, normalize=False)
        x = aim.forward(x)


        x = self.reduce(x)

        x = x.permute(0, 3, 1, 2).reshape(-1, 8 * model_dim, 64)
        x = self.encode(x)


        # x = x.permute(0, 2, 1, 3)
        # x = self.encode(x)
        # x = x.view(-1, 64, 8 * 17).permute(0, 2, 1)

        x = self.quantize(x)

        x = x.permute(0, 2, 1).reshape(-1, n_clusters)
        indices = torch.argmax(x, dim=-1)

        n = torch.zeros_like(x)
        n[torch.arange(x.shape[0]), indices] = x[torch.arange(x.shape[0]), indices]
        n = n / (n.max(dim=-1, keepdim=True)[0] + 1e-12)


        n = n.view(-1, 64, n_clusters).permute(0, 2, 1)

        x = self.decode(n)

        env = torch.abs(self.to_envelope(x))
        env = F.interpolate(env, size=n_samples, mode='linear')
        signal = self.audio(x)
        signal = signal * env
        return n, signal
    
ae = AutoEncoder().to(device)
optim = optimizer(ae, lr=1e-4)


def train_ae(batch):
    optim.zero_grad()
    encoded, decoded = ae.forward(batch)

    f = fb.forward(decoded, normalize=False)
    f = aim.forward(f)

    r = fb.forward(batch, normalize=False)
    r = aim.forward(r)

    loss = F.mse_loss(f, r)
    loss.backward()

    optim.step()
    return loss, encoded, decoded


@readme
class ScatteringTransformTest(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None

        self.encoded = None
        self.decoded = None

    def listen(self):
        return playable(self.decoded, samplerate)
    
    def code(self):
        return self.encoded.data.cpu().numpy()[0]

    def r(self):
        return playable(self.real, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item

            loss, self.encoded, self.decoded = train_ae(item)

            if i % 10 == 0:
                print(loss.item())
