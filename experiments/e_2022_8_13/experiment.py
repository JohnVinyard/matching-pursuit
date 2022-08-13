from config.dotenv import Config
from modules.linear import LinearOutputStack
from modules.reverb import NeuralReverb
from train.optim import optimizer
from upsample import ConvUpsample
from util import device, playable, readme
import zounds
import torch
from modules.ddsp import overlap_add
from torch import nn
from torch.nn import functional as F
from torch import jit
from modules.psychoacoustic import PsychoacousticFeature
from util import make_initializer


n_samples = 2 ** 15
samplerate = zounds.SR22050()

window_size = 512
step_size = window_size // 2
n_coeffs = (window_size // 2) + 1

n_frames = n_samples // step_size

model_dim = 128
latent_dim = 128
n_atoms = 32

n_bands = 128
kernel_size = 512

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


pif = PsychoacousticFeature([128] * 6).to(device)

init_weights = make_initializer(0.1)

def perceptual_feature(x):
    bands = pif.compute_feature_dict(x)
    return torch.cat(bands, dim=-2)


def perceptual_loss(a, b):
    return F.mse_loss(a, b)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)
        self.scale = nn.Conv1d(channels, channels, 3, 1,
                               dilation=dilation, padding=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1,
                              dilation=dilation, padding=dilation)

    def forward(self, x):
        batch = x.shape[0]
        skip = x
        scale = self.scale(x)
        gate = self.gate(x)
        x = torch.tanh(scale) * F.sigmoid(gate)
        out = self.out(x)
        next = self.next(x) + skip
        return next, out



def generate_event(envelope, transfer_functions, envelope_transfer):
    output = []

    transfer_functions = torch.complex(
        transfer_functions[:, :, 0, :], transfer_functions[:, :, 1, :])
    envelope_transfer = torch.complex(
        envelope_transfer[..., 0], envelope_transfer[..., 1])

    print(envelope.shape, transfer_functions.shape, envelope_transfer.shape)

    envelope = envelope.view(-1, 1, n_samples)
    transfer_functions = transfer_functions.view(-1, n_coeffs, n_frames)

    envelope_transfer = envelope_transfer.view(-1, n_coeffs)

    # apply the impulse transfer function to white noise
    impulse = torch.zeros(1, window_size).uniform_(-1, 1)
    impulse_spec = torch.fft.rfft(impulse, dim=-1, norm='ortho')
    impulse_spec = impulse_spec * envelope_transfer
    impulse = torch.fft.irfft(impulse_spec, dim=-1,
                              norm='ortho').view(-1, 1, window_size)

    for i in range(n_frames):
        start = i * step_size
        end = start + window_size

        env = envelope[:, :, start: end]
        if (env.shape[-1] < window_size):
            env = F.pad(env, (0, 256))

        # scale the impulse by the envelope
        excitation = impulse * env

        if i > 0:
            current = output[-1]
        else:
            current = torch.zeros_like(impulse)

        nxt = excitation + current
        spec = torch.fft.rfft(nxt, dim=-1, norm='ortho')
        spec = spec * transfer_functions[:, :, i]
        nxt = torch.fft.irfft(spec, dim=-1, norm='ortho')

        output.append(
            nxt * torch.hamming_window(window_size, device=envelope.device))

    output = torch.cat(output, dim=1)
    print('OUTPUT', output.shape)
    final = overlap_add(output[:, None, :, :])
    return final[..., :n_samples]


class EventGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.gen_env = ConvUpsample(
            latent_dim, model_dim, 8, n_frames, mode='nearest', out_channels=1)
        
        self.gen_transfer = ConvUpsample(
            latent_dim, model_dim, 8, n_frames, mode='nearest', out_channels=n_coeffs * 2)

        self.gen_impulse_transfer = LinearOutputStack(
            model_dim, 3, out_channels=n_coeffs * 2)

    def forward(self, x):
        x = x.view(-1, latent_dim)

        env = self.gen_env(x) ** 2
        env = F.upsample(env, size=n_samples, mode='linear')

        transfer = self.gen_transfer(x).view(-1, n_coeffs, 2, n_frames)
        impulse = self.gen_impulse_transfer(x).view(-1, n_coeffs, 2)

        return env, transfer, impulse


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        
        self.context = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 243),
            DilatedBlock(model_dim, 1),
        )

        self.gen_events = EventGenerator()

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate, n_samples)

        self.to_room = LinearOutputStack(
            model_dim, 3, out_channels=self.verb.n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)

        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]

        orig = x


        x = torch.zeros(batch, model_dim, x.shape[-1]).to(x.device)
        
        n = fb.forward(orig, normalize=False)

    
        for layer in self.context:
            n, o = layer.forward(n)
            x = x + o

        norms = torch.norm(x, dim=1)

        values, indices = torch.topk(norms, k=n_atoms, dim=-1)

        latents = []
        for b in range(batch):
            for i in range(n_atoms):
                latents.append(x[b, :, indices[b, i]][None, :])
            
        # latents = torch.gather(x, dim=-1, index=indices)
        latents = torch.cat(latents, dim=0).view(batch * n_atoms, latent_dim)

        env, transfer, impulse = self.gen_events(latents)

        atoms = generate_event(env, transfer, impulse).view(batch, n_atoms, n_samples)

        output = torch.zeros(orig.shape[0], 1, n_samples * 2)

        for b in range(x.shape[0]):
            for i in range(n_atoms):
                v = values[b, i]
                start = indices[b, i]
                end = start + n_samples
                output[b, :, start: end] += atoms[b, i]  * v
                
        
        final = output[..., :n_samples]

        agg, _ = torch.max(x, dim=-1)
        r = self.to_room(agg)
        m = self.to_mix(agg).view(-1, 1, 1)

        wet = self.verb.forward(final, r)
        final = (m * wet) + (final * (1 - m))

        return final


model = Model().to(device)
optim = optimizer(model)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return recon, loss


@readme
class WaveguideExpiriment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None

    def listen(self):
        return playable(self.fake, samplerate)

    def orig(self):
        return playable(self.real, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.fake, loss = train(item)

            print(i, loss.item())
