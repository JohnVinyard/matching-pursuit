import torch
from torch import nn
import zounds
from torch.nn import functional as F
from modules.linear import LinearOutputStack
from modules.phase import overlap_add
from modules.stft import stft
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules.sparse import sparsify_vectors
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer
from train.optim import optimizer
import numpy as np

n_samples = 2 ** 15
samplerate = zounds.SR22050()
window_size = 512
step_size = window_size // 2
n_coeffs = window_size // 2 + 1
n_frames = n_samples // step_size

model_dim = 128

n_events = 16
resolution = 8

env_size = resolution
transfer_size = n_coeffs * 2 * resolution

event_dim = 1 + env_size + transfer_size

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


class EventRenderer(object):
    """
    Take a batch of vectors representing energy/bang/actication and 
    a spectral transfer function and render to audio
    """

    def __init__(self):
        super().__init__()

    def render(self, x):
        x = x.view(-1, event_dim)
        batch = x.shape[0]


        # remove the absolute time, it's not relevant here
        x = x[:, 1:]

        # TODO: This could be sparse interpolation for much finer control
        env = x[:, :env_size].view(-1, 1, resolution)
        amp = F.interpolate(env, size=n_frames, mode='linear')
        noise = torch.zeros(batch, window_size, n_frames, device=x.device).uniform_(-1, 1)
        energy = amp * noise

        transfer = x[:, env_size:].reshape(-1, n_coeffs * 2, resolution)
        coeffs = F.interpolate(transfer, size=n_frames, mode='linear')
        # tf = torch.complex(coeffs[:, :n_coeffs, :], coeffs[:, n_coeffs:, :])
        real = (coeffs[:, :n_coeffs, :] * 2) - 1
        imag = (coeffs[:, n_coeffs:, :] * np.pi)
        imag = torch.cumsum(imag, dim=-1)
        tf = real * torch.exp(1j * imag)

        output_frames = []
        for i in range(n_frames):

            if len(output_frames):
                local_energy = output_frames[-1] + energy[:, :, i: i + 1]
            else:
                local_energy = energy[:, :, i: i + 1]

            spec = torch.fft.rfft(local_energy, dim=1, norm='ortho')
            spec = spec * tf[:, :, i: i + 1]
            new_frame = \
                torch.fft.irfft(spec, dim=1, norm='ortho') \
                * torch.hamming_window(window_size, device=x.device)[None, :, None]

            output_frames.append(new_frame)

        output_frames = torch.cat(output_frames, dim=-1)
        output_frames = output_frames.view(batch, 1, window_size, n_frames)
        output = overlap_add(output_frames)[..., :n_samples]
        output = output.view(batch, 1, n_samples)
        return output


class AudioSegmentRenderer(object):
    def __init__(self):
        super().__init__()
    
    def render(self, x):
        x = x.view(-1, n_events, n_samples)
        batch = x.shape[0]

        times = (x[:, :, 0] * n_samples).int()

        output = torch.zeros(batch, 1, n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(n_events):
                time = times[b, i]
                output[b, :, time: time + n_samples] += x[b, i][None, :]
        
        output = output[..., :n_samples]
        return output


class Summarizer(nn.Module):
    """
    Summarize a batch of audio samples into a set of sparse
    events, with each event described by a single vector.

    Output will be (batch, n_events, event_vector_size)

    """

    def __init__(self):
        super().__init__()

        encoder = nn.TransformerEncoderLayer(model_dim, 4, model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(encoder, 6, norm=None)
        self.reduce = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)

        self.attend = nn.Linear(model_dim, 1)
        self.to_events = LinearOutputStack(model_dim, 4, out_channels=event_dim)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, 1, n_samples)
        x = fb.forward(x, normalize=False)
        x = fb.temporal_pooling(x, window_size, step_size)[..., :n_frames]
        pos = pos_encoded(batch, n_frames, 16).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)
        x = x.permute(0, 2, 1)
        x = self.context(x)
        attn = torch.softmax(self.attend(x), dim=1)
        x = x.permute(0, 2, 1)
        x, _ = sparsify_vectors(x, attn, n_events)
        x = self.to_events(x)
        x = torch.sigmoid(x)
        return x


class LossPredictor(nn.Module):
    """
    Accepts vector-encoded events and corresponding 
    audio and attempts to predict a loss based on the
    event parameters

    events[:, :, :1] should be used to align the event
    vectors with the audio representation prior to processing
    
    """

    def __init__(self):
        super().__init__()
    
        encoder = nn.TransformerEncoderLayer(model_dim, 4, model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(encoder, 6, norm=None)
        self.reduce = nn.Conv1d(model_dim + 33 + event_dim, model_dim, 1, 1, 0)

        self.to_loss = LinearOutputStack(model_dim, 3, out_channels=n_coeffs)
        self.apply(init_weights)

    def forward(self, events, x):
        batch = events.shape[0]

        events = events.view(-1, n_events, event_dim)
        event_times = events[:, :, 0]
        event_indices = (event_times * n_frames).int()

        evt = torch.zeros(batch, n_frames, event_dim, device=x.device)
        for b in range(batch):
            for i in range(n_events):
                index = event_indices[b, i]
                evt[b, index] = events[b, i]

        x = x.view(events.shape[0], 1, n_samples)
        batch = x.shape[0]
        x = x.view(-1, 1, n_samples)
        x = fb.forward(x, normalize=False)
        x = fb.temporal_pooling(x, window_size, step_size)[..., :n_frames]
        pos = pos_encoded(batch, n_frames, 16).permute(0, 2, 1)

        x = torch.cat([x, pos, evt.permute(0, 2, 1)], dim=1)
        x = self.reduce(x)
        x = x.permute(0, 2, 1)
        x - self.context(x)
        
        x = self.to_loss(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.atom_renderer = EventRenderer()
        self.audio_renderer = AudioSegmentRenderer()
        self.apply(init_weights)
    
    def render(self, params):
        atoms = self.atom_renderer.render(params)
        audio = self.audio_renderer.render(atoms)
        return audio
    
    def forward(self, x):
        x = self.summary(x)
        return x


model = Model()
optim = optimizer(model, lr=1e-3)

predictor = LossPredictor()
loss_optim = optimizer(predictor, lr=1e-3)


def train(batch):
    optim.zero_grad()
    loss_optim.zero_grad()

    params = model.forward(batch)
    recon = model.render(params)

    real_spec = stft(batch, window_size, step_size, pad=True)
    fake_spec = stft(recon, window_size, step_size, pad=True)
    real_loss = (real_spec - fake_spec).view(-1, n_frames, n_coeffs)
    

    pred_loss = predictor.forward(params, batch).view(-1, n_frames, n_coeffs)

    p_loss = torch.abs(real_loss - pred_loss).sum()
    p_loss.backward()
    loss_optim.step()

    return torch.abs(real_loss).sum(), p_loss, recon

def train_model(batch):
    optim.zero_grad()
    loss_optim.zero_grad()

    params = model.forward(batch)
    pred_loss = torch.abs(predictor.forward(params, batch)).sum()
    pred_loss.backward()
    optim.step()

@readme
class TransferFunctionReinforcementLearning(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
    
    def orig(self):
        return playable(self.real, samplerate)
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item

            if i % 2 == 0:
                audio_loss, predictor_loss, self.fake = train(item)
                print(audio_loss.item(), ' | ', predictor_loss.item())
            else:
                train_model(item)
            