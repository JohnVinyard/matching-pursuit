
from typing import Callable, Dict
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from conjure import numpy_conjure, SupportedContentType
from modules import stft
from modules.ddsp import NoiseModel
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.normalization import max_norm
from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.atoms import unit_norm
from modules.linear import LinearOutputStack
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.unet import UNet
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 16
context_dim = 16
impulse_size = 4096
resonance_size = 32768
samplerate = 22050
n_samples = 32768



class NormPreservingNetwork(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.net = LinearOutputStack(
            channels, layers, out_channels=channels, norm=nn.LayerNorm((channels,)))
        
    
    def forward(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True)
        transformed = self.net(x)
        transformed = torch.relu(transformed)
        transformed = unit_norm(transformed)
        transformed = transformed * norms
        return transformed
        

class PhysicalModelNetwork(nn.Module):
    
    def __init__(self, window_size, n_frames):
        super().__init__()
        self.window_size = window_size
        self.step_size = window_size // 2
        self.coeffs = window_size // 2 + 1
        
        
        self.proj = nn.Linear(256, 257)
        
        self.embed = NormPreservingNetwork(self.coeffs, layers=3)
        self.transform = NormPreservingNetwork(self.coeffs, layers=3)
        self.embed_shape = NormPreservingNetwork(self.coeffs, layers=3)
        self.leakage = LinearOutputStack(
            self.coeffs, layers=3, out_channels=self.coeffs, norm=nn.LayerNorm((self.coeffs,)))
        self.n_frames = n_frames
        
        self.to_spec = nn.Linear(self.coeffs, self.coeffs * 2, bias=False)   
        
        self.register_buffer('group_delay', torch.linspace(0, np.pi, self.coeffs))
        
        self.max_leakage = 0.2
        
        self.apply(lambda x: exp.init_weights(x))
             
        
    def signal_to_latent(self, x):
        batch = x.shape[0]
        windowed = windowed_audio(x, self.window_size, self.step_size)
        spec = torch.fft.rfft(windowed, dim=-1)
        mag = torch.abs(spec)
        embedded = self.embed(mag)
        return embedded.view(batch, -1, self.coeffs)
    
    def forward(self, imp, embeddings):
        embeddings = self.proj(embeddings).view(-1, 1, 257).repeat(1, self.n_frames, 1)
        imp = imp.view(-1, 1, exp.n_samples)
        audio = self._forward(imp, embeddings)
        return audio.view(-1, n_events, exp.n_samples)
    
    def _forward(self, control_signal, shape):
        batch_size = control_signal.shape[0]
        
        control = self.signal_to_latent(control_signal)
        n_frames = control.shape[-2]

        hidden_state = torch.zeros(batch_size, self.coeffs, device=control.device)
        phase = torch.zeros(batch_size, self.coeffs, device=control.device)
        
        frames = []
        
        for i in range(n_frames):
            hidden_state = hidden_state + control[:, i, :]
            
            # TODO: the shape should influence the hidden state transformation
            # energy "bounces around" in the system
            hidden_state = self.transform(hidden_state)
            shape_latent = self.embed_shape(shape[:, i, :])
            
            # the shape determines how quickly (or not) energy 
            # leaves the system
            leakage_ratio = self.leakage(hidden_state + shape_latent)
            leakage_ratio = torch.sigmoid(leakage_ratio) * self.max_leakage
            
            # get the leaked energy and convert to samples
            leaked = hidden_state * leakage_ratio
            spec = self.to_spec(leaked)
            real, imag = spec[:, :self.coeffs], spec[:, self.coeffs:]
            
            phase = phase + (self.group_delay[None, :] * (torch.tanh(imag) * 0.1))
            spec = real * torch.exp(1j * phase)
            
            # spec = torch.complex(real, imag)
            samples = torch.fft.irfft(spec)
            
            frames.append(samples[:, None, :])

            # extract the leaked energy from the system
            hidden_state = hidden_state - leaked
        
        frames = torch.cat(frames, dim=1)
        audio = overlap_add(frames[:, None, :, :], apply_window=True)[..., :exp.n_samples]
        audio = audio.view(batch_size, 1, exp.n_samples)
        return audio


class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels, mixer_channels=2):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        mixer_channels = mixer_channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=mixer_channels, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

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
            self.n_frames * 16,
            self.n_samples,
            self.channels,
            batch_norm=True,
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
        )
        
        self.to_env = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):
        
        env = self.to_env(x) ** 2
        env = F.interpolate(env, mode='linear', size=self.n_samples)
        
        x = self.to_frames(x)
        x = self.noise_model(x)
        x = x.view(-1, n_events, self.n_samples)
        
        x = x * env
        return x




class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNet(1024)
    
        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)

        
        self.res = PhysicalModelNetwork(512, 128)

        self.to_amp = nn.Linear(256, 1)

        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))

        self.to_context_mean = nn.Linear(4096, context_dim)
        self.to_context_std = nn.Linear(4096, context_dim)
        self.embed_memory_context = nn.Linear(4096, context_dim)

        self.from_context = nn.Linear(context_dim, 256)
        

        self.apply(lambda x: exp.init_weights(x))


    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded = self.encoder.forward(x)
        encoded = F.dropout(encoded, 0.05)


        return encoded
    
    def sparse_encode(self, x):
        encoded = self.encode(x)
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        return encoded
    
    def from_sparse(self, sparse, ctxt):
        encoded, packed, one_hot = sparsify2(sparse, n_to_keep=n_events)
        x, imp = self.generate(encoded, one_hot, packed, ctxt)
        return x, imp, encoded

    def generate(self, encoded, one_hot, packed, dense):
        
        # one hot is n_events vectors
        proj = self.from_context(dense)
        oh = self.embed_one_hot(one_hot)
        
        embeddings = proj[:, None, :] + oh


        # impulses
        imp = self.imp.forward(embeddings)
        padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        mixed = self.res.forward(padded, embeddings)
        mixed = unit_norm(mixed)

        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps

        final = F.pad(mixed, (0, n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :n_samples]

        final = self.verb.forward(dense, final)

        return final, imp

    def forward(self, x):
        encoded = self.encode(x)
        
        dense = torch.mean(encoded, dim=-1)
        mean = self.to_context_mean(dense)
        std = self.to_context_std(dense)
        dense = mean + (torch.zeros_like(mean).normal_(0, 1) * std)
        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        encoded = torch.relu(encoded)
        

        final, imp = self.generate(encoded, one_hot, packed, dense)
        return final, encoded, imp
    
    


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def dict_op(
        a: Dict[int, torch.Tensor],
        b: Dict[int, torch.Tensor],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[int, torch.Tensor]:
    return {k: op(v, b[k]) for k, v in a.items()}


def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d2 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    return dict(**d1, **d2)
    

def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

    target = multiband_transform(target)

    full = torch.sum(recon, dim=1, keepdim=True)
    full = multiband_transform(full)

    residual = dict_op(target, full, lambda a, b: a - b)
    
    loss = 0
    
    # for i in range(n_events):
    i = np.random.randint(0, n_events)
    ch = recon[:, i: i + 1, :]
    
    ch = multiband_transform(ch)

    t = dict_op(residual, ch, lambda a, b: a + b)

    diff = dict_op(ch, t, lambda a, b: a - b)
    loss = loss + sum([torch.abs(y).sum() for y in diff.values()])

    return loss



def train(batch, i):
    optim.zero_grad()

    b = batch.shape[0]
    
    recon, encoded, imp = model.forward(batch)
    
    recon_summed = torch.sum(recon, dim=1, keepdim=True)

    loss = single_channel_loss(batch, recon)
    
    loss.backward()
    optim.step()

    recon = max_norm(recon_summed)
    encoded = max_norm(encoded)
    
    return loss, recon, encoded


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
class PhysicalModel(BaseExperimentRunner):
    
    encoded = MonitoredValueDescriptor(make_conjure)
    
    def __init__(
        self, 
        stream, 
        port=None, 
        save_weights=False, 
        load_weights=False):
        
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            save_weights=save_weights, 
            load_weights=load_weights)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, n_samples)
            l, r, e = train(item, i)


            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l, i)