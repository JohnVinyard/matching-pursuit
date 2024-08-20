from typing import Callable, Dict

import numpy as np
import torch
import zounds
from conjure import numpy_conjure, SupportedContentType
from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F

from modules.angle import windowed_audio
from config.experiment import Experiment
from modules.ddsp import NoiseModel
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.overlap_add import overlap_add
from modules.pif import fft_based_pif
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from modules.unet import UNet
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from modules.upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2 ** 15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=512)

n_events = 64
context_dim = 16
impulse_size = 4096
resonance_size = 32768


class RecurrentConservationOfEnergyModel(nn.Module):
    def __init__(self, latent_dim, channels, n_samples):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.n_samples = n_samples
        
        self.window_size = channels
        self.step_size = channels // 2
        
        self.n_frames = self.n_samples // self.step_size
        
        self.to_shape_deformations = ConvUpsample(
            latent_dim, 
            channels, 
            start_size=8, 
            end_size=self.n_frames, 
            mode='learned', 
            out_channels=channels, 
            from_latent=True, 
            batch_norm=True)
        
        self.embed_impulse = LinearOutputStack(
            channels, 3, out_channels=channels, norm=nn.LayerNorm((channels,)))
        
        self.to_output_latent = LinearOutputStack(
            channels, 3, out_channels=channels, norm=nn.LayerNorm((channels,)))
        
        self.to_output = LinearOutputStack(
            channels, 3, out_channels=channels, norm=nn.LayerNorm((channels,)))
        
        self.to_weight = nn.Linear(channels, channels)
        self.to_bias = nn.Linear(channels, channels)    
        self.to_leakage = nn.Linear(channels, 1)
        
    
    def forward(self, embedding: torch.Tensor, impulse: torch.Tensor):
        """
        embedding is (batch, n_events, latent_dim)
        impulse is   (batch, n_events, impulse_samples)
        """
        
        batch, n_events, _ = embedding.shape

        # embedding should generate a series of shape parameters
        deformations = self.to_shape_deformations(embedding)
        deformations = deformations.view(batch, n_events, self.channels, self.n_frames)
        deformations = unit_norm(deformations, dim=2)
        deformations = deformations.permute(0, 1, 3, 2)
        weights = self.to_weight(deformations)
        biases = self.to_bias(deformations)
        leakage = 0.1 + (torch.sigmoid(self.to_leakage(deformations)) * 0.98)
        
        
        # we should embed the windowed impulse, maintaining the original norm at each step
        windowed = windowed_audio(impulse, self.window_size, self.step_size)
        norms = torch.norm(windowed, dim=-1, keepdim=True)
        embedded = self.embed_impulse(windowed)
        embedded = unit_norm(embedded, dim=-1)
        embedded = embedded * norms
        
        n_impulse_frames = embedded.shape[2]
        
        hidden_state = torch.zeros(batch, n_events, self.channels, device=embedding.device)
        output_frames = []
        
        
        for i in range(self.n_frames):
            
            # embedded impulse is added to hidden state
            if i < n_impulse_frames:
                hidden_state = hidden_state + embedded[:, :, i, :]
            
            # how much energy is in the system?
            current_norm = torch.norm(hidden_state, dim=-1, keepdim=True)
            
            # the hidden state is warped by the current shape
            # but its norm is maintained
            hidden_state = (hidden_state * weights[:, :, i, :]) + biases[:, :, i, :]
            hidden_state = unit_norm(hidden_state) * current_norm
            
            # now, we generate output that is some fraction of the overall
            # energy
            output_latent = self.to_output_latent(hidden_state)
            output_latent = unit_norm(output_latent)
            output_latent = output_latent * (current_norm * leakage[:, :, i, :])
            
            # how much energy is leaving the system?
            output_norm = torch.norm(output_latent, dim=-1, keepdim=True)
            
            # energy leaves the system
            hidden_state = hidden_state - output_latent
            hidden_state = unit_norm(hidden_state, dim=-1) * (current_norm - output_norm)
            
            # the audio emitted has the same norm as the latent
            # leakage vector
            output_frame = self.to_output(output_latent)
            output_frame = output_frame * torch.hamming_window(self.window_size, device=embedded.device)[None, None, :]
            output_frame = unit_norm(output_frame)
            output_frame = output_frame * output_norm
            
            output_frames.append(output_frame[:, :, None, :])
        
        output_frames = torch.cat(output_frames, dim=2)
        output_frames = output_frames.view(batch, n_events, self.n_frames, self.window_size)
        output_frames = overlap_add(output_frames, apply_window=False)[..., :self.n_samples]
        output_frames = output_frames.view(-1, n_events, self.n_samples)
        return output_frames
            

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

    def forward(self, x):
        x = self.to_frames(x)
        x = self.noise_model(x)
        return x.view(-1, n_events, self.n_samples)




class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNet(1024)

        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)
        
        self.res = RecurrentConservationOfEnergyModel(256, 512, resonance_size)
        
        self.verb = ReverbGenerator(
            context_dim, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((context_dim,)))

        self.embed_memory_context = nn.Linear(4096, context_dim)

        self.from_context = nn.Linear(context_dim, 4096)

        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(
            self.refractory_period, power=10, device=device))

        self.apply(lambda x: exp.init_weights(x))

    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded = self.encoder.forward(x)
        encoded = F.dropout(encoded, 0.05)

        ref = F.pad(self.refractory,
                    (0, encoded.shape[-1] - self.refractory_period))
        encoded = fft_convolve(encoded, ref)[..., :encoded.shape[-1]]

        return encoded
    
    def sparse_encode(self, x):
        encoded = self.encode(x)
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        return encoded

    def generate(self, encoded, one_hot, packed):
        ctxt = torch.sum(encoded, dim=-1)
        dense = self.embed_memory_context(ctxt)  # (batch, context_dim)

        # ctxt is a single vector
        ce = self.embed_context(ctxt)

        # one hot is n_events vectors
        oh = self.embed_one_hot(one_hot)

        embeddings = ce[:, None, :] + oh

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)

        # resonances
        mixed = self.res.forward(embeddings, imp)
        
        padded = F.pad(imp, (0, resonance_size - impulse_size))
        
        mixed = mixed + padded


        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]

        final = self.verb.forward(dense, final)

        return final, imp

    def forward(self, x):
        encoded = self.encode(x)

        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        encoded = torch.relu(encoded)
        

        final, imp = self.generate(encoded, one_hot, packed)
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
    
    # pif = exp.perceptual_feature(x)
    # pif = fft_based_pif(x, 256, 64)
    # return { 'pif': pif }




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

    # b = batch.shape[0]
    
    recon, encoded, imp = model.forward(batch)
    
    energy_loss = torch.abs(imp).sum(dim=-1).mean() * 1e-5
    print('ENERGY LOSS', energy_loss.item())
    
    recon_summed = torch.sum(recon, dim=1, keepdim=True)

    loss = (single_channel_loss(batch, recon) * 1e-6) + energy_loss
    

    loss.backward()
    optim.step()

    print('GEN', loss.item())

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
class ConservationOfEnergy(BaseExperimentRunner):
    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None, load_weights=True, save_weights=False, model=model):
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            load_weights=load_weights, 
            save_weights=save_weights, 
            model=model)

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)


            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l, i)
