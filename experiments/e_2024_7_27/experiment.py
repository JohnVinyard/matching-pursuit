
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules import stft
from modules.normalization import max_norm
from modules.reds import RedsLikeModel, fft_shift
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

def generate_random(batch_size, n_events=1):
    
    noise_osc_mix = torch.softmax(torch.zeros((batch_size, n_events, 2), device=device).uniform_(0, 1), dim=-1)
    
    f0 = torch.zeros((batch_size, n_events, 1), device=device).uniform_(1e-4, 0.75) ** 4
    
    decay_choice = torch.zeros((batch_size, n_events, 1), device=device).uniform_(0.92, 0.9999)
    
    freq_spacing = torch.zeros((batch_size, n_events, 1), device=device).uniform_(0.25, 2)
    
    noise_filters = torch.zeros((batch_size, n_events, 2), device=device).uniform_(1e-3, 1)
    
    filter_decays = torch.zeros((batch_size, n_events, 1), device=device).uniform_(0.2, 0.9999)
    
    res_filter = torch.zeros((batch_size, n_events, 2), device=device).uniform_(1e-3, 1)
    res_filter_2 = torch.zeros((batch_size, n_events, 2), device=device).uniform_(1e-3, 1)
    
    decays = torch.zeros((batch_size, n_events, 1), device=device).uniform_(0.8, 0.9999)
    
    env = torch.zeros((batch_size, n_events, 2), device=device).uniform_(1e-3, 1)
    room_choice = F.gumbel_softmax(torch.zeros((batch_size, n_events, reds.n_reverb_rooms), device=device).uniform_(1e-3, 1), dim=-1, hard=True)
    
    verb_mix = torch.softmax(torch.zeros((batch_size, n_events, 2), device=device).uniform_(0, 1), dim=-1)
    
    shifts = torch.zeros((batch_size, n_events, 1), device=device).uniform_(0, 1)
    amplitudes = torch.zeros((batch_size, n_events, 1), device=device).uniform_(0, 1)
    
    samples = reds.generate_test_data(
        noise_osc_mix=noise_osc_mix,
        f0_choice=f0,
        decay_choice=decay_choice,
        freq_spacing=freq_spacing,
        noise_filter=noise_filters,
        filter_decays=filter_decays,
        resonance_filter=res_filter,
        resonance_filter2=res_filter_2,
        decays=decays,
        env=env,
        verb_room_choice=room_choice,
        verb_mix=verb_mix,
        shifts=shifts,
        amplitudes=amplitudes
    )
    
    test_batch = torch.sum(samples, dim=1, keepdim=True)
    target_labels = torch.cat([
        noise_osc_mix,
        f0,
        decay_choice,
        freq_spacing,
        noise_filters,
        filter_decays,
        res_filter,
        res_filter_2,
        decays,
        env,
        room_choice,
        verb_mix,
        shifts,
        amplitudes
    ], dim=-1)
    
    return test_batch, target_labels

def generate_from_dense(labels: torch.Tensor):
    synth = reds.generate_test_data(
            noise_osc_mix=labels[:, :, :2],
            f0_choice=labels[:, :, 2:3],
            decay_choice=labels[:, :, 3:4],
            freq_spacing=labels[:, :, 4:5],
            noise_filter=labels[:, :, 5:7],
            filter_decays=labels[:, :, 7:8],
            resonance_filter=labels[:, :, 8:10],
            resonance_filter2=labels[:, :, 10:12],
            decays=labels[:, :, 12:13],
            env=labels[:, :, 13:15],
            verb_room_choice=labels[:, :, 15:53],
            verb_mix=labels[:, :, 53:55],
            shifts=labels[:, :, 55:56],
            amplitudes=labels[:, :, 56: 57]
        )
    recon = torch.sum(synth, dim=1, keepdim=True)
    return recon

reds = RedsLikeModel(
    n_resonance_octaves=16, 
    n_samples=exp.n_samples, 
    samplerate=exp.samplerate, 
    use_wavetables=False).to(device)


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.l1 = nn.Linear(in_channels, out_channels)
        self.l2 = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = torch.relu(x)
        x = self.l2(x)
        x = x + skip
        return x


class NERF(nn.Module):
    def __init__(self, encoding_channels, channels, layers):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.layers = layers
        
        
        self.embed_params = nn.Linear(57, encoding_channels)
        self.pos = nn.Parameter(torch.zeros(1, 2**15, encoding_channels).uniform_(-0.02, 0.02))
        
        self.up = nn.Linear(encoding_channels, channels)
        self.layers = nn.ModuleList([Layer(channels, channels) for _ in range(layers)])
        
        
        self.to_shifts = nn.Linear(57, 1)
        
        self.to_amps = nn.Linear(channels, 128)
        self.to_sampled = nn.Linear(channels, 128)
        self.to_phase = nn.Linear(channels, 128)
        self.to_freq = nn.Linear(channels, 128)
        
        
        def init_weights(p):
            with torch.no_grad():
                try:
                    p.weight.uniform_(-0.05, 0.05)
                except AttributeError:
                    pass

                try:
                    p.bias.fill_(0)
                except AttributeError:
                    pass
        self.apply(lambda x: init_weights(x))
        
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        batch, _, _ = params.shape
        # encoding = encoding.permute(0, 2, 1)
        
        shifts = self.to_shifts(params)
        
        params = self.embed_params(params)
        
        pos = fft_shift(self.pos, shifts)
        encoding = pos + params
        
        x = self.up(encoding)
        for layer in self.layers:
            x = layer(x)
        
        
        a = self.to_amps(x) ** 2
        s = self.to_sampled(x)
        
        # f = self.to_freq(x) ** 2
        # p = self.to_phase(x)
        
        # x = a * torch.sin(f + p)
        
        x = a * s
        x = x.permute(0, 2, 1)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x.view(batch, 1, exp.n_samples)
        return x


model = NERF(encoding_channels=16, channels=128, layers=5).to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i, rnd: Tuple[torch.Tensor, torch.Tensor]):
    optim.zero_grad()
    
    target, params = rnd
    
    recon = model.forward(params)
    
    real = stft(target, 2048, 256, pad=True)
    fake = stft(recon, 2048, 256, pad=True)
    loss = torch.abs(real - fake).sum()
    loss.backward()
    optim.step()
    
    return loss, recon, target
    
    

@readme
class EventNerf(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    
    def run(self):
        
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            
            rnd = generate_random(self.batch_size)
            
            
            loss, recon, real = train(item, i, rnd)
            
            self.fake = max_norm(recon)
            self.real = max_norm(real)
            print(loss.item())
            
            self.after_training_iteration(loss, i)