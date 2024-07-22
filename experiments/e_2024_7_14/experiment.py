
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.anticausal import AntiCausalStack
from modules.normalization import max_norm
from modules.pointcloud import CanonicalOrdering
from modules.pos_encode import pos_encoded
from modules.reds import RedsLikeModel
from modules.softmax import sparse_softmax
from modules.sparse import sparsify, sparsify_vectors
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from conjure import numpy_conjure, SupportedContentType

exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

n_events = 16

def experiment_spectrogram(x: torch.Tensor):
    batch_size = x.shape[0]
    
    x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
    return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        analysis_channels = 1024
        
        self.embed_spec = nn.Conv1d(analysis_channels, analysis_channels, 1, 1, 0)
        self.embed_pos = nn.Conv1d(33, analysis_channels, 1, 1, 0)
        
        self.encoder = AntiCausalStack(analysis_channels, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 1], do_norm=True)
        self.to_event_vectors = nn.Conv1d(analysis_channels, analysis_channels, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(analysis_channels, 1, 1, 1, 0)
        
        self.to_params = nn.Linear(analysis_channels, 57)
        
        self.apply(lambda x: exp.init_weights(x))
        

    def encode(self, x: torch.Tensor):
        batch_size = x.shape[0]

        if x.shape[1] == 1:
            x = experiment_spectrogram(x)
        
        pos = pos_encoded(batch_size, x.shape[-1], n_freqs=16, device=device).permute(0, 2, 1)
        pos = self.embed_pos(pos)
        x = self.embed_spec(x)
        
        x = x + pos

        encoded = self.encoder.forward(x)
        event_vecs = self.to_event_vectors(encoded).permute(0, 2, 1) # batch, time, channels
        
        event_switch = self.to_event_switch(encoded)
        attn = torch.abs(event_switch).permute(0, 2, 1).view(batch_size, 1, -1)
        
        attn, attn_indices, values = sparsify(attn, n_to_keep=n_events, return_indices=True)
        
        vecs, indices = sparsify_vectors(event_vecs.permute(0, 2, 1), attn, n_to_keep=n_events, normalize=False)
        
        vecs = self.to_params(vecs)
        # vecs = torch.relu(vecs)
        
        # vecs[:, :, 15:53] = sparse_softmax(vecs[:, :, 15:53], dim=-1, normalize=True)
        # vecs = torch.abs(vecs)
        
        return vecs
    

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        return self.encode(x)

reds = RedsLikeModel(
    n_resonance_octaves=16, 
    n_samples=exp.n_samples, 
    samplerate=exp.samplerate, 
    use_wavetables=False).to(device)

transform = torch.zeros(57)
transform[-2] = 1

canonical = CanonicalOrdering(57, transform=transform).to(device)

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def generate_random(batch_size):
    
    noise_osc_mix = torch.softmax(torch.zeros((batch_size, n_events, 2), device=device).uniform_(0, 1), dim=-1)
    
    f0 = torch.zeros((batch_size, n_events, 1), device=device).uniform_(1e-4, 0.75) ** 4
    
    decay_choice = torch.zeros((batch_size, n_events, 1), device=device).uniform_(0.9, 0.9999)
    
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

        
def train(batch, i, rnd):
    batch_size, _, samples = batch.shape
    
    # with torch.no_grad():
    #     test_batch, target_labels = generate_random(batch_size)
    test_batch, target_labels = rnd
    
    
    optim.zero_grad()
    
    recon_labels = model.forward(test_batch)
    target_ordered = target_labels
    recon_ordered = recon_labels
    
    target_ordered = canonical.forward(target_labels)
    recon_ordered = canonical.forward(recon_labels)
    
    loss = F.mse_loss(target_ordered, recon_ordered)
    loss.backward()
    optim.step()
    
    # recon = generate_from_dense(recon_labels)
    # recon = torch.sum(recon, dim=1, keepdim=True)
    
    with torch.no_grad():
        labels = model.forward(batch)
        synth = generate_from_dense(labels)
        recon = max_norm(torch.sum(synth, dim=1, keepdim=True))
    
    return loss, recon, test_batch, recon_labels

def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)

def make_sched_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def events(x: torch.Tensor):
        x = x.data.cpu().numpy()[0]
        return x

    return (events,)


@readme
class SyntheticTestData(BaseExperimentRunner):
    
    real_params = MonitoredValueDescriptor(make_conjure)
    fake_params = MonitoredValueDescriptor(make_sched_conjure)
    
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    
    
    def run(self):
        
        
        
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            
            rnd = generate_random(self.batch_size)
            
            test_batch, target_labels = rnd
            self.real_params = target_labels    
            
            loss, recon, real, recon_labels = train(item, i, rnd)
            self.fake_params = recon_labels
            
            self.fake = max_norm(recon)
            self.real = max_norm(real)
            print(loss.item())
            
            self.after_training_iteration(loss, i)