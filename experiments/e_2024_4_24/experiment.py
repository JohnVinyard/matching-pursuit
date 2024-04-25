
from typing import List, Tuple
import torch
from config.experiment import Experiment
from modules.multibanddict import BandSpec, GlobalEventTuple, MultibandDictionaryLearning
from modules.normalization import unit_norm
from modules.pointcloud import CanonicalOrdering
from modules.random import RandomProjection
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from util import device
from util.playable import playable
from util.readmedocs import readme
import zounds
from conjure import numpy_conjure, SupportedContentType, audio_conjure
from collections import Counter

exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_atoms = 512
sparse_coding_steps = 64


# TODO: signal_samples is not necessary and should be removed
model = MultibandDictionaryLearning([
    BandSpec(512,   n_atoms, 128, device=device, signal_samples=exp.n_samples, is_lowest_band=True),
    BandSpec(1024,  n_atoms, 128, device=device, signal_samples=exp.n_samples),
    BandSpec(2048,  n_atoms, 128, device=device, signal_samples=exp.n_samples),
    BandSpec(4096,  n_atoms, 128, device=device, signal_samples=exp.n_samples),
    BandSpec(8192,  n_atoms, 128, device=device, signal_samples=exp.n_samples),
    BandSpec(16384, n_atoms, 128, device=device, signal_samples=exp.n_samples),
    BandSpec(32768, n_atoms, 128, device=device, signal_samples=exp.n_samples),
], n_samples=exp.n_samples)


n_events = sparse_coding_steps * len(model)



upper_triangular = n_events * (n_events - 1) // 2


# upper_triangular = 131328

# random_proj = torch.zeros(upper_triangular, 128, device=device).uniform_(-1, 1)
random_proj = RandomProjection(
    upper_triangular, 128, norm=lambda x: unit_norm(x, dim=-1)).to(device)
# canonical_ordering = torch.zeros(14, 1, device=random_proj.device).uniform_(-1, 1)
canonical_ordering = CanonicalOrdering(14).to(device)

def build_graph_embedding(
    batch_size: int, 
    events: List[GlobalEventTuple]):
    
    # get a canonical ordering by projecting to a single
    # dimension and sorting
    ge = model.event_embeddings(batch_size, events)
    
    n_events = ge.shape[1]
    
    ge = canonical_ordering.forward(ge)
    
    print(ge.shape)
    
    # get a self-similarity matrix
    ssm = ge @ ge.permute(0, 2, 1)
    print(ssm.shape)
    
    indices = torch.triu_indices(ssm.shape[-1], ssm.shape[-1], offset=1)
    print(indices.shape)
    
    
    ut = ssm[:, indices[0], indices[1]]
    
    print(upper_triangular, ut.shape, n_events)
    
    proj = random_proj.forward(ut)
    proj = unit_norm(proj)
    return proj
        
        

def round_trip(batch: torch.Tensor, steps: int) -> Tuple[zounds.AudioSamples, torch.Tensor]:
    # size -> (all_instances, scatter, shape)
    encoding = model.encode(batch, steps=steps)
    
    total_events = sum(len(t[0]) for size, t in encoding.items())
    print('TOTAL EVENTS', total_events // batch.shape[0])
    
    flat = model.flattened_event_tuples(encoding)
    print('FLAT TOTAL EVENTS', len(flat) // batch.shape[0])
    
    embeddings = build_graph_embedding(batch.shape[0], flat)
    print('EMBEDDINGS', embeddings.shape)
    
    recon_encoding = model.hierarchical_event_tuples(flat, encoding)
    recon = model.decode(recon_encoding)
    return playable(recon, zounds.audio_sample_rate(exp.samplerate)), embeddings


# NOTE: 2023-12-18 has relevant code for the "approximate" portion of this 

def train(batch, i):
    
    # TODO: Support different sparse coding iterations for each band
    with torch.no_grad():
        rt, embeddings = round_trip(batch, steps=sparse_coding_steps)
        recon, events = model.recon(batch, steps=sparse_coding_steps)
        model.learn(batch, steps=sparse_coding_steps)
        # print(i, 'sparse coding step')
    
    
    fm_size = 512
    fm = torch.zeros(batch.shape[0], n_atoms * len(events), fm_size)
    
    offset = 0
    
    for size, evts in events.items():
        
        for evt in evts:
            # events are (index, batch, pos, atom)
            index, batch, pos, atom = evt
            atom_index = index
            
            
            # TODO: figure out why the plot isn't showing
            # the correct number of atoms
            # print(size, pos, atom_index)
            
            
            t = pos / size
            t = int(t * fm_size)
            
            fm[batch, offset + atom_index, t] = 1
        
        offset += n_atoms
        
   
    
    loss = torch.zeros(1, device=device)
    
    return loss, recon, fm, rt, embeddings

def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)


def make_conjure_embeddings(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded_atom_embeddings(x: torch.Tensor):
        x = x.data.cpu().numpy()
        return x

    return (encoded_atom_embeddings,)

def make_conjure_rt(experiment: BaseExperimentRunner):
    @audio_conjure(experiment.collection, identifier='roundtrip')
    def encoded_audio(x: zounds.AudioSamples):
        bio = x.encode()
        return bio.read()
        

    return (encoded_audio,)

@readme
class AudioSegmentEmbedding(BaseExperimentRunner):
    
    feature_map = MonitoredValueDescriptor(make_conjure)
    rt = MonitoredValueDescriptor(make_conjure_rt)
    embeddings = MonitoredValueDescriptor(make_conjure_embeddings)
    
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    
    def run(self):
        for i, batch in enumerate(self.stream):
            batch = batch.view(-1, 1, exp.n_samples)
            l, r, fm, rt, embeddings = train(batch, i)
            
            self.rt = rt
            self.real = batch
            self.fake = r
            self.feature_map = fm
            self.embeddings = embeddings
            
            print(l.item())
            self.after_training_iteration(l, i)