
import os
from typing import List, Tuple
import torch
from config.dotenv import Config
from config.experiment import Experiment
from data.datastore import iter_audio_segments, load_audio_chunk
from modules.multibanddict import BandSpec, GlobalEventTuple, MultibandDictionaryLearning
from modules.normalization import unit_norm
from modules.pointcloud import GraphEdgeEmbedding
from modules.search import BruteForceSearch
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from util import device
from util.playable import listen_to_sound, playable
from util.readmedocs import readme
import zounds
from conjure import numpy_conjure, SupportedContentType, audio_conjure, pickle_conjure
import numpy as np

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

atom_embeddings = model.atom_embeddings()

embedding_dim = int((model.total_atoms * (model.total_atoms - 1)) / 2)


# edge_embedding = GraphEdgeEmbedding(
#     n_items=model.event_count(sparse_coding_steps), 
#     embedding_dim=embedding_dim, 
#     out_channels=512).to(device)

def build_graph_embedding(
    model: MultibandDictionaryLearning,
    batch_size: int, 
    events: List[GlobalEventTuple],
    atom_embeddings):
    
    # get embeddings for each event
    ge = model.event_embeddings(batch_size, events, atom_embeddings)
    ge = torch.sum(ge, dim=1).view(batch_size, -1)
    return ge

    # get an embedding for the entire graph
    # edges = edge_embedding.forward(ge)
    # return edges
        

def compute_embedding(model: MultibandDictionaryLearning, batch: torch.Tensor, steps: int, atom_embeddings) -> torch.Tensor:
    # size -> (all_instances, scatter, shape)
    encoding = model.encode(batch, steps=steps)
    flat = model.flattened_event_tuples(encoding)
    embeddings = build_graph_embedding(model, batch.shape[0], flat, atom_embeddings)
    return embeddings


def round_trip(batch: torch.Tensor, steps: int) -> Tuple[zounds.AudioSamples, torch.Tensor]:
    # size -> (all_instances, scatter, shape)
    encoding = model.encode(batch, steps=steps)
    
    flat = model.flattened_event_tuples(encoding)
    print('FLAT TOTAL EVENTS', len(flat) // batch.shape[0])
    
    recon_encoding = model.hierarchical_event_tuples(flat, encoding)
    recon = model.decode(recon_encoding)
    return playable(recon, zounds.audio_sample_rate(exp.samplerate))


# NOTE: 2023-12-18 has relevant code for the "approximate" portion of this 

def train(batch, i):
    
    # TODO: Support different sparse coding iterations for each band
    with torch.no_grad():
        rt = round_trip(batch, steps=sparse_coding_steps)
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
    
    return loss, recon, fm, rt

def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)



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
    
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
        
        self.search = self.make_conjure_collection('search')
        
        
        @pickle_conjure(self.search, read_hook=lambda x: print('reading train_model() from cache'))
        def train_model(training_steps: int) -> MultibandDictionaryLearning:
        
            for i, batch in enumerate(self.stream):
                batch = batch.view(-1, 1, exp.n_samples)
                l, r, fm, rt = train(batch, i)
                
                self.rt = rt
                self.real = batch
                self.fake = r
                self.feature_map = fm
                # self.embeddings = embeddings
                
                print(l.item())
                self.after_training_iteration(l, i)
                
                if i >= training_steps:
                    break
            
            return model
        
        self.train_model = train_model
        
        @pickle_conjure(self.search, read_hook=lambda x: print('reading build_index() from cache'))
        def build_index(model: MultibandDictionaryLearning, index_size_limit: int) -> dict:
            
            atom_embeddings = model.atom_embeddings()
            
            def make_key(full_path: str, start: int, stop: int) -> str:
                path, filename = os.path.split(full_path)
                fn, ext = os.path.splitext(filename)
                return f'{fn}_{start}_{stop}'
            
            keys = []
            embeddings = []
            
            for key, chunk in iter_audio_segments(
                    Config.audio_path(), 
                    '*.wav', 
                    exp.n_samples, 
                    device=device, 
                    make_key=make_key):
                
                embedding = compute_embedding(model, chunk, steps=sparse_coding_steps, atom_embeddings=atom_embeddings)
                embedding = embedding.data.cpu().numpy().reshape(1, -1)
                
                print('\t', key, chunk.shape, embedding.shape)
                
                keys.append(key)
                embeddings.append(embedding)
                
                if len(keys) > index_size_limit:
                    break
            
            embeddings = np.concatenate(embeddings, axis=0)
            
            print(f'all done with {len(keys)} keys, {embeddings.shape} embeddings')
            
            db = dict(embeddings=embeddings, keys=keys)
            return db
        
        self.build_index = build_index
    
    
    def run(self):
        
        
        
        training_steps = 128
        chunks_to_idndex = 8192
        
        trained_model = self.train_model(training_steps=training_steps)
        db = self.build_index(trained_model, chunks_to_idndex)
        
        from subprocess import Popen, PIPE
        
        def filepath_from_key(key: str) -> str:
            _id, start, end = key.split('_')
            fp = os.path.join(Config.audio_path(), f'{_id}.wav')
            return fp
        
        def slice_from_key(key: str) -> slice:
            _id, start, end = key.split('_')
            return slice(int(start), int(end))
        
            
        keys = db['keys']
        embeddings = db['embeddings']
        
        total_items = len(keys)
        n_examples = 10
        
        embeddings = torch.from_numpy(embeddings).float()
        embeddings = unit_norm(embeddings)
        
        
        search = BruteForceSearch(embeddings, keys, n_results=5)
        
        for i in range(n_examples):
            print('==============================================')
            
            query_index = np.random.randint(0, total_items)
            query = embeddings[query_index]
            keys, e = search.search(query)
            
            for key, embedding in zip(keys, e):
                fp = filepath_from_key(key)
                slce = slice_from_key(key)
                chunk = load_audio_chunk(fp, slce, device=embeddings.device)
                
                p: zounds.AudioSamples = playable(chunk, exp.samplerate, normalize=True)
                listen_to_sound(p, wait_for_user_input=True)
            
        
        