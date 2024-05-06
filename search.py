from config.dotenv import Config
from data.datastore import iter_audio_segments, load_audio_chunk
from experiments.e_2024_3_21.inference import model, ResonanceInferenceModel
import zounds
import torch
from modules.normalization import unit_norm
# from modules.pointcloud import GraphEdgeEmbedding
from modules.search import BruteForceSearch
from util.playable import listen_to_sound, playable
import conjure
import os
from util import device
import numpy as np
from time import time
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg', force=True)

n_samples = 2**15
samplerate = zounds.SR22050()

model: ResonanceInferenceModel = model.to(device)

collection = conjure.LmdbCollection('search')

# graph_embedding = GraphEdgeEmbedding(16, 16, 128).to(device)

def compute_embedding(model: ResonanceInferenceModel, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        channels, encoding, schedules = model.iterative(batch)
        
        b, n_points, dim = encoding.shape
        
        embeddings = torch.sum(encoding, dim=1)
        
        # embeddings = graph_embedding.forward(encoding)
        return embeddings


# @conjure.pickle_conjure(
#     collection, 
#     read_hook=lambda x: print('reading build_index() from cache'))

@conjure.conjure(
    content_type='application/octet-steam', 
    storage=collection,
    func_identifier=conjure.LiteralFunctionIdentifier('index'),
    param_identifier=conjure.LiteralParamsIdentifier('index_params'),
    serializer=conjure.PickleSerializer(),
    deserializer=conjure.PickleDeserializer(),
    read_from_cache_hook=lambda x: print('reading build_index() from cache'))
def build_index(model: ResonanceInferenceModel, index_size_limit: int) -> dict:
    
    def make_key(full_path: str, start: int, stop: int) -> str:
        path, filename = os.path.split(full_path)
        fn, ext = os.path.splitext(filename)
        return f'{fn}_{start}_{stop}'
    
    keys = []
    embeddings = []
    
    for key, chunk in iter_audio_segments(
            Config.audio_path(), 
            '*.wav', 
            n_samples, 
            device=device, 
            make_key=make_key):
        
        start_time = time()
        embedding = compute_embedding(model, chunk)
        embedding = embedding.data.cpu().numpy().reshape(1, -1)
        end_time = time()
        
        print('\t', key, chunk.shape, embedding.shape, f'{end_time - start_time:.2f} seconds')
        
        keys.append(key)
        embeddings.append(embedding)
        
        if len(keys) > index_size_limit:
            break
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    print(f'all done with {len(keys)} keys, {embeddings.shape} embeddings')
    
    db = dict(embeddings=embeddings, keys=keys)
    return db

def main(model: ResonanceInferenceModel):
        
    chunks_to_idndex = 8192
    
    db = build_index(model, chunks_to_idndex)
    
    
    def filepath_from_key(key: str) -> str:
        _id, start, end = key.split('_')
        fp = os.path.join(Config.audio_path(), f'{_id}.wav')
        return fp
    
    def slice_from_key(key: str) -> slice:
        _id, start, end = key.split('_')
        return slice(int(start), int(end))
    
        
    db_keys = db['keys']
    embeddings = db['embeddings']
    
    total_items = len(db_keys)
    n_examples = 10
    
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = unit_norm(embeddings)
    
    
    search = BruteForceSearch(embeddings, db_keys, n_results=5, visualization_dim=2)
    
    points = search.visualization().data.cpu().numpy()
    
    
    plt.scatter(points[:, 0], points[:, 1], s=0.1)
    plt.show()
    plt.clf()
    
    for i in range(n_examples):
        print('==============================================')
        
        query_index = np.random.randint(0, total_items)
        
        
        query = embeddings[query_index]
        query_key = db_keys[query_index]
        
        keys, e = search.search(query)
        
        e = e.data.cpu().numpy()
        
        plt.scatter(e[:, 0], e[:, 1])
        ax = plt.gca()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        plt.show()
        plt.clf()
        
        for key, embedding in zip(keys, e):
            fp = filepath_from_key(key)
            slce = slice_from_key(key)
            chunk = load_audio_chunk(fp, slce, device=embeddings.device)
            
            print(f'Query key {query_key}, result key {key}')    
            
            p: zounds.AudioSamples = playable(chunk, samplerate, normalize=True)
            listen_to_sound(p, wait_for_user_input=True)
            
            
            
if __name__ == '__main__':
    main(model)