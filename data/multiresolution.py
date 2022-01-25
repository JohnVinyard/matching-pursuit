import torch
from data.datastore import batch_stream

from modules.decompose import fft_frequency_decompose
from modules.psychoacoustic import PsychoacousticFeature
import os

n_samples = 2**14
min_band_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature = PsychoacousticFeature().to(device)


path = os.environ['AUDIO_PATH']

def process_batch(s):
    s = s.reshape(-1, 1, n_samples)
    s = torch.from_numpy(s).to(device).float()
    bands = fft_frequency_decompose(s, min_band_size)
    return bands


def build_compute_feature_dict():
    stream = batch_stream(path, '*.wav', 16, n_samples)
    s = next(stream)

    bands = process_batch(s)
    feat = feature.compute_feature_dict(bands)

    print('Computing stats')
    means = {k: v.mean() for k, v in feat.items()}
    stds = {k: v.std() for k, v in feat.items()}

    def f(bands):
        x = feature.compute_feature_dict(bands)
        x = {k: v - means[k] for k, v in x.items()}
        x = {k: v / stds[k] for k, v in x.items()}
        return x

    return f


compute_feature_dict = build_compute_feature_dict()


def sample_stream(batch_size, overfit=False):
    stream = batch_stream(path, '*.wav', batch_size, n_samples, overfit=overfit)
    for s in stream:
        bands = process_batch(s)
        feat = compute_feature_dict(bands)
        yield bands, feat
