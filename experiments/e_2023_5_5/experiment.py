import numpy as np
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.normalization import max_norm
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from conjure import time_series_conjure, audio_conjure, numpy_conjure

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DilatedStack(exp.model_dim, [1, 3, 9, 27, 81, 1])
        self.decoder = ConvUpsample(128, 32, 8, end_size=exp.n_samples, mode='nearest', out_channels=1)
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        spec = exp.pooled_filter_bank(x)
        encoded = self.encoder.forward(spec)
        encoded, _ = encoded.max(dim=-1)
        decoded = self.decoder.forward(encoded)
        decoded = max_norm(decoded)
        return decoded


ae = AutoEncoder().to(device)
optim = optimizer(ae, lr=1e-3)


def train_ae(batch):
    optim.zero_grad()
    recon = ae.forward(batch)
    loss = F.mse_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


def train(batch, i):
    return train_ae(batch)


@readme
class PointcloudAutoencoder(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
        self.fake = None
        self.vec = None
        self.encoded = None
        self.model = ae
        self.loss_func = None

        self._orig_audio = None
        self._orig_spec = None
        self._fake_audio = None
        self._fake_spec = None
    
    def after_training_iteration(self, l):
        # TODO: Move into base class
        self.loss_func(l)
        self._orig_audio(self.orig())
        self._orig_spec(self.real_spec())
        self._fake_audio(self.listen())
        self._fake_spec(self.fake_spec())

    @property
    def conjure_funcs(self):
        
        d = {
            'values': np.zeros((1,))
        }

        # TODO: Move into base class
        @time_series_conjure(self.collection, 'loss')
        def loss_func(l):
            """
            Append values to a time series
            """
            d['values'] = np.concatenate([d['values'], l.data.cpu().numpy().reshape((1,))], axis=-1)
            return d['values'].reshape((1, -1))
        
        @audio_conjure(self.collection)
        def orig_audio(x: zounds.AudioSamples):
            bio = x.encode()
            return bio.read()
        
        @numpy_conjure(self.collection)
        def orig_spec(x: np.ndarray):
            return x / (x.max() + 1e-12)
        
        @audio_conjure(self.collection)
        def fake_audio(x: zounds.AudioSamples):
            bio = x.encode()
            return bio.read()
        
        @numpy_conjure(self.collection)
        def fake_spec(x: np.ndarray):
            return x / (x.max() + 1e-12)
        
        self.loss_func = loss_func
        self._orig_audio = orig_audio
        self._orig_spec = orig_spec
        self._fake_audio = fake_audio
        self._fake_spec = fake_spec

        return [
            loss_func,
            orig_audio,
            orig_spec,
            # fake_audio,
            # fake_spec
        ]

    
    