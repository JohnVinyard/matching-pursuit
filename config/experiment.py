import zounds
from modules.phase import MelScale
from modules.pif import AuditoryImage
from modules.psychoacoustic import PsychoacousticFeature

from util import device
from util.weight_init import make_initializer
import torch
from torch.nn import functional as F

from modules import AuditoryImage


class Experiment(object):
    def __init__(self, samplerate, n_samples, model_dim=128, weight_init=0.1, kernel_size=512):
        super().__init__()
        self.samplerate = samplerate
        self.n_samples = n_samples
        self.window_size = 512
        self.step_size = self.window_size // 2
        self.n_frames = n_samples // self.step_size

        self.n_bands = model_dim
        self.model_dim = model_dim
        self.kernel_size = kernel_size

        band = zounds.FrequencyBand(40, samplerate.nyquist)
        self.scale = zounds.MelScale(band, self.n_bands)
        self.fb = zounds.learn.FilterBank(
            samplerate,
            self.kernel_size,
            self.scale,
            0.1,
            normalize_filters=True,
            a_weighting=False).to(device)

        self.init_weights = make_initializer(weight_init)

        self.pif = PsychoacousticFeature().to(device)

        self.aim = AuditoryImage(
            512, 128, do_windowing=True, check_cola=True).to(device)

    def perceptual_feature(self, x):
        # bands = self.pif.compute_feature_dict(x)
        # return torch.cat(list(bands.values()), dim=-1)

        x = self.fb.forward(x, normalize=False)
        x = self.aim.forward(x)
        return x

    def perceptual_loss(self, a, b):
        a = self.perceptual_feature(a)
        b = self.perceptual_feature(b)
        return F.mse_loss(a, b)
