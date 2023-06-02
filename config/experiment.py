import zounds
from modules.psychoacoustic import PsychoacousticFeature
# from modules.pif import AuditoryImage
# from modules.psychoacoustic import PsychoacousticFeature

from util import device
from util.weight_init import make_initializer
import torch
from torch.nn import functional as F

from modules import AuditoryImage


class Experiment(object):
    def __init__(
        self, 
        samplerate, 
        n_samples, 
        model_dim=128, 
        weight_init=0.1, 
        kernel_size=512, 
        residual_loss=False):

        super().__init__()
        self.samplerate = samplerate
        self.n_samples = n_samples
        self.window_size = 512
        self.step_size = self.window_size // 2
        self.n_frames = n_samples // self.step_size
        self.residual_loss = residual_loss

        self.n_bands = model_dim
        self.model_dim = model_dim
        self.kernel_size = kernel_size

        band = zounds.FrequencyBand(1, self.samplerate.nyquist)

        self.scale = zounds.MelScale(band, model_dim)
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
            512, 
            128, 
            do_windowing=False, 
            check_cola=False, 
            residual=self.residual_loss).to(device)
    
    def pooled_filter_bank(self, x):
        orig_shape = x.shape[-1]
        x = self.fb.forward(x, normalize=False)
        x = self.fb.temporal_pooling(x, 512, 256)
        x = x[..., :orig_shape // 256]
        return x

    def perceptual_feature(self, x):
        # bands = self.pif.compute_feature_dict(x)
        # return torch.cat(list(bands.values()), dim=-1)

        x = self.fb.forward(x, normalize=False)
        x = self.aim.forward(x)
        return x

    def perceptual_loss(self, a, b, norm='l2'):
        a = self.perceptual_feature(a)
        b = self.perceptual_feature(b)
        if norm == 'l2':
            return F.mse_loss(a, b)
        else:
            return torch.abs(a - b).sum()
