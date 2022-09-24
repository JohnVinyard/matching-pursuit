import torch
from modules import TransferFunction
import zounds

if __name__ == '__main__':
    n_bands = 512
    n_frames = 128
    n_samples = 2 ** 15

    resolution = 16

    samplerate = zounds.SR22050()
    band = zounds.FrequencyBand(30, samplerate.nyquist)
    scale = zounds.MelScale(band, n_bands)

    tf = TransferFunction(
        samplerate, scale, n_frames, resolution, n_samples)
    

    x = torch.zeros(3, n_bands, resolution).normal_(0, 1)
    x = tf.forward(x)
    print(x.shape)