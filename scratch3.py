import torch

from modules.sparse import to_key_points

if __name__ == '__main__':
    # n_bands = 512
    # n_frames = 128
    # n_samples = 2 ** 15

    # resolution = 16

    # samplerate = zounds.SR22050()
    # band = zounds.FrequencyBand(30, samplerate.nyquist)
    # scale = zounds.MelScale(band, n_bands)

    # tf = TransferFunction(
    #     samplerate, scale, n_frames, resolution, n_samples)
    

    # x = torch.zeros(3, n_bands, resolution).normal_(0, 1)
    # x = tf.forward(x)
    # print(x.shape)



    x = torch.zeros(8, 64, 128).normal_(0, 1)
    print(x.shape)
    kp = to_key_points(x)
    print(kp.shape)