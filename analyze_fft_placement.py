import torch
import zounds
from data.audioiter import AudioIterator
from torch.nn import functional as F
import numpy as np
from functools import reduce

import matplotlib
from modules import stft
from modules.normalization import max_norm

from util import playable
matplotlib.use('qt5agg', force=True)
import matplotlib.pyplot as plt

matplotlib.rcParams['agg.path.chunksize'] = 2**15


batch_size = 1
stem_samples = 64
n_samples = 256
samplerate = zounds.SR22050()


def fft_shift(a, shift):
    n_samples = a.shape[-1]

    # this will be a circular shift, so we want to pad to 
    # avoid wraparound artifacts
    shift_samples = (shift * n_samples) * 0.5
    a = F.pad(a, (0, n_samples))

    # take the real fft of the signal we want to shift
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')
    n_coeffs = spec.shape[-1]

    # magnitude is one, while phase is the correct shift for the bin's frequency
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
    shift = 1 * torch.exp(-shift * shift_samples)


    ang = torch.angle(shift)
    ang = np.unwrap(ang.data.cpu().numpy())
    mag = torch.abs(shift)

    # ang = np.unwrap(ang)
    # mag = torch.ones_like(shift)

    # real = mag * torch.cos(ang)
    # imag = mag * torch.sin(ang)
    # shift = torch.complex(real, imag)

    # multiplying in the frequency domain will produce
    # the product of magnitudes and the *sum* of angles
    shifted = spec * shift

    samples = torch.fft.irfft(shifted, dim=-1, norm='ortho')
    samples = samples[..., :n_samples]
    return samples, ang, mag

def fft_convolve(*args):
    n_samples = args[0].shape[-1]
    # pad to avoid wraparound artifacts
    padded = [F.pad(x, (0, x.shape[-1])) for x in args]
    specs = [torch.fft.rfft(x, dim=-1) for x in padded]

    shift = specs[1]
    ang = torch.angle(shift).data.cpu().numpy()
    ang = np.unwrap(ang)

    mag = torch.abs(shift)

    spec = reduce(lambda accum, current: accum * current, specs[1:], specs[0])
    final = torch.fft.irfft(spec, dim=-1)
    # remove padding
    return final[..., :n_samples], ang, mag


def fft_shift_by_conv(stem, amt):
    if isinstance(amt, float):
        one_hot = torch.zeros(n_samples)
        one_hot[int(amt * n_samples)] = 1
    else:
        one_hot = torch.zeros(amt.shape[0], stem.shape[1], requires_grad=True)
        indices = (amt * n_samples * 0.9999).long()
        one_hot = torch.scatter(one_hot, dim=-1, index=indices, src=torch.ones_like(amt))

    return fft_convolve(stem, one_hot)


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    audio_iter = AudioIterator(
        batch_size, stem_samples, samplerate, normalize=True)
    clip = next(audio_iter.__iter__())

    stem = torch.zeros(n_samples)

    stem[:stem_samples] = clip

    impulse = torch.zeros(n_samples)
    impulse[n_samples // 2] = 1

    c = torch.zeros(n_samples)
    c[n_samples // 2: (n_samples // 2) + stem_samples] = stem[:stem_samples]

    # look at losses and gradients
    positions = torch.linspace(0, 1, 100, requires_grad=True)

    p2 = torch.zeros(100, 256, requires_grad=True)
    indices = (positions * n_samples * 0.9999).long()
    p2 = torch.scatter(p2, dim=-1, index=indices[:, None], src=torch.ones(100, requires_grad=True)[:, None])
    p2.retain_grad()

    # plt.matshow(p2.detach())    
    # plt.show()

    renders, ang, mag = fft_shift(stem[None, ...], positions[..., None] )
    r, a, m = fft_convolve(stem[None, ...], p2)

    target = c.view(1, -1).repeat(100, 1)

    l = ((target - r) ** 2).mean(dim=(-1))
    lz = l.mean()
    lz.backward()

    plt.matshow(p2.grad, cmap='gray')
    plt.title('Eye Grad')
    plt.show()

    loss = ((target - renders) ** 2).mean(dim=(-1))

    plt.title('losses')
    plt.plot(loss.detach())
    plt.plot(l.detach())
    plt.show()

    loss = loss.mean()
    loss.backward()

    plt.plot(positions.grad)
    plt.title('position grad')
    plt.show()

    # candidate
    # grads = []
    # for i, item in enumerate(p2.grad):

    #     # TODO: This should be based on the original
    #     # position.  I can do this safely because our
    #     # candidate positions are linear and monotonically
    #     # ascending
    #     index = int((i / 100) * 256)
    #     left, right = item[:index], item[index:]
    #     scalar_grad = (left.mean() - right.mean()).view(-1)
    #     grads.append(scalar_grad)
    
    # grads = torch.cat(grads)
    # plt.plot(grads.detach())
    # plt.title('grad candidate')
    # plt.show()

    # grad = (p2.grad.T @ p2).sum(dim=0)
    # plt.plot(grad.detach())
    # plt.title('grad candidate')
    # plt.show()


    # a = fft_convolve(stem, impulse)
    # a = max_norm(a)

    # view phase and magnitude for shift spectrum
    # a1, ang1, mag1 = fft_shift(stem, -1)
    # a2, ang2, mag2 = fft_shift(stem, -0.5)
    # a3, ang3, mag3 = fft_shift(stem, 0)
    # a4, ang4, mag4 = fft_shift(stem, 0.5)
    # a5, ang5, mag5 = fft_shift(stem, 0.9)
    # a6, ang6, mag6 = fft_shift(stem, 1)

    # plt.plot(ang1)
    # plt.plot(ang2)
    # plt.plot(ang3)
    # plt.plot(ang4)
    # plt.plot(ang5)
    # plt.title('phase')
    # plt.show()

    # plt.plot(mag1)
    # plt.plot(mag2)
    # plt.plot(mag3)
    # plt.plot(mag4)
    # plt.plot(mag5)
    # plt.title('mag')
    # plt.show()

    # plt.plot(c)
    # plt.title('target')
    # plt.show()

    # plt.plot(a1)
    # plt.plot(a6)
    # plt.title('-1 to 1')
    # plt.show()

    # plt.plot(a2)
    # plt.title('-0.5')
    # plt.show()

    # plt.plot(a3)
    # plt.title('0')
    # plt.show()

    # plt.plot(a4)
    # plt.title('0.5')
    # plt.show()

    # plt.plot(a5)
    # plt.title('0.9')
    # plt.show()


    

