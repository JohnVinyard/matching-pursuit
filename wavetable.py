import numpy as np
from scipy.signal import sawtooth, square
from data.audioiter import AudioIterator
import zounds
from sklearn.decomposition import PCA

# samplerate = zounds.SR22050()
# n_samples = 2**15

# n_frames = 128

# wavetable_size = 1024
# wavetable = torch.sin(torch.linspace(-np.pi, np.pi, steps=wavetable_size))


'''
for i in range(n_transfer_functions // 3):
            rps = f0s[i] * np.pi
            radians = np.linspace(0, rps * n_samples, n_samples)
            sin = np.sin(radians)[None, ...]
            sq = square(radians)[None, ...]
            st = sawtooth(radians)[None, ...]
            tfs.extend([sin, sq, st])
'''

def make_waves(n_samples, f0s, samplerate):
    sawtooths = []
    squares = []
    triangles = []

    for f0 in f0s:
        f0 = f0 / (samplerate // 2)
        rps = f0 * np.pi
        radians = np.linspace(0, rps * n_samples, n_samples)
        sq = square(radians)[None, ...]
        squares.append(sq)
        st = sawtooth(radians)[None, ...]
        sawtooths.append(st)
        tri = sawtooth(radians, 0.5)[None, ...]
        triangles.append(tri)
    
    sawtooths = np.concatenate(sawtooths, axis=0)
    squares = np.concatenate(squares, axis=0)
    triangles = np.concatenate(triangles, axis=0)
    return sawtooths, squares, triangles




# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.p = nn.Parameter(torch.zeros(n_frames).uniform_(-0.001, 0.001))

#     def forward(self, x):
#         p = self.p.view(-1, 1, n_frames)
#         p = F.upsample(p, size=n_samples, mode='linear').view(-1)
#         indices = torch.cumsum(p, dim=-1)
#         # indices = torch.sin(indices)

#         values = diff_index(wavetable, indices)
#         return values



if __name__ == '__main__':
    # app = zounds.ZoundsApp(globals=globals(), locals=locals())
    # app.start_in_thread(9999)

    # synth = zounds.SineSynthesizer(samplerate)
    # target = synth.synthesize(samplerate.frequency * n_samples, [220])
    # target = torch.from_numpy(target).float()

    # def wt():
    #     return wavetable.data.cpu().numpy().squeeze()

    # def listen():
    #     return playable(estimate.view(1, -1), samplerate)

    # while True:
    #     optim.zero_grad()
    #     estimate = model.forward(None)
    #     loss = F.mse_loss(estimate, target)
    #     loss.backward()
    #     optim.step()
    #     print(loss.item())

    #     listen()


    # samplerate = 22050
    # n_samples = 1024
    # n_f0s = 128
    # st, sq, tri = make_waves(n_samples, np.linspace(40, 2000, n_f0s), samplerate)

    # index = 100
    # plt.plot(np.abs(np.fft.rfft(st[index])))
    # plt.show()
    # plt.clf()

    # plt.plot(np.abs(np.fft.rfft(sq[index])))
    # plt.show()
    # plt.clf()

    # plt.plot(np.abs(np.fft.rfft(tri[index])))
    # plt.show()
    # plt.clf()

    # seq_len = 128
    # n_features = 64

    # memory = np.linspace(0, 1, seq_len) ** 4
    # plt.plot(memory)
    # plt.show()
    # plt.clf()

    # encoded = np.random.binomial(1, 0.01, (n_features, seq_len))
    # plt.matshow(encoded)
    # plt.show()
    # plt.clf()

    # encoded_spec = np.fft.rfft(encoded, axis=-1)
    # memory_spec = np.fft.rfft(memory[None, :], axis=-1)
    # spec = encoded_spec * memory_spec
    # context = np.fft.irfft(spec)
    # plt.matshow(context)
    # plt.show()
    # plt.clf()

    n_samples = 1024
    n_features = 768

    stream = AudioIterator(
        n_samples,
        2**15,
        zounds.SR22050(),
        normalize=True,
        overfit=False,
        step_size=1,
        pattern='*.wav',
        as_torch=False)
    
    batch = next(stream.__iter__()).reshape((n_samples, 2**15))

    spec = np.fft.rfft(batch, axis=-1)

    n_coeffs = spec.shape[-1]

    mags = np.abs(spec)
    phase = np.angle(spec)
    phase = (phase + np.pi) % (2 * np.pi) - np.pi

    feature = np.concatenate([mags, phase], axis=1)
    print(feature.shape)

    pca = PCA(n_features)
    pca.fit(feature)

    recon_feature = pca.transform(feature)
    recon_feature = pca.inverse_transform(recon_feature)

    recon_mags = recon_feature[:, :n_coeffs]
    recon_phase = recon_feature[:, n_coeffs:]
    recon_spec = recon_mags * np.exp(1j * recon_phase)
    approx_recon = np.fft.irfft(recon_spec)
    approx_residual = ((batch - approx_recon) ** 2).mean()
    print(np.linalg.norm(approx_residual))


    spec = mags * np.exp(1j * phase)
    recon = np.fft.irfft(spec)
    residual = ((batch - recon) ** 2).mean()
    print(np.linalg.norm(residual))

    orig = zounds.AudioSamples(batch[0], zounds.SR22050())
    orig.save('fft_orig.wav')

    audio = zounds.AudioSamples(approx_recon[0], zounds.SR22050())
    audio.save('fft_pca.wav')
    
