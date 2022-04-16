import torch
from modules.atoms import AudioEvent
import zounds
import numpy as np
from modules.ddsp import band_filtered_noise
from modules.normal_pdf import pdf

from util import playable

if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    n_events = 1
    batch_size = 1
    sequence_length = 64
    n_harmonics = 32
    sr = zounds.SR22050()
    n_samples = 2**14

    ae = AudioEvent(
        sequence_length=sequence_length,
        n_samples=2**14,
        n_events=n_events,
        min_f0=20,
        max_f0=800,
        n_harmonics=n_harmonics,
        sr=sr)


    f0 = torch.zeros(batch_size, n_events, sequence_length).fill_(0.1)
    osc_env = torch.linspace(1, 0, sequence_length).view(1, 1, sequence_length) ** 2

    harm_env = torch\
        .linspace(0.1, 0, n_harmonics)\
        .view(1, 1, n_harmonics, 1)\
        .repeat(1, 1, 1, sequence_length) ** 2

    
    noise_env = torch.linspace(1, 0, sequence_length).view(1, 1, sequence_length) ** 2
    noise_std = torch.zeros(batch_size, n_events, sequence_length).fill_(0.5)

    overall_env = torch.zeros(batch_size, n_events, sequence_length).fill_(0)

    result = ae.forward(f0, overall_env, osc_env, noise_env, harm_env, noise_std)
    audio = playable(result, sr)
    spec = np.abs(zounds.spectral.stft(audio))


    # batch, atoms, time
    # mean = torch.linspace(0.001, 0.1, 64).view(1, 1, 64).repeat(2, 3, 1)
    # std = torch.linspace(0.5, 0.001, 64).view(1, 1, 64).repeat(2, 3, 1)
    # filt = pdf(torch.arange(0, 257, 1).view(1, 1, 257, 1), mean, std)
    # filt = filt.data.cpu().numpy()

    # noise = band_filtered_noise(n_samples, mean=mean, std=std)
    # audio = playable(noise.view(-1, n_samples), sr)
    # spec = np.abs(zounds.spectral.stft(audio))
    input('Waiting...')