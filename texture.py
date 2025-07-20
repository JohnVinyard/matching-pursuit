import torch
from conjure import LmdbCollection, loggers, serve_conjure

from data import get_one_audio_segment
from torch import nn
from torch.optim import Adam

from modules import fft_frequency_decompose, max_norm, unit_norm
from modules.auditory import gammatone_filter_bank
from modules.overfitraw import OverfitRawAudio
from modules.transfer import fft_convolve
from itertools import count

from modules.upsample import ensure_last_axis_length
from util import device, encode_audio


def calculate_kurtosis(tensor, dim=-1):
    mean = torch.mean(tensor, dim--1, keepdim=True)
    std_dev = torch.std(tensor, dim=-1, keepdim=True)
    # Calculate the fourth central moment
    fourth_moment = torch.mean((tensor - mean)**4, dim=-1, keepdim=True)
    kurtosis = (fourth_moment / (std_dev**4)) - 3 # Excess kurtosis
    return kurtosis

class AudioFeatures(nn.Module):

    def __init__(self, n_samples: int, n_filters: int, filter_size: int, samplerate: int):
        super().__init__()
        self.n_samples = n_samples
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.samplerate = samplerate
        fb = gammatone_filter_bank(
            n_filters,
            filter_size,
            20,
            samplerate // 2 - 10,
            samplerate,
            freq_spacing_type='linear')
        fb = torch.from_numpy(fb).float()
        fb = unit_norm(fb)

        self.register_buffer('fb', fb)

    def forward(self, audio):
        batch_size = audio.shape[0]
        audio = audio.view(-1, 1, self.n_samples)
        bands = fft_frequency_decompose(audio, min_size=512)
        results = []
        for size, band in bands.items():
            fb = self.fb.view(1, self.n_filters, self.filter_size)
            fb = ensure_last_axis_length(fb, size)

            # compute the envelope of each filter bank
            spec = fft_convolve(fb, band)
            spec = spec ** 2

            # compute auto-correlation
            reverse = spec.flip(-1)
            fwd = torch.abs(torch.fft.rfft(spec, dim=-1))
            bwd = torch.abs(torch.fft.rfft(reverse, dim=-1))

            # correlation within band
            corr_1 = fwd * bwd
            # corr_1 = corr_1.std(dim=-1)

            # correlation with neighboring band
            corr_2 = fwd[:, 1:, :] * bwd[:, :-1, :]

            corr = torch.cat((corr_1.view(batch_size, -1), corr_2.view(batch_size, -1)), dim=-1)

            results.append(corr)

        results = torch.cat(results, dim=-1)
        return results


def overfit(n_samples: int, device: torch.device):

    collection = LmdbCollection('texture')

    log_recon, log_target = loggers(
        ['recon', 'target'] ,'audio/wav', encode_audio, collection)

    serve_conjure([log_recon, log_target], port=9999, n_workers=1)

    segment = get_one_audio_segment(n_samples)
    segment = max_norm(segment)

    recon = OverfitRawAudio(shape=segment.shape, std=0.01, normalize=True).to(device)
    model = AudioFeatures(n_samples, 64, 64, 22050).to(device)
    optim = Adam(recon.parameters(), lr=1e-3)

    with torch.no_grad():
        log_target(segment)
        target = model.forward(segment)

    for i in count():
        optim.zero_grad()
        r = recon.forward(None)
        log_recon(r)
        r = model.forward(r)
        loss = torch.abs(r - target).sum()
        loss.backward()
        optim.step()
        print(i, loss.item())



if __name__ == '__main__':
    overfit(2**16, device)
