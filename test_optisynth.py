import zounds
from torch import nn
from torch.nn import functional as F

from featuresynth import audio, batch_stream, device, torch, xavier_normal_, \
    calculate_gain
from torch.optim import Adam
import numpy as np
from librosa.filters import chroma

from featuresynth.audio.transform import fft_frequency_decompose
from featuresynth.generator.ddsp import oscillator_bank, noise_bank2, \
    smooth_upsample2

samplerate = zounds.SR22050()
n_samples = 16384

scale = zounds.MelScale(zounds.FrequencyBand(20, samplerate.nyquist), 128)
fb = zounds.learn.FilterBank(
    samplerate,
    512,
    scale,
    0.01,
    normalize_filters=True,
    a_weighting=False).to(device)

def local_gram_matrix(x, shape, step, filter_bank):
    x = x.view(x.shape[0], 1, -1)
    x = filter_bank.forward(x, normalize=False)
    return x

    # pooled = filter_bank.temporal_pooling(x, 64, 32).view(-1)

    dim = 1
    for stride, step in zip(shape, step):
        x = x.unfold(dim, stride, step)
        dim += 1

    x = x.reshape(-1, *shape)
    x = torch.bmm(x, x.permute(0, 2, 1))
    print(x.shape)
    # x = x / len(x)
    # x = x - x.mean()
    # x = x / (x.std() + 1e-12)
    # return torch.cat([pooled, x.view(-1)])
    return x


def autocorrelation_loss(x):
    bands = fft_frequency_decompose(x, 512)
    features = []
    for size, band in bands.items():
        x = local_gram_matrix(band, (3, 64), (1, 32), fb).reshape(-1)
        features.append(x)
    return torch.cat(features)


def oscillator_bank3(frequency, phase, amplitude):
    audio = torch.sin(frequency + phase) * amplitude
    return audio.sum(dim=1)


def oscillator_bank(frequency, amplitude, sample_rate):
    """
    frequency and amplitude are (batch, n_oscillators, n_samples)
    sample rate is a scalar
    """
    # constrain frequencies
    frequency = torch.clamp(frequency, 1., sample_rate / 2.)

    # translate frequencies in hz to radians
    omegas = frequency * (2 * np.pi)
    omegas = omegas / sample_rate

    phases = torch.cumsum(omegas, dim=-1)

    wavs = torch.cos(phases)

    audio = wavs * amplitude
    audio = torch.sum(audio, dim=1)
    return audio


def oscillator_bank2(frequency, amplitude, sample_rate):
    n_samples = frequency.shape[-1]

    frequency = torch.clamp(frequency, 0, 0.5)
    r = torch.linspace(0, n_samples, steps=n_samples).to(frequency.device)

    samples = amplitude * torch.sin(
        (2 * np.pi) * frequency * r[None, None, :])

    samples = samples.sum(dim=1)
    return samples


def fm_oscillator_bank(c_freq, c_amp, frequency, amplitude, sample_rate):
    """
    frequency and amplitude are (batch, n_oscillators, n_samples)
    sample rate is a scalar
    """
    # constrain frequencies
    # frequency = torch.clamp(frequency, 20., sample_rate / 2.)
    # c_freq = torch.clamp(c_freq, 20., sample_rate / 2.)
    #
    # # translate frequencies in hz to radians
    # omegas = frequency * (2 * np.pi)
    # omegas = omegas / sample_rate
    #
    # c_omegas = c_freq * (2 * np.pi)
    # c_omegas = c_omegas / sample_rate
    #
    #
    # phases = torch.cumsum(omegas, dim=-1)
    # c_phases = torch.cumsum(c_omegas, dim=-1)
    #
    # samples = torch.sin(phases + (c_amp * torch.sin(c_phases)))



    r = torch.linspace(0, n_samples, steps=n_samples).to(frequency.device)
    samples = torch.sin(
        (2 * np.pi) * frequency * r[None, None, :] + \
        (c_amp * torch.sin((2 * np.pi) * c_freq * r[None, None, :])))

    audio = samples * amplitude
    audio = torch.sum(audio, dim=1)
    return audio


def stream(total_samples=n_samples, batch_size=1):
    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'

    # path = '/hdd/sine'
    # pattern = '*'

    samplerate = zounds.SR22050()
    feature_spec = {
        'audio': (total_samples, 1)
    }

    feature_funcs = {
        'audio': (audio, (samplerate,))
    }

    bs = batch_stream(
        path, pattern, batch_size, feature_spec, 'audio', feature_funcs)
    for batch, in bs:
        yield batch


class Synth(nn.Module):
    def __init__(self):
        super().__init__()

        start = np.random.normal(0, 0.1, n_samples)
        spec = np.fft.rfft(start, norm='ortho')
        curve = np.geomspace(1, 0.00001, len(spec))
        samples = np.fft.irfft(curve * spec, norm='ortho')
        param = torch.from_numpy(samples).float()
        self.params = nn.Parameter(param)

    def forward(self, *args):
        return self.params

    def synthesize(self, *args, **kwargs):
        recon = self.forward(*args, **kwargs)
        return zounds.AudioSamples(
            recon.data.cpu().numpy().squeeze(),
            samplerate).pad_with_silence()


class FilterBankSynth(Synth):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(
            torch.FloatTensor(1, 128, 16385).normal_(0, 1))

    def forward(self, *args):
        x = fb.transposed_convolve(self.params)
        return x, None, None


def erb(f):
    return 0.108 * f + 24.7


class DDSPSynth(Synth):
    def __init__(self):
        super().__init__()

        n_osc = 128
        frequencies = np.geomspace(20, 10000, num=n_osc)
        bandwidths = erb(frequencies)
        self.register_buffer(
            'bandwidths', torch.from_numpy(bandwidths).float())

        self.centers = nn.Parameter(
            torch.from_numpy(frequencies).float()
        )

        self.amp_params = nn.Parameter(
            torch.FloatTensor(1, n_osc, 64).uniform_(0, 0.01))

        self.freq_params = nn.Parameter(
            torch.FloatTensor(1, n_osc, 64).uniform_(-10, 10))

        self.noise_params = nn.Parameter(
            torch.FloatTensor(1, 17, 1024).uniform_(0, 0.01))

    def forward(self, noise_only=False, osc_only=False):
        orig_freq = freq = F.tanh(self.freq_params)

        # freq = self.starts[None, :, None] + (freq * self.diffs[None, :, None])
        freq = self.centers[None, :, None] + (
            freq * (self.bandwidths[None, :, None] / 2))

        orig_amp = amp = F.relu(self.amp_params)
        noise = F.relu(self.noise_params)

        freq = F.upsample(freq, size=n_samples, mode='linear')
        amp = F.upsample(amp, size=n_samples, mode='linear')

        x = oscillator_bank(freq, amp, int(samplerate))

        n = noise_bank2(noise)
        x = x.view(*n.shape)

        if noise_only:
            return n
        elif osc_only:
            return x
        else:
            return x + n, orig_freq, orig_amp, x, n


class FMSynth(Synth):
    def __init__(self):
        super().__init__()
        n_osc = 32

        self.osc_params = nn.Parameter(
            torch.FloatTensor(1, 2, n_osc, 64).normal_(0, 0.01))
        self.c_params = nn.Parameter(
            torch.FloatTensor(1, 2, n_osc, 64).normal_(0, 0.01))
        self.noise_params = nn.Parameter(
            torch.FloatTensor(1, 17, 1024).normal_(0, 0.01))

        # n_osc = 16

        # stops = np.geomspace(20, samplerate.nyquist, num=n_osc, endpoint=False)
        stops = np.geomspace(0.0001, 0.36, num=n_osc, endpoint=False)
        # n_osc + 1
        freqs = [0] + list(stops)
        # n_osc
        diffs = np.diff(freqs)
        # n_osc
        starts = stops - diffs

        self.starts = torch.from_numpy(starts).to(device).float()
        self.diffs = torch.from_numpy(diffs).to(device).float()

    def _process(self, p, constrain=False, carrier=None):

        # freq = freq
        # freq * freq * samplerate.nyquist

        if constrain:
            freq = torch.clamp(p[:, 0, ...], 0, 1) * 0.5
            # freq = self.starts[None, :, None] + (freq * self.diffs[None, :, None])
        elif carrier is not None:
            factor = F.relu(p[:, 0, ...])

            c = torch.clamp(carrier, 0, 1) * 0.5
            # c = self.starts[None, :, None] + (c * self.diffs[None, :, None])
            # c = carrier * samplerate.nyquist
            freq = factor * c
        else:
            freq = F.sigmoid(p[:, 0, ...])
            freq = freq * samplerate.nyquist

        amp = p[:, 1, ...]

        # freq = F.upsample(freq, size=n_samples, mode='linear')
        # amp = F.upsample(amp, size=n_samples, mode='linear')

        freq = smooth_upsample2(freq, size=n_samples)
        amp = smooth_upsample2(amp, size=n_samples)

        return freq, amp

    def forward(self, noise_only=False, osc_only=False, remove_fm=False):
        freq, amp = self._process(self.osc_params, constrain=True)
        c_freq, c_amp = self._process(self.c_params,
                                      carrier=self.osc_params[:, 0, ...])

        if remove_fm:
            c_amp = 0

        amp = F.relu(amp)
        c_amp = F.relu(c_amp)

        x = fm_oscillator_bank(c_freq, c_amp, freq, amp, int(samplerate))
        n = noise_bank2(self.noise_params)
        x = x.view(*n.shape)
        if noise_only:
            return n
        elif osc_only:
            return x
        else:
            return x + n, freq, None


class IdentityFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FFTFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(1, -1)
        window = torch.hann_window(512).to(x.device)
        x = torch.stft(x, 512, 256, 512, window)
        return x


class FFTRelativePhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FFTFeature()

    def forward(self, x):
        x = self.feature(x)

        # compute angle
        angle = torch.atan2(x[..., 1], x[..., 0])
        # unwrap
        # angle = (angle + np.pi) % (2 * np.pi) - np.pi
        # compute diff
        angle = angle[..., 1:] - angle[..., :-1]
        angle = F.pad(angle, (1, 0))

        # mag = torch.log(1 + torch.abs(x[..., 0]) + 1e-12)
        mag = x[..., 0]
        # x = torch.cat([mag[..., None], angle[..., None]], dim=-1)

        # remove phase above 5khz
        # x[:, 128:, :, 1] = 0

        # return x
        # , angle[:, 128:, :].reshape(-1)
        return torch.cat([mag.reshape(-1), angle.reshape(-1)])


class WeirdIdea(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # window = torch.hann_window(1024).to(x.device)
        # coeffs = torch.stft(x.view(1, -1), 1024, 256, 1024, window)
        # coeffs = torch.abs(coeffs[..., 0])

        coeffs = fb.forward(x, normalize=False)
        coeffs = fb.temporal_pooling(coeffs, 512, 256)

        x = x.view(1, 1, n_samples)
        x = x[:, :, None, :]
        x = x.unfold(-1, 1024, 512)
        kernel = x.reshape(31, 1, 1024)
        x = x.reshape(1, 31, 1024)
        x = F.conv1d(x, kernel, groups=31, padding=512)

        return torch.cat([
            x.reshape(-1),
            coeffs.reshape(-1)
        ])


class FFTNoPhaseFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FFTFeature()

    def forward(self, x):
        return torch.log(1 + torch.abs(self.feature(x)[..., 0]))


class LoudnessFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FFTNoPhaseFeature()

    def forward(self, x):
        x = self.feature(x)
        x = torch.norm(x, dim=1)
        return x


class MFCCFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FFTNoPhaseFeature()

    def forward(self, x):
        x = torch.abs(self.feature(x))
        x = torch.log(x + 1e-12)
        x = x.permute(0, 2, 1)
        x = torch.rfft(x, 1)
        x = x[..., 0][..., 1:13]
        return x


class ChromaFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FFTNoPhaseFeature()
        basis = torch.from_numpy(chroma(int(samplerate), 512))[None, ...]
        self.register_buffer('basis', basis)

    def forward(self, x):
        x = torch.abs(self.feature(x))
        x = torch.bmm(self.basis, x)
        return x


class FilterBankFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = fb.forward(x, normalize=False)
        return x


class PhaseInvariantFilterBankFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = torch.ones(1, 1, 3, 3).to(device)
        self.spec_dilations = [1]
        self.time_dilations = [1, 3, 9, 27, 81, 243]

    def forward(self, x):
        x = fb.convolve(x)

        features = []
        for sd in self.spec_dilations:
            for td in self.time_dilations:
                r = F.conv2d(
                    x[:, None, :, :],
                    self.kernel,
                    stride=(1, 1),
                    padding=(sd, td),
                    dilation=(sd, td))
                features.append(r)

        x = torch.cat(features, dim=1)
        # x = torch.sum(x, dim=1, keepdim=True)
        # print(x.shape)
        x = F.max_pool2d(x, (1, 512), (1, 256), padding=(0, 256))
        return x


class PooledFilterBankFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FilterBankFeature()

    def forward(self, x):
        x = self.feature(x)
        # x1 = F.max_pool1d(x, 512, 256, padding=256)
        # x2 = F.avg_pool1d(x, 512, 256, padding=256)
        # x = torch.cat([x1[:, None, ...], x2[:, None, ...]], dim=1)
        x = x.unfold(-1, 512, 256)
        x = torch.norm(x, dim=-1)
        print(x.reshape(-1).shape[0] / n_samples)
        return x


feats = {'f': None}

# TODO: How does this measure distance between audio with "rolled" phase?
# TODO: What about autocorrelation?
class PsychoacousticFeature2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        spec = fb.convolve(x)

        kernel_size = 512
        step = 256

        spec = F.pad(spec, (step // 2, step // 2))
        spec = spec.unfold(-1, kernel_size, step)

        discard_phase = False

        if discard_phase:
            spec = torch.norm(spec, dim=-1)
            return spec
        else:
            spec = torch.rfft(spec, signal_ndim=1)
            spec = torch.norm(spec, dim=-1)
            feats['f'] = spec.data.cpu().numpy()

        return spec


class PsychoacousticFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.banks = self._build_filter_banks()
        self.weights = {
            512: 1,
            1024: 1,
            2048: 1,
            4096: 1,
            8192: 1,
            16384: 1
        }

    def _build_filter_banks(self):
        spans = [
            (20, 344),
            (344, 689),
            (689, 1378),
            (1378, 2756),
            (2756, 5512),
            (5512, 11025)
        ]
        kernel_sizes = [
            32,
            64,
            128,
            256,
            512,
            1024
        ]
        self.kernel_sizes = kernel_sizes
        keys = [
            512,
            1024,
            2048,
            4096,
            8192,
            16384
        ]
        bank_dict = {}
        self.kernel_sizes = {}
        for span, size, key in zip(spans, kernel_sizes, keys):
            self.kernel_sizes[key] = size // 2 + 1

            band = zounds.FrequencyBand(*span)
            print(band)
            scale = zounds.MelScale(band, 64)
            samples_per_second = span[1] * 2
            print(samples_per_second)
            freq = zounds.Picoseconds(1e12) / samples_per_second
            sr = zounds.SampleRate(freq, freq)
            fb = zounds.learn.FilterBank(
                sr,
                size,
                scale,
                0.01,
                normalize_filters=True,
                a_weighting=False).to(device)
            bank_dict[key] = (fb, span, size)
        return bank_dict

    @property
    def band_sizes(self):
        return list(self.banks.keys())

    def decompose(self, x):
        return fft_frequency_decompose(x, 512)

    def compute_feature_dict(self, x):


        # TODO: Don't perform the decomposition if x is already
        # a dictionary
        if not isinstance(x, dict):
            batch_size = x.shape[0]

            x = x.view(batch_size, 1, -1)

            x = self.decompose(x)

        bands = dict()

        for size, data in self.banks.items():
            fb, span, kernel_size = data
            band = x[size]

            spec = fb.forward(band, normalize=False)

            spec = F.pad(spec, (kernel_size // 4, kernel_size // 4))
            spec = spec.unfold(-1, kernel_size, kernel_size // 2)


            spec = torch.norm(torch.rfft(spec, signal_ndim=1), dim=-1)
            bands[size] = spec

        return bands

    def forward(self, x):

        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
        else:
            batch_size = x.shape[0]

        # Don't perform the decomposition if x is already
        # a dictionary
        if not isinstance(x, dict):

            x = x.view(batch_size, 1, -1)
            x = self.decompose(x)


        results = []
        specs = []

        for size, data in self.banks.items():
            fb, span, kernel_size = data
            band = x[size]

            spec = fb.forward(band, normalize=False)

            spec = F.pad(spec, (kernel_size // 4, kernel_size // 4))
            spec = spec.unfold(-1, kernel_size, kernel_size // 2)


            if span[0] > 99999:
                spec = torch.norm(spec, dim=-1)
            else:
                freq = torch.rfft(spec, signal_ndim=1)
                spec = torch.norm(freq, dim=-1)
                specs.append(spec.data.cpu().numpy().squeeze())

            results.append(spec.reshape(batch_size, -1))

        x = torch.cat(results, dim=1)
        # print('RATIO', x.shape, x.shape[0] / n_samples)
        return x, specs


class DecompositionFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(1, 1, -1)
        x = fft_frequency_decompose(x, 128)

        result = []
        for v in x.values():
            result.append(v.view(-1) / torch.norm(v.view(-1)))

        # x = torch.cat([v.view(-1) for v in x.values()])
        x = torch.cat(result)
        return x


class GramFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def local_gram_matrix(self, x, shape, step, stride):
        sa, sb = shape
        step_a, step_b = step
        # size = 2
        x = x[..., ::stride]
        x = x.squeeze().unfold(0, sa, step_a).unfold(1, sb, step_b)
        # shape = x.shape
        dims = x.shape[:2]
        if len(x.shape) == 4:
            w, h = sa, sb
        else:
            n_features = np.prod(x.shape[2:])
            actual_size = np.sqrt(n_features)
            w = h = actual_size = int(actual_size)
        x = x.contiguous().view(-1, w, h)
        x = torch.bmm(x, x.permute(0, 2, 1))
        x = x.view(*dims, -1)
        # x = x - x.mean()
        # x = x / (x.std() + 1e-8)
        return x

    def forward(self, x):
        x = fb.forward(x, normalize=False)
        # x = fb.temporal_pooling(x, 512, 256)
        # x = x[:, :, None, :]
        # x = self.gram_matrix(x)

        x = torch.cat([
            # self.local_gram_matrix(x, (3, 512), (1, 256), 1).view(-1),
            # self.local_gram_matrix(x, (3, 128), (1, 32), 1).view(-1),
            x.view(-1)
        ])
        return x
        # x = self.local_gram_matrix(x, (3, 128), (1, 32))
        # return x.view(-1)


if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    target = next(stream())
    target /= np.abs(target).max()
    envelope = np.linspace(0, 1, 10)
    target[:, :,  :10] *= envelope
    target[:, :, -10:] *= envelope[::-1]

    target_audio = zounds.AudioSamples(target.squeeze(),
                                       samplerate).pad_with_silence()
    target = torch.from_numpy(target).float().to(device)
    print(target.shape)


    model = Synth().to(device)


    def listen():
        return model.synthesize()


    def spec():
        return np.abs(zounds.spectral.stft(listen()))


    def gradients(network):
        for n, p in network.named_parameters():
            g = p.grad
            if g is None:
                continue
            yield n, g, g.min().item(), g.max().item(), g.mean().item()


    optim = Adam(model.parameters(), lr=1e-3, betas=(0, 0.9))


    # feature = WeirdIdea().to(device)
    # feature = DecompositionFeature().to(device)
    feature = PsychoacousticFeature().to(device)
    # feature = IdentityFeature()

    while True:
        optim.zero_grad()

        recon = model().view(1, 1, n_samples)

        # h = zounds.AudioSamples(harm.data.cpu().numpy().squeeze(),
        #                         samplerate).pad_with_silence()
        # n = zounds.AudioSamples(noise.data.cpu().numpy().squeeze(),
        #                         samplerate).pad_with_silence()

        print('==============================================')
        raw_loss = ((target - recon) ** 2).mean()
        print('MSE', raw_loss.item())

        a = np.array(target.data.cpu().numpy().squeeze())[:1024]
        b = np.array(recon.data.cpu().numpy()).squeeze()[:1024]

        t, specs = feature(target)
        x, _ = feature(recon)

        loss = F.mse_loss(t, x)

        # f = freq.data.cpu().numpy().squeeze().T
        # a = amp.data.cpu().numpy().squeeze().T

        loss.backward()

        # for name, grad, mn, mx, mean in gradients(models['ddsp']):
        #     print(name, mn, mx, mean)
        #     if name == 'amp_params':
        #         amp_grad = grad.data.cpu().numpy().squeeze().T
        #     if name == 'freq_params':
        #         freq_grad = grad.data.cpu().numpy().squeeze().T
        #     if name == 'noise_params':
        #         noise_grad = grad.data.cpu().numpy().squeeze().T

        optim.step()
        print('PIF', loss.item())

    input('waiting...')
