import torch
import zounds
from torch import nn
from data.audiostream import audio_stream
from modules.latent_loss import latent_loss
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from modules.pif import AuditoryImage
from train.optim import optimizer
from upsample import ConvUpsample
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer

init_weights = make_initializer(0.12)

basis = MelScale()
codec = AudioCodec(basis)

samplerate = zounds.SR22050()
freq_band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(freq_band, 128)
fb = zounds.learn.FilterBank(
    samplerate,
    256,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)
aim = AuditoryImage(
    256, 128, do_windowing=True, check_cola=True).to(device)

n_samples = 2 ** 14


class Discriminator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()


        self.periodicity = LinearOutputStack(32, 2, out_channels=1, in_channels=128)

        self.down = nn.Sequential(
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 2, 2, 0),
        )


        self.apply(init_weights)

    def forward(self, x):
        x = fb.forward(x, normalize=False)
        x = aim(x)
        x = x.permute(0, 3, 1, 2)
        x = x[:, :128, :128, :128]
        x = self.periodicity(x).view(-1, 128, 128)
        x = self.down(x)
        x = x.view(-1, 128)
        return x


class LossEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Discriminator(n_samples)

        self.params_1 = nn.Sequential(
            nn.Conv1d(128 * 3, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.spec_1 = LinearOutputStack(128, 3, out_channels=1)


        self.net = nn.Sequential(
            nn.Conv1d(128 + 64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, 1, 1, 0)
        )

        self.apply(init_weights)

    def forward(self, x, freq_params, amp_params, noise_params):
        freq_params = freq_params[:, :128, :128]
        amp_params = freq_params[:, :128, :128]
        noise_params = noise_params[:, :128, :128]
        params = torch.cat([freq_params, amp_params, noise_params], dim=1)
        params = self.params_1(params) 

        x = aim(fb(x, normalize=False))[..., :-1]

        x = self.spec_1(x)
        x = x.view(-1, 128, 128)

        x = torch.cat([params, x], dim=1)

        

        x = self.net(x)
        return torch.abs(x)


class Generator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        self.up = ConvUpsample(
            128, 128, 4, 128, mode='learned', out_channels=128)

        self.harm = nn.Conv1d(128, 128, 1, 1, 0)
        self.n = nn.Conv1d(128, 128, 1, 1, 0)

        self.n_samples = n_samples

        self.osc = OscillatorBank(
            input_channels=128,
            n_osc=128,
            n_audio_samples=n_samples,
            activation=torch.sigmoid,
            amp_activation=torch.abs,
            return_params=True,
            constrain=True,
            log_frequency=False,
            lowest_freq=20 / samplerate.nyquist,
            sharpen=False,
            compete=False)

        self.noise = NoiseModel(
            input_channels=128,
            input_size=128,
            n_noise_frames=128,
            n_audio_samples=n_samples,
            channels=128,
            activation=lambda x: x,
            squared=False,
            mask_after=1,
            return_params=True)

        self.apply(init_weights)

    def forward(self, x):
        x = self.up(x)

        h = self.harm(x)
        n = self.n(x)

        h, freq_params, amp_params = self.osc(h)
        n, noise_params = self.noise(n)
        audio = h + n
        return audio, freq_params, amp_params, noise_params


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Discriminator(n_samples)
        self.decoder = Generator(n_samples)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded, freq_params, amp_params, noise_params = self.decoder(encoded)
        return encoded, decoded, freq_params, amp_params, noise_params


ae = AutoEncoder().to(device)
optim = optimizer(ae, lr=1e-4)

estimator = LossEstimator().to(device)
estim_optim = optimizer(estimator, lr=1e-4)


def train_estimator(batch):
    estim_optim.zero_grad()
    optim.zero_grad()

    with torch.no_grad():
        encoded, decoded, freq_params, amp_params, noise_params = ae(batch)
        real_feat = aim(fb(batch, normalize=False))
        fake_feat = aim(fb(decoded, normalize=False))
        spec_loss = ((real_feat - fake_feat) ** 2).mean(dim=-1)

    est = estimator(batch, freq_params, amp_params, noise_params)
    spec_loss = spec_loss.view(-1, 128, 128)
    est = est.view(-1, 128, 128)
    loss = ((spec_loss - est) ** 2).mean()
    loss.backward()
    estim_optim.step()
    print('ESTIMATE', loss.item(), 'LOSS', spec_loss.mean().item(), 'EST', est.mean().item())


def train_ae(batch):
    estim_optim.zero_grad()
    optim.zero_grad()
    encoded, decoded, freq_params, amp_params, noise_params = ae(batch)

    real_feat = aim(fb(batch, normalize=False))
    fake_feat = aim(fb(decoded, normalize=False))
    spec_loss = ((real_feat - fake_feat) ** 2).mean()

    est_loss = estimator(batch, freq_params, amp_params, noise_params).mean() 

    loss = est_loss + spec_loss
    loss.backward()
    optim.step()
    print('AE', loss.item())
    return encoded, decoded


@readme
class ReinforcementLearningQuestionMark(object):
    def __init__(self, overfit, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.overfit = overfit

        self.encoded = None
        self.z = None
        self.generated = None

    def real(self):
        return playable(self.orig, samplerate)

    def real_spec(self):
        spec = codec.to_frequency_domain(self.orig.view(-1, n_samples))
        return spec[..., 0].data.cpu().numpy().squeeze()[0]

    def fake(self):
        return playable(self.generated, samplerate)

    def fake_spec(self):
        spec = codec.to_frequency_domain(self.generated.view(-1, n_samples))
        return spec[..., 0].data.cpu().numpy()[0]

    def latent(self):
        return self.z.data.cpu().numpy().squeeze()

    def run(self):

        # TODO: Pass in the audio stream
        stream = audio_stream(
            self.batch_size, n_samples, self.overfit, normalize=True, as_torch=True)

        for i, item in enumerate(stream):
            if i % 2 == 0:
                self.orig = item
                e, d = train_ae(item)
                self.generated = d
                self.z = e
            else:
                train_estimator(item)
