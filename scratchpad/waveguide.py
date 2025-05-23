import numpy as np
import zounds
import torch
from torch.nn import functional as F
from torch import nn
from subprocess import Popen, PIPE

# from modules.ddsp import overlap_add
# from modules.phase import STFT, AudioCodec, MelScale
# from modules.pif import AuditoryImage
# from util import device, playable
# from modules.psychoacoustic import PsychoacousticFeature

"""
noise- + --------> output /\/\/\/\/\/\/\/
       ^       |
    filter <- delay
"""

samplerate = zounds.SR22050()

piano_samples = None

# pif = PsychoacousticFeature().to(device)
# print(pif.band_sizes)
#
# mel_scale = MelScale()
# codec = AudioCodec(mel_scale)

n_bands = 512
kernel_size = 512

# band = zounds.FrequencyBand(40, samplerate.nyquist)
# scale = zounds.MelScale(band, n_bands)
# fb = zounds.learn.FilterBank(
#     samplerate,
#     kernel_size,
#     scale,
#     0.01,
#     normalize_filters=True,
#     a_weighting=False).to(device)

# aim = AuditoryImage(512, 128, do_windowing=False, check_cola=False).to(device)


# def perceptual_feature(x):
#     # x = x.view(1, 1, 2**15)
#     # spec = stft(x, 512, 256, log_amplitude=True)
#     # return spec
#
#     # x = torch.abs(fb.convolve(x))
#     # x = fb.temporal_pooling(x, 512, 256)
#
#     # x = fb.forward(x, normalize=False)
#     # x = aim.forward(x)
#
#     bands = pif.compute_feature_dict(x.view(1, 1, -1))
#     bands = torch.cat(list(bands.values()), dim=-1)
#     return bands
#
#     # env = torch.abs(x).view(1, 1, -1)
#     # env = F.avg_pool1d(env, 64, 32)
#
#     # x = codec.to_frequency_domain(x.view(1, -1))
#     # x = torch.abs(mel_scale.to_frequency_domain(x.view(1, -1)))
#     # x = torch.cat([
#     #     # spec.view(-1),
#     #     bands.view(-1),
#     #     # env.view(-1)
#     # ])
#     return x


# def perceptual_loss(a, b):
#     a = perceptual_feature(a)
#     b = perceptual_feature(b)
#     loss = F.mse_loss(a, b)
#     # loss = torch.abs(a - b).sum() / a.shape[0]
#     return loss


class LayeredTransferFunctionModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.n_frames = 128
        self.n_coeffs = n_samples // 2 + 1

        self.register_buffer('env', torch.linspace(1, 0, 128).view(1, 1, 128) ** 20)

        self.transfer = nn.Parameter(
            torch.zeros(1, 1, self.n_frames, self.n_samples).uniform_(-0.01, 0.01),
        )

        self.transfer = nn.Parameter(
            torch.complex(
                torch.zeros(1, 1, self.n_frames, self.n_samples // 2 + 1).uniform_(-0.01, 0.01),
                torch.zeros(1, 1, self.n_frames, self.n_samples // 2 + 1).uniform_(-0.01, 0.01),
            )
        )

    def forward(self, x):

        # get the impulses at the correct sample rate
        noise = torch.zeros(self.n_samples).uniform_(-1, 1)
        env = F.interpolate(self.env, size=self.n_samples, mode='linear')
        env = torch.abs(env)
        env = env * noise

        # window and pad the envelope
        env = F.pad(env, (0, 256))
        env = env.unfold(-1, 512, 256)
        env = env * torch.hamming_window(512)[None, None, None:]

        remaining = torch.zeros(1, 1, 128, self.n_samples - 512)
        env = torch.cat([env, remaining], dim=-1)
        env = torch.fft.rfft(env, dim=-1, norm='ortho')

        # tf = torch.fft.rfft(self.transfer, dim=-1, norm='ortho')
        tf = self.transfer

        # r = torch.clamp(self.transfer.real, 0, 1)
        # a = self.transfer.imag

        # tf = torch.complex(
        #     r * torch.cos(a),
        #     r * torch.sin(a)
        # )

        spec = env * tf
        long_frames = torch.fft.irfft(spec, dim=-1, norm='ortho')

        final = torch.zeros(1, 1, self.n_samples * 2)

        for i in range(128):
            start = i * 256
            stop = start + self.n_samples
            final[:, :, start: stop] += long_frames[:, :, i, :]

        final = final[..., :self.n_samples]
        return final


class SuperSimpleSourceExcitationModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        # self.env = nn.Parameter(torch.zeros(1, 1, 128).uniform_(-0.01, 0.01))
        self.register_buffer('env', torch.hamming_window(128).view(1, 1, 128))
        self.transfer = nn.Parameter(torch.complex(
            torch.zeros(1, 1, n_samples // 2 + 1).uniform_(-0.01, 0.01),
            torch.zeros(1, 1, n_samples // 2 + 1).uniform_(-0.01, 0.01)
        ))
        self.n_samples = n_samples

    def forward(self, x):
        noise = torch.zeros(self.n_samples).uniform_(-1, 1)
        env = F.interpolate(self.env, size=self.n_samples, mode='linear')
        env = torch.abs(env)
        env = env * noise

        env = torch.fft.rfft(env, dim=-1, norm='ortho')
        spec = env * self.transfer
        final = torch.fft.irfft(spec, dim=-1, norm='ortho')
        return final.view(1, 1, self.n_samples)


class KarplusStrong(nn.Module):
    def __init__(self, memory_size, n_samples):
        super().__init__()
        self.memory_size = memory_size
        self.n_samples = n_samples

        self.impulse = nn.Parameter(torch.zeros(512).uniform_(-1, 1))

        n_frames = (piano_samples // 256) + 2

        self.excitation_env = nn.Parameter(
            torch.zeros(n_frames * 32).uniform_(0, 0.01))

        transfer_functions = torch.zeros(2, n_frames, 257).fill_(0.2)
        self.transfer_functions = nn.Parameter(torch.complex(
            transfer_functions[0, ...], transfer_functions[1, ...]))

        excitation_transfer_function = torch.zeros(2, 257).fill_(0.2)
        self.excitation_transfer_functions = nn.Parameter(torch.complex(
            excitation_transfer_function[0, ...], excitation_transfer_function[1, ...]))

        self.learned_impulse = False

        # it's a bit choppy, but it's possible to learn the appropiate
        # impulse for the piano signal
        self.continuous_excitation = True

        # it's possible to learn something akin to the piano signal with
        # a single transfer function

        # it's also possible to learn something akin to piano signal with
        # a series of transfer functions
        self.dynamic_transfer = True

    def forward(self, x):
        output = []

        # for i in range((self.n_samples // 256) - 1):
        i = 0

        excitation = (self.excitation_env ** 2).view(1, 1, -1)
        excitation = F.upsample(
            excitation, scale_factor=256 // 32, mode='linear')

        while (i * 256) < piano_samples:

            # excitation = self.excitation_env[i * 8: i * 8 + 8].view(1, 1, -1) ** 2
            # excitation = F.upsample(excitation, size=512)
            exc = excitation[:, :, i * 256: i * 256 + 512]

            impulse = (torch.zeros(512).uniform_(-1, 1).to(device) * exc)
            spec = torch.fft.rfft(impulse, norm='ortho')
            spec = spec * self.excitation_transfer_functions
            impulse = torch.fft.irfft(spec, norm='ortho')

            if len(output) == 0:
                if self.learned_impulse:
                    output.append(self.impulse)
                else:
                    output.append(impulse)
            elif self.continuous_excitation:
                output[-1] = output[-1] + impulse

            spec = torch.fft.rfft(output[-1], dim=-1, norm='ortho')
            filtered = spec * \
                self.transfer_functions[i if self.dynamic_transfer else 0]
            new_block = torch.fft.irfft(filtered, dim=-1, norm='ortho')

            new_block = new_block * torch.hamming_window(512).to(device)
            output.append(new_block)
            i += 1

        output = torch.cat(output).view(1, 1, len(output), 512)
        output = overlap_add(output)

        return output.view(-1)[:piano_samples]


def karplus_strong(total_samples, delay_samples, filter_memory_size, damping):
    excitation = np.random.uniform(-1, 1, 512)

    output = []
    filter_memory = []

    for i in range(total_samples):
        o = 0

        if len(filter_memory) > filter_memory_size:
            avg = sum(filter_memory[-filter_memory_size:]) / filter_memory_size
            o += avg

        if len(output) > delay_samples:
            filter_memory.append(output[-delay_samples] * damping)

        if i < len(excitation):
            o += excitation[i]

        output.append(o)

    return zounds.AudioSamples(np.array(output), zounds.SR22050())


# TODO: It might be nice to move this into zounds
def listen_to_sound(samples: zounds.AudioSamples, wait_for_user_input: bool = True) -> None:
    proc = Popen(f'aplay', shell=True, stdin=PIPE)

    if proc.stdin is not None:
        proc.stdin.write(samples.encode().read())
        proc.communicate()

    if wait_for_user_input:
        input('Next')


def simulate(entry_coord, entry_block, sampling_coord, width, height, n_steps=128):
    """
    room properties = (B, filter coeffs, width, height)

    room state = (B, block, width, height)

    Simulation consists of two steps:
        1. Apply filter to block state
        2. propagate with a 3x3 filter:
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ] and add to current state
    """

    output_blocks = []
    states = []

    block_size = entry_block.shape[-1]

    n_coeffs = (entry_block.shape[-1] // 2) + 1

    # block-wise transfer functions in the frequency domain
    transfer_functions = torch.zeros(
        1, 2, n_coeffs, width, height).fill_(1,)
    # reshape
    transfer_functions = torch.complex(
        transfer_functions[:, 0, ...], transfer_functions[:, 1, ...])
    # apply a frequency-wise exponential slope across frequencies
    transfer_functions *= (torch.linspace(0.285, 0, n_coeffs)
                           ** 2)[None, :, None, None]

    room_state = torch.zeros(1, entry_block.shape[-1], width, height)

    entry_block = entry_block.view(1, block_size)

    room_state[:, :, entry_coord[0], entry_coord[1]] += entry_block

    # (out, in, w, h)
    propagation_kernel = torch.from_numpy(np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float32)).view(1, 1, 3, 3, 1, 1)

    for i in range(n_steps):

        # take the real FFT of the room sample blocks
        spec = torch.fft.rfft(room_state, dim=1, norm='ortho')
        # apply the transfer functions
        spec = spec * transfer_functions
        # transform back into the time domain
        state = torch.fft.irfft(spec, dim=1, norm='ortho')

        # propagate energy
        state = F.pad(state, [1, 1, 1, 1], mode='reflect')

        windowed = F.unfold(
            state,
            kernel_size=(3, 3),
            # padding=(1, 1),
            stride=(1, 1))
        windowed = windowed.view(1, block_size, 3, 3, width, height)

        room_state = torch.sum(windowed * propagation_kernel, dim=(2, 3))

        states.append(room_state.mean(dim=1))

        out = room_state[:, :, sampling_coord[0], sampling_coord[1]]
        output_blocks.append(out.view(-1))

        print(f'completed step {i}, room_state, {room_state.std().item()}')

    samples = torch.cat(output_blocks).data.cpu().numpy()
    samples /= np.abs(samples.max()) + 1e-8
    samples = zounds.AudioSamples(samples, zounds.SR22050()).pad_with_silence()

    all_states = torch.cat(states, dim=0).data.cpu().numpy()
    return samples, all_states



if __name__ == '__main__':
    s, ast = simulate(
        entry_coord=[1, 1],
        entry_block=torch.zeros(128).uniform_(-1, 1),
        sampling_coord=[8, 8],
        width=12,
        height=12,
        n_steps=512)
    listen_to_sound(s)
