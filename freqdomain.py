from typing import Union

import torch
from modules.overlap_add import overlap_add
from modules.phase import windowed_audio
from util import playable
from util.playable import listen_to_sound
from torch import nn

class Block(nn.Module):

    def __init__(self, window_size: int, transfer: torch.Tensor, gain: Union[float, torch.Tensor]):
        super().__init__()
        self.window_size = window_size
        self.n_coeffs = self.window_size // 2 + 1

        self.transfer = nn.Parameter(torch.zeros((self.n_coeffs,)))
        self.transfer.data[:] = transfer

        self.gain = nn.Parameter(torch.ones((1,)).fill_(gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time = x.shape
        windowed = windowed_audio(x, self.window_size, self.window_size // 2)
        spec = torch.fft.rfft(windowed, dim=-1)
        batch, _, frames, coeffs = spec.shape

        output_frames = []

        for i in range(frames):
            current = spec[:, :, i: i + 1, :]

            if i > 0:
                current = current + output_frames[i - 1]

            filtered = current * self.transfer[None, None, None, :]

            output_frames.append(filtered)

        output = torch.cat(output_frames, dim=2)
        audio_windows = torch.fft.irfft(output, dim=-1)
        samples = overlap_add(audio_windows, apply_window=True)
        samples = samples * self.gain
        samples = torch.tanh(samples)
        return samples[..., :time]



class AudioNetwork(nn.Module):
    def __init__(self, window_size: int, n_blocks: int):
        super().__init__()
        self.window_size = window_size
        self.n_blocks = n_blocks
        self.mixer = nn.Parameter(torch.zeros((n_blocks + 1,)))

        self.blocks = nn.ModuleList([
            Block(window_size, self.init_transfer(), torch.zeros(1).uniform_(10, 200).item())
            for _ in range(self.n_blocks)
        ])

    def init_transfer(self):
        resonances = torch.zeros((batch_size, n_coeffs)).bernoulli_(p=0.01)
        scaling = torch.zeros_like(resonances).uniform_(0.9, 0.998)
        pitch = torch.linspace(1, 0, n_coeffs) ** 2
        scaling = scaling * pitch
        scaled_resonances = resonances * scaling
        return scaled_resonances


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x[..., None]]
        inp = x

        for block in self.blocks:
            inp = block(inp)
            outputs.append(inp[..., None])


        result = torch.cat(outputs, dim=-1)
        mixer_values = torch.softmax(self.mixer, dim=-1)
        mixed = (result * mixer_values[None, None, None, :]).sum(dim=-1)
        return mixed



if __name__ == '__main__':
    batch_size = 1
    window_size = 2048
    n_frames = 128
    n_coeffs = window_size // 2 + 1
    gain = 500


    resonances = torch.zeros((batch_size, n_coeffs)).bernoulli_(p=0.01)
    scaling = torch.zeros_like(resonances).uniform_(0.99, 0.998)
    pitch = torch.linspace(1, 0, n_coeffs) ** 2
    scaling = scaling * pitch
    scaled_resonances = resonances * scaling
    #
    # samples = freq_domain_transfer_function_to_resonance(window_size, scaled_resonances, 256, apply_decay=True)
    # print(samples.min(), samples.max())
    # samples = samples * gain
    # samples = torch.tanh(samples)
    #

    impulse_size = 64
    impulse = torch.zeros(impulse_size).uniform_(-1, 1)
    impulse = impulse * torch.hann_window(impulse_size)

    inp = torch.zeros(1, 1, 2**18)

    inp[:, :, 1024: 1024 + impulse_size] += impulse * 0.1
    inp[:, :, 16384: 16384 + impulse_size] += impulse

    network = AudioNetwork(window_size, n_blocks=8)
    samples = network.forward(inp)
    # block = Block(window_size=window_size, transfer=scaled_resonances, gain=gain)
    # samples = block.forward(inp)

    print(samples.min(), samples.max())
    samples = playable(samples, samplerate=22050, normalize=True)
    listen_to_sound(samples)