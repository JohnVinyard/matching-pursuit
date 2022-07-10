from torch import nn
import torch


class NeuralReverb(nn.Module):
    def __init__(self, size, n_rooms, impulses=None):
        super().__init__()
        self.size = size
        self.n_rooms = n_rooms

        if impulses is None:
            imp = torch.FloatTensor(self.n_rooms, self.size).uniform_(-0.01, 0.01)
        else:
            imp = torch.from_numpy(impulses)
            if imp.shape != (self.n_rooms, self.size):
                raise ValueError(
                    f'impulses must have shape ({self.n_rooms}, {self.size}) but had shape {imp.shape}')

        self.rooms = nn.Parameter(imp)

    def forward(self, x, reverb_mix):

        mx, _ = torch.max(self.rooms, dim=-1, keepdim=True)
        rooms = self.rooms / (mx + 1e-12)

        # choose a linear mixture of "rooms"
        mix = (reverb_mix[:, None, :] @ rooms)

        reverb_spec = torch.fft.rfft(mix, dim=-1, norm='ortho')
        signal_spec = torch.fft.rfft(x, dim=-1, norm='ortho')

        # convolution in the frequency domain
        x = reverb_spec * signal_spec

        x = torch.fft.irfft(x, dim=-1, n=self.size, norm='ortho')

        return x


if __name__ == '__main__':

    n_samples = 2**14
    n_rooms = 1
    batch_size = 4

    signal = torch.FloatTensor(batch_size, 1, n_samples).uniform_(-1, 1)
    mix = torch.FloatTensor(batch_size, n_rooms).uniform_(-1, 1)

    reverb = NeuralReverb(n_samples, n_rooms)

    z = reverb.forward(signal, mix)

    print(z.shape)
