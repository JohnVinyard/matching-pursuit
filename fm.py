from torch import nn
import torch

from dho import damped_harmonic_oscillator
from util import playable
from util.playable import listen_to_sound


class FM(nn.Module):

    def __init__(self, n_osc: int, n_samples: int):
        super().__init__()
        self.n_osc = n_osc
        self.n_samples = n_samples

        self.masses = nn.Parameter(torch.zeros(n_osc, 1).uniform_(-6, 6))
        self.tensions = nn.Parameter(torch.zeros(n_osc, 1).uniform_(3, 6))
        self.damping = nn.Parameter(torch.zeros(n_osc, 1).uniform_(-6, 6))
        self.initial_displacement = nn.Parameter(torch.zeros(n_osc, 1).uniform_(-0.1, 0.1))

        self.masses2 = nn.Parameter(torch.zeros(n_osc, 1).uniform_(-6, 6))
        self.tensions2 = nn.Parameter(torch.zeros(n_osc, 1).uniform_(-6, 6))
        self.damping2 = nn.Parameter(torch.zeros(n_osc, 1).uniform_(-6, 6))
        self.initial_displacement2 = nn.Parameter(torch.zeros(n_osc, 1).uniform_(-0.1, 0.1))

        self.scale = nn.Parameter(torch.zeros(1).uniform_(0.01, 1))

    def forward(self) -> torch.Tensor:

        time = torch.linspace(0, 10, self.n_samples, device=self.masses.device)

        m1 = torch.sigmoid(self.masses[..., None])
        d1 = torch.sigmoid(self.damping[..., None]) * 10
        t1 = 10 ** self.tensions[..., None]
        _id1 = self.initial_displacement[..., None]

        tension = damped_harmonic_oscillator(
            time=time,
            mass=m1,
            damping=d1,
            tension=t1,
            initial_displacement=_id1,
            initial_velocity=0
        )

        m2 = torch.sigmoid(self.masses2[..., None])
        d2 = torch.sigmoid(self.damping2[..., None]) * 10
        t2 = 10 ** (self.tensions2[..., None] + (tension * self.scale))
        _id2 = self.initial_displacement2[..., None]


        x = damped_harmonic_oscillator(
            time=time,
            mass=m2,
            damping=d2,
            tension=t2,
            initial_displacement=_id2,
            initial_velocity=0
        )

        print('============================')
        print(self.scale)
        print(m1, d1, t1, _id1)
        print(m2, d2, t2, _id2)

        x = torch.sum(x, dim=1, keepdim=True)
        return x

if __name__ == '__main__':

    while True:
        model = FM(n_osc=8, n_samples=2**16)
        samples = model.forward().view(-1)
        samples[0] = 0
        samples[-1] = 0
        samples = playable(samples, 22050, normalize=True)
        try:
            listen_to_sound(samples, wait_for_user_input=True)
        except KeyboardInterrupt:
            continue