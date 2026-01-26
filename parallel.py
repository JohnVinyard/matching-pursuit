from io import BytesIO
from subprocess import Popen, PIPE
from typing import Tuple, Callable, List

from conjure import SupportedContentType, NumpySerializer, NumpyDeserializer, IdentitySerializer, IdentityDeserializer
from torch import nn

import torch
from matplotlib import pyplot as plt
import numpy as np
from soundfile import SoundFile
from torch.nn import functional as F

import conjure
from conjure.logger import encode_audio
from modules import sparsify
from modules.upsample import upsample_with_holes
from util import device
from util.overfit import overfit_model

Solution = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

"""
TODO:
- understand and integrate sample-rate free decays
- add momentum/springiness to tension and damping
"""


def stft(
        x: torch.Tensor,
        ws: int = 512,
        step: int = 256,
        pad: bool = False):
    frames = x.shape[-1] // step

    if pad:
        x = F.pad(x, (0, ws))
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')
    x = torch.abs(x)
    x = x[:, :, :frames, :]
    return x


# TODO: It might be nice to move this into zounds
def listen_to_sound(
        samples: np.ndarray,
        wait_for_user_input: bool = True) -> None:
    bio = BytesIO()
    with SoundFile(bio, mode='w', samplerate=22050, channels=1, format='WAV', subtype='PCM_16') as sf:
        sf.write(samples.astype(np.float32))

    bio.seek(0)
    data = bio.read()

    proc = Popen(f'aplay', shell=True, stdin=PIPE)

    if proc.stdin is not None:
        proc.stdin.write(data)
        proc.communicate()

    if wait_for_user_input:
        input('Next')


@torch.jit.script
def damped_harmonic_oscillator(
        energy: torch.Tensor,
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor
) -> torch.Tensor:
    x = (damping / (2 * mass))

    omega = torch.sqrt(torch.abs(tension - (x ** 2)))

    phi = torch.atan2(
        (x * initial_displacement),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    # z = a * torch.exp(-x * time) * torch.cos(omega * time - phi)
    z = a * energy * torch.cos(omega * time - phi)
    return z


def sequential(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(forces)
    for i in range(forces.shape[-1]):
        output[i] = (forces[i] + output[i - 1]) * damping[i]
    return output


def parallel_sr_independent(
        forces: torch.Tensor,
        lambda_: torch.Tensor,  # continuous-time damping rate (1/sec)
        sample_rate: float,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    forces:    (..., T)
    lambda_:   (..., T)   continuous-time damping rates
    returns:   (..., T)
    """
    dt = 1.0 / sample_rate

    # Discretize the continuous-time system
    alpha = torch.exp(-lambda_ * dt)

    # Stable computation of beta
    beta = torch.where(
        lambda_.abs() > eps,
        (1.0 - alpha) / lambda_,
        torch.full_like(lambda_, dt),  # limit as lambda -> 0
    )

    # Recurrence: o[n] = alpha[n] * o[n-1] + beta[n] * forces[n]
    b = beta * forces

    # Parallel scan formulation
    p = torch.cumprod(alpha, dim=-1)
    s = torch.cumsum(b / p, dim=-1)

    return p * s


def parallel(forces: torch.Tensor, damping: torch.Tensor, log_space_decay: bool = False) -> torch.Tensor:
    # a[i] = d[i]
    # b[i] = d[i] * f[i]
    b = damping * forces

    if log_space_decay:
        damping = torch.log(damping + 1e-12)
        damping = torch.cumsum(damping, dim=-1)
        p = torch.exp(damping)
    else:
        # p[i] = prod_{j<=i} d[j]
        p = torch.cumprod(damping, dim=-1)

    # sum_{k<=i} b[k] / p[k]
    s = torch.cumsum(b / p, dim=-1)

    # o[i] = p[i] * s[i]
    return p * s


def generate_params(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    f = torch.zeros(n_samples).bernoulli_(p=0.001)
    d = torch.zeros(n_samples).fill_(0.9991)
    return f, d


def test(nsamples: int):
    f, d = generate_params(nsamples)
    x = sequential(f, d)
    return x


class Layer(nn.Module):
    def __init__(self, n_nodes: int, n_samples: int, control_rate: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate

        # TODO: eventually, this will vary at control rate and will have "momentum",
        # springing back to a baseline value

        self.damp = nn.Parameter(torch.zeros(1, self.n_nodes, 1).uniform_(-6, 6))
        # damp = torch.zeros(1, self.n_nodes, 1).uniform_(0.9997, 0.9998)
        # self.register_buffer('damp', damp)

        self.mass = nn.Parameter(torch.zeros(1, self.n_nodes, 1).uniform_(-6, 6))

        # TODO: eventually, this will vary at control rate and will have "momentum",
        # springing back to a baseline value
        self.tension = nn.Parameter(torch.zeros(1, self.n_nodes, 1).uniform_(4, 9))

        d = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('d', d)

        _id = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('_id', _id)

        t = torch.linspace(0, 10, self.n_samples)
        self.register_buffer('t', t)

        # self.influence = nn.Parameter(torch.zeros(1, self.n_nodes, 1).uniform_(-0.01, 0.01))

        self.force_router = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes).uniform_(-0.01, 0.01))
        self.tension_router = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes).uniform_(-0.001, 0.001))

    def forward(self, forces: torch.Tensor, tension_modifier: torch.Tensor = None) -> torch.Tensor:
        forces = torch.einsum('abc,bd->adc', forces, self.force_router)

        damp = 0.9997 + (torch.sigmoid(self.damp) * 0.0002)
        damp = damp.repeat(1, 1, self.n_samples)
        energy = parallel(forces, damp)

        mass = torch.sigmoid(self.mass) * 500
        tension = self.tension

        if tension_modifier is not None:
            tension_modifier = torch.einsum('abc,bd->adc', tension_modifier, self.tension_router)
            tension = tension + tension_modifier

        x = damped_harmonic_oscillator(
            energy=energy,
            time=self.t,
            mass=mass,
            damping=self.d,
            tension=10 ** tension,
            initial_displacement=self._id,
        )

        return x


class LayerController(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_nodes: int,
            n_samples: int,
            control_rate: int,
            n_to_keep: int = 1024):

        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate
        self.n_to_keep = n_to_keep

        self.forces = nn.Parameter(torch.zeros(1, n_nodes, self.n_frames).bernoulli_(p=1e-4))

        self.layers = nn.ModuleList([Layer(n_nodes, n_samples, control_rate) for _ in range(self.n_layers)])

    def materialize_forces(
            self, forces:
            torch.Tensor = None,
            do_upsample: bool = True,
            n_to_keep: int = None) -> torch.Tensor:

        if forces is not None:
            f = forces
        else:
            f = self.forces

        f = f + f.mean()

        if do_upsample:
            f = upsample_with_holes(f, desired_size=self.n_samples)

        f = sparsify(f, n_to_keep=n_to_keep or self.n_to_keep)
        return f

    def random_forward(self, sum_output: bool = True, n_to_keep: int = None) -> torch.Tensor:
        f = torch.zeros_like(self.forces).uniform_(0, 1)
        return self.forward(sum_output=sum_output, forces=f, n_to_keep=n_to_keep)

    def forward(self, sum_output: bool = True, forces: torch.Tensor = None, n_to_keep: int = None) -> torch.Tensor:

        # forces = forces or self.forces
        # forces = forces + forces.mean()
        # sparse_forces = sparsify(forces, n_to_keep=self.n_to_keep)

        if forces is not None:
            sparse_forces = self.materialize_forces(forces, n_to_keep=n_to_keep)
        else:
            sparse_forces  = self.materialize_forces(self.forces, n_to_keep=n_to_keep)

        tm = None
        for i, layer in enumerate(self.layers):
            tm = layer.forward(forces=sparse_forces, tension_modifier=tm)

        if sum_output:
            tm = torch.sum(tm, dim=1, keepdim=True)

        return tm


def test_osc(n_nodes: int, n_samples: int) -> torch.Tensor:
    controller = LayerController(
        n_layers=3,
        n_nodes=n_nodes,
        n_samples=n_samples,
        control_rate=128
    )
    x = controller.forward()
    x = torch.sum(x, dim=1, keepdim=True)
    return x


def loss_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = stft(a)
    b = stft(b)
    return torch.abs(a - b).sum()




def overfit_osc(n_nodes: int, n_samples: int, n_layers: int):
    controller = LayerController(
        n_layers=n_layers,
        n_nodes=n_nodes,
        n_samples=n_samples,
        control_rate=128
    ).to(device)

    def logger_factory(collection):

        frce, = conjure.loggers(
            ['forces'],
            SupportedContentType.Spectrogram.value,
            lambda x: x.data.cpu().numpy()[0],
            collection,
            NumpySerializer(),
            NumpyDeserializer()
        )

        rnd,  = conjure.loggers(
            ['random'],
            SupportedContentType.Audio.value,
            encode_audio,
            collection,
            IdentitySerializer(),
            IdentityDeserializer())

        return [frce, rnd]

    def training_loop_hook(
            iteration: int,
            loggers: List[conjure.Conjure],
            model: LayerController):

        # TODO: index loggers by name
        rnd_logger = loggers[-1]
        frce_logger = loggers[-2]

        with torch.no_grad():
            rnd = model.random_forward(n_to_keep=16)
            rnd_logger(rnd)
            f = model.materialize_forces(do_upsample=False)
            frce_logger(f)



    overfit_model(
        n_samples=n_samples,
        model=controller,
        loss_func=loss_func,
        collection_name='parallel',
        logger_factory=logger_factory,
        training_loop_hook=training_loop_hook,
        learning_rate=1e-4
    )


def display_osc(n_nodes: int, n_samples: int):
    x = test_osc(n_nodes, n_samples)
    print(x.max().item())
    x = x / x.max()

    spec = stft(x.view(1, 1, -1))
    spec = spec.view(-1, spec.shape[-1])
    spec = spec.data.cpu().numpy()

    arr = x.data.cpu().numpy().reshape((-1,))
    plt.plot(arr)
    plt.show()

    plt.matshow(spec)
    plt.show()

    listen_to_sound(arr, wait_for_user_input=True)


def harness(n_samples: int, *solutions: Solution):
    f, d = generate_params(n_samples)
    for solution in solutions:
        x = solution(f, d)
        plt.plot(x.data.cpu().numpy())

    plt.show()


if __name__ == '__main__':
    # harness(2**15, sequential, parallel)
    # display_osc(32, 2**17)
    overfit_osc(n_nodes=64, n_samples=2 ** 17, n_layers=3)
