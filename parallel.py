from io import BytesIO
from subprocess import Popen, PIPE
from typing import Tuple, Callable, List

import torch
from matplotlib import pyplot as plt
import numpy as np
from soundfile import SoundFile

Solution = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

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
    lambda_: torch.Tensor,   # continuous-time damping rate (1/sec)
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


def parallel(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    # a[i] = d[i]
    # b[i] = d[i] * f[i]
    b = damping * forces

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

def test_osc(n_samples:int) -> torch.Tensor:
    f = torch.zeros(n_samples).bernoulli_(p=0.0001)
    d = torch.zeros(n_samples).fill_(0.9998)
    e = parallel(f, d)
    m = torch.zeros(1).uniform_(0.1, 0.5)
    damping = torch.zeros(1).fill_(1)
    tension = torch.zeros(1).uniform_(10**4, 10**9)
    initial_displacement = torch.zeros(1).fill_(1)

    x = damped_harmonic_oscillator(
        energy=e,
        time=torch.linspace(0, 1, n_samples),
        mass=m,
        damping=damping,
        tension=tension,
        initial_displacement=initial_displacement,
    )
    return x

def display_osc(n_samples: int):
    x = test_osc(n_samples)
    x = x / x.max()
    arr = x.data.cpu().numpy()
    plt.plot(arr)
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
    display_osc(2**16)