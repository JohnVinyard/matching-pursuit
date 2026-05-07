from dataclasses import dataclass
from typing import Tuple

import torch
import jax
import jax.numpy as np
import jax.scipy as sp
from jax import random
import matplotlib
from matplotlib import pyplot as plt
from time import time
from io import BytesIO
from soundfile import SoundFile
from subprocess import Popen, PIPE


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


type Position = np.ndarray
type Velocity = np.ndarray
type Forces = np.ndarray
type HomePosition = np.ndarray
type Carry = Tuple[Position, Velocity]
type Inputs = Tuple[Forces, HomePosition]


@dataclass
class LayerParameters:
    limits: np.ndarray
    tensions: np.ndarray
    masses: np.ndarray
    damping: np.ndarray


def create_layer_parameters(n_masses: int, dim: int, seed:int) -> LayerParameters:
    rk = random.key(seed)
    rk, t, m, l, d = random.split(rk, 5)

    limits = random.uniform(l, (1, n_masses, dim), minval=-10, maxval=10)
    tensions = random.uniform(t, (1, n_masses, dim), minval=0.001, maxval=2)
    masses = random.uniform(m, (1, n_masses, 1), minval=1, maxval=1000)
    damping = random.uniform(d, (1, n_masses, 1), minval=0.9990, maxval=0.998)

    return LayerParameters(
        limits=limits,
        tensions=tensions,
        masses=masses,
        damping=damping
    )


def create_iter_func(n_masses: int, dim: int, seed: int):
    layer_params = create_layer_parameters(n_masses, dim, seed)

    tensions = layer_params.tensions
    masses = layer_params.masses
    limits = layer_params.limits
    damping = layer_params.damping

    # damping = 0.999

    @jax.jit
    def one_iter(carry: Carry, inputs: Inputs) -> Tuple[Carry, np.ndarray]:
        position, velocity = carry

        forces, home_pos = inputs

        # find displacement
        direction = home_pos - position

        # compute acceleration based on displacement
        acc = forces + (((tensions + home_pos) * direction) / masses)

        # update velocity
        new_velocity = (velocity + acc) * damping

        # update position
        new_position = position + new_velocity

        # first, check boundary conditions
        clamped_pos = np.clip(new_position, -np.abs(limits), np.abs(limits))
        diff = (np.abs(new_position) - np.abs(clamped_pos)) + 1e-12
        s = np.sign(diff)

        new_position = clamped_pos - (1e-12 * -s)
        new_velocity = new_velocity * s

        # this is really just displacement
        force = home_pos - new_position
        
        # force = masses * acc

        return ((new_position, new_velocity), force)

    return one_iter


def make_oscillator_stack(n_layers: int, batch_size: int, n_masses: int, dim: int):
    pass

def tryjax():

    # This should list your GPU (e.g., CudaDevice(id=0))
    print(jax.devices())

    # This should return 'gpu'
    print(jax.default_backend())

    batch_size = 1
    n_masses = 16
    dim = 3
    n_samples = 2 ** 16

    rk = random.key(int(time()))
    rk, m, m2, f, p, inf = random.split(rk, 6)

    mics = random.uniform(
        m, (1, batch_size, n_masses, dim, 1), minval=-0.01, maxval=0.01)
    
    mics2 = random.uniform(
        m2, (1, batch_size, n_masses, dim, 1), minval=-0.01, maxval=0.01)
    
    influence = random.uniform(
        inf, (n_masses, n_masses, dim), minval=-0.05, maxval=0.05
    )

    one_iter = create_iter_func(n_masses, dim, seed=int(time()) + 11)
    two_iter = create_iter_func(n_masses, dim, seed=int(time()) + 21)

    initial_pos = np.zeros((batch_size, n_masses, dim))
    initial_pos_2 = np.zeros((batch_size, n_masses, dim))
    velocity = np.zeros((batch_size, n_masses, dim))
    velocity_2 = np.zeros((batch_size, n_masses, dim))

    start = time()

    forces = random.bernoulli(
        f, p=1e-5, shape=(n_samples, batch_size, n_masses, 1))
    zero_force = np.zeros_like(forces)

    # per-sample position that the nodes are trying to return to
    # via a spring mechanism
    home_pos = np.zeros((n_samples, batch_size, n_masses, dim))

    forces = forces * \
        random.uniform(p, shape=(n_samples, batch_size,
                       n_masses, dim), minval=-1, maxval=1)

    inputs = (forces, home_pos)

    _, stacked_force = jax.lax.scan(
        f=one_iter,
        init=(initial_pos, velocity),
        xs=inputs,
        length=n_samples,
    )

    force_with_influence = np.einsum('abcd,ecd->abcd', stacked_force, influence)
    
    inputs2 = (zero_force, force_with_influence)
        
    _, stacked_force_2 = jax.lax.scan(
        f=two_iter,
        init=(initial_pos_2, velocity_2),
        xs=inputs2,
        length=n_samples,
    )

    # samples = np.einsum('abcd,abcde->ba', stacked_force, mics)
    samples2 = np.einsum('abcd,abcde->ba', stacked_force_2, mics2)
    samples = samples2

    normalized_samples = samples / (samples.max(axis=-1, keepdims=True) + 1e-8)

    end = time()

    n_seconds = n_samples / 22050
    t = end - start
    print(stacked_force.shape, n_seconds, t)
    
    listen_to_sound(normalized_samples[0], wait_for_user_input=True)

    _, _, spec = sp.signal.stft(
        normalized_samples[0], nperseg=512, noverlap=256)

    spec = np.abs(spec) ** 0.5

    plt.matshow(np.flipud(spec[:128, :]))
    plt.show()


if __name__ == '__main__':
    tryjax()
    tryjax()
    tryjax()
