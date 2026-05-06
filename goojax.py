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

# def generate_params(batch_size: int, n_masses: int, dim: int):
#     key = random.key(1324)
    
#     tensions = random.uniform(key, (batch_size, n_masses, dim))
#     masses = random.uniform(key, (batch_size, n_masses, dim))
#     pass

# def sim():
#     pass


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

def tryjax():
    
    # This should list your GPU (e.g., CudaDevice(id=0))
    print(jax.devices())

    # This should return 'gpu'
    print(jax.default_backend())
    
    
    batch_size = 1
    n_masses = 16
    dim = 8
    n_samples = 2 ** 16
    
    rk = random.key(int(time()))
    rk, f, t, m, mic, l = random.split(rk, 6)
    
    damping = 0.9998
    
    # TODO: add a utility to generate a layer
    # direction of mics, which project ND vibration into a 1D signal
    mics = random.uniform(mic, (1, batch_size, n_masses, dim, 1), minval=-0.01, maxval=0.01)
    limits = random.uniform(l, (1, n_masses, dim), minval=-0.01, maxval=0.01)
    tensions = random.uniform(t, (1, n_masses, dim), minval=0.00001, maxval=5)
    masses = random.uniform(m, (1, n_masses, 1), minval=100, maxval=500)
    
    
    # home_pos = np.zeros((1, n_masses, dim))
    
    
    
    
    # TODO: add a time-varying home-position here
    @jax.jit
    def one_iter(carry: Carry, inputs: Inputs) -> Tuple[Carry, np.ndarray]:
        position, velocity = carry
        
        forces, home_pos = inputs
        
        
        # find displacement
        direction = home_pos - position
        
        # compute acceleration based on displacement
        acc = forces + ((tensions * direction) / masses)
        
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
        
                
        force = home_pos - new_position
        
        return ((new_position, new_velocity), force)
    
    
    initial_pos = np.zeros((batch_size, n_masses, dim))
    velocity = np.zeros((batch_size, n_masses, dim))
    
    start = time()
    
    forces = random.bernoulli(f, p=1e-5, shape=(n_samples, batch_size, n_masses, 1))
    
    
    home_pos = np.zeros((n_samples, batch_size, n_masses, dim))
    forces = forces * random.uniform(f, shape=(n_samples, batch_size, n_masses, dim ), minval=-0.1, maxval=0.1)
    
    inputs = (forces, home_pos)
    
    _, stacked_force = jax.lax.scan(
        f=one_iter,
        init=(initial_pos, velocity),
        xs=inputs,
        length=n_samples,
    )
    

    print(stacked_force.min(), stacked_force.max())    
    
    samples = np.einsum('abcd,abcde->ba', stacked_force, mics)
    
    
    normalized_samples = samples / (samples.max(axis=-1, keepdims=True) + 1e-8)
    
    
    
    end = time()
    
    n_seconds = n_samples / 22050
    t = end - start
    print(stacked_force.shape, n_seconds, t)
    
    # listen_to_sound(normalized_samples[0], wait_for_user_input=True)
    
    # plt.plot(normalized_samples[0])
    # plt.show()
    
    _, _, spec = sp.signal.stft(normalized_samples[0], nperseg=2048, noverlap=256)
    
    spec = np.log(1e-12 + np.abs(spec))
    
    
    plt.matshow(np.flipud(spec[:256, :]))
    plt.show()




if __name__ == '__main__':
    tryjax()
    tryjax()
    tryjax()