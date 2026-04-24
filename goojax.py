from typing import Tuple

import torch
import jax
import jax.numpy as np
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


def tryjax():
    
    # This should list your GPU (e.g., CudaDevice(id=0))
    print(jax.devices())

    # This should return 'gpu'
    print(jax.default_backend())
    
    
    batch_size = 1
    n_masses = 16
    dim = 4
    n_samples = 2 ** 16
    
    rk = random.key(int(time()))
    rk, f, t, m, mic = random.split(rk, 5)
    
    damping = 0.9998
    
    # direction of mics, which project ND vibration into a 1D signal
    mics = random.uniform(mic, (1, batch_size, n_masses, dim, 1), minval=-0.01, maxval=0.01)
    
    tensions = random.uniform(t, (1, n_masses, dim), minval=30, maxval=300)
    home_pos = np.zeros((1, n_masses, dim))
    
    masses = random.uniform(m, (1, n_masses, 1), minval=50, maxval=5000)
    
    
    type Position = np.ndarray
    type Velocity = np.ndarray
    
    type Carry = Tuple[Position, Velocity]
    
    
    @jax.jit
    def one_iter(carry: Carry, forces: np.ndarray) -> Tuple[Carry, np.ndarray]:
        position, velocity = carry
        
        direction = home_pos - position
        
        acc = forces + ((tensions * direction) / masses)
        
        new_velocity = (velocity + acc) * damping
        new_position = position + new_velocity
        force = masses * acc
        
        return ((new_position, new_velocity), force)
    
    
    initial_pos = np.zeros((batch_size, n_masses, dim))
    velocity = np.zeros((batch_size, n_masses, dim))
    
    start = time()
    
    force_shape = (n_samples, batch_size, n_masses, dim)
    forces = random.bernoulli(f, p=1e-5, shape=force_shape)
    forces = forces * random.uniform(f, shape=force_shape, minval=-0.1, maxval=0.1)
    
    final_carry, stacked_force = jax.lax.scan(
        f=one_iter,
        init=(initial_pos, velocity),
        xs=forces,
        length=n_samples,
    )
    
    print(final_carry[0].shape, final_carry[1].shape)
    
    samples = np.einsum('abcd,abcde->ba', stacked_force, mics)
    print(samples.shape)
    
    
    normalized_samples = samples / (samples.max(axis=-1, keepdims=True) + 1e-8)
    
    print(samples.min(), samples.max(), samples)
    
    
    end = time()
    
    n_seconds = n_samples / 22050
    t = end - start
    # print(force.shape, n_seconds, t)
    
    listen_to_sound(normalized_samples[0], wait_for_user_input=True)
    
    plt.plot(normalized_samples[0])
    plt.show()
    
        

# @torch.jit.script
# def sim(
#         home: torch.Tensor,
#         tensions: torch.Tensor,
#         masses: torch.Tensor,
#         damping: float,
#         # gains: torch.Tensor,
#         mics: torch.Tensor,
#         forces: torch.Tensor,
#         home_modifier: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

#     batch, n_masses, dim, n_samples = home_modifier.shape

#     # NOTE: home is defined as zero for each node/mass, so this
#     # could simply be home_modifier directly
#     h = home + home_modifier

#     position = torch.zeros(batch, n_masses, dim, 1, device=forces.device)
#     velocity = torch.zeros(batch, n_masses, dim, 1, device=forces.device)

#     f = torch.zeros(batch, n_masses, dim, n_samples, device=forces.device)
    
#     velocity_recording = torch.zeros(batch, n_masses, dim, n_samples, device=forces.device)
    
#     # displacement = torch.zeros(batch, n_masses, dim, n_samples, device=forces.device)

#     for i in range(n_samples):
#         direction = h[..., i: i + 1] - position
        
#         # displacement[..., i: i + 1] = direction
        

#         acc = forces[..., i: i + 1] + ((tensions * direction) / masses)
#         velocity = velocity + acc
#         velocity = velocity * damping
        
#         velocity_recording[..., i: i + 1] = velocity
        
#         position = position + velocity

#         r = masses * acc
#         f[:, :, :, i: i + 1] = r

#     recording = f.permute(0, 1, 3, 2) @ mics
    

#     return recording, velocity_recording


if __name__ == '__main__':
    tryjax()
    tryjax()
    tryjax()