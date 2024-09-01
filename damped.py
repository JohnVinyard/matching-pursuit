from typing import Tuple
import torch
import numpy as np
from soundfile import SoundFile
from io import BytesIO
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
from numba import jit


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


n_samples = 2**15
dimension = 3

def step(
        home: torch.Tensor,
        velocity: torch.Tensor,
        position: torch.Tensor,
        tension: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch, n_events, dim = home.shape


    direction = home - position
    acceleration = (tension * direction) / mass
    velocity += acceleration
    velocity *= damping
    position += velocity

    return home, velocity, position


def layer(
        home_pos: torch.Tensor,
        velocity: torch.Tensor,
        position: torch.Tensor,
        tension: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        n_samples: int) -> torch.Tensor:


    batch_size, n_events, dim = velocity.shape

    if home_pos.ndim == 3:
        home_pos = home_pos[..., None].repeat(1, 1, 1, n_samples)

    # a place to record the node's position
    rec = torch.zeros(batch_size, n_events, dimension, n_samples)

    for i in range(n_samples):
        _, vel, pos = step(
            home=home_pos[..., i],
            velocity=velocity,
            position=position,
            tension=tension,
            mass=mass,
            damping=damping)
        rec[:, :, :, i] = pos

    return rec


def main():
    batch_size = 4
    n_events = 3

    home_pos = torch.zeros(batch_size, n_events, dimension)
    vel = torch.zeros(batch_size, n_events, dimension)
    pos = torch.zeros(batch_size, n_events, dimension).fill_(10)

    tension = torch.zeros(batch_size, n_events, 1).fill_(0.01)
    mass = torch.zeros(batch_size, n_events, 1).fill_(100)
    damping = torch.zeros(batch_size, n_events, 1).fill_(0.9998)

    # a place to record the node's position
    # rec = torch.zeros(batch_size, n_events, dimension, n_samples)
    #
    # for i in range(n_samples):
    #     home_pos, vel, pos = step(
    #         home=home_pos,
    #         velocity=vel,
    #         position=pos,
    #         tension=tension,
    #         mass=mass,
    #         damping=damping)
    #     # print(rec.shape, pos.shape)
    #     rec[:, :, :, i] = pos

    rec = layer(
        home_pos, vel, pos, tension, mass, damping, n_samples=n_samples)

    samples = rec[:, :, 0, :]
    return samples


if __name__ == '__main__':
    s = main()
    plt.plot(s[0, 0, :].data.cpu().numpy()[:8192])
    plt.show()