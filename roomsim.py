from io import BytesIO
import os
from subprocess import PIPE, Popen
from typing import Tuple, Union
from matplotlib.animation import PillowWriter
from torch.nn import functional as F
import torch
from matplotlib import pyplot as plt
from soundfile import SoundFile
import numpy as np

from goo import listen_to_sound
from modules.normalization import unit_norm
from modules.overlap_add import overlap_add
from conjure.movie import FuncAnimation, tensor_movie


def encode_audio(
        x: Union[torch.Tensor, np.ndarray],
        samplerate: int = 22050,
        format='WAV',
        subtype='PCM_16') -> bytes:
    if isinstance(x, torch.Tensor):
        x = x.data.cpu().numpy()

    if x.ndim > 1:
        x = x[0]

    x = x.reshape((-1,))
    io = BytesIO()

    with SoundFile(
            file=io,
            mode='w',
            samplerate=samplerate,
            channels=1,
            format=format,
            subtype=subtype) as sf:
        sf.write(x)

    io.seek(0)
    return io.read()


def listen_to_sound(samples: bytes, wait_for_user_input: bool = True) -> None:
    proc = Popen(f'aplay', shell=True, stdin=PIPE)
    print('PROC', proc.stdin)
    if proc.stdin is not None:
        proc.stdin.write(samples)
        proc.communicate()

    if wait_for_user_input:
        input('Next')


def tensor_movie(
        filepath: os.PathLike,
        arr: Union[np.ndarray, torch.Tensor],
        fps: int=10):

    if isinstance(arr, torch.Tensor):
        arr = arr.data.cpu().numpy()

    if len(arr.shape) != 3:
        raise ValueError('make_movie expects a 3D array with shape (time, width, height)')

    n_frames, width, height = arr.shape
    
    
    
    try:

        data = []
        for i in range(n_frames):
            data.append(arr[i])

        fig = plt.figure()
        plt.axis('off')
        plt.margins(0, 0)

        plot = plt.imshow(data[0], cmap='gray')

        def init():
            plot.set_data(data[0])
            return [plot]

        def update(frame):
            plot.set_data(data[frame])
            return [plot]

        frame_delay = int(1000 / fps)
        ani = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, n_frames, 1),
            init_func=init,
            blit=True,
            interval=frame_delay)
        
        ani.save(filepath, writer=PillowWriter(fps=fps))
        plt.close()
        
    except Exception as e:
        print(f'Could not write movie due to')
   

def roomsim(
        samplerate: int,
        blocksize: int,
        transfer: torch.Tensor,
        n_frames: int,
        control_signal: torch.Tensor):
    """

    sound travels at 1125.33 feet per second

    wavelength = speed / frequency


    Then entire simulation will be represented as (batch=1, block_size, width, height, depth)

    The room is defined as a three-dimensional grid of real-valued transfer functions.
    Each of these will be a vector of size (blocksize // 2 + 1), with values in the range [0-1].

    Room boundaries will have perfectly-efficient reflection, i.e., they will lose no energy.  This will represent a
    very "live" room.  Future iterations of the code will implement energy loss/absorption by room boundaries.  All
    energy-loss will occur via sub-one transfer function values.

    mic_position will define the block that we "record" from.  This will be a three-dimensional coordinate
    in the "room".

    n_frames will determine how many blocks in the time dimension that the simulation should be run

    The control signal will represent the "injection" of energy into the room system

    At each (block) time step, we'll:

        - take th real-valued FFT of the control signal
        - apply the transfer function
        - propagate energy via an averaging kernel
    """

    print(control_signal.shape)

    wavelength_in_feet = 1125.33 / (samplerate / blocksize)
    total_seconds = (control_signal.shape[1] * blocksize) / samplerate

    print(
        f'Wavelength in feet {wavelength_in_feet}, total secondss {total_seconds}')
    print(
        f'Room size is ({wavelength_in_feet * control_signal.shape[-3]} ft x {wavelength_in_feet * control_signal.shape[-2]} ft x {wavelength_in_feet * control_signal.shape[-1]} ft)')

    # kernel = torch.ones(blocksize, blocksize, 3, 3, 3)

    # print('KERNEL', kernel.shape, kernel)

    recording = []

    room_state = torch.zeros(
        1,  # batch
        blocksize,  # channels
        width,  # width
        height,  # height
        depth  # depth
    )
    
    frames = torch.zeros(n_frames, width, height)

    for i in range(n_frames):
        
        room_state = room_state + control_signal[i: i + 1, ...]
        
        to_display = torch.norm(room_state[0, :, :, :, 4], dim=0)
        frames[i] = to_display 
        

        # apply transfer function in the frequency domain
        t = transfer[None, ...]
        cs = torch.fft.rfft(room_state, dim=1, norm='ortho')
        
        
        with_transfer_func = t * cs

        # return filtered control signal to time domain
        room_state = torch.fft.irfft(with_transfer_func, dim=1, norm='ortho')
        recorded_block = torch.sum(room_state[0: 1, ...], dim=(2, 3, 4))
        
        
        room_state = F.pad(room_state, [
            1, 1,
            1, 1,
            1, 1
        ], mode='reflect')
        
        neighborhood = room_state.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)
        room_state = torch.mean(room_state, dim=(-1, -2, -3))
        
        
        recording.append(recorded_block)

    # overlap add wants batch, channels, frames, samples
    
    # recording = torch.stack(recording, dim=1)[:, None, :, :]
    # recording = overlap_add(recording)    
    recording = torch.cat(recording, dim=0)
    # print('FINAL BLOCK SIZE', recording.shape)
    print(recording.shape)
    
    
    tensor_movie('movie.gif', frames, fps=1)
    
    return recording.view(-1)


if __name__ == '__main__':
    block_size = 64
    n_frames = 512
    width = 5
    height = 17
    depth = 9
    samplerate = 22050
    n_fft_coeffs = block_size // 2 + 1

    # create control signal of the correct shape (block-size, time and space dimensions)
    control_signal = torch.zeros(n_frames, block_size, width, height, depth)
    # input energy at a particular time and location
    control_signal[0, :, 3, 3, 3] = torch.zeros(block_size).uniform_(-1, 1)
    
    
    # first, generate 2d numbers on the unit sphere
    # t = unit_norm(torch.zeros(width, height, depth, n_fft_coeffs, 2).uniform_(-1, 1))
    # then, generate the magnitude of these points, bringing them off the surface and toward the center
    # mag = torch.zeros(width, height, depth, n_fft_coeffs, 1, ).fill_(0.999999)
    # t = (t * mag)
    # transfer = torch.view_as_complex(t).permute(3, 0, 1, 2)
    
    transfer_shape = (n_fft_coeffs, width, height, depth)
    
    transfer = torch.zeros(*transfer_shape).uniform_(0.99, 0.9999) * torch.linspace(1, 0, n_fft_coeffs)[:, None, None, None] ** 0.5
    transfer[0, ...] = 0

    # transfer = torch.zeros(n_fft_coeffs, width, height, depth,  dtype=torch.complex64).uniform_(-2.8, 2.8) #* torch.zeros(n_fft_coeffs).bernoulli_(p=0.1)

    samples = roomsim(
        samplerate=samplerate,
        blocksize=block_size,
        # frequency-domain transfer function should always be less-than one
        transfer=transfer,
        n_frames=n_frames,
        control_signal=control_signal
    )

    # print('TOTAL SAMPLES', samples.shape)
    print('TOTAL SECONDS', len(samples) / samplerate)

    plt.plot(samples)
    plt.show()

    listen_to_sound(
        encode_audio(samples)
    )
