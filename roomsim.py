from typing import Tuple
from torch.nn import functional as F
import torch
from matplotlib import pyplot as plt


def roomsim(
        samplerate: int,
        blocksize: int,
        transfer: torch.Tensor,
        mic_position: Tuple[int, int, int],
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

    print(f'Wavelength in feet {wavelength_in_feet}, total secondss {total_seconds}')
    print(
        f'Room size is ({wavelength_in_feet * control_signal.shape[-3]} ft x {wavelength_in_feet * control_signal.shape[-2]} ft x {wavelength_in_feet * control_signal.shape[-1]} ft)')

    kernel = torch.ones(blocksize, blocksize, 3, 3, 3) * (1 / 18) * 0.9

    # print('KERNEL', kernel.shape, kernel)

    recording = []

    room_state = torch.zeros(
        1,  # batch
        blocksize,  # channels
        width,  # width
        height,  # height
        depth  # depth
    )

    for i in range(n_frames):
        # print(i, '========================================')
        # apply transfer function to input control signal
        # print('ROOM STATE', room_state.shape)
        # print('CONTROL SIGNAL', control_signal[i: i + 1, ...].shape)

        print('BEFORE TRANSFER', i, torch.norm(control_signal[i: i + 1, ...]))

        # apply transfer function in the frequency domain
        t = transfer[None, :, None, None, None]
        cs = torch.fft.rfft(control_signal[i: i + 1, ...], dim=1, norm='ortho')
        with_transfer_func = t * cs

        # return filtered control signal to time domain
        time_domain = torch.fft.irfft(with_transfer_func, dim=1, norm='ortho')
        print('AFTER TRANSFER', i, torch.norm(time_domain))
        # print('TIME DOMAIN', time_domain.shape)

        room_state = room_state + time_domain

        # print(room_state.shape)
        room_state = F.pad(room_state, [1, 1, 1, 1, 1, 1], mode='reflect')
        # print(i, room_state.shape)
        room_state = F.conv3d(room_state, kernel)
        # print('AFTER CONV', i, room_state.shape)

        recorded_block = room_state[0: 1, :, mic_position[0], mic_position[1], mic_position[2]]
        print(i, 'RECORDED BLOCK', torch.norm(recorded_block))
        recording.append(recorded_block)

    recording = torch.cat(recording, dim=0)
    # print('FINAL BLOCK SIZE', recording.shape)
    print(recording.shape)
    return recording.view(-1)


if __name__ == '__main__':
    block_size = 32
    n_frames = 128
    width = 16
    height = 16
    depth = 16
    samplerate = 22050
    n_fft_coeffs = block_size // 2 + 1

    # create control signal of the correct shape (block-size, time and space dimensions)
    control_signal = torch.zeros(n_frames, block_size, width, height, depth)
    # input energy at a particular time and location
    control_signal[0, :, 3, 3, 3] = torch.zeros(block_size).uniform_(-1, 1)

    samples = roomsim(
        samplerate=samplerate,
        blocksize=block_size,
        # frequency-domain transfer function should always be less-than one
        transfer=torch.ones(n_fft_coeffs) * 0.99,
        mic_position=(width // 2, height // 2, depth // 2),
        n_frames=n_frames,
        control_signal=control_signal
    )

    # print('TOTAL SAMPLES', samples.shape)
    print('TOTAL SECONDS', len(samples) / samplerate)

    plt.plot(samples)
    plt.show()
