import numpy as np
import zounds
from subprocess import Popen, PIPE
from scipy.interpolate import interp1d

# TODO: It might be nice to move this into zounds
def listen_to_sound(samples: zounds.AudioSamples, wait_for_user_input: bool = True) -> None:
    proc = Popen(f'aplay', shell=True, stdin=PIPE)
    
    if proc.stdin is not None:
        proc.stdin.write(samples.encode().read())
        proc.communicate()
    
    if wait_for_user_input:
        input('Next')


def harmonics(f0: float, n_octaves: int, spacing: float):
    return f0 * np.linspace(1, spacing * (n_octaves + 1), n_octaves)

def osc_bank(hz: np.ndarray, n_samples: int, samplerate: int):
    mask = hz >= (samplerate // 2)
    hz[mask] = 0
    radians = (hz / samplerate) * np.pi
    radians = radians.reshape(-1, 1).repeat(n_samples, axis=-1)
    osc = np.sin(np.cumsum(radians, axis=-1))
    return osc

def exponential_decays(factors: np.ndarray, n_samples: int, n_frames: int):
    factors = factors.reshape(-1, 1).repeat(n_frames, axis=-1)
    factors = np.cumprod(factors, axis=-1)
    x1 = np.linspace(0, 1, n_frames)
    func = interp1d(x1, factors, kind='linear', axis=-1)
    amps = func(np.linspace(0, 1, n_samples))
    return amps





if __name__ == '__main__':
    
    n_samples = 2 ** 16
    harm = harmonics(300, 64, spacing=0.8)
    
    start_amps = np.linspace(1, 0.01, 64) 
    amps = exponential_decays(np.linspace(0.99, 0.001, 64) ** 2, n_samples, 128)
    env = start_amps[:, None] * amps
    
    x = osc_bank(harm, n_samples, 22050) * start_amps[:, None]
    
    final = (x * env).sum(axis=0)
    final = final / (final.max() + 1e-8)
    
    x = zounds.AudioSamples(final, zounds.SR22050())
    listen_to_sound(x)