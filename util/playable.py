from io import BytesIO
from typing import Union
import numpy as np
import zounds
import torch
from subprocess import Popen, PIPE
from soundfile import SoundFile


def decode_audio(data: bytes) -> np.ndarray:
    bio = BytesIO(data)
    with SoundFile(bio, mode='rb') as sound:
        return sound.read()


def encode_audio(
        x: Union[torch.Tensor, np.ndarray],
        samplerate: int = 22050,
        format='WAV',
        subtype='PCM_16'):

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

def playable(
    x: Union[torch.Tensor, np.ndarray], 
    samplerate: Union[zounds.SampleRate, int], 
    normalize: bool =False, 
    pad_with_silence: bool = True) -> zounds.AudioSamples:
    
    if not isinstance(x, np.ndarray):
        x = x.data.cpu().numpy()
    
    if len(x.shape) != 1:
        x = x[0].reshape(-1)
    
    if isinstance(samplerate, int):
        samplerate = zounds.audio_sample_rate(samplerate)
    
    samples  = zounds.AudioSamples(x, samplerate).pad_with_silence()
    
    if normalize:
        mx = samples.max()
        samples = samples / (mx + 1e-8)
    
    return samples


# TODO: It might be nice to move this into zounds
def listen_to_sound(samples: zounds.AudioSamples, wait_for_user_input: bool = True) -> None:
    proc = Popen(f'aplay', shell=True, stdin=PIPE)
    
    if proc.stdin is not None:
        proc.stdin.write(samples.encode().read())
        proc.communicate()
    
    if wait_for_user_input:
        input('Next')
    

def viewable(x, samplerate, normalize=False):
    p = playable(x, samplerate)
    return np.abs(zounds.spectral.stft(p))