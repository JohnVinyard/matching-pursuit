from typing import Union, Optional

from .audiostream import audio_stream
import torch


class AudioIterator(object):

    def __init__(
            self,
            batch_size,
            n_samples,
            samplerate,
            normalize=False,
            overfit=False,
            step_size=1,
            pattern='*.wav',
            as_torch=True,
            return_indices=False,
            audio_path: str = None):

        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.normalize = normalize
        self.overfit = overfit
        self.step_size = step_size
        self.pattern = pattern
        self.as_torch = as_torch
        self.return_indices = return_indices
        self.audio_path = audio_path

    def __iter__(self):
        return audio_stream(
            self.batch_size,
            self.n_samples,
            self.overfit,
            self.normalize,
            as_torch=self.as_torch,
            step_size=self.step_size,
            pattern=self.pattern,
            return_indices=self.return_indices,
            audio_path=self.audio_path)


def get_one_audio_segment(
        n_samples: int,
        samplerate: int = 22050,
        device: Union[torch.device, None] = None,
        pattern: str = '*.wav',
        audio_path: Optional[str] = None) -> torch.Tensor:
    ai = AudioIterator(
        batch_size=1,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True,
        overfit=True,
        pattern=pattern,
        audio_path=audio_path)
    target: torch.Tensor = next(iter(ai)).to(device).view(-1, 1, n_samples)
    return target

def get_one_audio_batch(
        batch_size: int,
        n_samples: int,
        samplerate: int = 22050,
        device: Union[torch.device, None] = None,
        pattern: str = '*.wav',
        audio_path: Optional[str] = None) -> torch.Tensor:
    ai = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True,
        overfit=True,
        pattern=pattern,
        audio_path=audio_path)
    target: torch.Tensor = next(iter(ai)).to(device).view(-1, 1, n_samples)
    return target
