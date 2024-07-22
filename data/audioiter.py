from .audiostream import audio_stream


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
            return_indices=False):

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

    def __iter__(self):
        return audio_stream(
            self.batch_size,
            self.n_samples,
            self.overfit,
            self.normalize,
            as_torch=self.as_torch,
            step_size=self.step_size,
            pattern=self.pattern,
            return_indices=self.return_indices)
