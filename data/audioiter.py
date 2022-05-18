from .audiostream import audio_stream


class AudioIterator(object):

    def __init__(
            self,
            batch_size,
            n_samples,
            samplerate,
            normalize=False,
            overfit=False):

        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.normalize = normalize
        self.overfit = overfit

    def __iter__(self):
        return audio_stream(
            self.batch_size,
            self.n_samples,
            self.overfit,
            self.normalize,
            as_torch=True)
