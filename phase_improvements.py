
from config.experiment import Experiment
from data.audiostream import audio_stream
from modules.phase import AudioCodec, MelScale
import zounds

from util.playable import playable

codec = AudioCodec(MelScale())


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    exp = Experiment(zounds.SR22050(), 2**15)

    stream = audio_stream(1, exp.n_samples, as_torch=True)
    for item in stream:
        spec = codec.to_frequency_domain(item)
        recon = codec.to_time_domain(spec)

        mag = spec[..., 0].data.cpu().numpy().squeeze()
        phase = spec[..., 1].data.cpu().numpy().squeeze()


        r = playable(recon, exp.samplerate)
        

        input('next...')
        