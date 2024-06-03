import soundfile
import subprocess
import numpy as np
from io import BytesIO


class Mp3Encoder(object):
    def __init__(self):
        super().__init__()

    def __call__(self, flo):
        with soundfile.SoundFile(flo) as f:
            samples = f.read()
            samples *= ((2 ** 16) // 2)
            samples = samples.astype(np.int16)
            bio = BytesIO()
            proc = subprocess.Popen(
                args=[
                    'ffmpeg',
                    '-y',
                    '-loglevel', 'error',
                    '-f', 's16le',
                    '-ac', str(f.channels),
                    '-ar', str(f.samplerate),
                    '-i', '-',
                    '-acodec', 'libmp3lame',
                    '-f', 'mp3',
                    '-'
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate(input=samples.tobytes())
            bio.write(stdout)
            bio.seek(0)
            return bio


def encode_mp3(flo):
    encoder = Mp3Encoder()
    return encoder(flo)