from enum import auto
import torch
import zounds
from modules.reverb import NeuralReverb
from util import playable


sr = zounds.SR22050()
n_samples = 2**15
n_rooms = 1



if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    synth = zounds.TickSynthesizer(sr)
    signal = synth.synthesize(sr.frequency * (n_samples + 1), sr.frequency * 8192)
    signal[:8192] = 0
    signal[-9000:] = 0
    signal = torch.from_numpy(signal).float().view(1, 1, -1)

    impulses = torch.zeros(1, n_samples).normal_(0, 1)
    env = torch.linspace(1, 0, steps=n_samples) ** 8

    impulses = impulses * env.view(1, -1)

    verb = NeuralReverb(n_samples, n_rooms, impulses=impulses.data.cpu().numpy())
    rm = torch.zeros(1, n_rooms).normal_(0, 1)
    rm = torch.softmax(rm, dim=-1)


    wet = verb.forward(signal[..., :n_samples], rm)
    signal = (signal * 0.5) + (wet * 0.5)

    signal = playable(signal, sr)


    input('Waiting...')