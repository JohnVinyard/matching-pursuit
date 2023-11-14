import zounds
import numpy as np
from scipy.signal import chirp, sawtooth


def energy_preservation():
    shape_latent = np.random.normal(0, 1, (128,))
    control_latent = np.random.normal(0, 1, (128,))

    network_weights = np.random.uniform(0, 1, (128, 128))
    network_weights /= (np.linalg.norm(network_weights, axis=-1 ,keepdims=True))
    network_weights = network_weights.T
    network_weights /= (np.linalg.norm(network_weights, axis=-1 ,keepdims=True))
    network_weights = network_weights.T
    

    inp = shape_latent + control_latent

    for i in range(1000):
        inp = inp @ network_weights
        print(i, inp.shape, np.linalg.norm(inp))


def work_it_out(control_signal):
    '''
    I keep going in circles.  This model is nice because it 
    allows me to do things like vibrato, which is difficult/impossible
    with the convolution model.  That said, the relationship between
    the control signal and the spectral shape isn't strong enough.  Typically,
    I take the control signal, apply some momentum, and then apply that envelope
    to a constant spectral shape.

    I keep returning to convolution because it makes the natural-sounding, complex
    relationship between the control signal and the spectral shape easier to model,
    that said, things like vibrato would require a large number of "atoms" to make
    work, since varying pitches all get blended together.

    resonance/spectral shape is a function of the:
        - control signal at this moment = CS
        - other parameters representing the instrument/shape of the body = BODY
        - some integration-like representation of the control signal going infinitely back in time (or some approx. thereof) = PAST
    
    CS varies over time
    BODY may vary over time (think bending strings or varying fingerings on a wind/brass instrument)
    PAST is some fixed exponential lookback/integration

    To simulate the fact that physical bodies may only change so quickly, we can instead
    create an initial shape parameter and then a time-series of shape deltas

    f(CS, BODY, PAST) = [f0, harmonic_ratios, amp]

    One approach is to window the audio and compute this frame-wise

    Another would be to simply produce many "atoms" from a single event, each with a control
    signal, and any number of resonances.

    The "many atoms" approach doesn't perfectly handle this scenario:

    A string is picked or plucked, and then grazed with a finger or bow soon after.

    The state of the string and instrument body must be taken into account to model the new event.

    I _think_ the atoms model can handle things like the spectral shape responding to impulse volume
    '''

    # control signal will be (batch, n_events, impulse_samples)
    # get its envelope (batch, n_events, n_frames)
    
    pass

if __name__ == '__main__':


    energy_preservation()
    exit()

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    samplerate = zounds.SR22050()
    n_samples = 2**15

    impulse_samples = n_samples

    impulse = np.random.uniform(-1, 1, impulse_samples) * np.hamming(impulse_samples)
    bow_amp = sawtooth(np.linspace(0, np.pi * 150, impulse_samples))
    impulse = impulse * bow_amp

    impulse = np.pad(impulse, [(0, n_samples - impulse_samples)])


    # start_hz = 220
    # end_hz = 300

    # start_radians = (start_hz / samplerate.nyquist) * np.pi
    # stop_radians = (end_hz / samplerate.nyquist) * np.pi

    # tremolo_freq = 6
    # tremolo_radians = (tremolo_freq / samplerate.nyquist) * np.pi
    # f = np.zeros(n_samples)
    # f[:] = tremolo_radians
    # tr = np.sin(np.cumsum(f)) * 0.002

    # # freqs = np.linspace(start_radians, stop_radians, n_samples)

    # freqs = start_hz + tr
    # signal = np.sin(np.cumsum(freqs))

    signal = chirp(np.linspace(0, 1, n_samples), 440, 0.9, 250)
    signal *= np.linspace(1, 0, n_samples) ** 2


    impulse_spec = np.fft.rfft(impulse, axis=-1, norm='ortho')
    signal_spec = np.fft.rfft(signal, axis=-1, norm='ortho')
    conv = signal_spec * impulse_spec
    final = np.fft.irfft(conv, norm='ortho')


    samples = zounds.AudioSamples(final, samplerate).pad_with_silence()

    spec = np.abs(zounds.spectral.stft(samples))

    input('waiting...')