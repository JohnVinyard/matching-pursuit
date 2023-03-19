import numpy as np
from config.experiment import Experiment
from modules.matchingpursuit import dictionary_learning_step, flatten_atom_dict, sparse_code
from modules.normalization import unit_norm
from train.experiment_runner import BaseExperimentRunner
from util.readmedocs import readme
import zounds
from util import device, playable
import torch


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_atoms = 512
atom_size = 512
steps = 256
approx = None

d = torch.zeros(n_atoms, atom_size, device=device).uniform_(-1, 1)
d = unit_norm(d)


def events_to_tensor(events):
    """
    - (atom_index, batch, position, atom)
    """
    ai, b, p, a = zip(*events)
    batch_size = max(b) + 1
    n_events = len(events)

    t = torch.zeros(batch_size, 3, n_events)
    for i in range(n_events):
        b_index = b[i]
        at_index = ai[i]
        pos = p[i]
        amp = torch.norm(a[i])

        t[b_index, 0, i] = at_index
        t[b_index, 1, i] = pos
        t[b_index, 2, i] = amp
    
    return t

def tensor_to_events(events, d):
    output = []

    for b, seq in enumerate(events):
        for i in range(seq.shape[-1]):
            ai, p, a = events[b, :, i]
            output.append((ai, b, p, d[int(ai)] * a))
    
    return output


def train():
    pass

@readme
class SingleBandMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.encoded = None
    
    def roundtrip(self, steps=steps):
        with torch.no_grad():
            target = self.real[:1, ...]
            instances, scatter_segments = sparse_code(
                target, d, steps, device=device, approx=approx)
            instances = flatten_atom_dict(instances)
            t = events_to_tensor(instances)
            events = tensor_to_events(t, d)
            recon = scatter_segments(target.shape, events)
            return playable(recon, exp.samplerate)

    def recon(self, steps=steps):
        with torch.no_grad():
            target = self.real[:1, ...]
            instances, scatter_segments = sparse_code(
                target, d, steps, device=device, approx=approx)
            recon = scatter_segments(target.shape, flatten_atom_dict(instances))
            return playable(recon, exp.samplerate)

    def view_dict_spec(self):
        return np.abs(np.fft.rfft(d.data.cpu().numpy(), axis=-1)).T
    
    def view_dict(self):
        return d.data.cpu().numpy()

    def spec(self, steps=steps):
        return np.abs(zounds.spectral.stft(self.recon(steps)))

    def run(self):
        for i, item in enumerate(self.iter_items()):
            with torch.no_grad():
                self.real = item
                new_d = dictionary_learning_step(
                    item,
                    d,
                    steps,
                    device=device,
                    approx=approx)

                d[:] = unit_norm(new_d)
