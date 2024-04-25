import torch
from torch.nn import functional as F

def torch_conv(signal, atom):
    n_samples = signal.shape[-1]
    n_atoms, atom_size = atom.shape
    padded = F.pad(signal, (0, atom_size))
    fm = F.conv1d(padded, atom.view(n_atoms, 1, atom_size))[..., :n_samples]
    return fm

def fft_convolve(signal, atoms, approx=None):
    batch = signal.shape[0]
    n_samples = signal.shape[-1]
    n_atoms, atom_size = atoms.shape
    diff = n_samples - atom_size
    half_width = atom_size // 2

    signal = F.pad(signal, (0, atom_size))
    padded = F.pad(atoms, (0, signal.shape[-1] - atom_size))

    sig = torch.fft.rfft(signal, dim=-1)
    atom = torch.fft.rfft(torch.flip(padded, dims=(-1,)), dim=-1)[None, ...]

    if isinstance(approx, slice):
        slce = approx
        fm_spec = torch.zeros(
            batch, n_atoms, sig.shape[-1], device=signal.device, dtype=sig.dtype)
        app = sig[..., slce] * atom[..., slce]
        fm_spec[..., slce] = app
    elif isinstance(approx, int) and approx < n_samples:
        fm_spec = torch.zeros(
            batch, n_atoms, sig.shape[-1], device=signal.device, dtype=sig.dtype)

        # choose local peaks
        mags = torch.abs(sig)
        # window_size = atom_size // 64 + 1
        # padding = window_size // 2
        # avg = F.avg_pool1d(mags, window_size, 1, padding=padding)
        # mags = mags / avg

        values, indices = torch.topk(mags, k=approx, dim=-1)
        sig = torch.gather(sig, dim=-1, index=indices)

        # TODO: How can I use broadcasting rules to avoid this copy?
        atom = torch.gather(atom.repeat(batch, 1, 1), dim=-1, index=indices)
        sparse = sig * atom
        fm_spec = torch.scatter(fm_spec, dim=-1, index=indices, src=sparse)
    else:
        fm_spec = sig * atom

    fm = torch.fft.irfft(fm_spec, dim=-1)
    fm = torch.roll(fm, 1, dims=(-1,))
    return fm[..., :n_samples]

def correct_fft_convolve(signal: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    return fft_convolve(signal, atoms)


'''
With a projection, we'll:

take the fft of the signal
flip the atoms and take the fft

translate both to real
project both to lower dimension

NOTE: To see benefits, the dictionary needs to _stay_
in the lower dimensional space during training and inference.

Only when subtracting an atom should we project back to sample space
'''


def compare_conv():
    # n_samples = 16
    # atom_size = 4
    # diff = 12
    # half_width = 2
    signal = torch.zeros(1, 1, 16).normal_(0, 1)
    atoms = torch.zeros(8, 4).normal_(0, 1)
    fm_fft = fft_convolve(signal, atoms)
    fm_torch = torch_conv(signal, atoms)
    return fm_fft, fm_torch