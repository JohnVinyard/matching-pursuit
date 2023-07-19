
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.matchingpursuit import sparse_code, sparse_code_to_differentiable_key_points, sparse_feature_map
from modules.normalization import max_norm, unit_norm
from modules.overfitraw import OverfitRawAudio
from modules.pos_encode import pos_encode_feature
from modules.softmax import hard_softmax
from modules.sparse import soft_dirac
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


d_size = 256
kernel_size = 256
sparse_coding_iterations = 16


band = zounds.FrequencyBand(20, 2000)
scale = zounds.MelScale(band, d_size)

d = morlet_filter_bank(exp.samplerate, kernel_size, scale, 0.1, normalize=True).real
d = torch.from_numpy(d).float().to(device)
d = unit_norm(d, dim=-1)


def generate(batch_size):
    total_events = batch_size * sparse_coding_iterations
    amps = torch.zeros(total_events, device=device).uniform_(0.9, 1)
    positions = torch.zeros(total_events, device=device).uniform_(0, 1)
    atom_indices = (torch.zeros(total_events).uniform_(0, 1) * d_size).long()

    output = _inner_generate(
        batch_size, total_events, amps, positions, atom_indices)
    
    output = max_norm(output)
    return output

def _inner_generate(batch_size, total_events, amps, positions, atom_indices):
    output = torch.zeros(total_events, exp.n_samples, device=device)
    for i in range(total_events):
        index = atom_indices[i]
        pos = positions[i]
        amp = amps[i]
        signal = torch.zeros(exp.n_samples, device=device)
        signal[:kernel_size] = unit_norm(d[index]) * amp
        signal = fft_shift(signal, pos)[..., :exp.n_samples]
        output[i] = signal

    output = output.view(batch_size, sparse_coding_iterations, exp.n_samples)
    output = torch.sum(output, dim=1, keepdim=True)
    return output



model = OverfitRawAudio((1, 1, exp.n_samples), std=0.01, normalize=False).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    pass

def sample_loss(a, b):
    """
    This is our baseline.  It works, obviously
    """
    return F.mse_loss(a, b)

def spectral_loss(a, b):
    """
    This didn't do what I was expecting.  It smears everything out
    """
    a = torch.abs(torch.fft.rfft(a, dim=-1, norm='ortho'))
    b = torch.abs(torch.fft.rfft(b, dim=-1, norm='ortho'))
    return F.mse_loss(a, b)


def extract_keypoints(a, n_steps=sparse_coding_iterations):
    points = []

    def visit(fm, atom_index, position, atom):
        hard = soft_dirac(fm.view(-1), dim=-1).view(fm.shape)

        time = (hard.sum(dim=0) @ torch.linspace(0, 1, exp.n_samples, device=device)).view(1).repeat(256).view(256)
        
        points.append(torch.cat([
            atom, 
            time
        ])[None, ...])

    fm, _, residual = sparse_code(
        a, d, n_steps=n_steps, device=a.device, flatten=True, visit_key_point=visit, return_residual=True)
    
    return torch.cat(points, dim=0), residual
    


def local_keypoint_loss(a, b):
    akp, a_res = extract_keypoints(a, n_steps=sparse_coding_iterations)
    bkp, b_res = extract_keypoints(b, n_steps=sparse_coding_iterations)
    return torch.abs(akp - bkp).sum() + torch.abs(torch.norm(a_res) - torch.norm(b_res)).sum()


def extract_sparse_feature_map(a):
    a, res = sparse_feature_map(a, d, n_steps=sparse_coding_iterations, device=device, return_residual=True)

    return torch.cat([
        a.unfold(-1, 16384, 8192).sum(dim=-1).reshape(-1),
        a.unfold(-1, 8192, 4096).sum(dim=-1).reshape(-1) * 0.5,
        a.unfold(-1, 2048, 1024).sum(dim=-1).reshape(-1) * 0.25,
        a.unfold(-1, 512, 256).sum(dim=-1).reshape(-1) * 0.125,
        a.unfold(-1, 128, 64).sum(dim=-1).reshape(-1) * 0.0625,
        a.reshape(-1) * 0.03,
    ]), res

def sparse_feature_map_loss(a, b):
    """
    This learns some random, out-of-order version of the original
    """
    a, a_res = extract_sparse_feature_map(a)
    b, b_res = extract_sparse_feature_map(b)
    return torch.abs(a - b).sum() + torch.abs(torch.norm(a_res) - torch.norm(b_res)).sum()

def atom_comparison_loss(recon, target):
    """
    This learns a noisy, but precisely-timed reconstruction
    """
    events, scatter = sparse_code(
        target, d, n_steps=sparse_coding_iterations, device=device, flatten=True)
    loss = 0
    for ai, j, p, a in events:
        start = p
        stop = min(exp.n_samples, p + kernel_size)
        size = stop - start
        r = recon[j, :, start: start + size]
        at = a[:, :size]
        loss = loss + torch.abs(r - at).sum()
    return loss


def experiment_loss(a, b):
    return sparse_feature_map_loss(a, b)
    # return local_keypoint_loss(a, b)

@readme
class DenseToSparse(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    

    def run(self):
        item = generate(1)
        self.real = item

        i = 0
        while True:
            optim.zero_grad()
            recon = model.forward(None)
            self.fake = recon
            loss = experiment_loss(recon, item)
            loss.backward()
            optim.step()
            print(i, loss.item())

            self.after_training_iteration(loss)
            i += 1