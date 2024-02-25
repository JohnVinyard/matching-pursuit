from typing import List, Tuple
from torch import nn
import torch
from modules.decompose import fft_frequency_decompose
from modules.normalization import unit_norm
from modules.pos_encode import two_d_pos_encode
from torch.nn import functional as F
from dataclasses import dataclass
from modules.softmax import sparse_softmax
from modules.stft import stft
import numpy as np

from util.weight_init import make_initializer

@dataclass
class PatchBandSpec:
    band_size: int
    stft_window_size: int
    stft_step_size: int
    patch_size: Tuple[int]
    patch_step: Tuple[int]
    n_centroids: int


def patches2(spec: torch.Tensor, size: Tuple[int], step: Tuple[int]):
    batch, channels, time = spec.shape
    
    w, h = size
    ws, hs = step
    
    # patch_size = w * h
    final_size = (w // 2 + 1) * h
    
    p = spec.unfold(1, w, ws).unfold(2, h, hs)
    p = torch.abs(torch.fft.rfft2(p, dim=(-1, -2)))
    p = p.reshape(batch, -1, final_size)
    norms = torch.norm(p, dim=-1, keepdim=True)
    normed = p / (norms + 1e-12)
    return p, norms, normed



def patches(spec: torch.Tensor, size: int = 16, step: int = 8):
    batch, channels, time = spec.shape
    p = spec.unfold(1, size, step).unfold(2, size, step)
    p = torch.abs(torch.fft.rfft2(p, dim=(-1, -2)))
    last_dim = (size // 2 + 1) * size
    p = p.reshape(batch, -1, last_dim)
    norms = torch.norm(p, dim=-1, keepdim=True)
    normed = p / (norms + 1e-12)
    return p, norms, normed


class MultiWindowSpectralInfoLoss(nn.Module):
    def __init__(self, specs):
        super().__init__()
        models = []
        for size, step in specs:
            model = SpectralInfoLoss(2048, 256, patch_size=size, patch_step=step, n_centroids=256)
            models.append(model)
        
        self.models = nn.ModuleList(models)
    
    def loss(self, target: torch.Tensor, recon: torch.Tensor):
        losses = [model.loss(target, recon) for model in self.models]
        loss = sum(losses)
        return loss
    
    def encode(self, signal: torch.Tensor):
        results = [model.encode(signal) for model in self.models]
        return results
    
    def forward(self, signal: torch.Tensor):
        batch_size = signal.shape[0]
        results = [model.forward(signal) for model in self.models]
        recon = torch.cat([x[0].view(batch_size, -1) for x in results], dim=-1)
        target = torch.cat([x[1].view(batch_size, -1) for x in results], dim=-1)
        return recon, target

class MultiBandSpectralInfoLoss(nn.Module):
    def __init__(self, specs: List[PatchBandSpec]):
        super().__init__()
        self.specs = specs
        self.models = nn.ModuleDict({
            str(spec.band_size): SpectralInfoLoss(
                stft_window_size=spec.stft_window_size,
                stft_step_size=spec.stft_step_size,
                patch_size=spec.patch_size,
                patch_step=spec.patch_step,
                n_centroids=spec.n_centroids
            ) for spec in specs}
        )
        
    def loss(self, target: torch.Tensor, recon: torch.Tensor):
        target_bands = fft_frequency_decompose(target, 512)
        recon_bands = fft_frequency_decompose(recon, 512)
        losses = {k: model.loss(target_bands[int(k)], recon_bands[int(k)]) for k, model in self.models.items()}
        loss = sum(losses.values())
        return loss
        
    
    def encode(self, signal: torch.Tensor):
        bands = fft_frequency_decompose(signal, 512)
        results = {k: model.encode(bands[int(k)]) for k, model in self.models.items()}
        return results
    
    def forward(self, signal: torch.Tensor):
        batch_size = signal.shape[0]
        
        bands = fft_frequency_decompose(signal, 512)
        results = {k: model.forward(bands[int(k)]) for k, model in self.models.items()}
        
        recon = torch.cat([x[0].view(batch_size, -1) for x in results.values()], dim=-1)
        target = torch.cat([x[1].view(batch_size, -1) for x in results.values()], dim=-1)

        return recon, target



class SpectralInfoLoss(nn.Module):
    
    def __init__(
            self, 
            stft_window_size=2048, 
            stft_step_size=256, 
            patch_size=(16, 16), 
            patch_step=(8, 8), 
            embedding_channels=32,
            pos_encoding_channels=34,
            n_centroids=1024):
        
        super().__init__()
        self.stft_window_size = stft_window_size
        self.stft_step_size = stft_step_size
        self.start_channels = self.stft_window_size // 2 + 1
        self.pos_encoding_channels = pos_encoding_channels
        self.n_centroids = n_centroids
        
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.embedding_channels = embedding_channels
        
        
        self.full_patch_size = (patch_size[0] // 2 + 1) * patch_size[1]
        
        self.patch_embed = nn.Linear(self.full_patch_size, self.embedding_channels)
        self.proj = nn.Linear(self.embedding_channels, self.embedding_channels)
        self.up = nn.Linear(self.embedding_channels, self.n_centroids)
        self.down = nn.Linear(self.n_centroids, self.embedding_channels)
        self.recon = nn.Linear(self.embedding_channels, self.full_patch_size)
        
        init = make_initializer(0.02)
        self.apply(init)
        
        
    
    def loss(self, target: torch.Tensor, recon: torch.Tensor):
        """
        Get categories, weights and norms from target
        
        Get categories, _ and norms from recon
        
        loss = cross_entropy(fake_cat, real_cat, real_weights) + l1(fake_norms, real_norms)
        """
        toh, tc, tw, tnorms, tnormed, traw = self.encode(target)
        foh, fc, fw, fnorms, fnormed, fraw = self.encode(recon)
        
        
        cat_loss = F.cross_entropy(foh.view(-1, self.n_centroids), tc.view(-1), weight=tw)
        coarse_loss = F.mse_loss(fnorms, tnorms.detach()) * 1e-3
        
        return cat_loss + coarse_loss
    
    # def compute_total_information(self, signal: torch.Tensor):
        
    #     if signal.shape[1] != 1:
    #         frames = signal.shape[1]
    #         spec = signal.view(-1, frames, self.start_channels)
    #     else:
    #         frames = signal.shape[-1] // self.stft_step_size
            
    #         spec = stft(signal, self.stft_window_size, self.stft_step_size, pad=True)\
    #             .view(-1, frames, self.start_channels)
        
    #     raw, norms, normed = patches2(spec, size=self.patch_size, step=self.patch_step)
    #     # print(raw.shape, norms.shape, normed.shape, self.patch_embed)
    #     spec_embed = self.patch_embed(normed)
        
    #     # TODO: the embedding should only be for frequency bin;  time is shift-invariant
    #     x = spec_embed
    #     x = self.proj(x)
    #     x = self.up(x)
    #     x = sparse_softmax(x, normalize=True)
    #     one_hot = x
        
    #     codes = torch.argmax(x, dim=-1, keepdim=True).view(-1)
    #     normed = normed.view(-1)
        
    #     # batch_size = signal.shape[0]
        
    #     # group patches according to code
    #     accum = torch.zeros(1024, device=signal.device)

    #     for i, code in enumerate(codes):
    #         # this is kind of a trick.  Each norm of normed will always
    #         # be 1, so this is a differentiable way to measure the total
    #         # amount of information
    #         accum[code] = accum[code] + torch.norm(normed[i])

    #     # this is the total information present in each sample
    #     inverse = 1 / (accum + 1)
    #     total_info = inverse.sum()
    #     return total_info
        
    
    def encode(self, signal: torch.Tensor):
        
        if signal.shape[1] != 1:
            frames = signal.shape[1]
            spec = signal.view(-1, frames, self.start_channels)
        else:
            frames = signal.shape[-1] // self.stft_step_size
            
            spec = stft(signal, self.stft_window_size, self.stft_step_size, pad=True)\
                .view(-1, frames, self.start_channels)
        
        raw, norms, normed = patches2(spec, size=self.patch_size, step=self.patch_step)
        # print(raw.shape, norms.shape, normed.shape, self.patch_embed)
        spec_embed = self.patch_embed(normed)
        
        # TODO: the embedding should only be for frequency bin;  time is shift-invariant
        x = spec_embed
        x = self.proj(x)
        x = self.up(x)
        x = sparse_softmax(x, normalize=True)
        one_hot = x
        
        codes = torch.argmax(x, dim=-1, keepdim=True)
        
        total_elements = codes.nelement()
        counts = torch.bincount(codes.view(-1), minlength=self.n_centroids) + 1
        weights = 1 / (counts / total_elements) # per-class weighting
        return one_hot, codes, weights, norms, normed, raw
        
    
    def forward(self, signal: torch.Tensor):
        # print(signal.shape, self.stft_window_size, self.stft_step_size, self.patch_size)
        x, codes, weights, norms, normed, raw = self.encode(signal)
        # print(list(codes[0].data.cpu().numpy().squeeze())[:128])
        x = self.down(x)
        x = self.recon(x)
        recon = unit_norm(x, dim=-1)
        normed_recon = recon * norms
        return normed_recon, normed
        
        
        
            
        
    