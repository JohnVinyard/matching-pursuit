from unittest2 import TestCase
import torch

from modules.anticausal import AntiCausalAnalysis
from modules.auditory import STFT
from modules.iterative import iterative_loss
from modules.upsample import upsample_with_holes
import numpy as np

class ModuleTests(TestCase):
    
    def test_stft_single_channel(self):
        transform = STFT(2048, 256)
        audio = torch.zeros(8, 1, 2**15)
        spec = transform.forward(audio)
        self.assertEqual((8, 1, 1025, 128), spec.shape)
    
    def test_stft_multi_channel(self):
        transform = STFT(2048, 256)
        audio = torch.zeros(8, 3, 2**15)
        spec = transform.forward(audio)
        self.assertEqual((8, 3, 1025, 128), spec.shape)
    
    def test_upsample_with_holes(self):
        x = torch.ones(3, 6, 128)
        us = upsample_with_holes(low_sr=x, desired_size=2**15)
        self.assertEqual((3, 6, 2**15), us.shape)
        total = torch.sum(us[0, 0, :]).item()
        self.assertEqual(128, total)
    
    def test_iterative_loss_smoke_test_with_stft(self):
        target = torch.zeros(3, 1, 2**15)
        recon = torch.zeros(3, 16, 2**15)
        transform = STFT(2048, 256)
        
        transform_shape = transform.forward(target).shape
        flattened = np.product(transform_shape[1:])
        
        residual, loss = iterative_loss(
            target_audio=target,
            recon_channels=recon,
            transform=transform.forward,
            return_residual=True)
        
        self.assertEqual((3, flattened), residual.shape)
    
    def test_can_analyze_stft(self):
        batch = 3
        audio = torch.zeros(batch, 1, 2**15)
        stft = STFT(2048, 256)
        analysis = AntiCausalAnalysis(stft.n_coeffs, 256, 2, dilations=[1, 2, 4, 1])
        spec = stft.forward(audio)
        spec = spec.view(batch, stft.n_coeffs, -1)
        x = analysis.forward(spec)
        self.assertEqual((batch, 256, spec.shape[-1]), x.shape)
    
    def test_can_analyze_pif(self):
        self.fail()
        
        