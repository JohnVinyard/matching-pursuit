from unittest2 import TestCase
import torch

from modules.anticausal import AntiCausalAnalysis
from modules.auditory import STFT
from modules.iterative import iterative_loss
from modules.quantize import select_items
from modules.transfer import hierarchical_dirac
from modules.upsample import interpolate_last_axis, upsample_with_holes
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
    
    def test_can_interpolate_1d(self):
        signal = torch.zeros(128)
        upsampled = interpolate_last_axis(signal, 1024)
        self.assertEqual((1024,), upsampled.shape)
    
    def test_can_interpolate_2d(self):
        signal = torch.zeros(2, 128)
        upsampled = interpolate_last_axis(signal, 1024)
        self.assertEqual((2, 1024), upsampled.shape)
    
    def test_can_interpolate_3d(self):
        signal = torch.zeros(3, 2, 128)
        upsampled = interpolate_last_axis(signal, 1024)
        self.assertEqual((3, 2, 1024), upsampled.shape)
        
    def test_can_interpolate_4d(self):
        signal = torch.zeros(5, 3, 2, 128)
        upsampled = interpolate_last_axis(signal, 1024)
        self.assertEqual((5, 3, 2, 1024), upsampled.shape)
    
    def test_can_select_single_item(self):
        selection = torch.zeros(3).uniform_(-1, 1)
        items = torch.zeros(3, 16).uniform_(-1, 1)
        selected = select_items(selection, items)
        self.assertEqual((16,), selected.shape)
    
    def test_can_select_2d(self):
        selection = torch.zeros(21, 3).uniform_(-1, 1)
        items = torch.zeros(3, 16).uniform_(-1, 1)
        selected = select_items(selection, items)
        self.assertEqual((21, 16), selected.shape)
    
    def test_can_select_3d(self):
        selection = torch.zeros(18, 21, 3).uniform_(-1, 1)
        items = torch.zeros(3, 16).uniform_(-1, 1)
        selected = select_items(selection, items)
        self.assertEqual((18, 21, 16), selected.shape)
    
    def test_can_select_4d(self):
        selection = torch.zeros(13, 18, 21, 3).uniform_(-1, 1)
        items = torch.zeros(3, 16).uniform_(-1, 1)
        selected = select_items(selection, items)
        self.assertEqual((13, 18, 21, 16), selected.shape)
    
    def test_can_produce_hierarchical_dirac_1d(self):
        desired_size = 1024
        n_elements = int(np.log2(desired_size))
        pos = torch.zeros(n_elements, 2).uniform_(-1, 1)
        x = hierarchical_dirac(pos)
        self.assertEqual((1024,), x.shape)
    
    def test_can_produce_hierarchical_dirac_2d(self):
        desired_size = 1024
        n_elements = int(np.log2(desired_size))
        pos = torch.zeros(3, n_elements, 2).uniform_(-1, 1)
        x = hierarchical_dirac(pos)
        self.assertEqual((3, 1024,), x.shape)
    
    def test_can_produce_hierarchical_dirac_3d(self):
        desired_size = 1024
        n_elements = int(np.log2(desired_size))
        pos = torch.zeros(5, 3, n_elements, 2).uniform_(-1, 1)
        x = hierarchical_dirac(pos)
        self.assertEqual((5, 3, 1024,), x.shape)
        
    def test_can_produce_hierarchical_dirac_4d(self):
        desired_size = 1024
        n_elements = int(np.log2(desired_size))
        pos = torch.zeros(4, 5, 3, n_elements, 2).uniform_(-1, 1)
        x = hierarchical_dirac(pos)
        self.assertEqual((4, 5, 3, 1024,), x.shape)