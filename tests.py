from unittest import TestCase
import torch
from modules.conv import correct_fft_convolve, torch_conv

class SmokeTests(TestCase):
    
    def test_check_conv(self):
        signal = torch.zeros(3, 1, 1024).uniform_(-1, 1)
        atoms = torch.zeros(16, 1024).uniform_(-1, 1)
        
        fft_result = correct_fft_convolve(signal, atoms)
        torch_result = torch_conv(signal, atoms)
        
        torch.testing.assert_allclose(fft_result, torch_result, atol=1e-4, rtol=1e-3)