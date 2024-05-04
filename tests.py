from unittest import TestCase
import torch
from modules.conv import correct_fft_convolve, torch_conv
from modules.pointcloud import flattened_upper_triangular, pairwise_differences

class SmokeTests(TestCase):
    
    def test_check_conv(self):
        signal = torch.zeros(3, 1, 1024).uniform_(-1, 1)
        atoms = torch.zeros(16, 1024).uniform_(-1, 1)
        
        fft_result = correct_fft_convolve(signal, atoms)
        torch_result = torch_conv(signal, atoms)
        
        torch.testing.assert_allclose(fft_result, torch_result, atol=1e-4, rtol=1e-3)
    
    def test_check_pairwise_differences(self):
        batch, n_points, dim = 8, 128, 16
        
        features = torch.zeros(batch, n_points, dim).uniform_(-1, 1)
        diff = pairwise_differences(features)
        
        self.assertEqual(diff.shape, (batch, dim, n_points, n_points))
    
    def test_check_upper_triangular(self):
        batch, n_points, dim = 8, 16, 16
        
        features = torch.zeros(batch, n_points, dim).uniform_(-1, 1)
        diff = pairwise_differences(features)
        ut = flattened_upper_triangular(diff)
        expected_dim = (n_points * (n_points - 1)) / 2
        self.assertEqual(ut.shape, (batch, dim, expected_dim))
        self.fail()