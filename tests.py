from unittest import TestCase, skip
import torch
from modules.conv import correct_fft_convolve, torch_conv
from modules.pointcloud import GraphEdgeEmbedding, flattened_upper_triangular, pairwise_differences

@skip('These are no longer needed')
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
        
        # element-wise different between every pair of point embeddings
        self.assertEqual(diff.shape, (batch, dim, n_points, n_points))
    
    def test_check_upper_triangular(self):
        batch, n_points, dim = 8, 16, 16
        
        features = torch.zeros(batch, n_points, dim).uniform_(-1, 1)
        diff = pairwise_differences(features)
        
        # Note: this is just the same assertion as above
        self.assertEqual(diff.shape, (batch, dim, n_points, n_points))
        
        ut = flattened_upper_triangular(diff)
        expected_dim = (n_points * (n_points - 1)) / 2
        self.assertEqual(ut.shape, (batch, dim, expected_dim))
    
    def test_check_graph_edge_embedding(self):
        batch, n_points, dim = 8, 16, 16
        
        features = torch.zeros(batch, n_points, dim).uniform_(-1, 1)
        
        embedding_dim = 32
        gee = GraphEdgeEmbedding(n_points, dim, out_channels=embedding_dim)
        
        result = gee.forward(features)
        
        self.assertEqual(result.shape, (batch, embedding_dim))