from unittest2 import TestCase
import torch

from modules.components.auditory import STFTTransform

class AuditoryTests(TestCase):
    
    def test_raises_for_wrong_shape(self):
        self.fail()
    
    def test_stft_returns_expected_shape(self):
        samples = torch.zeros(4, 1, 2**15)
        transform = STFTTransform(window_size=2048, step_size=256)
        spec = transform.forward(samples)
        self.assertEqual((4, 1025, 128), spec.shape)
    
    def test_stft_returns_expected_shape_with_slice(self):
        self.fail()