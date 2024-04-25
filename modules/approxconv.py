from torch.jit import ScriptModule, script_method
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
import numpy as np




class ApproximateConvolution(ScriptModule):
    def __init__(self):
        super().__init__()

    @script_method
    def forward(self, a, b, percent_sparse):
        n_samples = a.shape[-1]

        a = F.pad(a, (0, a.shape[-1]))
        b = F.pad(b, (0, b.shape[-1]))

        n_coeffs = ((a.shape[-1] // 2) + 1)
        n_elements = int(n_coeffs * percent_sparse)

        a_spec = torch.fft.rfft(a, dim=-1, norm='ortho')[..., :n_elements]
        b_spec = torch.fft.rfft(b, dim=-1, norm='ortho')[..., :n_elements]

        x = a_spec * b_spec

        x = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], n_coeffs -
                      n_elements, dtype=x.dtype, device=x.device)], dim=-1)

        x = torch.fft.irfft(x, dim=-1, norm='ortho')[..., :n_samples]

        return x
