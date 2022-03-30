from scipy.misc import face
import zounds
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np


def sampling_exp():

    # original image, or wavetable
    n_pixels = 8
    pallette = np.random.normal(0, 1, (n_pixels,))
    pixels = np.linspace(0, 1, n_pixels)

    # sampling grid
    n_grid_points = 16
    sampling_pos = np.linspace(0, 1, n_grid_points, endpoint=False)

    # build the sampling kernel
    kernel = np.clip(
        1 - np.abs(sampling_pos[None, :] - pixels[:, None]), 0, np.inf)
    kernel[kernel < 0.9] = 0

    # choose the indices at which we'll sample from the grid
    indices = np.arange(0, n_grid_points, 1)[::-1]

    sampled = pallette @ kernel[:, indices]

    return kernel, pallette, sampled


def to_hard_indices(soft, size):
    indices = torch.clamp(soft, -0.999, 0.999).view(-1)
    hard_indices = (((indices + 1) / 2) * size).long()
    return hard_indices

def sampling_kernel(size, device):
    # build the sampling kernel
    x = torch.linspace(0, 1, size).to(device)
    kernel = torch.clamp(1 - torch.abs(x[None, :] - x[:, None]), 0, np.inf)
    # how many neighors should we take into account when sampling?
    # kernel[kernel < 0.9] = 0
    return kernel

class DifferentiableIndex(torch.autograd.Function):

    def forward(self, pallette, indices):
        """
        indices => pallette @ sampling_kernel[indices] => values => ||target - values||
        """
        orig_shape = indices.shape
        p = pallette.view(-1)
        p_size = p.shape[0]
        hard_indices = to_hard_indices(indices, p_size)
        sampled = pallette[hard_indices]

        self.save_for_backward(pallette, indices, hard_indices, sampled)

        return sampled.reshape(*orig_shape)

    def backward(self, *grad_outputs):
        
        grad_output, = grad_outputs
        pallette, indices, hard_indices, sampled = self.saved_tensors

        # get neighboring indices (only look at the nearest neighbor)
        left = torch.clamp(hard_indices - 1, 0, pallette.shape[0] - 1)
        right = torch.clamp(hard_indices + 1, 0, pallette.shape[0] - 1)

        left_samples = pallette[left]
        right_samples = pallette[right]

        error = grad_output.view(-1)

        left_grad = torch.abs(error - (sampled - left_samples) - error)
        right_grad = torch.abs(error - (sampled - right_samples))
        step = 2 / pallette.shape[0]

        grad = torch.sign(right_grad - left_grad) * step
        grad = grad.reshape(*grad_output.shape)

        return None, grad


diff_index = DifferentiableIndex.apply


class Lookup(nn.Module):
    def __init__(self, pallette, shape):
        super().__init__()
        self.register_buffer('pallette', pallette)
        self.indices = nn.Parameter(
            torch.FloatTensor(*shape).uniform_(-0.01, 0.01))
        self.shape = shape

    def forward(self, x):
        return diff_index(self.pallette, self.indices).reshape(*self.shape)


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    # load a black and white image in the range -1, 1
    img = (face(gray=True) / 128) - 1

    target = torch.from_numpy(img).float()

    # called i in the paper, the number of points
    # on a regularly-spaced sampling grid
    sampling_granularity = 100
    pallette = torch.linspace(-1, 1, sampling_granularity)

    model = Lookup(pallette, img.shape)

    optim = Adam(model.parameters(), lr=1e-4, betas=(0, 0.9))

    while True:
        optim.zero_grad()
        recon = model(None)
        loss = F.mse_loss(recon, target)
        loss.backward()
        optim.step()
        print(loss.item())
        r = recon.data.cpu().numpy()

    input('input...')
