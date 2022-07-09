import torch


def to_hard_indices(soft, size):
    indices = torch.clamp(soft, -0.999, 0.999).view(-1)
    hard_indices = torch.round((((indices + 1) / 2) * size)).long()
    return hard_indices


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
        # TODO: make this adjustable, with the option to take more
        # neighbors into account using different windows
        left = torch.clamp(hard_indices - 1, 0, pallette.shape[0] - 1)
        right = torch.clamp(hard_indices + 1, 0, pallette.shape[0] - 1)

        # sample neighboring values
        left_samples = pallette[left]
        right_samples = pallette[right]

        error = grad_output.view(-1)

        # which direction better matches the error gradient?
        left_grad = torch.abs(error - (sampled - left_samples) - error)
        right_grad = torch.abs(error - (sampled - right_samples))

        # set grad in that direction, scaled to our [-1, 1] indexing space
        step = 2 / pallette.shape[0]
        grad = torch.sign(right_grad - left_grad) * step
        grad = grad.reshape(*grad_output.shape)

        return None, grad


diff_index = DifferentiableIndex.apply
