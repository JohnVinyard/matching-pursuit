import torch
from torch.nn import functional as F


def dtw_loss(input, target, window_size=2, step=2):

    inp = input.unfold(0, window_size, step).permute(0, 2, 1).contiguous()
    targ = target.unfold(0, window_size, step).permute(0, 2, 1).contiguous()

    dist = torch.cdist(inp, targ)

    indices = torch.argmin(dist, dim=1)

    # KLUDGE: It should be possible to do this with indexing
    # alone, i.e., without a loop
    aligned = []
    for i, index in enumerate(indices):
        aligned.append(targ[i, index][None, ...])
    aligned = torch.cat(aligned, dim=0)

    loss = F.mse_loss(inp, aligned)
    return loss


if __name__ == '__main__':

    x = torch.FloatTensor(16, 7).normal_(0, 1)
    # y = torch.FloatTensor(100, 12).normal_(0, 1)

    dtw_loss(x, x)
