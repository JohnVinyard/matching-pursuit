import torch


def sparsify(x, n_to_keep):
    orig_shape = x.shape
    x = x.view(x.shape[0], -1)
    values, indices = torch.topk(x, n_to_keep, dim=-1)
    out = torch.zeros_like(x)
    out = torch.scatter(out, dim=-1, index=indices, src=values)
    out = out.view(*orig_shape)
    return out
