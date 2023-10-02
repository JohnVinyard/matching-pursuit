import torch


def sparsify2(x: torch.Tensor, n_to_keep: int = 8):
    """
    input: (batch, channels, time)

    outputs:
        sparse:  (batch, channels, time)
        packed:  (batch, n_to_keep, time)
        one_hot: (batch, n_to_keep, channels)
    
        
    - one_hot can be used to generate events
    - packed can be used to convolve generated events with 
      original activations
    """
    batch, channels, time = x.shape
    # orig = x

    x = x.view(batch, -1)
    values, indices = torch.topk(x, k=n_to_keep, dim=-1)

    ch = indices // time
    t = indices % time

    

    packed_range = torch.arange(0, n_to_keep, step=1, device=x.device)
    packed_indices = (packed_range[None, :] * time) + t

    context_indices = (packed_range[None, :] * channels) + ch

    sparse = torch.zeros_like(x, device=x.device)
    sparse = torch.scatter(sparse, dim=-1, index=indices, src=values)
    sparse = sparse.view(batch, channels, time)

    context = torch.zeros(batch, n_to_keep * channels, device=x.device)
    context = torch.scatter(context, dim=-1, index=context_indices, src=values)
    context = context.view(batch, n_to_keep, channels)
    
    packed = torch.zeros(batch, n_to_keep * time, device=x.device)
    packed = torch.scatter(packed, dim=-1, index=packed_indices, src=values)
    packed = packed.view(batch, n_to_keep, time)

    return sparse, packed, context



if __name__ == '__main__':
    t = torch.zeros(2, 8, 4).uniform_(-1, 1)
    s, p, c = sparsify2(t, n_to_keep=3)
    print(s)
    print(c.shape)
    print(c)

    # print(16 * 17)
    # nz = (p > 0).sum()
    # print(nz)