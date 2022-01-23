import torch

# def pdf(domain, mean, std):
#     twopi = torch.FloatTensor([2 * torch.pi]).to(domain.device)
#     return (1 / (std * torch.sqrt(twopi))) * (torch.e ** (0.5 * (((domain - mean) / std) ** 2)))


def pdf(x, mean, sd):
    var = sd ** 2
    denom = (2* torch.pi * var) ** .5
    num = torch.exp(-(x-mean) ** 2/ (2*var))
    return num / denom
