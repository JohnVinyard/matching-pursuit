import torch

def covariance(x):
    m = x.mean(dim=0, keepdim=True)
    x = x - m
    cov = torch.matmul(x.T, x.clone().detach()) * (1 / x.shape[1])
    return cov

def latent_loss(enc, mean_weight=1, std_weight=1):
    mean_loss = torch.abs(0 - enc.mean(dim=0)).mean()
    std_loss = torch.abs(1 - enc.std(dim=0)).mean()
    cov = covariance(enc)
    d = torch.sqrt(torch.diag(cov))
    cov = cov / d[None, :]
    cov = cov / d[:, None]
    cov = torch.abs(cov)
    cov = cov.mean()
    return (mean_loss * mean_weight) + (std_loss * std_weight) + cov