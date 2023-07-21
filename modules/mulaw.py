import torch

def mu_law(x: torch.Tensor, mu=255) -> torch.Tensor:
    s = torch.sign(x)
    x = torch.abs(x)
    z = torch.zeros(1, device=x.device).fill_(1 + mu)
    return s * (torch.log(1 + (mu * x)) / torch.log(z))


def inverse_mu_law(x: torch.Tensor, mu=255) -> torch.Tensor:
    s = torch.sign(x)
    x = torch.abs(x)
    z = torch.zeros(1, device=x.device).fill_(1 + mu)
    x *= torch.log(z)
    x = (torch.exp(x) - 1) / mu
    return x * s