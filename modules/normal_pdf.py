import torch
import numpy as np
from torch.distributions import Normal, Gamma


def pdf(x, mean, sd, epsilon=1e-8):
    var = (sd ** 2) + epsilon
    denom = (2 * np.pi * var) ** .5
    num = torch.exp(-(x-mean) ** 2 / (2*var))
    return num / denom


def  pdf2(means, stds, n_elements, normalize=True):
    """
    Probability distribution for a normal distribution
    """
    dist = Normal(means[..., None], stds[..., None])
    prob = dist.log_prob(torch.linspace(0, 1, n_elements, device=means.device).view(*([1] * len(stds.shape)), n_elements))
    prob = torch.exp(prob)
    
    if normalize:
        prob = prob / (prob.max(dim=-1, keepdim=True)[0] + 1e-8)
        
    return prob


def gamma_pdf(
        shape: torch.Tensor,
        rate: torch.Tensor,
        n_elements: int,
        normalize: bool = True):

    """
    Probability density function for a gamma distribution
    """
    dist = Gamma(shape[..., None], rate[..., None])
    # KLUDGE: This range is arbitrarily based on the wikipedia graph here:
    # https://en.wikipedia.org/wiki/Gamma_distribution#/media/File:Gamma_distribution_pdf.svg
    prob = dist.log_prob(
        torch.linspace(1e-12, 20, n_elements, device=shape.device).view(*([1] * len(rate.shape)), n_elements))
    prob = torch.exp(prob)
    
    if normalize:
        prob = prob / (prob.max(dim=-1, keepdim=True)[0] + 1e-8)
    
    return prob
                         