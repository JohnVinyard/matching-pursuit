import torch
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from itertools import cycle

gan_cycle = cycle(['gen', 'disc'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_latent(batch_size, dim):
    return torch.zeros((batch_size, dim), device=device).normal_(0, 1)


def train_gen(batch, gen, disc, gen_optim, get_latent, loss=least_squares_generator_loss):
    gen_optim.zero_grad()
    latent = get_latent()
    recon = gen.forward(latent)
    j = disc.forward(recon)
    l = loss(j)
    l.backward()
    gen_optim.step()
    print('G', l.item())
    return recon

def train_disc(batch, disc, gen, disc_optim, get_latent, loss=least_squares_disc_loss):
    disc_optim.zero_grad()
    latent = get_latent()
    recon = gen.forward(latent)
    fj = disc.forward(recon)
    rj = disc.forward(batch)
    l = loss(rj, fj)
    l.backward()
    disc_optim.step()
    print('D', l.item())