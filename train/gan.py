import torch
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from itertools import cycle

gan_cycle = cycle(['gen', 'disc'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_latent(batch_size, dim):
    return torch.FloatTensor(batch_size, dim).normal_(0, 1).to(device)


def train_gen(batch, gen, disc, gen_optim, get_latent, loss=least_squares_generator_loss):
    gen_optim.zero_grad()
    latent = get_latent()
    recon = gen.forward(latent)
    j = disc.forward(recon)
    # loss = torch.abs(1 - j).mean()
    loss = least_squares_generator_loss(j)
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return recon

def train_disc(batch, disc, gen, disc_optim, get_latent, loss=least_squares_disc_loss):
    disc_optim.zero_grad()
    latent = get_latent()
    recon = gen.forward(latent)
    fj = disc.forward(recon)
    rj = disc.forward(batch)
    loss = torch.abs(0 - fj).mean() + torch.abs(1 - rj).mean()
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())