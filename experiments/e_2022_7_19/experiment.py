

from modules.decompose import fft_frequency_decompose
from modules.phase import stft
from upsample import FFTUpsampleBlock
from util.readmedocs import readme

import numpy as np
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
import zounds
import torch
from torch import nn
from torch.nn import functional as F

from util.weight_init import make_initializer

n_samples = 2 ** 14
samplerate = zounds.SR22050()

n_steps = 25

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.99
    return torch.linspace(beta_start, beta_end, timesteps)

betas = cosine_beta_schedule(n_steps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, pos_embeddings[t_index]) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for i in range(n_steps - 1, -1, -1):
        img = p_sample(
            model, 
            img, 
            torch.full((b,), i, device=device, dtype=torch.long), 
            i)
    
    return img

@torch.no_grad()
def sample(model):
    return p_sample_loop(model, shape=(1, 1, n_samples))


pos_embeddings = pos_encoded(1, n_steps, 16, device=device).reshape(n_steps, 33)


def forward_process(audio, n_steps):
    noise = torch.randn_like(audio)

    degraded = q_sample(audio, n_steps, noise)
    return audio, degraded, noise

    # degraded = audio
    # for i in range(n_steps):
    #     noise = torch.zeros_like(audio).normal_(means[i], stds[i]).to(device)
    #     degraded = degraded + noise
    # return audio, degraded, noise


def reverse_process(model):
    # degraded = torch.zeros(
    #     1, 1, n_samples).normal_(0, 1.6).to(device)
    
    # for i in range(n_steps - 1, -1, -1):
    #     pred_noise = model.forward(degraded, pos_embeddings[:1, i, :])
    #     degraded = degraded - pred_noise

    # return degraded
    return sample(model)


init_weights = make_initializer(0.1)


def activation(x):
    return F.leaky_relu(x, 0.2)


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return activation(x)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = activation(x)
        return x

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, 1, 3)
        self.context = nn.Sequential(
            DilatedBlock(out_channels, 1),
            DilatedBlock(out_channels, 3),
        )
        self.conv2 = nn.Conv1d(out_channels + 33, out_channels, 7, 4, 3)
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = None
        

    def forward(self, x, step):
        x = self.conv1(x)
        x = activation(x)

        x = self.context(x)
        
        step = step.view(x.shape[0], 33, 1).repeat(1, 1, x.shape[-1])
        x = torch.cat([x, step], dim=1)
        x = self.conv2(x)
        x = activation(x)
        # x = F.avg_pool1d(x, 7, 4, 3)

        if self.norm is None:
            self.norm = nn.LayerNorm(x.shape[1:]).to(x.device)
        x = self.norm(x)

        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels + 33, in_channels, 7, 1, 3)
        self.context = nn.Sequential(
            DilatedBlock(in_channels, 1),
            DilatedBlock(in_channels, 3),
        )
        self.up = nn.ConvTranspose1d(in_channels, in_channels, 12, 4, 4)
        # self.norm = nn.BatchNorm1d(in_channels)
        self.norm = None

        self.conv2 = nn.Conv1d(in_channels, out_channels, 7, 1, 3)


    def forward(self, x, d, step):

        x = x + d

        step = step.view(x.shape[0], 33, 1).repeat(1, 1, x.shape[-1])
        x = torch.cat([x, step], dim=1)
        x = self.conv1(x)
        x = activation(x)

        x = self.context(x)

        # x = F.upsample(x, scale_factor=4, mode='nearest')
        x = self.up(x)
        
        x = activation(x)

        if self.norm is None:
            self.norm = nn.LayerNorm(x.shape[1:]).to(x.device)
        x = self.norm(x)

        x = self.conv2(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.Sequential(
            DownsamplingBlock(1, 16),
            DownsamplingBlock(16, 32),
            DownsamplingBlock(32, 64),
            DownsamplingBlock(64, 128),
            DownsamplingBlock(128, 256),
        )

        self.up = nn.Sequential(
            UpsamplingBlock(256, 128),
            UpsamplingBlock(128, 64),
            UpsamplingBlock(64, 32),
            UpsamplingBlock(32, 16),
            UpsamplingBlock(16, 1),
        )


        self.apply(init_weights)

    def forward(self, audio, pos_embedding):
        x = audio

        d = {}

        # initial shape assertions
        for layer in self.down:
            x = layer.forward(x, pos_embedding)
            d[x.shape[-1]] = x

        for layer in self.up:
            z = d[x.shape[-1]]
            x = layer.forward(x, z, pos_embedding)
        
        return x


gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-3)


def train_gen(batch):
    gen_optim.zero_grad()

    step = torch.randint(0, n_steps, (batch.shape[0],)).to(device)
    pos = pos_embeddings[step]

    orig, degraded, noise = forward_process(batch, step)

    pred_noise = gen.forward(degraded, pos)

    # noise_bands = fft_frequency_decompose(noise, 512)
    # pred_bands = fft_frequency_decompose(pred_noise, 512)

    # loss = 0
    # for a, b in zip(pred_bands.values(), noise_bands.values()):
    #     loss = loss + torch.abs(a - b).sum()


    loss = F.smooth_l1_loss(pred_noise, noise)
    
    loss.backward()
    gen_optim.step()
    return loss, pred_noise


@readme
class DiffusionWithUNet(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.fake = None
        self.real = None
        self.pred = None
        self.gen = gen
        self.betas = betas

    def listen(self):
        return playable(self.fake, samplerate)
    
    def predicted(self):
        return self.pred.data.cpu().numpy().squeeze()
    
    def orig(self):
        return playable(self.real, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.denoise()))

    def check_degraded(self, t=n_steps-1):
        with torch.no_grad():
            audio, degraded, noise = forward_process(
                self.real, 
                torch.zeros(self.real.shape[0]).fill_(t).long().to(device))
            return playable(degraded, samplerate)

    def denoise(self):
        with torch.no_grad():
            result = reverse_process(gen)
            return playable(result, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            gen_loss, self.pred = train_gen(item)
            print('GEN', i, gen_loss.item())