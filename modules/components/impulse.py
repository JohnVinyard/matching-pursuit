from torch import nn
import torch

class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels, mixer_channels=2):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        mixer_channels = mixer_channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=mixer_channels, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

    def forward(self, x):
        x = self.to_mix(x)
        x = x.view(-1, self.encoding_channels, 1)
        x = torch.softmax(x, dim=-1)
        return x

class GenerateImpulse(nn.Module):

    def __init__(self, latent_dim, channels, n_samples, n_filter_bands, encoding_channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.n_frames = n_samples // 256
        self.n_filter_bands = n_filter_bands
        self.channels = channels
        self.filter_kernel_size = 16
        self.encoding_channels = encoding_channels

        self.to_frames = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            mode='learned',
            end_size=self.n_frames,
            out_channels=channels,
            # batch_norm=True
            weight_norm=True
        )

        self.noise_model = NoiseModel(
            channels,
            self.n_frames,
            self.n_frames * 4,
            self.n_samples,
            self.channels,
            # batch_norm=True,
            weight_norm=True,
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
        )
        
        self.to_env = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):
        batch_size = x.shape[0]
        
        env = self.to_env(x) ** 2
        env = F.interpolate(env, mode='linear', size=self.n_samples)
        
        x = self.to_frames(x)
        x = self.noise_model(x)
        x = x.view(batch_size, -1, self.n_samples)
        
        x = x * env
        return x



class SimpleGenerateImpulse(nn.Module):
    
    def __init__(self, latent_dim, channels, n_samples, n_filter_bands, encoding_channels):
        super().__init__()
        
        self.n_samples = n_samples
        
        self.filter_size = 64
        
        self.to_envelope = LinearOutputStack(
            channels, layers=3, out_channels=self.n_samples // 128, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        
        self.to_filt = LinearOutputStack(
            channels, layers=3, out_channels=self.filter_size, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        env = self.to_envelope(x)
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        
        # TODO: consider making this a hard choice via gumbel softmax as well
        env = torch.abs(env).view(x.shape[0], -1, self.n_samples)
        
        filt = self.to_filt(x).view(x.shape[0], -1, self.filter_size)
        
        noise = torch.zeros(x.shape[0], 1, self.n_samples, device=x.device).uniform_(-1, 1)
        
        noise = noise * env
        
        filt = F.pad(filt, (0, self.n_samples - self.filter_size))
        
        final = fft_convolve(noise, filt)
        return final
