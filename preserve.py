from io import BytesIO
from subprocess import PIPE, Popen
from typing import Iterable, Tuple

import conjure
import torch
from torch import nn
from torch.optim import Adam
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm
from modules.upsample import ensure_last_axis_length, upsample_with_holes
from util import device
from matplotlib import pyplot as plt
from torch.nn import functional as F
import numpy as np
from soundfile import SoundFile

from util.playable import encode_audio
from util.weight_init import make_initializer


initializer = make_initializer(0.01)


# TODO: It might be nice to move this into zounds
def listen_to_sound(
        samples: np.ndarray,
        wait_for_user_input: bool = True) -> None:

    bio = BytesIO()
    with SoundFile(bio, mode='w', samplerate=22050, channels=1, format='WAV', subtype='PCM_16') as sf:
        sf.write(samples.astype(np.float32))

    bio.seek(0)
    data = bio.read()

    proc = Popen(f'aplay', shell=True, stdin=PIPE)

    if proc.stdin is not None:
        proc.stdin.write(data)
        proc.communicate()

    if wait_for_user_input:
        input('Next')



class NonLinearity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 0.2)

class Layer(nn.Module):
    
    def __init__(self, n_channels: int, frame_size: int, n_samples: int, hidden_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.frame_size = frame_size
        self.n_samples = n_samples
        self.hidden_channels = hidden_channels
        
        self.n_frames = n_samples // self.frame_size
        
        self.time_embedding = nn.Parameter(torch.zeros(1, hidden_channels, self.n_frames).uniform_(-0.01, 0.01))
        
        self.embed_damping = nn.Sequential(
            nn.Conv1d(n_channels, hidden_channels, 1, 1, 0),
            NonLinearity(),
            nn.Conv1d(hidden_channels, hidden_channels, 1, 1, 0),
            NonLinearity(),
            
        )
        
        self.embed_control = nn.Sequential(
            nn.Conv1d(n_channels, hidden_channels, 1, 1, 0),
            NonLinearity(),
            nn.Conv1d(hidden_channels, hidden_channels, 1, 1, 0),
            NonLinearity(),
        )
        
        self.embed_all = nn.Conv1d(hidden_channels, 1, 1, 1, 0)
        
        self.time_step_input = nn.Linear(self.n_frames, hidden_channels)
        
        self.audio_output = LinearOutputStack(
            channels=hidden_channels,
            layers=3,
            out_channels=self.frame_size,
            activation=NonLinearity(),
            shortcut=True
        )
        
        self.apply(initializer)
        
    
    def forward(
            self, 
            control_plane: torch.Tensor, 
            damping: torch.Tensor) -> torch.Tensor:
        
        # TODO: first, compress into a single time dimension
        embedded_control = self.embed_control(control_plane)
        embedded_damping = self.embed_damping(damping)
        
        x = embedded_control + embedded_damping
        x = self.embed_all(x)
        
        x = self.time_step_input(x)
        x = x.permute(0, 2, 1)
        
        x = x * self.time_embedding
        
        x = x.permute(0, 2, 1)
        x = self.audio_output(x)
        
        x = x.view(-1, 1, self.n_samples)
        
        x = x * torch.zeros_like(x).uniform_(-1, 1)
        
        return x
    


# @torch.jit.script
def with_damping(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    """
    Apply damping/decay to acquire an envelope
    """
    
    forces = torch.abs(forces)
    
    output = torch.zeros_like(forces)
    for i in range(forces.shape[-1]):
        if i == 0:
            output[..., i] = forces[..., i]
        else:
            output[..., i] = (forces[..., i] + output[..., i - 1]) * damping[..., i]
    return output


def envelope(signal: torch.Tensor, frame_size: int) -> torch.Tensor:
    """
    Compute a lower-frequency enelope from a higher-sample-rate signal 
    """
    
    windowed = signal.unfold(-1, size=frame_size, step=frame_size)
    avg = torch.mean(torch.abs(windowed), dim=-1)
    return avg

def damping_loss(control: torch.Tensor, output: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    """
    Compare output energy at each timestep to input energy plus the influence of damping
    """
    input_with_damping = with_damping(control, damping)
    
    output_with_damping = output
    return torch.abs(input_with_damping - output_with_damping).sum()

def energy_loss(control: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """
    Compare total output enegy to total input energy
    """
    input_energy = torch.sum(control, dim=(1, 2), keepdim=False)
    output_energy = torch.sum(output, dim=(1, 2), keepdim=False)
    return torch.abs(input_energy - output_energy).sum()


def produce_batch(
    batch_size: int, 
    frame_size: int, 
    n_channels: int, 
    n_samples: int, 
    device: torch.device = device) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    
    n_frames = n_samples // frame_size
    
    while True:
        control = torch.zeros(batch_size, n_channels, n_frames, device=device).bernoulli_(p=0.001)
        control = control * torch.zeros_like(control).uniform_(0, 1)
        
        damping = torch.zeros(batch_size, n_channels, 1, device=device).uniform_(0.9, 0.9998)
        damping = damping.repeat(1, 1, n_frames)
        
        damped = with_damping(control, damping)
        
        yield (control, damping, damped)
        


if __name__ == '__main__':
    n_samples = 2 ** 16
    frame_size = 128
    n_channels = 16
    hidden_channels = 512
    
    batch_size = 8
    
    model = Layer(n_channels, frame_size, n_samples, hidden_channels=hidden_channels).to(device)
    # model = AlternateLayer(n_channels, frame_size, n_samples, hidden_channels).to(device)

    optim = Adam(model.parameters(), lr=1e-3)
    
    collection = conjure.LmdbCollection(path='preserve')

    r, = conjure.loggers(
        ['recon' ],
        'audio/wav',
        encode_audio,
        collection)


    conjure.serve_conjure(
        [r],
        port=9999,
        n_workers=1,
        web_components_version='0.0.101')
    
    for i, (forces, damping, expected_envelope) in enumerate(produce_batch(
            batch_size, 
            frame_size, 
            n_channels, 
            n_samples, 
            device=device)):
        
        optim.zero_grad()
        
        # print(forces.shape, forces.mean())
        
        summed_expected_envelope = torch.sum(expected_envelope, dim=1, keepdim=True)
        
        embedded = model.forward(forces, damping)
        
        r(max_norm(embedded[0].view(-1)))
        
        actual_envelope = envelope(embedded, frame_size)
        
        # if i % 1000 == 0 and i > 0:
        #     plt.plot(summed_expected_envelope.view(-1).data.cpu().numpy())
        #     plt.plot(actual_envelope.view(-1).data.cpu().numpy())
        #     plt.show()
        #     listen_to_sound(
        #         torch.sum(embedded, dim=1, keepdim=True)[0, ...].view(-1).data.cpu().numpy(), 
        #         wait_for_user_input=True
        #     )
        
        loss = torch.abs(summed_expected_envelope - actual_envelope).sum()
        
        loss.backward()
        optim.step()
    
        print(i, loss.item())        
    
    