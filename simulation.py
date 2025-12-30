from typing import Tuple
import torch
from torch import nn

from modules import stft, unit_norm, flattened_multiband_spectrogram, interpolate_last_axis
from modules.decompose import fft_resample
from modules.overlap_add import overlap_add
from modules.transfer import fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util.overfit import overfit_model


def ensure_symmetric(x: torch.Tensor) -> None:
    if not torch.all(x == x.T):
        raise ValueError('tensions must be a symmetric matrix')


def sparse_forces(shape: Tuple, probability: float):
    sparse = torch.zeros(shape).bernoulli_(p=probability)
    rand = torch.zeros(shape).uniform_(-0.0001, 0.0001)
    return sparse * rand


@torch.jit.script
def _torch_spring_mesh(
        node_positions: torch.Tensor,
        masses: torch.Tensor,
        tensions: torch.Tensor,
        damping: float,
        n_samples: int,
        mixer: torch.Tensor,
        constrained_mask: torch.Tensor,
        forces: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    forces is (n_samples, n_nodes, dim) representing any outside forces applied to each
    node at each timestep
    """

    n_masses = masses.shape[0]

    if not torch.all(tensions == tensions.T):
        raise ValueError('tensions must be a symmetric matrix')

    connectivity_mask: torch.Tensor = tensions > 0

    # compute vectors representing the resting states of the springs
    resting = node_positions[None, :] - node_positions[:, None]

    # initialize a vector to hold recorded samples from the simulation
    recording: torch.Tensor = torch.zeros(n_samples, device=node_positions.device)
    node_forces: torch.Tensor = torch.zeros(n_samples, n_masses, device=node_positions.device)

    # first derivative of node displacement
    velocities = torch.zeros_like(node_positions)

    ones = torch.ones(n_masses, n_masses, 1, device=node_positions.device)
    upper_mask = torch.triu(ones)
    lower_mask = torch.tril(ones)

    z = tensions[..., None] * connectivity_mask[..., None]

    upper_tensions = z * upper_mask
    lower_tensions = z * lower_mask

    m = masses[..., None]

    for t in range(n_samples):
        # Note that we're overwriting accelerations rather than modifying it
        accelerations = forces[t]

        current = node_positions[None, :] - node_positions[:, None]

        # update m1
        x = torch.einsum('ikj,ikj->kj', -resting + current, upper_tensions)
        a = x / m

        # update m2
        x = torch.einsum('ikj,ikj->kj', resting - current, lower_tensions)
        b = x / m

        accelerations = accelerations + a + b

        # update velocities and apply damping
        velocities = velocities + accelerations

        # update positions for nodes that are not constrained/fixed
        node_positions = node_positions + (velocities * constrained_mask[..., None])

        f = m * accelerations
        node_forces[t] = torch.sum(f, dim=-1)

        mixed = mixer @ f[:, 0]

        # recording[t] = velocities[32, 0]
        recording[t] = mixed

        # clear all the accumulated forces
        velocities = velocities * damping

    return recording, node_forces


def torch_spring_mesh(
        node_positions: torch.Tensor,
        masses: torch.Tensor,
        tensions: torch.Tensor,
        damping: float,
        n_samples: int,
        mixer: torch.Tensor,
        constrained_mask: torch.Tensor,
        forces: torch.Tensor,
        interpolate: int = 1):
    recording, node_forces = _torch_spring_mesh(
        node_positions, masses, tensions, damping, n_samples, mixer, constrained_mask, forces)

    if interpolate > 1:
        recording = fft_resample(recording.view(1, 1, -1), desired_size=n_samples * interpolate, is_lowest_band=True)
        recording = recording.view(-1)

    return recording.view(1, 1, -1), node_forces


class Model(nn.Module):

    def __init__(
            self,
            n_nodes: int,
            node_dim: int,
            control_frame_rate: int,
            n_samples: int,
            n_filters: int = 64,
            filter_latent_dim: int = 8,
            filter_size: int = 2048):
        super().__init__()

        self.filter_size = filter_size
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.control_frame_rate = control_frame_rate
        self.n_samples = n_samples
        self.n_frames = n_samples // control_frame_rate
        self.window_size = control_frame_rate * 2
        self.n_filters = n_filters
        self.filter_latent_dim = filter_latent_dim

        self.n_coeffs = self.window_size // 2 + 1

        n_filters_coeffs = self.filter_size // 2 + 1

        self.ratio = int((self.filter_size / self.n_samples) * n_filters_coeffs)

        self.latents = nn.Parameter(torch.zeros(n_nodes, filter_latent_dim).uniform_(-1, 1))
        self.filter_map = nn.Parameter(torch.zeros(filter_latent_dim, n_filters).uniform_(-1, 1))
        self.filters = nn.Parameter(torch.zeros(n_filters, self.filter_size).uniform_(-1, 1))
        self.high_freq_factor = nn.Parameter(torch.zeros(n_nodes, 1).uniform_(-0.00001, 0.00001))

        self.nodes = nn.Parameter(torch.zeros(n_nodes, node_dim).uniform_(-1, 1))
        self.masses = nn.Parameter(torch.zeros(n_nodes, ).uniform_(15, 18))
        # TODO: There should be time-varying deformations applied to the tension
        self.tensions = nn.Parameter(torch.zeros(n_nodes, n_nodes).uniform_(10, 11))

        self.damping = 0.98

        # How much does each mass contribute to the final recording?
        self.mixer = nn.Parameter(torch.zeros(n_nodes, ).uniform_(-0.1, 0.1))

        forces = sparse_forces((self.n_frames // 16, self.n_nodes, self.node_dim), probability=0.001)

        self.forces = nn.Parameter(forces)

        self.constrained_mask = nn.Parameter(torch.zeros(n_nodes, ).bernoulli_(0.1))

        # self.gains = nn.Parameter(torch.zeros(n_nodes, 1).uniform_(-1, 1))

    @property
    def force_norm(self):
        return torch.norm(self.forces.view(-1, self.node_dim), dim=-1, p=1).sum()

    @property
    def constrained(self):
        fwd = (self.constrained_mask > 0).float()
        back = self.constrained_mask
        y = back + (fwd - back).detach()
        return y

    @property
    def symmetric_tensions(self) -> torch.Tensor:
        upper = torch.triu(self.tensions, diagonal=1)
        symmetric = upper + upper.T
        return symmetric

    @property
    def interpolated_forces(self):
        # permute to (nodes, node_dim, time)
        x = upsample_with_holes(self.forces.permute(1, 2, 0), self.n_frames)
        x = x.permute(2, 0, 1)
        return x

    def _forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, node_forces = torch_spring_mesh(
            node_positions=self.nodes,
            masses=torch.abs(self.masses) * 10000 + 1e-8,
            tensions=torch.abs(self.symmetric_tensions) * 4 + 1e-8,
            damping=self.damping,
            n_samples=self.n_frames,
            mixer=torch.softmax(self.mixer, dim=-1),
            constrained_mask=self.constrained,
            forces=self.interpolated_forces,
            interpolate=self.control_frame_rate
        )
        return x, node_forces

    def forward(self) -> torch.Tensor:
        x, node_forces = self._forward()

        # envelopes = interpolate_last_axis(node_forces.T, desired_size=self.n_samples)
        #
        # # map the latents for each node to a linear combination of filteres
        # filter_choice = self.filter_map.T @ self.latents.T
        # filters = (self.filters.T @ filter_choice).T
        # filters = ensure_last_axis_length(filters, desired_size=self.n_samples)
        # filters = unit_norm(filters)
        # filters = torch.fft.rfft(filters, dim=-1)
        #
        # # TODO: the cutoff should be all frequencies below the
        # # control/simulation rate
        # filters[..., :self.ratio] = 0
        # filters = torch.fft.irfft(filters, dim=-1)
        #
        # noise = torch.zeros_like(envelopes).uniform_(-0.01, 0.01) * envelopes
        #
        # conv = fft_convolve(filters, noise)
        #
        # conv = conv * self.high_freq_factor.view(self.n_nodes, 1)
        # conv = conv.view(1, self.n_nodes, self.n_samples)
        #
        # # conv = torch.tanh(conv * self.gains)
        #
        # conv = torch.sum(conv, dim=1, keepdim=True)


        # x = x + (conv * 0)


        return x


def compute_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    t = stft(target, 2048, 256, pad=True)
    r = stft(recon, 2048, 256, pad=True)
    return torch.abs(t - r).sum()


if __name__ == '__main__':
    n_samples = 2 ** 15

    model = Model(
        n_nodes=512,
        node_dim=2,
        control_frame_rate=32,
        n_filters=128,
        filter_latent_dim=8,
        n_samples=n_samples
    )


    def full_loss_func(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        recon_loss = compute_loss(target, recon)
        energy_loss = model.force_norm
        # encourage sparse energy
        return recon_loss + (energy_loss * 100)


    overfit_model(
        n_samples=n_samples,
        model=model,
        loss_func=full_loss_func,
        collection_name='simulation',
        learning_rate=1e-2
    )
