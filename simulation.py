import torch
from torch.nn import functional as F
from torch import nn
from modules.decompose import fft_resample


# @torch.jit.script
def torch_spring_mesh(
        node_positions: torch.Tensor,
        masses: torch.Tensor,
        tensions: torch.Tensor,
        damping: float,
        n_samples: int,
        constrained_mask: torch.Tensor,
        forces: torch.Tensor,
        interpolate: int = 1
) -> torch.Tensor:
    """
    forces is (n_samples, n_nodes, dim) representing any outside forces applied to each
    node at each timestep
    """

    recording_index = 32

    n_masses = masses.shape[0]

    if not torch.all(tensions == tensions.T):
        raise ValueError('tensions must be a symmetric matrix')

    # orig_positions = node_positions.clone()

    connectivity_mask: torch.Tensor = tensions > 0

    # compute vectors representing the resting states of the springs
    resting = node_positions[None, :] - node_positions[:, None]

    # initialize a vector to hold recorded samples from the simulation
    recording: torch.Tensor = torch.zeros(n_samples, device=node_positions.device)

    # first derivative of node displacement
    velocities = torch.zeros_like(node_positions)

    # accelerations = torch.zeros_like(node_positions)

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
        x = ((-resting + current) * upper_tensions).sum(dim=0)
        a = x / m

        # update m2
        x = ((resting - current) * lower_tensions).sum(dim=0)
        b = x / m

        accelerations = accelerations + a + b

        # update velocities and apply damping
        velocities = velocities + accelerations

        # update positions for nodes that are not constrained/fixed
        node_positions = node_positions + (velocities * constrained_mask[..., None])

        recording[t] = velocities[recording_index, 0]

        # clear all the accumulated forces
        velocities = velocities * damping

    if interpolate > 1:
        recording = F.interpolate(recording.view(1, 1, -1), scale_factor=interpolate, mode='linear')
        # recording = fft_resample(recording, desired_size=n_samples * interpolate, is_lowest_band=True)
        recording = recording.view(-1)

    return recording


class Model(nn.Module):

    def __init__(self, n_nodes: int, node_dim: int, ):
        super().__init__()

    def forward(self) -> torch.Tensor:
        pass

if __name__ == '__main__':
    pass