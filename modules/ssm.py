from torch import nn
import torch
import numpy as np

from modules import max_norm, unit_norm
from modules.overlap_add import overlap_add
from ssmcompression import max_efficiency

max_efficiency = 0.999

def project_and_limit_norm(
        vector: torch.Tensor,
        matrix: torch.Tensor,
        max_efficiency: float = max_efficiency) -> torch.Tensor:
    # get the original norm, this is the absolute max norm/energy we should arrive at,
    # given a perfectly efficient physical system
    # original_norm = torch.norm(vector, dim=-1, keepdim=True)
    # project
    x = vector @ matrix
    return x

    # TODO: clamp norm should be a utility that lives in normalization
    # find the norm of the projection
    new_norm = torch.norm(x, dim=-1, keepdim=True)
    # clamp the norm between the allowed values
    clamped_norm = torch.clamp(new_norm, min=None, max=original_norm * max_efficiency)

    # give the projected vector the clamped norm, such that it
    # can have lost some or all energy, but not _gained_ any
    normalized = unit_norm(x, axis=-1)
    x = normalized * clamped_norm
    return x


class SSM(nn.Module):
    """
    A state-space model-like module, with one additional matrix, used to project the control
    signal into the shape of each audio frame.

    The final output is produced by overlap-adding the windows/frames of audio into a single
    1D signal.
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int, windowed: bool = True):
        super().__init__()
        self.state_matrix_dim = state_matrix_dim
        self.input_dim = input_dim
        self.control_plane_dim = control_plane_dim
        self.windowed = windowed

        # matrix mapping control signal to audio frame dimension
        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        )

        # state matrix mapping previous state vector to next state vector
        self.state_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        # matrix mapping audio frame to hidden/state vector dimension
        self.input_matrix = nn.Parameter(
            torch.zeros(input_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        # matrix mapping hidden/state vector to audio frame dimension
        self.output_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, input_dim).uniform_(-0.01, 0.01)
        )

        # skip-connection-like matrix mapping input audio frame to next
        # output audio frame
        self.direct_matrix = nn.Parameter(
            torch.zeros(input_dim, input_dim).uniform_(-0.01, 0.01)
        )

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)

        proj = control @ self.proj
        assert proj.shape == (batch, frames, self.input_dim)

        results = []
        state_vec = torch.zeros(batch, self.state_matrix_dim, device=control.device)

        for i in range(frames):
            inp = proj[:, i, :]

            state_vec = project_and_limit_norm(state_vec, self.state_matrix)
            b = project_and_limit_norm(inp, self.input_matrix)
            c = project_and_limit_norm(state_vec, self.output_matrix)
            d = project_and_limit_norm(inp, self.direct_matrix)

            state_vec = state_vec + b
            output = c + d

            # state_vec = (state_vec @ self.state_matrix) + (inp @ self.input_matrix)
            # output = (state_vec @ self.output_matrix) + (inp @ self.direct_matrix)

            # state_vec = state_vec + b
            # output = c + d
            # state_vec = (state_vec @ self.state_matrix) + (inp @ self.input_matrix)
            # output = (state_vec @ self.output_matrix) + (inp @ self.direct_matrix)
            results.append(output.view(batch, 1, self.input_dim))


        result = torch.cat(results, dim=1)
        result = result[:, None, :, :]

        result = overlap_add(result, apply_window=self.windowed)
        return result[..., :frames * (self.input_dim // 2)]



class OverfitControlPlane(nn.Module):
    """
    Encapsulates parameters for control signal and state-space model
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int, n_samples: int, windowed: bool = True):
        super().__init__()
        self.ssm = SSM(control_plane_dim, input_dim, state_matrix_dim, windowed=windowed)
        self.n_samples = n_samples
        self.n_frames = int(n_samples / (input_dim // 2))

        self.control = nn.Parameter(
            torch.zeros(1, control_plane_dim, self.n_frames).uniform_(-0.01, 0.01))

    @property
    def control_signal_display(self) -> np.ndarray:
        return self.control_signal.data.cpu().numpy().reshape((-1, self.n_frames))

    @property
    def control_signal(self) -> torch.Tensor:
        return torch.relu(self.control)

    def random(self, p=0.001):
        """
        Produces a random, sparse control signal, emulating short, transient bursts
        of energy into the system modelled by the `SSM`
        """
        cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=p)
        audio = self.forward(sig=cp)
        return max_norm(audio)

    def forward(self, sig=None):
        """
        Inject energy defined by `sig` (or by the `control` parameters encapsulated by this class)
        into the system modelled by `SSM`
        """
        return self.ssm.forward(sig if sig is not None else self.control_signal)