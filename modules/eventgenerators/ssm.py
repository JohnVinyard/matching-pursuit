from modules import HyperNetworkLayer
from modules.atoms import unit_norm
from modules.eventgenerators.generator import EventGenerator, ShapeSpec
import torch
from torch import nn

from modules.eventgenerators.schedule import DiracScheduler
from modules.iterative import TensorTransform
from modules.overlap_add import overlap_add


def project_and_limit_norm(
        vector: torch.Tensor,
        forward: TensorTransform,
        max_efficiency: float = 0.999) -> torch.Tensor:

    # get the original norm, this is the absolute max norm/energy we should arrive at,
    # given a perfectly efficient physical system
    original_norm = torch.norm(vector, dim=-1, keepdim=True)

    # project
    x = forward(vector)
    return x

    # TODO: clamp norm should be a utility that lives in normalization
    # find the norm of the projection
    new_norm = torch.norm(x, dim=-1, keepdim=True)

    # clamp the norm between the allowed values
    mx_value = original_norm.reshape(*new_norm.shape) * max_efficiency
    clamped_norm = torch.clamp(new_norm, min=None, max=mx_value)

    # give the projected vector the clamped norm, such that it
    # can have lost some or all energy, but not _gained_ any
    normalized = unit_norm(x, axis=-1)
    x = normalized * clamped_norm
    return x


def state_space_model(
        control: torch.Tensor,
        proj_matrix: TensorTransform,
        state_matrix: TensorTransform,
        input_matrix: TensorTransform,
        output_matrix: TensorTransform,
        direct_matrix: TensorTransform,
        state_matrix_dim: int,
        input_dim: int,
        n_samples: int):

    batch, cpd, frames = control.shape
    control = control.permute(0, 2, 1)

    # project the control signal into the input dimension
    proj = project_and_limit_norm(control, proj_matrix)

    results = []

    state_vec = torch.zeros(
        batch,
        state_matrix_dim,
        device=control.device)

    for i in range(frames):
        inp = proj[:, i, :]

        assert inp.shape == (batch, input_dim)

        state_vec = project_and_limit_norm(state_vec, state_matrix)

        b = project_and_limit_norm(inp, input_matrix)

        c = project_and_limit_norm(state_vec, output_matrix)

        d = project_and_limit_norm(inp, direct_matrix)

        state_vec = state_vec + b
        output = c + d

        results.append(output.view(batch, 1, input_dim))

    result = torch.cat(results, dim=1)
    result = result[:, None, :, :]

    result = overlap_add(result, apply_window=True)
    return result[..., :n_samples]


class StateSpaceModelEventGenerator(EventGenerator, nn.Module):

    def __init__(
            self,
            context_dim: int,
            control_plane_dim: int,
            input_dim: int,
            state_dim: int,
            hypernetwork_dim: int,
            hypernetwork_latent: int,
            n_samples: int,
            samplerate: int,
            n_frames: int):
        super().__init__()

        self.hypernetwork_latent = hypernetwork_latent
        self.context_dim = context_dim
        self.control_plane_dim = control_plane_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hypernetwork_dim = hypernetwork_dim
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.n_frames = n_frames

        self.proj_matrix_hypernetwork = HyperNetworkLayer(
            hypernetwork_dim, hypernetwork_latent, context_dim, input_dim)

        self.state_matrix_hypernetwork = HyperNetworkLayer(
            hypernetwork_dim, hypernetwork_latent, state_dim, state_dim)

        self.input_matrix_hypernetwork = HyperNetworkLayer(
            hypernetwork_dim, hypernetwork_latent, input_dim, state_dim)

        self.output_matrix_hypernetwork = HyperNetworkLayer(
            hypernetwork_dim, hypernetwork_latent, state_dim, input_dim)

        self.direct_matrix_hypernetwork = HyperNetworkLayer(
            hypernetwork_dim, hypernetwork_latent, input_dim, input_dim)

        self.scheduler = DiracScheduler(1, n_frames, n_samples)

    @property
    def shape_spec(self) -> ShapeSpec:
        return dict(
            control_signal=(self.control_plane_dim, self.n_frames),
            state_matrix_hypervector=(1, self.hypernetwork_dim,),
            output_matrix_hypervector=(1, self.hypernetwork_dim,),
            input_matrix_hypervector=(1, self.hypernetwork_dim,),
            direct_matrix_hypervector=(1, self.hypernetwork_dim,),
            proj_matrix_hypervector=(1, self.hypernetwork_dim,)
        )

    def forward(
            self,
            control_signal: torch.Tensor,
            state_matrix_hypervector: torch.Tensor,
            output_matrix_hypervector: torch.Tensor,
            input_matrix_hypervector: torch.Tensor,
            direct_matrix_hypervector: torch.Tensor,
            proj_matrix_hypervector: torch.Tensor,
            times: torch.Tensor) -> torch.Tensor:

        # control_signal = torch.relu(control_signal)
        control_signal = control_signal ** 2

        state_matrix, state_matrix_forward = self.state_matrix_hypernetwork(state_matrix_hypervector)
        _, input_matrix_forward = self.input_matrix_hypernetwork(input_matrix_hypervector)
        _, direct_matrix_forward = self.direct_matrix_hypernetwork(direct_matrix_hypervector)
        _, output_matrix_forward = self.output_matrix_hypernetwork(output_matrix_hypervector)
        _, proj_matrix_forward = self.proj_matrix_hypernetwork(proj_matrix_hypervector)


        final = state_space_model(
            control_signal.view(-1, self.context_dim, self.n_frames),
            proj_matrix_forward,
            state_matrix_forward,
            input_matrix_forward,
            output_matrix_forward,
            direct_matrix_forward,
            state_matrix_dim=self.state_dim,
            input_dim=self.input_dim,
            n_samples=self.n_samples)

        final = self.scheduler.schedule(times, final)

        return final

