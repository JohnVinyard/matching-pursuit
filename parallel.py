from dataclasses import dataclass
from io import BytesIO
from subprocess import Popen, PIPE
from typing import Iterable, Tuple, Callable, List

from conjure import SupportedContentType, NumpySerializer, NumpyDeserializer, IdentitySerializer, IdentityDeserializer
from torch import nn

import torch
from matplotlib import pyplot as plt
import numpy as np
from soundfile import SoundFile
from torch.nn import functional as F

import conjure
from conjure.logger import encode_audio
from torch.optim import Optimizer
from config.dotenv import Config
from modules import sparsify, interpolate_last_axis, max_norm
from modules.normalization import unit_norm
from modules.reverb import NeuralReverb
from modules.transfer import fft_convolve
from modules.upsample import ensure_last_axis_length
from util import device, count_parameters
from util.overfit import overfit_model

from matplotlib import pyplot as plt

Solution = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

@dataclass
class HyperParameters:
    n_samples: int
    n_frames: int
    base_resonance: float
    resonance_diff: float
    mass_coeff: float


@dataclass
class StaticTensors:
    """
    Tensors that are static and do not require any gradients throughout
    the entire experiment.  
    """
    time: torch.Tensor
    damping: torch.Tensor
    initial_displacement: torch.Tensor


@dataclass
class PerformanceTensors:
    """
    Tensors that represent actions from something with agency, like a human 
    performer
    """
    forces: torch.Tensor
    damp_mod: torch.Tensor
    tension_mod: torch.Tensor


@dataclass
class InstrumentDefinitionTensors:
    """
    Tensors representing the physical properties of a subset of a physical
    object
    """
    mass: torch.Tensor
    tension: torch.Tensor
    filters: torch.Tensor
    filters_mix: torch.Tensor
    force_router: torch.Tensor
    tension_router: torch.Tensor
    damping: torch.Tensor
    noise_mix: torch.Tensor
    
    def display_shapes(self):
        print(f'''
            {self.mass.shape}
            {self.tension.shape}
            {self.filters.shape}
            {self.filters_mix.shape}
            {self.force_router.shape}
            {self.tension_router.shape}
            {self.damping.shape}
            {self.noise_mix.shape}
        ''')


class ParameterGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ln1 = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        return x

class InstrumentHyperNetwork(nn.Module):
    def __init__(self, latent_dim: int, n_nodes: int, filter_size: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.router_size = self.n_nodes ** 2
        self.filter_size = filter_size
        
        self.to_masses = ParameterGenerator(latent_dim, n_nodes)
        self.to_tension = ParameterGenerator(latent_dim, n_nodes)
        self.damping = ParameterGenerator(latent_dim, n_nodes)
        
        self.filters = ParameterGenerator(latent_dim, n_nodes * filter_size)
        self.filters_mix = ParameterGenerator(latent_dim, n_nodes * 2)
        
        self.force_router = ParameterGenerator(latent_dim, self.router_size)
        self.tension_router = ParameterGenerator(latent_dim, self.router_size)
        
        self.noise_mix = ParameterGenerator(latent_dim, n_nodes * 2)
          
    
    def forward(self, latent: torch.Tensor) -> InstrumentDefinitionTensors:
        batch, latent_dim = latent.shape
        
        print(latent.shape)
        
        
        m = self.to_masses(latent).view(batch, self.n_nodes, 1)
        t = self.to_tension(latent).view(batch, self.n_nodes, 1)
        d = self.damping(latent).view(batch, self.n_nodes, 1)
        
        filt = self.filters(latent).view(batch, self.n_nodes, self.filter_size)
        mx = self.filters_mix(latent).view(batch, self.n_nodes, 2)
        
        fr = self.force_router(latent).view(batch, self.n_nodes, self.n_nodes)
        tr = self.tension_router(latent).view(batch, self.n_nodes, self.n_nodes)
        
        nm = self.noise_mix(latent).view(batch, self.n_nodes, 2)
        
        return InstrumentDefinitionTensors(
            mass=m,
            tension=t,
            damping=d,
            filters=filt,
            filters_mix=mx,
            force_router=fr,
            tension_router=tr,
            noise_mix=nm
        )
        
        

def execute_parallel_layer(
    hyper: HyperParameters,
    static: StaticTensors,
    instrument: InstrumentDefinitionTensors,
    forces: torch.Tensor,
    tension_modifier: torch.Tensor = None,
    damp_mod: torch.Tensor = None,
        tension_mod: torch.Tensor = None) -> torch.Tensor:

    forces = torch.einsum('abc,bd->adc', forces, instrument.force_router)

    d = instrument.damping.repeat(1, 1, hyper.n_frames)
    if damp_mod is not None:
        d = d + damp_mod

    # DECISION: Is it better to clamp or to use sigmoid?
    damp = hyper.base_resonance + \
        (torch.clamp(d, 0, 1) * hyper.resonance_diff)

    damp = interpolate_last_axis(damp, desired_size=hyper.n_samples)

    mass = torch.sigmoid(instrument.mass) * hyper.mass_coeff

    energy = sequential(forces / mass, d)
    energy = interpolate_last_axis(energy, desired_size=hyper.n_samples)

    noisy_energy = torch.zeros_like(energy).uniform_(-1, 1)
    noisy_energy = noisy_energy * energy
    energy = torch.stack([energy, noisy_energy], dim=-1)
    energy = torch.einsum('abcd,bd->abc', energy, instrument.noise_mix)

    tension = instrument.tension

    if tension_modifier is not None:
        tension_modifier = torch.einsum(
            'abc,bd->adc', tension_modifier, instrument.tension_router)
        tension = instrument.tension + tension_modifier

    if tension_mod is not None:
        tm = interpolate_last_axis(tension_mod, desired_size=hyper.n_samples)
        tension = tension + tm

    x = damped_harmonic_oscillator(
        energy=energy,
        time=static.time,
        mass=mass,
        damping=static.damping,
        tension=10 ** tension,
        initial_displacement=static.initial_displacement,
    )

    filt = fft_convolve(x, ensure_last_axis_length(
        unit_norm(instrument.filters), desired_size=hyper.n_samples))
    x = torch.stack([x, filt], dim=-1)

    x = torch.einsum('abcd,bd->abc', x, instrument.filters_mix)

    return x


def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()

    forward = mask
    backward = x

    y = backward + (forward - backward).detach()

    return y.sum()


def l1_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x).sum()


def stft(
        x: torch.Tensor,
        ws: int = 512,
        step: int = 256,
        pad: bool = False):
    frames = x.shape[-1] // step

    if pad:
        x = F.pad(x, (0, ws))
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')
    x = torch.abs(x)
    x = x[:, :, :frames, :]
    return x


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


@torch.jit.script
def damped_harmonic_oscillator(
        energy: torch.Tensor,
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor
) -> torch.Tensor:
    x = (damping / (2 * mass))

    omega = torch.sqrt(torch.abs(tension - (x ** 2)))

    phi = torch.atan2(
        (x * initial_displacement),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    # z = a * torch.exp(-x * time) * torch.cos(omega * time - phi)
    z = a * energy * torch.cos(omega * time - phi)
    return z


@torch.jit.script
def sequential(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(forces)
    for i in range(forces.shape[-1]):
        if i == 0:
            output[..., i] = forces[..., i]
        else:
            output[..., i] = (
                forces[..., i] + output[..., i - 1]) * damping[..., i]
    return output


def parallel_sr_independent(
        forces: torch.Tensor,
        lambda_: torch.Tensor,  # continuous-time damping rate (1/sec)
        sample_rate: float,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    forces:    (..., T)
    lambda_:   (..., T)   continuous-time damping rates
    returns:   (..., T)
    """
    dt = 1.0 / sample_rate

    # Discretize the continuous-time system
    alpha = torch.exp(-lambda_ * dt)

    # Stable computation of beta
    beta = torch.where(
        lambda_.abs() > eps,
        (1.0 - alpha) / lambda_,
        torch.full_like(lambda_, dt),  # limit as lambda -> 0
    )

    # Recurrence: o[n] = alpha[n] * o[n-1] + beta[n] * forces[n]
    b = beta * forces

    # Parallel scan formulation
    p = torch.cumprod(alpha, dim=-1)
    s = torch.cumsum(b / (p + 1e-12), dim=-1)

    return p * s


def new_parallel(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    print(forces.shape, damping.shape)
    raise NotImplementedError('')


def parallel(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    # a[i] = d[i]
    # b[i] = d[i] * f[i]
    b = damping * forces

    # p[i] = prod_{j<=i} d[j]
    p = torch.cumprod(damping, dim=-1)

    # sum_{k<=i} b[k] / p[k]
    s = torch.cumsum(b / p, dim=-1)

    # o[i] = p[i] * s[i]
    return p * s


    

class Layer(nn.Module):
    def __init__(
            self, 
            n_nodes: int, 
            n_samples: int, 
            control_rate: int, 
            filter_size: int):
        
        super().__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate
        self.filter_size = filter_size
        
        # static hyperparameter stuff
        self.mass_coeff = 1
        d = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('d', d)

        _id = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('_id', _id)
        
        t = torch.linspace(0, 10, self.n_samples)
        self.register_buffer('t', t)
        

        self.mass = nn.Parameter(torch.zeros(
            1, self.n_nodes, 1).uniform_(-6, 6))

        # TODO: eventually, tension and damping will vary at control rate and will have "momentum",
        # springing back to a baseline value, just as the N-dim nodes do
        self.tension = nn.Parameter(torch.zeros(
            1, self.n_nodes, 1).uniform_(4, 9))
        
        
        self.damp = nn.Parameter(torch.zeros(
            1, self.n_nodes, 1).uniform_(1e-12, 0.9999))


        self.filt = nn.Parameter(torch.zeros(
            1, self.n_nodes, self.filter_size).uniform_(-0.01, 0.01))
        self.filt_mix = nn.Parameter(torch.zeros(
            self.n_nodes, 2).uniform_(-0.01, 0.01))

        # DECISION:  Should there be separate routing matrices for force and tension, or just one?

        self.force_router = nn.Parameter(torch.zeros(
            self.n_nodes, self.n_nodes).uniform_(-0.01, 0.01) + torch.eye(self.n_nodes, self.n_nodes))
        self.tension_router = nn.Parameter(torch.zeros(
            self.n_nodes, self.n_nodes).uniform_(-0.01, 0.01) + torch.eye(self.n_nodes, self.n_nodes))

        self.noise_mix = nn.Parameter(
            torch.zeros(self.n_nodes, 2).uniform_(-1, 1))

        self.base_resonance = 0.02
        self.max_resonance = 0.999
        self.diff = (1 - self.base_resonance)

    def total_mass_cost(self):
        return (torch.sigmoid(self.mass) * self.mass_coeff).sum()

    def total_tension_cost(self):
        return torch.abs(self.tension).sum()

    def total_damp_cost(self):
        damp = self.base_resonance + (torch.sigmoid(self.damp) * self.diff)
        return damp.sum()

    def forward(
            self,
            forces: torch.Tensor,
            tension_modifier: torch.Tensor = None,
            damp_mod: torch.Tensor = None,
            tension_mod: torch.Tensor = None) -> torch.Tensor:

        hyper = HyperParameters(
            n_samples=self.n_samples,
            n_frames=self.n_frames,
            base_resonance=self.base_resonance,
            resonance_diff=self.diff,
            mass_coeff=self.mass_coeff
        )

        static = StaticTensors(
            time=self.t,
            damping=self.d,
            initial_displacement=self._id
        )

        instrument = InstrumentDefinitionTensors(
            mass=self.mass,
            tension=self.tension,
            filters=self.filt,
            filters_mix=self.filt_mix,
            force_router=self.force_router,
            tension_router=self.tension_router,
            damping=self.damp,
            noise_mix=self.noise_mix
        )
        

        x = execute_parallel_layer(
            hyper=hyper,
            static=static,
            instrument=instrument,
            forces=forces,
            tension_modifier=tension_modifier,
            damp_mod=damp_mod,
            tension_mod=tension_mod,
        )
        return x





class LayerController(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_nodes: int,
            n_samples: int,
            control_rate: int,
            n_to_keep: int = 1024,
            filter_size: int = 32,
            latent_dim: int = 64):

        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate
        self.n_to_keep = n_to_keep
        self.filter_size = filter_size
        self.latent_dim = latent_dim
        
        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate=22050, n_samples=n_samples)
        self.room_mix = nn.Parameter(torch.zeros(
            1, self.verb.n_rooms).uniform_(-1, 1))
        self.wet_dry_mix = nn.Parameter(torch.zeros(2).uniform_(-0.01, 0.01))
        
        self.latents = nn.Parameter(torch.zeros(1, latent_dim).uniform_(-1, 1))
        self.hypernetwork = InstrumentHyperNetwork(self.latent_dim, self.n_nodes, filter_size=self.filter_size)

        self.forces = nn.Parameter(torch.zeros(
            1, n_nodes, self.n_frames).uniform_(-0.01, 0.01))
        self.damp_mod = nn.Parameter(torch.zeros(
            1, n_nodes, self.n_frames).uniform_(-1, 1))
        self.tension_mod = nn.Parameter(torch.zeros(
            1, n_nodes, self.n_frames).uniform_(-1, 1))

        self.layers: Iterable[Layer] = nn.ModuleList(
            [Layer(n_nodes, n_samples, control_rate, filter_size) for _ in range(self.n_layers)])
        
        self.mix = nn.Parameter(torch.zeros(
            self.n_layers).uniform_(-0.01, 0.01))

    @property
    def total_params(self):
        return \
            64 + 16 + 16 \
            + sum([count_parameters(x) for x in self.layers]) \
            + self.room_mix.numel() \
            + self.wet_dry_mix.numel() \
            + self.mix.numel()

    def total_mass_cost(self):
        return sum([l.total_mass_cost() for l in self.layers])

    def total_tension_cost(self):
        return sum([l.total_tension_cost() for l in self.layers])

    def total_damp_cost(self):
        return sum([l.total_damp_cost() for l in self.layers])

    def materialize_damping_mod(self):
        dm = self.damp_mod * 0.001
        dm = sparsify(dm, n_to_keep=16, salience=torch.abs(dm))
        return dm

    def materialize_tension_mod(self):
        tm = self.tension_mod * 0.001
        tm = sparsify(tm, n_to_keep=16, salience=torch.abs(tm))
        return tm

    def materialize_forces(
            self,
            forces: torch.Tensor = None,
            do_upsample: bool = True,
            n_to_keep: int = None) -> torch.Tensor:

        if forces is not None:
            f = forces
        else:
            f = self.forces

        f = torch.abs(f)
        f = f - f.mean()
        f = torch.relu(f)

        f = sparsify(f, n_to_keep=n_to_keep or self.n_to_keep)
        return f

    def compression_ratio(self):
        # For each sparse event, we need amplitude, time, and control_plane dimension

        params_per_event = 3
        params = count_parameters(self.layers) + \
            (self.n_to_keep * params_per_event)

        return params / self.n_samples

    def random_forward(self, sum_output: bool = True, n_to_keep: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            f = torch.zeros_like(self.forces).uniform_(
                0, self.forces.max().item()) ** 2
            f = sparsify(f, n_to_keep=n_to_keep)
        except:
            f = torch.zeros_like(self.forces).uniform_(0, 1)

        return self.forward(sum_output=sum_output, forces=f, n_to_keep=n_to_keep)

    def forward(
            self,
            sum_output: bool = True,
            forces: torch.Tensor = None,
            n_to_keep: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        params = self.hypernetwork.forward(self.latents)
        params.display_shapes()
        

        if forces is not None:
            sparse_forces = self.materialize_forces(
                forces, n_to_keep=n_to_keep)
        else:
            sparse_forces = self.materialize_forces(
                self.forces, n_to_keep=n_to_keep)

        dm = self.materialize_damping_mod()
        tm = self.materialize_tension_mod()

        tm = None
        outputs = []
        for i, layer in enumerate(self.layers):
            tm = layer.forward(
                forces=sparse_forces,
                tension_modifier=tm,
                damp_mod=dm,
                tension_mod=tm
            )
            outputs.append(tm)

        # DECISION: should all outputs be stacked and mixed, or only the last layer, exclusively?
        # CONCLUSION: Stacking leads to bad results with too many high frequencies
        # tm = torch.stack(outputs, dim=-1)
        # tm = tm @ self.mix

        # DECISION: Does the conv reverb cause issues because it's outside the energy calculation?

        # apply conv reverb
        wet = self.verb.forward(tm, self.room_mix)

        # wet/dry reverb mix
        x = torch.stack([tm, wet], dim=-1)
        x = x * self.wet_dry_mix
        tm = torch.sum(x, dim=-1)

        if sum_output:
            tm = torch.sum(tm, dim=1, keepdim=True)

        return tm, sparse_forces




def loss_func(a: torch.Tensor, b: torch.Tensor, control: torch.Tensor, model: LayerController) -> torch.Tensor:
    
    a = stft(a, 2048, 256, pad=False)
    b = stft(b, 2048, 256, pad=False)
    
    # means = torch.mean(a, dim=-1, keepdim=True)
    # stds = torch.std(a, dim=-1, keepdim=True)
    
    # a = (a - means) / (stds + 1e-12)
    # b = (b - means) / (stds + 1e-12)
    
    base_loss = torch.abs(a - b).sum()
    return base_loss


def overfit_osc(n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):

    controller = LayerController(
        n_layers=n_layers,
        n_nodes=n_nodes,
        n_samples=n_samples,
        control_rate=512,
        n_to_keep=n_to_keep,
        latent_dim=32
    ).to(device)

    def logger_factory(collection):

        frce, = conjure.loggers(
            ['forces'],
            SupportedContentType.Spectrogram.value,
            lambda x: x.data.cpu().numpy()[0],
            collection,
            NumpySerializer(),
            NumpyDeserializer()
        )

        rnd,  = conjure.loggers(
            ['random'],
            SupportedContentType.Audio.value,
            encode_audio,
            collection,
            IdentitySerializer(),
            IdentityDeserializer())

        return [frce, rnd]

    def training_loop_hook(
            iteration: int,
            loggers: List[conjure.Conjure],
            model: LayerController,
            optim: Optimizer):

        # params_count = count_parameters(model)
        print('N PARAMS', model.total_params)

        if iteration == 5000:
            print('Changing learning rate')
            for g in optim.param_groups:
                g['lr'] = 1e-4

        # TODO: index loggers by name
        rnd_logger = loggers[-1]
        frce_logger = loggers[-2]

        with torch.no_grad():
            rnd, _ = model.random_forward(n_to_keep=4)
            rnd_logger(max_norm(rnd))
            f = model.materialize_forces(do_upsample=False)
            f = f / (f.max() + 1e-12)
            frce_logger(f)

    def model_eval(model: LayerController, _, target: torch.Tensor):
        recon, control_signal = model.forward()
        # TODO: Should we normalize model output amplitude?
        loss = loss_func(target, recon, control_signal, model)
        return recon, loss

    overfit_model(
        n_samples=n_samples,
        model=controller,
        loss_func=loss_func,
        model_eval=model_eval,
        collection_name='parallel',
        logger_factory=logger_factory,
        training_loop_hook=training_loop_hook,
        learning_rate=1e-3,
        port=9998
    )


def display_osc(n_nodes: int, n_samples: int):
    x = test_osc(n_nodes, n_samples)
    print(x.max().item())
    x = x / x.max()

    spec = stft(x.view(1, 1, -1))
    spec = spec.view(-1, spec.shape[-1])
    spec = spec.data.cpu().numpy()

    arr = x.data.cpu().numpy().reshape((-1,))
    plt.plot(arr)
    plt.show()

    plt.matshow(spec)
    plt.show()

    listen_to_sound(arr, wait_for_user_input=True)


def compare():
    n_samples = 2 ** 15
    frame_size = 1
    n_frames = n_samples // frame_size

    decay = 0.9
    sr_dependent = torch.zeros(n_frames).fill_(decay)
    forces = torch.zeros(n_frames).bernoulli_(p=1e-3) * \
        torch.zeros(n_frames).uniform_(0, 1)

    a = parallel(forces, sr_dependent)
    # a = parallel_sr_independent(forces, sr_dependent, 1)

    # a = interpolate_last_axis(a, n_samples)
    # f = upsample_with_holes(forces, n_samples)

    # print('NaN', torch.isnan(b).sum().item(), 'inf', torch.isinf(b).sum().item(), b.numel())
    print(torch.isnan(a).sum() / n_samples)
    # b = interpolate_last_axis(b, n_samples)

    plt.plot(a)
    plt.plot(forces)
    # plt.plot(b)
    plt.show()


def harness(n_samples: int, *solutions: Solution):
    f, d = generate_params(n_samples)
    for solution in solutions:
        x = solution(f, d)
        plt.plot(x.data.cpu().numpy())

    plt.show()
    plt.show()


if __name__ == '__main__':
    # compare()

    # DECISION:  How many layers are appropriate/sufficient?
    overfit_osc(n_nodes=32, n_samples=2 ** 17, n_layers=2, n_to_keep=64)
