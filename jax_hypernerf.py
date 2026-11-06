"""
An audio autoencoder whose decoder is an implicit neural representation
(in the style of NeRF/SIREN), with weights produced by a hyper-network.

Encoder ("classic"): log-magnitude STFT -> 1x1 projection -> a stack of
strided (stride 2) 1D convolutions over time, halving the number of frames at
each layer until a single frame remains -> a linear layer to the latent
vector.

Decoder: rather than upsampling from the latent vector, a hyper-network maps
the latent vector to the weights and biases of an N-layer SIREN, a
per-sample MLP with sine activations, `sin(w0 * (x @ W + b))`, and a final
linear layer producing one audio sample.  The SIREN's input, for each sample
of the segment, is a *learned* positional encoding: a table of shape
(n_samples, pe_dim), shared across all examples, initialized to sines and
cosines at log-spaced frequencies (from about one cycle per segment up to
near Nyquist), and trained along with everything else.  So the positional
encodings say "where", and are the same for every segment; the generated
weights say "what", and are specific to each segment.

The generated weights are scaled to match SIREN's initialization: the
hyper-network's per-layer heads produce values of roughly unit variance
(the trunk's output is layer-normalized, the heads' weights have variance
1 / fan_in), which are then multiplied by the std of SIREN's uniform init
for that layer, so at initialization each generated network looks like a
freshly initialized SIREN.

The loss is the mean absolute error between log-mel spectrograms of the
target and the reconstruction.
"""
import argparse
import itertools
from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from conjure import LmdbCollection, SupportedContentType, NumpySerializer, NumpyDeserializer, loggers, serve_conjure
from conjure.logger import encode_audio

from hyper_rnn import batch_stream, make_log_mel_spectrogram, max_norm, stft


# ---------------------------------------------------------------------------
# encoder

class Encoder(eqx.Module):
    """
    STFT -> strided conv stack -> latent vector.
    """
    proj: eqx.nn.Conv1d
    downs: List[eqx.nn.Conv1d]
    norms: List[eqx.nn.LayerNorm]
    to_latent: eqx.nn.Linear

    window: int = eqx.field(static=True)
    hop: int = eqx.field(static=True)

    def __init__(self, n_samples: int, window: int, hop: int, channels: int, latent_dim: int, key):
        self.window = window
        self.hop = hop

        n_frames = n_samples // hop
        n_layers = int(np.log2(n_frames))
        if 2 ** n_layers != n_frames:
            raise ValueError(f'n_samples // hop ({n_frames}) must be a power of two')

        keys = jax.random.split(key, n_layers + 2)
        self.proj = eqx.nn.Conv1d(window // 2 + 1, channels, 1, key=keys[0])
        self.downs = [
            eqx.nn.Conv1d(channels, channels, 3, stride=2, padding=1, key=k)
            for k in keys[1:-1]
        ]
        # normalize over channels at each frame
        self.norms = [eqx.nn.LayerNorm(channels) for _ in range(n_layers)]
        self.to_latent = eqx.nn.Linear(channels, latent_dim, key=keys[-1])

    def __call__(self, audio: jnp.ndarray) -> jnp.ndarray:
        """
        audio: (n_samples,)
        returns: (latent_dim,)
        """
        spec = stft(audio, ws=self.window, step=self.hop, pad=True)  # (n_frames, n_coeffs)
        x = jnp.log(spec.T + 1e-4)                                     # (n_coeffs, n_frames)
        x = self.proj(x)
        for conv, norm in zip(self.downs, self.norms):
            x = conv(x)
            x = jax.vmap(norm, in_axes=1, out_axes=1)(x)
            x = jax.nn.leaky_relu(x, 0.2)
        return self.to_latent(x[:, 0])


# ---------------------------------------------------------------------------
# hyper-network and SIREN

def sinusoidal_table(n_samples: int, pe_dim: int) -> jnp.ndarray:
    """
    (n_samples, pe_dim): sines and cosines at `pe_dim // 2` log-spaced
    frequencies, from one cycle per segment up to a quarter of the sample
    rate (a period of 4 samples).
    """
    n_freqs = pe_dim // 2
    t = jnp.arange(n_samples)
    cycles_per_sample = jnp.geomspace(1. / n_samples, 0.25, n_freqs)
    angles = 2 * jnp.pi * t[:, None] * cycles_per_sample[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def siren_std(fan_in: int, w0: float, first: bool) -> float:
    """
    The std of SIREN's uniform weight initialization: U(-1/fan_in, 1/fan_in)
    for the first layer, U(-sqrt(6/fan_in)/w0, sqrt(6/fan_in)/w0) after.
    U(-a, a) has std a / sqrt(3).
    """
    bound = 1. / fan_in if first else np.sqrt(6. / fan_in) / w0
    return float(bound / np.sqrt(3.))


class LayerShape(eqx.Module):
    fan_in: int = eqx.field(static=True)
    fan_out: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)


class HyperNetwork(eqx.Module):
    """
    latent -> (W, b) for each layer of a SIREN, i.e., `n_layers` sine layers
    (pe_dim -> hidden, then hidden -> hidden) followed by a linear output
    layer (hidden -> 1).
    """
    trunk: List[eqx.nn.Linear]
    trunk_norm: eqx.nn.LayerNorm
    heads: List[eqx.nn.Linear]
    shapes: Tuple[LayerShape, ...]

    def __init__(
            self,
            latent_dim: int,
            trunk_dim: int,
            pe_dim: int,
            hidden: int,
            n_layers: int,
            w0: float,
            w0_first: float,
            key):

        trunk_key, *head_keys = jax.random.split(key, n_layers + 2)

        k1, k2 = jax.random.split(trunk_key)
        self.trunk = [
            eqx.nn.Linear(latent_dim, trunk_dim, key=k1),
            eqx.nn.Linear(trunk_dim, trunk_dim, key=k2),
        ]
        self.trunk_norm = eqx.nn.LayerNorm(trunk_dim)

        fan_ins = [pe_dim] + [hidden] * n_layers
        fan_outs = [hidden] * n_layers + [1]
        shapes = []
        for i, (fi, fo) in enumerate(zip(fan_ins, fan_outs)):
            if i == 0:
                scale = siren_std(fi, w0_first, first=True)
            else:
                scale = siren_std(fi, w0, first=False)
            shapes.append(LayerShape(fi, fo, scale))
        self.shapes = tuple(shapes)

        # heads: unit-variance outputs from the unit-variance trunk output
        heads = []
        for shape, k in zip(self.shapes, head_keys):
            n_out = shape.fan_in * shape.fan_out + shape.fan_out
            head = eqx.nn.Linear(trunk_dim, n_out, key=k)
            weight = jax.random.normal(k, head.weight.shape) / np.sqrt(trunk_dim)
            head = eqx.tree_at(lambda h: (h.weight, h.bias), head, (weight, jnp.zeros(n_out)))
            heads.append(head)
        self.heads = heads

    def __call__(self, latent: jnp.ndarray) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        latent: (latent_dim,)
        returns: [(W (fan_in, fan_out), b (fan_out,)), ...], one per layer
        """
        x = latent
        for layer in self.trunk:
            x = jax.nn.gelu(layer(x))
        x = self.trunk_norm(x)

        params = []
        for head, shape in zip(self.heads, self.shapes):
            flat = head(x) * shape.scale
            n_w = shape.fan_in * shape.fan_out
            w = flat[:n_w].reshape(shape.fan_in, shape.fan_out)
            b = flat[n_w:]
            params.append((w, b))
        return params


def siren(params: List[Tuple[jnp.ndarray, jnp.ndarray]], pe: jnp.ndarray, w0: float, w0_first: float) -> jnp.ndarray:
    """
    params: the hyper-network's output
    pe: (n_samples, pe_dim)
    returns: (n_samples,)
    """
    x = pe
    *hidden, (w_out, b_out) = params
    for i, (w, b) in enumerate(hidden):
        x = jnp.sin((w0_first if i == 0 else w0) * (x @ w + b))
    return (x @ w_out + b_out)[:, 0]


# ---------------------------------------------------------------------------
# model

class HyperNerf(eqx.Module):
    encoder: Encoder
    latent_norm: eqx.nn.LayerNorm
    hyper: HyperNetwork
    pe: jnp.ndarray

    w0: float = eqx.field(static=True)
    w0_first: float = eqx.field(static=True)

    def __init__(
            self,
            n_samples: int,
            window: int,
            hop: int,
            channels: int,
            latent_dim: int,
            trunk_dim: int,
            pe_dim: int,
            hidden: int,
            n_layers: int,
            w0: float,
            w0_first: float,
            key):
        self.w0 = w0
        self.w0_first = w0_first

        enc_key, hyper_key = jax.random.split(key)
        self.encoder = Encoder(n_samples, window, hop, channels, latent_dim, key=enc_key)
        self.latent_norm = eqx.nn.LayerNorm(latent_dim)
        self.hyper = HyperNetwork(latent_dim, trunk_dim, pe_dim, hidden, n_layers, w0, w0_first, key=hyper_key)
        self.pe = sinusoidal_table(n_samples, pe_dim)

    def encode(self, audio: jnp.ndarray) -> jnp.ndarray:
        """(n_samples,) -> (latent_dim,), layer-normalized"""
        return self.latent_norm(self.encoder(audio))

    def decode(self, latent: jnp.ndarray) -> jnp.ndarray:
        """(latent_dim,) -> (n_samples,)"""
        return siren(self.hyper(latent), self.pe, self.w0, self.w0_first)

    def __call__(self, audio: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        latent = self.encode(audio)
        return self.decode(latent), latent


# ---------------------------------------------------------------------------
# training

def make_compute_loss(args):
    log_mel = make_log_mel_spectrogram(
        args.samplerate, n_fft=args.loss_window, hop_length=args.loss_hop, n_mels=args.n_mels)

    def compute_loss(model: HyperNerf, batch: jnp.ndarray):
        """
        batch: (batch, n_samples)
        """
        recon, latent = jax.vmap(model)(batch)
        target_mel = jax.vmap(log_mel)(batch)
        recon_mel = jax.vmap(log_mel)(recon)
        loss = jnp.abs(target_mel - recon_mel).mean()
        return loss, {'recon': recon, 'recon_mel': recon_mel, 'latent': latent}

    return compute_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'overfit'], required=True)
    parser.add_argument('--batch-size', type=int, default=4, help='ignored in overfit mode')
    parser.add_argument('--n-samples', type=int, default=2 ** 15)
    parser.add_argument('--samplerate', type=int, default=22050)
    parser.add_argument('--window', type=int, default=2048, help='encoder STFT window size')
    parser.add_argument('--hop', type=int, default=256,
                        help='encoder STFT hop size; n_samples // hop must be a power of two')
    parser.add_argument('--channels', type=int, default=256, help='encoder channels')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--trunk-dim', type=int, default=256, help="hyper-network trunk's width")
    parser.add_argument('--pe-dim', type=int, default=64, help='learned positional encoding size')
    parser.add_argument('--hidden', type=int, default=64, help="generated SIREN's width")
    parser.add_argument('--n-layers', type=int, default=4, help="generated SIREN's sine layers")
    parser.add_argument('--w0', type=float, default=30., help='SIREN frequency scale, hidden layers')
    parser.add_argument('--w0-first', type=float, default=30., help='SIREN frequency scale, first layer')
    parser.add_argument('--loss-window', type=int, default=2048)
    parser.add_argument('--loss-hop', type=int, default=256)
    parser.add_argument('--n-mels', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collection', type=str, default='jax_hypernerf')
    parser.add_argument('--port', type=int, default=9998)
    args = parser.parse_args()

    if args.n_samples % args.hop != 0:
        raise ValueError('--n-samples must be divisible by --hop')
    if args.pe_dim % 2 != 0:
        raise ValueError('--pe-dim must be even')

    collection = LmdbCollection(path=args.collection)

    orig_audio, recon_audio, random_audio = loggers(
        ['orig', 'recon', 'random-latent'],
        'audio/wav',
        lambda x: encode_audio(x, samplerate=args.samplerate),
        collection)

    recon_mel_logger, = loggers(
        ['recon-mel'],
        SupportedContentType.Spectrogram.value,
        lambda x: x,
        collection,
        NumpySerializer(),
        NumpyDeserializer())

    serve_conjure(
        [orig_audio, recon_audio, random_audio, recon_mel_logger],
        port=args.port,
        n_workers=1,
        web_components_version='0.0.101')

    model_key, random_key = jax.random.split(jax.random.key(args.seed))

    model = HyperNerf(
        n_samples=args.n_samples,
        window=args.window,
        hop=args.hop,
        channels=args.channels,
        latent_dim=args.latent_dim,
        trunk_dim=args.trunk_dim,
        pe_dim=args.pe_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
        w0=args.w0,
        w0_first=args.w0_first,
        key=model_key)

    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.lr)),
        max_consecutive_errors=100)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    compute_loss = make_compute_loss(args)

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    @eqx.filter_jit
    def decode_random(model, key):
        """
        Decode a random latent; latents are layer-normalized, so a standard
        normal vector is roughly in-distribution.
        """
        return model.decode(model.latent_norm(jax.random.normal(key, (args.latent_dim,))))

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f'jax backend: {jax.default_backend()}, devices: {jax.devices()}, params: {n_params:,}')

    stream = batch_stream(args.mode, args.batch_size, args.n_samples, args.samplerate)

    for i, batch in zip(itertools.count(), stream):
        model, opt_state, loss, aux = train_step(model, opt_state, batch)

        orig_audio(max_norm(np.asarray(batch[0])))
        recon_audio(max_norm(np.asarray(aux['recon'][0])))

        random_key, render_key = jax.random.split(random_key)
        random_audio(max_norm(np.asarray(decode_random(model, render_key))))

        m = np.asarray(aux['recon_mel'][0])
        m = m - m.min()
        recon_mel_logger(m / (m.max() + 1e-12))

        latent = np.asarray(aux['latent'])
        print(f'{i} mel loss {float(loss):.4f}  latent spread {latent.std(axis=0).mean():.3f}')


if __name__ == '__main__':
    main()
