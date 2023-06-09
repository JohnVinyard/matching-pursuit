import torch
from torch import jit
from torch import nn
from torch.nn import functional as F


def soft_dirac(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = torch.softmax(x, dim=dim)

    values, indices = torch.max(x, dim=dim, keepdim=True)

    output = torch.zeros_like(x, requires_grad=True)
    ones = torch.ones_like(values, requires_grad=True)

    output = torch.scatter(output, dim=dim, index=indices, src=ones)

    forward = output
    backward = x

    y = backward + (forward - backward).detach()
    return y



def sparsify(x, n_to_keep, return_indices=False, soft=False, sharpen=False, salience=None):

    orig = x
    orig_shape = x.shape

    if sharpen:
        x = x.view(-1, 1, x.shape[1], x.shape[-1])
        pooled = F.avg_pool2d(x, (9, 27), stride=(1, 1), padding=(4, 13))

        sharpened = x - pooled

        sharpened = sharpened.view(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
    elif salience is not None:
        sharpened = salience.view(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
    else:
        x = x.reshape(x.shape[0], -1)
        sharpened = x

    # get peaks from sharpened
    values, indices = torch.topk(sharpened, n_to_keep, dim=-1)
    # get values from original
    values = torch.gather(x, dim=-1, index=indices)

    out = torch.zeros_like(x, requires_grad=True)
    out = torch.scatter(out, dim=-1, index=indices, src=values)
    out = out.reshape(*orig_shape)

    if salience is not None:
        salience = salience.view(*out.shape)
        out = out * salience

    if soft:
        backward = orig
        backard_norm = torch.norm(backward, dim=(1, 2), keepdim=True)
        backward = backward / (backard_norm + 1e-12)

        forward = out
        forward_norm = torch.norm(forward, dim=(1, 2), keepdim=True)
        backward = backward * forward_norm

        forward = backward + (forward - backward).detach()
        out = forward

    if return_indices:
        return out, indices, values
    else:
        return out


def to_sparse_vectors_with_context(x, n_elements):
    batch, channels, time = x.shape

    one_hot = torch.zeros(batch, n_elements, channels, device=x.device)
    indices = torch.nonzero(x)

    # time positions at which non-zero elements occur
    positions = [idx[-1] for idx in indices]

    for i in range(indices.shape[0]):
        batch = i // n_elements
        el = i % n_elements
        index = indices[i]
        val = x[tuple(index)]
        one_hot[batch, el, index[1]] = val

    context = torch.sum(x, dim=-1, keepdim=True).repeat(1,
                                                        1, n_elements).permute(0, 2, 1)

    return one_hot, context, positions


def sparsify_vectors(x, attn, n_to_keep, normalize=True, dense=False):
    batch, channels, time = x.shape

    attn = attn.view(batch, time)
    values, indices = torch.topk(attn, k=n_to_keep, dim=-1)

    if normalize:
        values = values + (1 - values)

    if dense:
        output = torch.zeros_like(x)

    latents = []
    for b in range(batch):
        for i in range(n_to_keep):
            latent = x[b, :, indices[b, i]][None, :]
            v = values[b, i]
            if dense:
                output[b, indices[b, i]] = latent
            else:
                latents.append(latent * v.view(1, 1, 1))

    if dense:
        return output
    else:
        latents = torch.cat(latents, dim=0).view(batch, n_to_keep, channels)
        return latents, indices


def to_key_points_one_d(fm: torch.Tensor, n_to_keep: int = 64) -> torch.Tensor:
    points = []
    batch, channels, time = fm.shape


    sp, indices, values = sparsify(fm, n_to_keep, return_indices=True)


    rng = torch.linspace(0, 1, time, device=fm.device, requires_grad=True)

    for i in range(batch):
        for j in range(n_to_keep):
            index = indices[i, j]
            value = values[i, j]

            time_index = index % time
            channel_index = index // time

            ch_span = torch.zeros(channels, device=fm.device)
            ch_span[channel_index] = value
            # ch_span = fm[i, :, time_index]
            ch_span = soft_dirac(ch_span)

            span = torch.zeros(time, device=fm.device)
            span[time_index] = value
            # span = fm[i, channel_index, :]
            span = soft_dirac(span)
            span = rng @ span

            vec = torch.cat([value.view(1), span.view(1), ch_span.view(channels)]) 
            points.append(vec)
    
    points = torch.cat(points, dim=0)
    return points.view(batch, -1, channels + 2)


def to_key_points(x: torch.Tensor, n_to_keep: int = 64) -> torch.Tensor:
    points = []
    batch, width, height = x.shape

    orig_x = x

    x = x.reshape(batch, -1)
    indices = torch.argsort(x, dim=-1).flip(dims=(-1,))

    w_range = torch.linspace(0, 1, width, device=x.device, requires_grad=True)
    h_range = torch.linspace(0, 1, height, device=x.device, requires_grad=True)

    for i in range(batch):
        for j in range(n_to_keep):
            index = indices[i, j]

            row_index = index % width
            col_index = index // height

            value = x[i, index]
            width_span = torch.zeros(width, device=x.device)
            height_span = torch.zeros(height, device=x.device)

            # width_span[row_index] = value
            width_span[:] = orig_x[:, :, col_index]
            width_span = soft_dirac(width_span, dim=-1)

            # height_span[col_index] = value
            height_span[:] = orig_x[:, row_index, :]
            height_span = soft_dirac(height_span, dim=-1)

            w_loc = w_range @ width_span
            h_loc = h_range @ height_span

            vec = torch.cat([value.view(1), w_loc.view(1), h_loc.view(1)])[None, :]

            points.append(vec)
    
    points = torch.cat(points, dim=0)


    return points.view(batch, -1, 3)


class AtomPlacement(jit.ScriptModule):
    def __init__(self, n_samples, n_events, step_size):
        super().__init__()
        self.n_samples = n_samples
        self.n_events = n_events
        self.step_size = step_size

    @jit.script_method
    def render(self, x, indices):
        x = x.view(-1, self.n_events, self.n_samples)
        batch = x.shape[0]

        times = indices * self.step_size

        output = torch.zeros(batch, 1, self.n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(self.n_events):
                time = times[b, i]
                output[b, :, time: time + self.n_samples] += x[b, i][None, :]

        output = output[..., :self.n_samples]
        return output


class SparseAudioModel(nn.Module):
    def __init__(self, n_samples, n_atoms, atom_size, to_keep):
        super().__init__()
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.to_keep = to_keep

        self.atoms = nn.Parameter(
            torch.zeros(1, n_atoms, atom_size).uniform_(-0.01, 0.01))
        self.placement = AtomPlacement(
            n_samples, n_atoms, atom_size, to_keep)

    def forward(self, x):
        a = self.atoms.repeat(x.shape[0], 1, 1)
        x = self.placement.forward(x, a)
        return x


class ElementwiseSparsity(nn.Module):
    def __init__(self, model_dim, high_dim=2048, keep=64, dropout=None, softmax=False):
        super().__init__()
        self.expand = nn.Conv1d(model_dim, high_dim, 1, 1, 0)
        self.contract = nn.Conv1d(high_dim, model_dim, 1, 1, 0)
        self.keep = keep
        self.dropout = dropout
        self.softmax = softmax

    def forward(self, x):
        if self.dropout is not None:
            x = torch.dropout(x, self.dropout, self.training)

        x = self.expand(x)

        if self.softmax:
            x = torch.softmax(x, dim=1)

        sparse = sparsify(x, self.keep)
        x = self.contract(sparse)
        return x, sparse


class VectorwiseSparsity(nn.Module):
    def __init__(self, model_dim, keep=16, channels_last=True, dense=True, normalize=False, time=128):
        super().__init__()
        self.channels_last = channels_last
        self.attn = nn.Linear(model_dim, 1)
        self.keep = keep
        self.dense = dense
        self.normalize = normalize

    def forward(self, x):
        if self.channels_last:
            x = x.permute(0, 2, 1)

        batch, channels, time = x.shape

        x = x.permute(0, 2, 1)

        attn = self.attn(x)
        attn = attn.view(batch, time)
        # attn = torch.softmax(attn, dim=1)

        x = sparsify_vectors(
            x.permute(0, 2, 1), attn, n_to_keep=self.keep, dense=self.dense, normalize=self.normalize)

        if not self.dense:
            return x

        if self.channels_last:
            x = x.permute(0, 2, 1)

        return x


# class SparseEncoderModel(nn.Module):
#     def __init__(
#             self,
#             atoms,
#             samplerate,
#             n_samples,
#             model_dim,
#             n_bands,
#             n_events,
#             total_params,
#             filter_bank,
#             scale,
#             window_size,
#             step_size,
#             n_frames,
#             unit_activation,
#             collapse=False,
#             transformer_context=False):

#         super().__init__()
#         self.atoms = atoms
#         self.samplerate = samplerate
#         self.n_samples = n_samples
#         self.model_dim = model_dim
#         self.n_bands = n_bands
#         self.n_events = n_events
#         self.total_params = total_params
#         self.filter_bank = filter_bank
#         self.scale = scale
#         self.window_size = window_size
#         self.step_size = step_size
#         self.n_frames = n_frames
#         self.unit_activation = unit_activation
#         self.collapse = collapse
#         self.transformer_context = transformer_context

#         self.hearing_model = CochleaModel(
#             samplerate,
#             zounds.MelScale(zounds.FrequencyBand(20, samplerate.nyquist - 10), 128),
#             kernel_size=512)

#         self.periodicity = Periodicity(512, 256)

#         self.audio_feature = NormalizedSpectrogram(
#             pool_window=512,
#             n_bins=128,
#             loudness_gradations=256,
#             embedding_dim=64,
#             out_channels=128)

#         if self.transformer_context:
#             encoder = nn.TransformerEncoderLayer(model_dim, 4, model_dim, batch_first=True)
#             self.context = nn.TransformerEncoder(encoder, 6)
#         else:
#             self.context = DilatedStack(model_dim, [1, 3, 9, 27, 1])

#         self.verb = NeuralReverb.from_directory(
#             Config.impulse_response_path(), samplerate, n_samples)

#         self.n_rooms = self.verb.n_rooms


#         self.reduce = nn.Conv1d(scale.n_bands + 33, model_dim, 1, 1, 0)
#         self.norm = ExampleNorm()

#         self.sparse = VectorwiseSparsity(
#             model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)

#         self.to_mix = LinearOutputStack(model_dim, 2, out_channels=1)
#         self.to_room = LinearOutputStack(
#             model_dim, 2, out_channels=self.n_rooms)

#         self.to_params = LinearOutputStack(
#             model_dim, 2, out_channels=total_params)

#         if collapse:
#             self.to_latent = PosEncodedUpsample(
#                 model_dim,
#                 model_dim,
#                 size=n_events,
#                 out_channels=model_dim,
#                 layers=4)

#     def forward(self, x):
#         batch = x.shape[0]

#         # x = self.hearing_model.forward(x)
#         # p = self.periodicity.forward(x)
#         # x = self.audio_feature.forward(x)
#         # print(x.shape, p.shape)

#         x = self.filter_bank.forward(x, normalize=False)
#         x = self.filter_bank.temporal_pooling(
#             x, self.window_size, self.step_size)[..., :self.n_frames]
#         # x = self.norm(x)

#         pos = pos_encoded(
#             batch, self.n_frames, 16, device=x.device).permute(0, 2, 1)

#         x = torch.cat([x, pos], dim=1)
#         x = self.reduce(x)

#         if self.transformer_context:
#             x = x.permute(0, 2, 1)
#             x = self.context(x)
#             x = x.permute(0, 2, 1)
#         else:
#             x = self.context(x)

#         # x = self.norm(x)

#         x, indices = self.sparse(x)

#         orig_verb_params, _ = torch.max(x, dim=1)
#         verb_params = orig_verb_params

#         # expand to params
#         mx = torch.sigmoid(self.to_mix(verb_params)).view(batch, 1, 1)
#         rm = torch.softmax(self.to_room(verb_params), dim=-1)

#         if self.collapse:
#             x = self.to_latent(orig_verb_params).permute(0, 2, 1)
#             x = x.reshape(-1, self.model_dim)

#         params = self.unit_activation(self.to_params(x))
#         atoms = self.atoms(params)

#         wet = self.verb.forward(atoms, torch.softmax(rm, dim=-1))
#         final = (mx * wet) + ((1 - mx) * atoms)
#         return final
