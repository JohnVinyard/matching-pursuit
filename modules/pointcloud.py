'''
(ai, j, p, a)

Each event is in the form
(atom_index, batch, position, amplitude)

transformers should intake a concatenation of:
- (band, atom_embedding, position_embedding, amplitude_embedding)

Tasks:
- transformer training objective (correct atom, relative time, relative magnitude)
- batch HPSS algorithm
- set autoencoder, using alignment loss
'''
import torch
import numpy as np
from collections import defaultdict


def greedy_set_alignment(
        a: torch.Tensor,
        z: torch.Tensor,
        return_indices=False,
        full_feature_count=None):

    batch, n_elements, n_features = a.shape
    diff = torch.cdist(a, z)
    indices = torch.argsort(diff, dim=-1)

    output_indices = []

    for b in range(batch):
        seen = set()
        sort_indices = []

        for e in range(n_elements):
            fits = indices[b, e, :]
            i = 0

            while fits[i].item() in seen:
                i += 1

            sort_indices.append(fits[i].item())
            seen.add(fits[i].item())

        sort_indices = torch.from_numpy(np.array(sort_indices))[
            None, ...].to(a.device)
        output_indices.append(sort_indices)

    output_indices = torch.cat(output_indices, dim=0)
    output_indices = output_indices[..., None].repeat(
        1, 1, full_feature_count or n_features)

    if return_indices:
        return output_indices
    
    arranged = torch.gather(z, dim=1, index=output_indices)
    return arranged


def encode_events(
        event_dict: 'dict[int, tuple[int, int, float, float]]', 
        n_atoms: int, 
        dict_size: int):
    """
    Take a dictionary encoding:
        band_size -> [(index, batch, position, atom),]
    And convert it to a tensor of shape
        (batch, 3, n_events)
    """
    x = []
    i = 0

    for size, events in event_dict.items():
        atom_index, batch, position, amplitude = zip(*events)
        batch_size = max(batch) + 1

        # positions should be in the range 0-1
        p = torch.cat(position).float().view(batch_size, 1, -1) / size

        a = torch.cat(amplitude).float().view(batch_size, -1, n_atoms)
        a = torch.norm(a, dim=1, keepdim=True)

        at = torch.from_numpy(np.array(atom_index) + (i * dict_size)).long()
        at = at.view(batch_size, 1, -1).to(a.device)

        s = torch.zeros_like(at).fill_(size)

        z = torch.cat([at, p, a, s], dim=1)
        x.append(z)
        i += 1

    x = torch.cat(x, dim=-1)

    # sort all elements by ascending time
    p = x[:, 1:2, :]
    indices = torch.argsort(p, dim=-1).repeat(1, 4, 1)
    x = torch.gather(x, dim=-1, index=indices)

    # take the amplitude and position difference between successive atoms
    # pos_amp = x[:, 1:3, :]
    # pos_amp = torch.cat([torch.zeros(pos_amp.shape[0], pos_amp.shape[1], 1, device=pos_amp.device), pos_amp], dim=-1)
    # pos_amp = torch.diff(pos_amp, dim=-1)
    # x[:, 1:3, :] = pos_amp

    return x


def decode_events(events: torch.Tensor, band_dicts: 'dict[int, torch.Tensor]', n_atoms: int, dict_size: int):
    """
    Take a tensor of shape
        (batch, 4, n_events) - channels being (atom_index, position, amplitude, band_size)
    And conver it to a dictionary encoding:
        band_size => [(index, batch, position, atom),]
    """
    batch, _, n_events = events.shape

    # regain absolute positions and magnitudes
    # pos_amp = events[:, 1:3, :]
    # pos_amp = torch.cumsum(pos_amp, dim=-1)
    # events[:, 1:3, :] = pos_amp

    events = events.view(-1, 4, n_events).permute(2,
                                                  0, 1)  # (n_events, batch, 4)
    event_dict = defaultdict(list)

    size_index = {size: i for i, size in enumerate(band_dicts.keys())}

    for i, event in enumerate(events):
        # (batch, 4)
        for b, item in enumerate(event):
            # (4,)
            ai = int(item[0].item())
            p = item[1].item()
            a = item[2].item()
            s = item[3].item()

            # positions should be a number of samples
            p *= s

            i = size_index[int(s)]
            ai = ai % dict_size

            a = band_dicts[s][ai] * a

            event_dict[s].append((ai, b, p, a))

    return event_dict


if __name__ == '__main__':

    a = torch.zeros(1, 7, 3).uniform_(-1, 1)
    b = torch.zeros(1, 7, 3).uniform_(-1, 1)

    srt = greedy_set_alignment(a, b)

    dist = ((a - srt) ** 2).mean()
    rand_dist = ((a - b) ** 2).mean()
    print('=========================')
    print(dist.item())
    print(rand_dist.item())


