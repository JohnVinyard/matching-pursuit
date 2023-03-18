'''
(ai, j, p, a)

Each event is in the form
(atom_index, batch, position, amplitude)

transformers should intake a concatenation of:
- (band, atom_embedding, position_embedding, amplitude_embedding)

Tasks:
- transform point clouds for transformer (or other GNN) consumption
- transformer training objective (correct atom, relative time, relative magnitude)
- greedy point cloud alignment and loss
- batch HPSS algorithm
- set autoencoder, using alignment loss
'''
import torch
import numpy as np
from collections import defaultdict


def encode_events(event_dict: 'dict[int, tuple[int, int, float, float]]', n_atoms: int):
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

        p = torch.cat(position).float().view(batch_size, 1, -1)

        a = torch.cat(amplitude).float().view(batch_size, -1, n_atoms)
        a = torch.norm(a, dim=1, keepdim=True)

        at = torch.from_numpy(np.array(atom_index) + (i * n_atoms)).long()
        at = at.view(batch_size, 1, -1).to(a.device)

        s = torch.zeros_like(at).fill_(size)

        z = torch.cat([at, p, a, s], dim=1)
        x.append(z)
        i += 1
    
    x = torch.cat(x, dim=-1)
    return x

def decode_events(events: torch.Tensor, band_dicts: 'dict[int, torch.Tensor]', n_atoms: int):
    """
    Take a tensor of shape
        (batch, 4, n_events) - channels being (atom_index, position, amplitude, band_size)
    And conver it to a dictionary encoding:
        band_size => [(index, batch, position, atom),]
    """
    batch, _, n_events = events.shape
    events = events.view(-1, 4, n_events).permute(2, 0, 1) # (n_events, batch, 4)
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

            i = size_index[s]
            ai -= i * n_atoms

            a = band_dicts[s][ai] * a

            event_dict[s].append((ai, b, p, a))
    
    return event_dict

    


