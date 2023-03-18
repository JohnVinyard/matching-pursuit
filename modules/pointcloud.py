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


def encode_events(event_dict: 'dict[int, tuple[int, int, float, float]]', n_atoms: int):
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

        z = torch.cat([at, p, a], dim=1)
        x.append(z)
        i += 1
    
    x = torch.cat(x, dim=-1)
    return x