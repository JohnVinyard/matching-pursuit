import numpy as np
from sklearn.decomposition import PCA, NMF
import torch
from itertools import product

def learn_atom_embeddings(atoms, mags, embedding_size, atom_dict):

    atoms = np.concatenate(list(atom_dict.values()), axis=0)
    print(atoms.shape)
    atoms = np.abs(np.fft.rfft(atoms, axis=-1))
    atom_embeddings = atoms

    # atom_embeddings = torch.zeros((3072, 3072))

    # for i, seq in enumerate(atoms):
    #     seq = torch.argmax(seq, axis=-1)
    #     print(i)
    #     # TODO: Should I be using the magnitude to denote "relevance?"
    #     seq = list(x.item() for x in seq)
    #     indices = list(zip(*product(seq, seq)))
    #     atom_embeddings[indices[0], indices[1]] += 1
    
    # atom_embeddings /= (atom_embeddings.std() + 1e-12)

    pca = PCA(n_components=embedding_size)
    print('Starting to learn PCA')
    pca.fit(atom_embeddings)
    coeffs = pca.transform(atom_embeddings)
    recon_embeddings = pca.inverse_transform(coeffs)
    # coeffs = unit_norm(coeffs)

    coeffs -= coeffs.mean()
    coeffs /= coeffs.std()
    return atom_embeddings, recon_embeddings, coeffs