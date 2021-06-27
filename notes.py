import numpy as np


if __name__ == '__main__':
    samples = 512
    components = 17
    batch = 8

    d = np.zeros((samples, components))
    sig = np.zeros((samples, batch))
    code = np.zeros((components, batch))

    residual = -(np.matmul(d, code)) + sig
    print('residual', residual.shape, '(samples, batch)')

    k = 1
    
    # add activations for a single atom back to the residual
    # this is as if we had removed all other atoms except this one
    atom_r = np.outer(d[:, k], code[k, :]) + residual

    print('atom_residual', atom_r.shape)
    print('code for single atom', code[k, :].shape)

    # update the atom
    updated_atom = np.dot(atom_r, code[k, :])
    print(updated_atom.shape, '(samples)')

    # subtract the updated atom from the residual