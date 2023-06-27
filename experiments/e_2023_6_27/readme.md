Start simply and figure out where scheduling goes wrong


TODO: hierarchical scheduling?

# Findings

Using softmax for training and a "hard" function for inference doesn't work as planned, 
because the model learns to use artifacts from the fuzzy scheduling to perform the reconstruction

## Mitigating Strategies to Test
- use hard function during training
    - `soft_dirac`
    - `hard_softmax`
- randomly interpolate between hard and soft, so the model cannot learn to rely on the artifacts
- coarse to fine scheduling (does this really address the problem?)
