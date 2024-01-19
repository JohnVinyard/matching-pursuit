Start simply and figure out where scheduling goes wrong


TODO: hierarchical scheduling?

# Things to Try
- `fft_shift` 🙄
- positional encodings
- coarse-to-fine scheduling
- coarse-to-fine loss

# Findings

Using softmax for training and a "hard" function for inference doesn't work as planned, 
because the model learns to use artifacts from the fuzzy scheduling to perform the reconstruction


## Mitigating Strategies to Test
- use hard function during training
    - `soft_dirac`
    - `hard_softmax`
- randomly interpolate between hard and soft, so the model cannot learn to rely on the artifacts
- coarse to fine scheduling (does this really address the problem?)

## Working Model

As of git commit 76a109030738b35157a4d606279cc1b2949a3b65, I have a model that
reconstructs synthetic generations

## Next Steps

- Add reverb in generating process and in model
- add bandpass-filtered noise in generating model and the same parameterized generator in model

### Question

How do I ensure that the model is not leveraging artifacts of the softmax for reconstructions.

IDEA: For each batch, choose a random linear combination of the `softmax` and `soft_dirac`.
This _should_ encourage the model to prefer sparser solutions that resemble `soft_dirac`, rendering
the linear combination effect-less