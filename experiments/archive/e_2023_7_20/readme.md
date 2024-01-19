Try out a few ideas that have a low bar to entry, and keep the sound encoding simple:

## Mu-law encoding prior to sparse coding
This makes things even noisier

## Local Contrast norm to pick the best atom at each step
Taking 9x9 patches and subtracting the local average before selecting the best atom
might actually make things a little more consistent with less noise and gaps

## Dense Network with Sparsification Bottleneck