Going back to the *very* beginning, can I replace
    - the direct convolution with a model that predicts the results of that convolution (approximately)
    - sample loss with a perceptual loss

and adds reverb to avoid lots of noise in the atom domain

Doing just (multi-band) sample-based loss results in a calliope sound, i.e., just fundamental frequencies for each band, I'm assuming

# Ideas

## Use Perceptual Rather than Sample Loss
- also try this with dropout if the first experiment doesn't work

## More Atoms

## Initialize to Waveforms

## Select Impulse and Transfer Functions to generate dictionary

## Sparse Generator from Perceptual Feature

## Hierarchical Matching Pursuit

## Approximate Conv.

## Set-based spectrogram loss