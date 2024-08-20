# TODO

## Experiment
- lose the context vector entirely;  everything depends on the event vector
- bump up sparsity penalty a bit
- try out other resonance model, just for fun (mostly pretty good, but I believe the original still performs better)


## Report UI

- use audio buffers rather than audio elements to play individual sounds
- show low-res view of audio "channels", not just event positions
- update architecture diagram



## Negative Results
- try out info/entropy-based loss again and see where it gets us


Possible Reasons the last run didn't work:

- I forgot to delete the discriminator weights before starting;  it was too powerful at the outset
- My changes to reverb have caused issues
- sparsity is too high
- there's something about `ResonanceModel2` and the current sparsity that don't go well together

Actual Reason is that there's something about `ResonanceModel2` and the current sparsity that don't play
well together, the sparsity penalty for event amplitudes was overtaking quality reconstructions


## Next Up

Understand why models do not capture violin vibrato.

### Theories:

- static resonances do not provide enough flexibility.  If I were to expand the total number of static
  resonances, perhaps this problem would go away
- in theory, a bank of oscillators with an f0 parameter would work better
- I could also try the stft based resonance model
- I could try a NERF-based model


