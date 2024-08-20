Try out the new spectral info loss model

Ideally, this would be covariance across the _entire_ spectrogram.  In lieu of that,
take a random subset at each batch, whatever number of samples is manageable.

# Things to Investigate
- reconstructions are reasonable
- codes are diverse
- frequency/weight makes sense
- visualize information content in a batch after learning for a bit


# Avenues of Improvement
- multiple patch sizes
- multiple levels of residual
- pyramidal decomposition
- some kind of frequency embedding OR constant-Q spectrogram
- try this with pif feature, so that patches are (16, 16, 257)...eek


Octave bank would be `(batch, 12, n_banks, n_frames)`
PIF feature is `(batch, n_banks, n_frames, periodicity)`