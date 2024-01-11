Problems to Solve:

- lots of redundant events

Questions to answer about current model performance:

- Is the single-channel loss to blame, what about orthogonal MSE?
- NERF-style networks tend to be fairly constant, would they be good at the kind of events I'm after?
- Would it be better to generate single events solely with "classic" convolutional filters
- could I use a U-net followed by a large and long bank of filters to get around the fixed event count? (l1 loss)


Things to try:

- plain MSE (is model regularization enough?)
- local contrast normalized spectrogram
- MSE loss with atoms being pushed apart
- maximizing norm reduction at each step
- MSE with energy penalty
- dense labels for spectogram -> cluster -> match event with nearest cluster


This could be a transform like `(batch, f0, octave, time)`

## Possibly Unstable Due to Adversarial Process
The problem with norm reduction at each step is that once an event has reached perfection, it will continue to try to explain more of the audio.  **The only way to fix this is to always sort channels by descending norm**.  We set up a competition between events, and maybe
and equilibrium is reached and we just oscillate at a poor loss?
    - PRO: the random order encourages channels to be independent
    - 

The problem with energy penalty is that it might be possible for the model to "make up" the volume in another parameter, and effectively change nothing.  We'd have to take great care that is wasn't the case, by making sure the resonance energy is entirely dependent on the impulse.