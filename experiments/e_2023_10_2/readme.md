Trying once more for a graph-based representation.  Getting the sparse model
to take advantage of resonance and respect cause and effect is my goal

Sept 4th experiment has a decent resonance model already, but takes the approach
of using context to generate a dictionary which is then convolved with the sparse representation

instead, the one_hot vectors + context should be used to generate all resonance model parameters


# TODO
- better frequency resolution in loss [X]
- try without dense context vector [X]
- add RELU + l1 loss for encoding sparsity


# Next Steps

- "easy" conv model with more correct matching pursuit loss 
  (each channel's loss computed in the absence of all others)

- point-cloud generator using dense/dirac scheduling.  What kind of clustering is possible?

- pure conv generator (no resonance/physical modelling) with losses to enforce
  causality and independence (few or no correlations between codes)

- better matching pursuit loss, stopping gradients when subtracting all other channels

- MLP/FFT mixer architecture.  Anti-causal masking?

- sparse scheduling will make runs of the same atom less likely?

- this is harder, because gradients must flow back to the "cause", rather than just the current moment

- 1d projection plus lateral inhibition


# Observations

It's pretty clear that gradients are much more likely to flow to _samples_, rather
than the control signal, e.g., a stem might end up pushed to have a transient in its
center because it was poorly positioned.  This suggests that the sparse control signal
and the stem-producing modules should be optimized separately.

## Ideas


### Greedy Algorithm

produce and remove one stem at a time

### Just Stems

Produce a control signal and stems.  We find the best position for each stem and compute loss one-by-one, independently.
Additionally, we compare the sparse control signal we produced to the optimal positioning for each stem

