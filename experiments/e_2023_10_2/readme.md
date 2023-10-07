Trying once more for a graph-based representation.  Getting the sparse model
to take advantage of resonance and respect cause and effect is my goal

Sept 4th experiment has a decent resonance model already, but takes the approach
of using context to generate a dictionary which is then convolved with the sparse representation

instead, the one_hot vectors + context should be used to generate all resonance model parameters


# Next Steps

- "easy" conv model with more correct matching pursuit loss 
  (each channel's loss computed in the absence of all others)

- point-cloud generator using dense/dirac scheduling.  What kind of clustering is possible?

- pure conv generator (no resonance/physical modelling) with losses to enforce
  causality and independence (few or no correlations between codes)

- MLP/FFT mixer architecture.  Anti-causal masking?

- sparse scheduling will make runs of the same atom less likely?

- this is harder, because gradients must flow back to the "cause", rather than just the current moment