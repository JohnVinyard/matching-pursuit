Trying once more for a graph-based representation.  Getting the sparse model
to take advantage of resonance and respect cause and effect is my goal

Sept 4th experiment has a decent resonance model already, but takes the approach
of using context to generate a dictionary which is then convolved with the sparse representation

instead, the one_hot vectors + context should be used to generate all resonance model parameters


# TODO
- Try out energy loss with pif and stft losses
- try out an anti-causal analysis step, i.e., only samples from _after_ the time-step in the feature
  map have contributed to that atom.
- Is the noise model overly complex?  What if it was simpler?


# Model Alternative

- one monolithic bank of atoms
- `N` resonance steps
- latent vector generates `N` "routing" vectors and `N` mixes

```
signal + (signal . resonance)
  |         |
  -----------
      [mix]
        |
        V
  
```


# Loss Alternative
From loudest to quietest channel, minimize the sum of ratios `new_norm / old_norm` at each step

## Question

Does it matter whether we order the channels first?

