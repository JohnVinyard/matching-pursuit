Working on better sparsty-inducing stuff

# Questions

Is sparsity all that important to my project?


What I want is a representation that can't help but produce musical audio.

Even though the sparse representation uses its own context, it can't help but be
severely limited.



# Things to Try

- does increasing the amount of noise/dropout early in training allow gradient to flow better?
- could a straight-through estimator help ( I don't think so )
- 

# Diagnostic Tools

- an interactive way to play individual events would be helpful

# Thoughts

Matching pursuit naturally leads to a jagged, sparse representation, which is what we want.

Ideally, we want to arrive at something similar, all in one shot

Individual active atoms are risk-averse.  An easy way to minimze risk is to only sound for the current
frame

To be widely applicable, atoms of a fixed-size dictionary must generally be short;  the longer the
sequence, the more variability there could be.  I think this is why my current sparse models still
tend toward a frame-based representation.